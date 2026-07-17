"""Bounded multilingual graph + episodic context refreshed per user turn.

The graph remains authoritative durable knowledge. Relevant session summaries
are separately labeled narrative context and never participate in graduation.
Embeddings and all owner text remain in the local SQLite/data directory.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypedDict

import numpy as np

from therapy.knowledge.embeddings import (
    EmbeddingBackend,
    FastEmbedBackend,
    cosine_similarity,
)
from therapy.knowledge.insight import DeliveredInsight, InsightRecord, InsightService
from therapy.knowledge.research import ResearchKB
from therapy.knowledge.user_model import (
    GraphContext,
    GraphEdge,
    GraphNode,
    UserModel,
    _tokens,
    render_context,
)
from therapy.memory.store import MemoryStore, RowDict

MEMORY_MARKER = "# Longitudinal context (refreshed per turn)"


class EpisodicContext(TypedDict):
    """Retrieved narrative summary, never an established graph claim."""

    session_id: str
    started_at: str
    summary: str
    relevance: float


class TurnContext(TypedDict):
    """One bounded memory refresh result."""

    note: str | None
    graph: GraphContext
    episodes: list[EpisodicContext]
    insight: DeliveredInsight | None
    resolution: InsightRecord | None
    research: str


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Strict per-turn memory limits."""

    max_nodes: int = 10
    max_edges: int = 10
    max_episodes: int = 3
    max_tokens: int = 1_200
    node_threshold: float = 0.46
    episode_threshold: float = 0.52

    def __post_init__(self) -> None:
        if not 1 <= self.max_nodes <= 50:
            raise ValueError("max_nodes must be between 1 and 50")
        if not 0 <= self.max_edges <= 50:
            raise ValueError("max_edges must be between 0 and 50")
        if not 0 <= self.max_episodes <= 10:
            raise ValueError("max_episodes must be between 0 and 10")
        if not 100 <= self.max_tokens <= 4_000:
            raise ValueError("max_tokens must be between 100 and 4000")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ContextAssembler:
    """Semantic graph/episode retriever with local versioned vector cache."""

    def __init__(
        self,
        model: UserModel,
        store: MemoryStore,
        *,
        embedder: EmbeddingBackend | None = None,
        budget: ContextBudget | None = None,
        insight_service: InsightService | None = None,
        research: ResearchKB | None = None,
    ) -> None:
        self.model = model
        self.store = store
        self.embedder = embedder or FastEmbedBackend(model.data_dir)
        self.budget = budget or ContextBudget()
        self.insights = insight_service or InsightService(model)
        self.research = research or ResearchKB(model.data_dir, embedder=self.embedder)
        self._db_path = model.data_dir / "therapy.db"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
        finally:
            connection.close()

    def _vectors(
        self, kind: str, items: Sequence[tuple[str, str]]
    ) -> dict[str, np.ndarray]:
        metadata = self.embedder.metadata
        result: dict[str, np.ndarray] = {}
        missing: list[tuple[str, str]] = []
        with self._connect() as connection:
            for entity_id, content in items:
                digest = _content_hash(content)
                row = connection.execute(
                    """
                    SELECT vector, dimension, content_hash
                    FROM semantic_embeddings
                    WHERE entity_kind = ? AND entity_id = ?
                      AND model_name = ? AND model_revision = ?
                    """,
                    (kind, entity_id, metadata.name, metadata.revision),
                ).fetchone()
                if (
                    row is None
                    or row["content_hash"] != digest
                    or int(row["dimension"]) != metadata.dimension
                ):
                    missing.append((entity_id, content))
                    continue
                vector = np.frombuffer(row["vector"], dtype=np.float32).copy()
                if vector.shape != (metadata.dimension,):
                    missing.append((entity_id, content))
                    continue
                result[entity_id] = vector
        if missing:
            embedded = self.embedder.embed_documents(
                [content for _, content in missing]
            )
            if len(embedded) != len(missing):
                raise RuntimeError("embedding backend returned wrong batch length")
            with self._connect() as connection:
                with connection:
                    for (entity_id, content), vector in zip(
                        missing, embedded, strict=True
                    ):
                        if vector.shape != (metadata.dimension,):
                            raise RuntimeError("embedding backend dimension mismatch")
                        normalized = vector.astype(np.float32, copy=False)
                        connection.execute(
                            """
                            INSERT INTO semantic_embeddings (
                                entity_kind, entity_id, model_name, model_revision,
                                dimension, content_hash, vector, indexed_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(
                                entity_kind, entity_id, model_name, model_revision
                            ) DO UPDATE SET
                                dimension = excluded.dimension,
                                content_hash = excluded.content_hash,
                                vector = excluded.vector,
                                indexed_at = excluded.indexed_at
                            """,
                            (
                                kind,
                                entity_id,
                                metadata.name,
                                metadata.revision,
                                metadata.dimension,
                                _content_hash(content),
                                normalized.tobytes(),
                                _utc_now(),
                            ),
                        )
                        result[entity_id] = normalized
        return result

    @staticmethod
    def _record_items(source: str, count: int) -> None:
        """Bounded per-source context evidence (plan O3.2): count buckets
        only; exact selected/rendered context stays restricted."""
        from therapy.observability.model import count_bucket
        from therapy.observability.telemetry import record_metric

        record_metric(
            "therapy_context_items_total",
            1,
            {"source": source, "bucket": count_bucket(count)},
        )

    def assemble(self, topic: str, session_id: str) -> TurnContext:
        """Refresh semantic/episodic memory and adjacent insight for one turn."""
        import time as time_module

        from therapy.observability.telemetry import record_metric

        if not session_id:
            raise ValueError("session_id is required")
        assemble_started = time_module.monotonic()
        outcome = "success"
        try:
            if not topic.strip():
                graph = self.model.assemble_context("")
                return {
                    "note": self._render(graph, [], None),
                    "graph": graph,
                    "episodes": [],
                    "insight": None,
                    "resolution": None,
                    "research": "",
                }
            resolution = self.insights.resolve_conversational(session_id, topic)
            query = self.embedder.embed_query(topic)
            graph = self._graph_context(topic, query)
            episodes = self._episodes(topic, query)
            adjacent_claim_ids = {
                node["id"]
                for key in ("identity", "preferences", "goals", "threads", "walk_nodes")
                for node in graph[key]
            }
            insight = (
                None
                if resolution is not None
                else self.insights.next_for_topic(
                    topic, session_id, adjacent_claim_ids=adjacent_claim_ids
                )
            )
            research = self.research.grounding_context(topic)
            note = self._render(graph, episodes, insight, research, resolution)
            self._record_items("graph", len(adjacent_claim_ids))
            self._record_items("episode", len(episodes))
            self._record_items("insight", 1 if insight else 0)
            self._record_items("research", 1 if research else 0)
            return {
                "note": note,
                "graph": graph,
                "episodes": episodes,
                "insight": insight,
                "resolution": resolution,
                "research": research,
            }
        except Exception:
            outcome = "error"
            raise
        finally:
            record_metric(
                "therapy_context_assembly_seconds",
                time_module.monotonic() - assemble_started,
                {"outcome": outcome},
            )

    def _allows_node(
        self, node: GraphNode, topic: str, semantic_similarity: float
    ) -> bool:
        lowered = topic.casefold()
        statement = node["statement"].casefold()
        private_patterns = [
            value.casefold() for value in self.model.never_initiate_topics()
        ]
        raised = [pattern for pattern in private_patterns if pattern in lowered]
        unraised = [pattern for pattern in private_patterns if pattern not in lowered]
        if any(pattern in statement for pattern in unraised):
            return False
        if not node["never_initiate"]:
            return True
        if any(pattern in statement for pattern in raised):
            return True
        if raised:
            return semantic_similarity >= max(0.85, self.budget.node_threshold)
        return bool(_tokens(topic) & _tokens(node["statement"]))

    def _graph_context(self, topic: str, query: np.ndarray) -> GraphContext:
        candidates = [
            node
            for node in self.model.nodes()
            if node["status"] not in {"rejected", "superseded", "needs_revalidation"}
        ]
        vectors = self._vectors(
            "node", [(str(node["id"]), node["statement"]) for node in candidates]
        )
        scored = sorted(
            (
                (cosine_similarity(query, vectors[str(node["id"])]), node)
                for node in candidates
            ),
            key=lambda item: (-item[0], item[1]["id"]),
        )
        allowed_ids = {
            node["id"]
            for similarity, node in scored
            if self._allows_node(node, topic, similarity)
        }
        candidates = [node for node in candidates if node["id"] in allowed_ids]
        semantic = [
            node
            for score, node in scored
            if node["id"] in allowed_ids and score >= self.budget.node_threshold
        ][: self.budget.max_nodes]
        standing_statuses = {"observation", "proposed", "confirmed"}
        identity = [
            node
            for node in candidates
            if node["type"] == "identity_fact" and node["status"] in standing_statuses
        ][:4]
        preferences = [
            node
            for node in candidates
            if node["type"] == "preference" and node["status"] in standing_statuses
        ][:4]
        goals = [
            node
            for node in candidates
            if node["type"] == "goal" and node["status"] in standing_statuses
        ][:4]
        threads = [
            node
            for node in candidates
            if node["type"] == "thread" and node["status"] in standing_statuses
        ][:4]
        selected: list[GraphNode] = []
        selected_ids: set[int] = set()
        for node in (*identity, *preferences, *goals, *threads, *semantic):
            if node["id"] not in selected_ids and len(selected) < self.budget.max_nodes:
                selected.append(node)
                selected_ids.add(node["id"])
        edges = [
            edge
            for edge in self.model.edges(status="confirmed")
            if edge["src"] in selected_ids or edge["dst"] in selected_ids
        ]
        confirmed_by_id = {
            node["id"]: node for node in candidates if node["status"] == "confirmed"
        }
        selected_edges: list[GraphEdge] = []
        for edge in edges:
            for endpoint in (edge["src"], edge["dst"]):
                if endpoint in selected_ids:
                    continue
                neighbor = confirmed_by_id.get(endpoint)
                if neighbor is not None and len(selected) < self.budget.max_nodes:
                    selected.append(neighbor)
                    selected_ids.add(endpoint)
            if edge["src"] in selected_ids and edge["dst"] in selected_ids:
                selected_edges.append(edge)
            if len(selected_edges) >= self.budget.max_edges:
                break
        standing_ids = {
            node["id"] for node in (*identity, *preferences, *goals, *threads)
        }
        return {
            "identity": identity,
            "preferences": preferences,
            "goals": goals,
            "threads": threads,
            "never_initiate": self.model.never_initiate_topics(),
            "walk_nodes": [node for node in selected if node["id"] not in standing_ids],
            "walk_edges": selected_edges,
        }

    def _episodes(self, topic: str, query: np.ndarray) -> list[EpisodicContext]:
        rows = self.store.episodic_summaries()
        boundary_patterns = [
            value.casefold() for value in self.model.never_store_topics()
        ]
        lowered = topic.casefold()
        boundary_patterns.extend(
            value.casefold()
            for value in self.model.never_initiate_topics()
            if value.casefold() not in lowered
        )
        rows = [
            row
            for row in rows
            if not any(
                pattern in str(row["summary"] or "").casefold()
                for pattern in boundary_patterns
            )
        ]
        vectors = self._vectors(
            "episode", [(str(row["id"]), str(row["summary"])) for row in rows]
        )
        scored: list[tuple[float, RowDict]] = []
        now = datetime.now(UTC)
        for row in rows:
            similarity = cosine_similarity(query, vectors[str(row["id"])])
            started = datetime.fromisoformat(str(row["started_at"]))
            if started.tzinfo is None:
                started = started.replace(tzinfo=UTC)
            recency = max(0.0, 1.0 - ((now - started).days / 365.0)) * 0.05
            scored.append((similarity + recency, row))
        scored.sort(
            key=lambda item: (-item[0], str(item[1]["started_at"])), reverse=False
        )
        episodes: list[EpisodicContext] = [
            EpisodicContext(
                session_id=str(row["id"]),
                started_at=str(row["started_at"]),
                summary=str(row["summary"]),
                relevance=round(score, 4),
            )
            for score, row in scored
            if score >= self.budget.episode_threshold
        ][: self.budget.max_episodes]
        return episodes

    def _render(
        self,
        graph: GraphContext,
        episodes: list[EpisodicContext],
        insight: DeliveredInsight | None,
        research: str = "",
        resolution: InsightRecord | None = None,
    ) -> str | None:
        graph_note = render_context(graph)
        sections = [MEMORY_MARKER]
        if graph_note:
            sections.append(graph_note)
        if episodes:
            sections.append(
                "# Relevant episodic narrative (context, not established truth)\n"
                + "\n".join(
                    f"- [{episode['started_at'][:10]}] {episode['summary']}"
                    for episode in episodes
                )
            )
        if research:
            sections.append("# Silent technique grounding\n" + research[:1_600])
        if insight is not None:
            sections.append(
                "# Adjacent reflection to raise once\n"
                + insight["reflection"]
                + "\nWait for an explicit answer; do not infer confirmation."
            )
        if resolution is not None:
            sections.append(
                "# Explicit insight response recorded\n"
                f"The owner {resolution['state']} the exact claim just raised. "
                "Acknowledge this briefly; do not raise another insight this turn."
            )
        if len(sections) == 1:
            return None
        note = "\n\n".join(sections)
        max_chars = self.budget.max_tokens * 4
        if len(note) <= max_chars:
            return note
        clipped = note[: max_chars - 40]
        boundary = clipped.rfind("\n")
        if boundary > max_chars // 2:
            clipped = clipped[:boundary]
        return clipped + "\n[Context clipped to token budget]"


def replace_longitudinal_context(
    messages: list[dict[str, Any]], note: str | None
) -> list[dict[str, Any]]:
    """Replace prior memory block while preserving system prompt and dialogue."""
    filtered = [
        message
        for message in messages
        if not (
            message.get("role") == "system"
            and isinstance(message.get("content"), str)
            and str(message["content"]).startswith(MEMORY_MARKER)
        )
    ]
    if note is None:
        return filtered
    insertion = 1 if filtered and filtered[0].get("role") == "system" else 0
    return [
        *filtered[:insertion],
        {"role": "system", "content": note},
        *filtered[insertion:],
    ]


__all__ = [
    "ContextAssembler",
    "ContextBudget",
    "EpisodicContext",
    "MEMORY_MARKER",
    "TurnContext",
    "replace_longitudinal_context",
]
