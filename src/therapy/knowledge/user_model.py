"""Privacy-safe property-graph self-model (SPEC Appendix A; Phase 4 A–D).

Nodes and edges share one evidence-gated lifecycle. Evidence is normalized and
auditable; model-supplied source authority is rejected; deletions create keyed
digests that survive endpoint recreation without retaining deleted prose.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, NotRequired, Protocol, TypedDict, cast
from uuid import uuid4

from therapy.knowledge.schema import (
    CLAIM_STATUSES,
    EDGE_TYPES,
    LEGACY_EDGE_TYPE_MAP,
    LEGACY_NODE_TYPE_MAP,
    NODE_TYPES,
    TOMBSTONE_KEY_FILE,
    canonical_edge_type,
    canonical_node_type,
    edge_base_tombstone_signature,
    load_or_create_tombstone_key,
    migrate_database,
    normalize_statement,
    tombstone_digest,
)

STATUSES = CLAIM_STATUSES
SOURCES: tuple[str, ...] = ("conversation", "user-stated", "ser")
GRADUATION_MIN_OCCURRENCES = 3
GRADUATION_MIN_SESSIONS = 2

_STATUS_WEIGHT = {
    "confirmed": 4.0,
    "proposed": 2.5,
    "observation": 1.0,
    "needs_revalidation": 0.5,
    "rejected": 0.0,
    "superseded": 0.0,
}
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_MAX_STATEMENT_CHARS = 4_000
_MAX_QUOTE_CHARS = 8_000

type ClaimKind = Literal["node", "edge"]
type GraphMutationOperation = Literal[
    "add_node",
    "add_edge",
    "edit",
    "delete",
    "purge",
    "revalidate",
    "distill_apply",
]
type GraphMutationOutcome = Literal["success", "error", "timeout", "rejected"]
type SuppressionTableClass = Literal["node", "edge", "file", "turn"]


def _record_graph_mutation(
    operation: GraphMutationOperation, outcome: GraphMutationOutcome
) -> None:
    """Record a content-free graph mutation outcome."""
    from therapy.observability import telemetry

    telemetry.record_metric(
        "therapy_graph_mutations_total",
        1,
        {"operation": operation, "outcome": outcome},
    )


def _record_suppression(table_class: SuppressionTableClass, count: int = 1) -> None:
    """Record never-store suppression by bounded storage class only."""
    if count <= 0:
        return
    from therapy.observability import telemetry

    telemetry.record_metric(
        "therapy_never_store_suppressions_total",
        count,
        {"table_class": table_class},
    )


def _graph_error_outcome(error: BaseException) -> Literal["error", "timeout"]:
    """Classify timeouts without inspecting or exposing exception text."""
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, sqlite3.OperationalError) and getattr(
        error, "sqlite_errorcode", None
    ) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
        return "timeout"
    return "error"


class GraphQuote(TypedDict):
    """Verified source quotation attached to claim evidence."""

    text: str
    language: str
    session_id: NotRequired[str]
    turn_id: NotRequired[int]
    observed_at: NotRequired[str]


class ClaimEvidence(TypedDict):
    """Decoded provenance row."""

    id: str
    claim_kind: ClaimKind
    claim_id: int
    source_type: str
    source_session_id: str | None
    source_turn_id: int | None
    observed_at: str
    language: str | None
    quote_text: str | None
    source_state: str
    source_deleted_at: str | None
    source_marker: str | None
    extractor_version: str
    validation_event_id: int | None
    evidence_key: str


class GraphNode(TypedDict):
    """Owner-visible node plus derived evidence counters."""

    id: int
    type: str
    statement: str
    statement_key: str
    status: str
    source: str
    never_initiate: bool
    user_edited: bool
    first_seen: str
    last_seen: str
    n_occurrences: int
    n_sessions: int
    n_auditable_occurrences: int
    n_auditable_sessions: int
    revalidation_due_at: str | None
    superseded_by: int | None
    quotes: list[GraphQuote]
    sessions: list[str]


class GraphEdge(TypedDict):
    """Owner-visible edge claim plus derived evidence counters."""

    id: int
    src: int
    dst: int
    type: str
    statement: str
    statement_key: str
    status: str
    source: str
    user_edited: bool
    first_seen: str
    last_seen: str
    n_occurrences: int
    n_sessions: int
    n_auditable_occurrences: int
    n_auditable_sessions: int
    revalidation_due_at: str | None
    superseded_by: int | None
    quotes: list[GraphQuote]
    sessions: list[str]


class GraphWalk(TypedDict):
    """Topic-relevant graph slice plus initiation boundaries."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    never_initiate: list[str]


class GraphContext(TypedDict):
    """Bounded continuity context rendered into dialogue policy."""

    identity: list[GraphNode]
    preferences: list[GraphNode]
    goals: list[GraphNode]
    threads: list[GraphNode]
    never_initiate: list[str]
    walk_nodes: list[GraphNode]
    walk_edges: list[GraphEdge]


class GraphContextProvider(Protocol):
    """Port for per-turn graph context assembly."""

    def assemble_context(self, topic: str = "", *, k: int = 8) -> GraphContext:
        """Build bounded context for current topic."""
        ...


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text)}


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _fact_kind_to_type(kind: str) -> str:
    mapping = {
        "identity": "identity_fact",
        "identity_fact": "identity_fact",
        "preference": "preference",
        "value": "value",
        "goal": "goal",
        "relationship": "person",
        "person": "person",
        "strength": "strength",
        "strategy": "strategy",
    }
    return mapping.get(kind, "pattern")


class UserModel:
    """SQLite property graph with evidence, lifecycle, and privacy invariants."""

    def __init__(self, data_dir: Path | None = None) -> None:
        """Open graph store and apply ordered migrations."""
        self.data_dir = data_dir or Path(os.environ.get("THERAPY_DATA_DIR", "./data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.data_dir / "therapy.db"
        self.schema_backup_path = migrate_database(self._db_path)
        self._tombstone_key = self._load_tombstone_key()
        self.migrate_v1_facts()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self._db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
        finally:
            connection.close()

    def _load_tombstone_key(self) -> bytes:
        return load_or_create_tombstone_key(self.data_dir)

    def reload_tombstone_key(self) -> None:
        """Reload keyed deletion identity after a validated owner-data restore."""
        self._tombstone_key = self._load_tombstone_key()

    def _digest(self, value: str) -> str:
        return tombstone_digest(self._tombstone_key, value)

    def _node_signature(self, type_: str, statement: str) -> str:
        return f"node:{canonical_node_type(type_)}:{normalize_statement(statement)}"

    def _edge_signature(
        self,
        connection: sqlite3.Connection,
        src: int,
        dst: int,
        type_: str,
        statement: str,
    ) -> str:
        return ":".join(
            (
                self._edge_base_signature(connection, src, dst, type_),
                normalize_statement(statement),
            )
        )

    def _edge_base_signature(
        self,
        connection: sqlite3.Connection,
        src: int,
        dst: int,
        type_: str,
    ) -> str:
        endpoints = connection.execute(
            "SELECT id, type, statement FROM nodes WHERE id IN (?, ?)", (src, dst)
        ).fetchall()
        by_id = {int(row["id"]): row for row in endpoints}
        if src not in by_id or dst not in by_id:
            raise ValueError("Edge endpoints must both exist")
        src_row = by_id[src]
        dst_row = by_id[dst]
        return edge_base_tombstone_signature(
            type_,
            str(src_row["type"]),
            str(src_row["statement"]),
            str(dst_row["type"]),
            str(dst_row["statement"]),
        )

    def _is_tombstoned(
        self, connection: sqlite3.Connection, claim_kind: ClaimKind, signature: str
    ) -> bool:
        row = connection.execute(
            "SELECT 1 FROM tombstones WHERE digest = ? AND claim_kind = ?",
            (self._digest(signature), claim_kind),
        ).fetchone()
        return row is not None

    def _tombstone(
        self, connection: sqlite3.Connection, claim_kind: ClaimKind, signature: str
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO tombstones (digest, claim_kind, created_at)
            VALUES (?, ?, ?)
            """,
            (self._digest(signature), claim_kind, _utc_now()),
        )

    # ------------------------------------------------------------------
    # Boundaries and privacy purge
    # ------------------------------------------------------------------

    def boundaries(self, kind: str | None = None) -> list[dict[str, object]]:
        """Return boundaries, optionally filtered by tier."""
        query = "SELECT id, kind, value, created_at FROM boundaries"
        params: tuple[str, ...] = ()
        if kind is not None:
            if kind not in {"never_store", "never_initiate"}:
                raise ValueError(f"Unknown boundary kind: {kind!r}")
            query += " WHERE kind = ?"
            params = (kind,)
        query += " ORDER BY kind, value"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, params)]

    def _never_store_patterns(self) -> list[str]:
        return [
            str(item["value"]).casefold() for item in self.boundaries("never_store")
        ]

    def never_store_topics(self) -> list[str]:
        """Return owner-configured storage exclusions."""
        return [str(item["value"]) for item in self.boundaries("never_store")]

    def is_never_store(self, text: str) -> bool:
        """Return whether text matches an owner `never_store` boundary."""
        lowered = text.casefold()
        return any(pattern in lowered for pattern in self._never_store_patterns())

    def never_initiate_topics(self) -> list[str]:
        """Return topics assistant must not raise unsolicited."""
        return [str(item["value"]) for item in self.boundaries("never_initiate")]

    def add_boundary(self, kind: str, value: str) -> dict[str, int]:
        """Add boundary and synchronously purge matching data for `never_store`."""
        if kind not in {"never_store", "never_initiate"}:
            raise ValueError(f"Unknown boundary kind: {kind!r}")
        value = value.strip()
        if not value or len(value) > 1_000:
            if kind == "never_store":
                _record_graph_mutation("purge", "rejected")
            raise ValueError("Boundary value must contain 1..1000 characters")
        suppression_counts: dict[SuppressionTableClass, int] = {
            "node": 0,
            "edge": 0,
            "file": 0,
            "turn": 0,
        }
        try:
            with self._connect() as connection:
                with connection:
                    connection.execute(
                        """
                        INSERT INTO boundaries (kind, value, created_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(kind, value) DO NOTHING
                        """,
                        (kind, value, _utc_now()),
                    )
                    counts = {
                        "graph_claims_removed": 0,
                        "inbox_rows_removed": 0,
                        "summaries_removed": 0,
                        "research_documents_removed": 0,
                    }
                    if kind == "never_store":
                        counts = self._purge_boundary(
                            connection, value, suppression_counts
                        )
                        connection.execute(
                            """
                            INSERT INTO privacy_purge_events (
                                boundary_digest, graph_claims_removed,
                                inbox_rows_removed, summaries_removed,
                                research_documents_removed, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                self._digest(f"boundary:{value}"),
                                counts["graph_claims_removed"],
                                counts["inbox_rows_removed"],
                                counts["summaries_removed"],
                                counts["research_documents_removed"],
                                _utc_now(),
                            ),
                        )
        except Exception as error:
            if kind == "never_store":
                _record_graph_mutation("purge", _graph_error_outcome(error))
            raise
        if kind == "never_store":
            for table_class, count in suppression_counts.items():
                _record_suppression(table_class, count)
            _record_graph_mutation("purge", "success")
        return counts

    def _purge_boundary(
        self,
        connection: sqlite3.Connection,
        value: str,
        suppression_counts: dict[SuppressionTableClass, int],
    ) -> dict[str, int]:
        pattern = value.casefold()
        node_ids = {
            int(row["id"])
            for row in connection.execute("SELECT id, statement FROM nodes")
            if pattern in str(row["statement"]).casefold()
        }
        node_ids.update(
            int(row["node_id"])
            for row in connection.execute("SELECT node_id, alias FROM node_aliases")
            if pattern in str(row["alias"]).casefold()
        )
        edge_ids = {
            int(row["id"])
            for row in connection.execute("SELECT id, statement FROM edges")
            if pattern in str(row["statement"]).casefold()
        }
        for evidence in connection.execute(
            """
            SELECT claim_kind, claim_id, quote_text FROM claim_evidence
            WHERE quote_text IS NOT NULL
            """
        ):
            if pattern not in str(evidence["quote_text"]).casefold():
                continue
            if evidence["claim_kind"] == "node":
                node_ids.add(int(evidence["claim_id"]))
            else:
                edge_ids.add(int(evidence["claim_id"]))
        if node_ids:
            placeholders = ",".join("?" for _ in node_ids)
            edge_ids.update(
                int(row["id"])
                for row in connection.execute(
                    f"SELECT id FROM edges WHERE src IN ({placeholders}) "
                    f"OR dst IN ({placeholders})",
                    (*node_ids, *node_ids),
                )
            )
        for node_id in node_ids:
            row = connection.execute(
                "SELECT type, statement FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if row is not None:
                self._tombstone(
                    connection,
                    "node",
                    self._node_signature(str(row["type"]), str(row["statement"])),
                )
        for edge_id in edge_ids:
            row = connection.execute(
                "SELECT * FROM edges WHERE id = ?", (edge_id,)
            ).fetchone()
            if row is not None:
                src = int(row["src"])
                dst = int(row["dst"])
                type_ = str(row["type"])
                self._tombstone(
                    connection,
                    "edge",
                    self._edge_signature(
                        connection, src, dst, type_, str(row["statement"])
                    ),
                )
                self._tombstone(
                    connection,
                    "edge",
                    self._edge_base_signature(connection, src, dst, type_),
                )
        if node_ids:
            placeholders = ",".join("?" for _ in node_ids)
            connection.execute(
                f"DELETE FROM pending_insights WHERE claim_kind = 'node' "
                f"AND claim_id IN ({placeholders})",
                tuple(node_ids),
            )
            connection.execute(
                f"DELETE FROM claim_evidence WHERE claim_kind = 'node' "
                f"AND claim_id IN ({placeholders})",
                tuple(node_ids),
            )
            connection.execute(
                f"DELETE FROM lifecycle_events WHERE claim_kind = 'node' "
                f"AND claim_id IN ({placeholders})",
                tuple(node_ids),
            )
            connection.execute(
                f"DELETE FROM semantic_embeddings WHERE entity_kind = 'node' "
                f"AND entity_id IN ({placeholders})",
                tuple(str(node_id) for node_id in node_ids),
            )
            connection.execute(
                f"DELETE FROM nodes WHERE id IN ({placeholders})", tuple(node_ids)
            )
        if edge_ids:
            placeholders = ",".join("?" for _ in edge_ids)
            connection.execute(
                f"DELETE FROM pending_insights WHERE claim_kind = 'edge' "
                f"AND claim_id IN ({placeholders})",
                tuple(edge_ids),
            )
            connection.execute(
                f"DELETE FROM claim_evidence WHERE claim_kind = 'edge' "
                f"AND claim_id IN ({placeholders})",
                tuple(edge_ids),
            )
            connection.execute(
                f"DELETE FROM lifecycle_events WHERE claim_kind = 'edge' "
                f"AND claim_id IN ({placeholders})",
                tuple(edge_ids),
            )
            connection.execute(
                f"DELETE FROM edges WHERE id IN ({placeholders})", tuple(edge_ids)
            )
        inbox_ids = [
            int(row["id"])
            for row in connection.execute("SELECT id, text FROM observation_inbox")
            if pattern in str(row["text"]).casefold()
        ]
        if inbox_ids:
            placeholders = ",".join("?" for _ in inbox_ids)
            connection.execute(
                f"DELETE FROM observation_inbox WHERE id IN ({placeholders})",
                tuple(inbox_ids),
            )
        summaries_removed = 0
        if self._table_exists(connection, "sessions"):
            for row in connection.execute("SELECT id, summary, title FROM sessions"):
                content = f"{row['summary'] or ''}\n{row['title'] or ''}".casefold()
                if pattern in content:
                    connection.execute(
                        "UPDATE sessions SET summary = NULL, title = NULL WHERE id = ?",
                        (row["id"],),
                    )
                    summaries_removed += 1
        research_removed = self._purge_research(connection, pattern)
        suppression_counts.update(
            {
                "node": len(node_ids),
                "edge": len(edge_ids),
                "file": research_removed,
                "turn": len(inbox_ids) + summaries_removed,
            }
        )
        return {
            "graph_claims_removed": len(node_ids) + len(edge_ids),
            "inbox_rows_removed": len(inbox_ids),
            "summaries_removed": summaries_removed,
            "research_documents_removed": research_removed,
        }

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            is not None
        )

    def _purge_research(self, connection: sqlite3.Connection, pattern: str) -> int:
        if not self._table_exists(connection, "research_documents"):
            return 0
        columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(research_documents)")
        }
        text_columns = [
            name
            for name in (
                "title",
                "source",
                "content",
                "extracted_text",
                "source_title",
                "source_ref",
                "filename",
                "extracted_markdown",
            )
            if name in columns
        ]
        if not text_columns:
            return 0
        matched: list[object] = []
        for row in connection.execute("SELECT * FROM research_documents"):
            block_match = False
            if self._table_exists(connection, "research_blocks"):
                block_match = any(
                    pattern in str(block["text"]).casefold()
                    for block in connection.execute(
                        "SELECT text FROM research_blocks WHERE doc_id = ?",
                        (row["id"],),
                    )
                )
            if block_match or any(
                pattern in str(row[name] or "").casefold() for name in text_columns
            ):
                matched.append(row["id"])
        if matched:
            if "artifact_path" in columns:
                artifact_root = (self.data_dir / "research" / "sources").resolve()
                for document_id in matched:
                    row = connection.execute(
                        "SELECT artifact_path FROM research_documents WHERE id = ?",
                        (document_id,),
                    ).fetchone()
                    if row is not None and row["artifact_path"]:
                        artifact = (self.data_dir / str(row["artifact_path"])).resolve()
                        if artifact.is_relative_to(artifact_root):
                            artifact.unlink(missing_ok=True)
            placeholders = ",".join("?" for _ in matched)
            connection.execute(
                f"DELETE FROM research_documents WHERE id IN ({placeholders})",
                tuple(matched),
            )
        return len(matched)

    def remove_boundary(self, kind: str, value: str) -> bool:
        """Remove one boundary."""
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM boundaries WHERE kind = ? AND value = ?", (kind, value)
                )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Inbox and distillation runs
    # ------------------------------------------------------------------

    def add_observation(
        self, text: str, *, session_id: str | None = None, language: str | None = None
    ) -> int | None:
        """Append session-scoped freeform observation unless blocked."""
        text = text.strip()
        if not text or len(text) > _MAX_QUOTE_CHARS:
            return None
        if self.is_never_store(text):
            _record_suppression("turn")
            return None
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO observation_inbox
                        (session_id, text, language, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, text, language, _utc_now()),
                )
        return cast(int, cursor.lastrowid)

    def pending_observations(
        self,
        session_id: str | None = None,
        *,
        include_processed: bool = False,
    ) -> list[dict[str, object]]:
        """Return unconsumed observations, scoped to session when provided."""
        clauses: list[str] = []
        params: list[object] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if not include_processed:
            clauses.append("processed_at IS NULL")
        query = (
            "SELECT id, session_id, text, language, created_at, processed_at, "
            "distillation_run_id FROM observation_inbox"
        )
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, tuple(params))]

    def mark_observations_processed(
        self, ids: Sequence[int], *, run_id: str | None = None
    ) -> None:
        """Mark observations consumed; retained for audit until pruning."""
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            with connection:
                connection.execute(
                    f"UPDATE observation_inbox SET processed_at = ?, "
                    f"distillation_run_id = ? WHERE id IN ({placeholders})",
                    (_utc_now(), run_id, *ids),
                )

    def prune_processed_observations(self, *, older_than_days: int = 30) -> int:
        """Delete processed inbox audit rows after bounded retention."""
        if older_than_days < 1:
            raise ValueError("older_than_days must be >= 1")
        threshold = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM observation_inbox "
                    "WHERE processed_at IS NOT NULL AND processed_at < ?",
                    (threshold,),
                )
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Evidence and claim writes
    # ------------------------------------------------------------------

    def _validate_content(self, statement: str, quotes: Sequence[GraphQuote]) -> str:
        statement = statement.strip()
        if not statement or len(statement) > _MAX_STATEMENT_CHARS:
            raise ValueError("statement must contain 1..4000 characters")
        if self.is_never_store(statement):
            raise PermissionError("statement matches a never_store boundary")
        for quote in quotes:
            text = quote.get("text", "").strip()
            language = quote.get("language", "").strip()
            if not text or len(text) > _MAX_QUOTE_CHARS or not language:
                raise ValueError("quotes require bounded text and language")
            if self.is_never_store(text):
                raise PermissionError("quote matches a never_store boundary")
        return statement

    def _insert_evidence(
        self,
        connection: sqlite3.Connection,
        *,
        claim_kind: ClaimKind,
        claim_id: int,
        source_type: str,
        session_id: str | None,
        quote: GraphQuote | None,
        extractor_version: str,
        evidence_key: str,
    ) -> bool:
        quote_text = quote["text"].strip() if quote is not None else None
        language = quote["language"].strip() if quote is not None else None
        turn_id = quote.get("turn_id") if quote is not None else None
        observed_at = (
            quote.get("observed_at", _utc_now()) if quote is not None else _utc_now()
        )
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO claim_evidence (
                id, claim_kind, claim_id, source_type, source_session_id,
                source_turn_id, observed_at, language, quote_text,
                source_state, extractor_version, evidence_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            """,
            (
                uuid4().hex,
                claim_kind,
                claim_id,
                source_type,
                session_id,
                turn_id,
                observed_at,
                language,
                quote_text,
                extractor_version,
                evidence_key,
            ),
        )
        return cursor.rowcount > 0

    def _refresh_counts(
        self, connection: sqlite3.Connection, claim_kind: ClaimKind, claim_id: int
    ) -> None:
        table = "nodes" if claim_kind == "node" else "edges"
        row = connection.execute(
            """
            SELECT count(*) AS total,
                   count(DISTINCT coalesce(source_session_id, source_marker)) AS sessions,
                   sum(CASE WHEN source_state = 'active' THEN 1 ELSE 0 END) AS auditable,
                   count(DISTINCT CASE WHEN source_state = 'active'
                                      THEN source_session_id END) AS auditable_sessions
            FROM claim_evidence
            WHERE claim_kind = ? AND claim_id = ?
            """,
            (claim_kind, claim_id),
        ).fetchone()
        connection.execute(
            f"""
            UPDATE {table}
            SET n_occurrences = ?, n_sessions = ?,
                n_auditable_occurrences = ?, n_auditable_sessions = ?
            WHERE id = ?
            """,
            (
                int(row["total"] or 0),
                int(row["sessions"] or 0),
                int(row["auditable"] or 0),
                int(row["auditable_sessions"] or 0),
                claim_id,
            ),
        )

    def _revalidation_due(self, type_: str, seen_at: str) -> str | None:
        days = {"thread": 14, "pattern": 56, "strategy": 90}.get(type_)
        if days is None:
            return None
        return (_parse_timestamp(seen_at) + timedelta(days=days)).isoformat()

    def upsert_node(
        self,
        type_: str,
        statement: str,
        *,
        source: str = "conversation",
        quotes: list[GraphQuote] | None = None,
        session_id: str | None = None,
        never_initiate: bool = False,
        extractor_version: str = "application-v1",
        evidence_key: str | None = None,
    ) -> int | None:
        """Insert/reinforce inferred claim; source authority stays application-owned."""
        try:
            if source == "user-stated":
                raise ValueError("Use add_user_statement for trusted direct knowledge")
            if source not in {"conversation", "ser"}:
                raise ValueError(f"Unknown source: {source!r}")
            node_id = self._upsert_node(
                type_,
                statement,
                source=source,
                quotes=quotes or [],
                session_id=session_id,
                never_initiate=never_initiate,
                extractor_version=extractor_version,
                evidence_key=evidence_key,
                direct=False,
            )
        except (PermissionError, ValueError):
            _record_graph_mutation("add_node", "rejected")
            raise
        except Exception as error:
            _record_graph_mutation("add_node", _graph_error_outcome(error))
            raise
        _record_graph_mutation("add_node", "success" if node_id is not None else "rejected")
        return node_id

    def add_user_statement(
        self,
        type_: str,
        statement: str,
        *,
        session_id: str | None = None,
        quote: GraphQuote | None = None,
        never_initiate: bool = False,
    ) -> int | None:
        """Store explicit owner statement as confirmed through trusted path."""
        try:
            node_id = self._upsert_node(
                type_,
                statement,
                source="user-stated",
                quotes=[quote] if quote is not None else [],
                session_id=session_id,
                never_initiate=never_initiate,
                extractor_version="direct-user-stated-v1",
                evidence_key=None,
                direct=True,
            )
        except (PermissionError, ValueError):
            _record_graph_mutation("add_node", "rejected")
            raise
        except Exception as error:
            _record_graph_mutation("add_node", _graph_error_outcome(error))
            raise
        _record_graph_mutation("add_node", "success" if node_id is not None else "rejected")
        return node_id

    def _upsert_node(
        self,
        type_: str,
        statement: str,
        *,
        source: str,
        quotes: list[GraphQuote],
        session_id: str | None,
        never_initiate: bool,
        extractor_version: str,
        evidence_key: str | None,
        direct: bool,
    ) -> int | None:
        node_type = canonical_node_type(type_)
        try:
            statement = self._validate_content(statement, quotes)
        except PermissionError:
            _record_suppression("node")
            return None
        with self._connect() as connection:
            with connection:
                return self._upsert_node_tx(
                    connection,
                    node_type,
                    statement,
                    source=source,
                    quotes=quotes,
                    session_id=session_id,
                    never_initiate=never_initiate,
                    extractor_version=extractor_version,
                    evidence_key=evidence_key,
                    direct=direct,
                )

    def _upsert_node_tx(
        self,
        connection: sqlite3.Connection,
        node_type: str,
        statement: str,
        *,
        source: str,
        quotes: list[GraphQuote],
        session_id: str | None,
        never_initiate: bool,
        extractor_version: str,
        evidence_key: str | None,
        direct: bool,
    ) -> int | None:
        signature = self._node_signature(node_type, statement)
        if self._is_tombstoned(connection, "node", signature):
            return None
        now = _utc_now()
        row = connection.execute(
            "SELECT * FROM nodes WHERE type = ? AND statement_key = ?",
            (node_type, normalize_statement(statement)),
        ).fetchone()
        if row is None:
            status = "confirmed" if direct else "observation"
            cursor = connection.execute(
                """
                INSERT INTO nodes (
                    type, statement, statement_key, status, source,
                    never_initiate, user_edited, first_seen, last_seen,
                    revalidation_due_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    node_type,
                    statement,
                    normalize_statement(statement),
                    status,
                    source,
                    int(never_initiate),
                    now,
                    now,
                    self._revalidation_due(node_type, now),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a node id")
            node_id = int(cursor.lastrowid)
            self._record_event(
                connection,
                "node",
                node_id,
                None,
                status,
                "direct_user_statement" if direct else "observed",
                session_id=session_id,
            )
        else:
            node_id = int(row["id"])
            status = "confirmed" if direct else str(row["status"])
            source_value = "user-stated" if direct else str(row["source"])
            connection.execute(
                """
                UPDATE nodes SET last_seen = ?, never_initiate = ?,
                    status = ?, source = ?, revalidation_due_at = ?
                WHERE id = ?
                """,
                (
                    now,
                    int(never_initiate or bool(row["never_initiate"])),
                    status,
                    source_value,
                    self._revalidation_due(node_type, now),
                    node_id,
                ),
            )
            if direct and row["status"] != "confirmed":
                self._record_event(
                    connection,
                    "node",
                    node_id,
                    str(row["status"]),
                    "confirmed",
                    "direct_user_statement",
                    session_id=session_id,
                )
        connection.execute(
            """
            INSERT OR IGNORE INTO node_aliases (
                node_id, alias, alias_key, language, created_at
            ) VALUES (?, ?, ?, 'en', ?)
            """,
            (node_id, statement, normalize_statement(statement), now),
        )
        evidence_items: list[GraphQuote | None] = list(quotes) or [None]
        base_key = evidence_key or uuid4().hex
        for index, quote in enumerate(evidence_items):
            self._insert_evidence(
                connection,
                claim_kind="node",
                claim_id=node_id,
                source_type=source,
                session_id=session_id,
                quote=quote,
                extractor_version=extractor_version,
                evidence_key=f"{base_key}:{index}",
            )
        self._refresh_counts(connection, "node", node_id)
        return node_id

    def upsert_edge(
        self,
        src: int,
        dst: int,
        type_: str,
        *,
        statement: str,
        source: str = "conversation",
        quotes: list[GraphQuote] | None = None,
        session_id: str | None = None,
        extractor_version: str = "application-v1",
        evidence_key: str | None = None,
    ) -> int | None:
        """Insert/reinforce inferred edge; empty statements are forbidden."""
        try:
            if source == "user-stated":
                raise ValueError("Use add_user_edge for trusted direct knowledge")
            if source not in {"conversation", "ser"}:
                raise ValueError(f"Unknown source: {source!r}")
            edge_id = self._upsert_edge(
                src,
                dst,
                type_,
                statement,
                source=source,
                quotes=quotes or [],
                session_id=session_id,
                extractor_version=extractor_version,
                evidence_key=evidence_key,
                direct=False,
            )
        except (PermissionError, ValueError):
            _record_graph_mutation("add_edge", "rejected")
            raise
        except Exception as error:
            _record_graph_mutation("add_edge", _graph_error_outcome(error))
            raise
        _record_graph_mutation("add_edge", "success" if edge_id is not None else "rejected")
        return edge_id

    def add_user_edge(
        self,
        src: int,
        dst: int,
        type_: str,
        statement: str,
        *,
        session_id: str | None = None,
        quote: GraphQuote | None = None,
    ) -> int | None:
        """Store explicit owner relationship as confirmed through trusted path."""
        try:
            edge_id = self._upsert_edge(
                src,
                dst,
                type_,
                statement,
                source="user-stated",
                quotes=[quote] if quote is not None else [],
                session_id=session_id,
                extractor_version="direct-user-stated-v1",
                evidence_key=None,
                direct=True,
            )
        except (PermissionError, ValueError):
            _record_graph_mutation("add_edge", "rejected")
            raise
        except Exception as error:
            _record_graph_mutation("add_edge", _graph_error_outcome(error))
            raise
        _record_graph_mutation("add_edge", "success" if edge_id is not None else "rejected")
        return edge_id

    def _upsert_edge(
        self,
        src: int,
        dst: int,
        type_: str,
        statement: str,
        *,
        source: str,
        quotes: list[GraphQuote],
        session_id: str | None,
        extractor_version: str,
        evidence_key: str | None,
        direct: bool,
    ) -> int | None:
        edge_type = canonical_edge_type(type_)
        try:
            statement = self._validate_content(statement, quotes)
        except PermissionError:
            _record_suppression("edge")
            return None
        with self._connect() as connection:
            with connection:
                return self._upsert_edge_tx(
                    connection,
                    src,
                    dst,
                    edge_type,
                    statement,
                    source=source,
                    quotes=quotes,
                    session_id=session_id,
                    extractor_version=extractor_version,
                    evidence_key=evidence_key,
                    direct=direct,
                )

    def _upsert_edge_tx(
        self,
        connection: sqlite3.Connection,
        src: int,
        dst: int,
        edge_type: str,
        statement: str,
        *,
        source: str,
        quotes: list[GraphQuote],
        session_id: str | None,
        extractor_version: str,
        evidence_key: str | None,
        direct: bool,
    ) -> int | None:
        signature = self._edge_signature(connection, src, dst, edge_type, statement)
        base_signature = self._edge_base_signature(connection, src, dst, edge_type)
        if self._is_tombstoned(
            connection, "edge", signature
        ) or self._is_tombstoned(connection, "edge", base_signature):
            return None
        now = _utc_now()
        row = connection.execute(
            """
            SELECT * FROM edges
            WHERE src = ? AND dst = ? AND type = ? AND statement_key = ?
            """,
            (src, dst, edge_type, normalize_statement(statement)),
        ).fetchone()
        if row is None:
            status = "confirmed" if direct else "observation"
            cursor = connection.execute(
                """
                INSERT INTO edges (
                    src, dst, type, statement, statement_key, status,
                    source, user_edited, first_seen, last_seen,
                    revalidation_due_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    src,
                    dst,
                    edge_type,
                    statement,
                    normalize_statement(statement),
                    status,
                    source,
                    now,
                    now,
                    (datetime.now(UTC) + timedelta(days=56)).isoformat(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an edge id")
            edge_id = int(cursor.lastrowid)
            self._record_event(
                connection,
                "edge",
                edge_id,
                None,
                status,
                "direct_user_statement" if direct else "observed",
                session_id=session_id,
            )
        else:
            edge_id = int(row["id"])
            status = "confirmed" if direct else str(row["status"])
            source_value = "user-stated" if direct else str(row["source"])
            connection.execute(
                """
                UPDATE edges SET last_seen = ?, status = ?, source = ?,
                    revalidation_due_at = ? WHERE id = ?
                """,
                (
                    now,
                    status,
                    source_value,
                    (datetime.now(UTC) + timedelta(days=56)).isoformat(),
                    edge_id,
                ),
            )
            if direct and row["status"] != "confirmed":
                self._record_event(
                    connection,
                    "edge",
                    edge_id,
                    str(row["status"]),
                    "confirmed",
                    "direct_user_statement",
                    session_id=session_id,
                )
        evidence_items: list[GraphQuote | None] = list(quotes) or [None]
        base_key = evidence_key or uuid4().hex
        for index, quote in enumerate(evidence_items):
            self._insert_evidence(
                connection,
                claim_kind="edge",
                claim_id=edge_id,
                source_type=source,
                session_id=session_id,
                quote=quote,
                extractor_version=extractor_version,
                evidence_key=f"{base_key}:{index}",
            )
        self._refresh_counts(connection, "edge", edge_id)
        return edge_id

    def apply_distillation(
        self,
        *,
        session_id: str,
        extractor_version: str,
        candidates: Sequence[Mapping[str, object]],
        inbox_ids: Sequence[int],
    ) -> tuple[str, list[int], list[int]]:
        """Apply one distillation transaction with a bounded mutation outcome."""
        try:
            result = self._apply_distillation(
                session_id=session_id,
                extractor_version=extractor_version,
                candidates=candidates,
                inbox_ids=inbox_ids,
            )
        except (KeyError, PermissionError, ValueError):
            _record_graph_mutation("distill_apply", "rejected")
            raise
        except Exception as error:
            _record_graph_mutation("distill_apply", _graph_error_outcome(error))
            raise
        _record_graph_mutation("distill_apply", "success")
        return result

    def _apply_distillation(
        self,
        *,
        session_id: str,
        extractor_version: str,
        candidates: Sequence[Mapping[str, object]],
        inbox_ids: Sequence[int],
    ) -> tuple[str, list[int], list[int]]:
        """Atomically apply validated candidates, evidence, and inbox consumption."""
        if not session_id:
            raise ValueError("session_id is required for distillation")
        if not extractor_version or len(extractor_version) > 200:
            raise ValueError("extractor_version must contain 1..200 characters")
        run_id = self._begin_distillation_run(session_id, extractor_version)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT state, result_json FROM distillation_runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if existing is not None and existing["state"] == "succeeded":
                result = json.loads(str(existing["result_json"] or "{}"))
                return (
                    run_id,
                    [int(item) for item in result.get("node_ids", [])],
                    [int(item) for item in result.get("edge_ids", [])],
                )
        try:
            with self._connect() as connection:
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    locked_run = connection.execute(
                        "SELECT state, result_json FROM distillation_runs WHERE id = ?",
                        (run_id,),
                    ).fetchone()
                    if locked_run is not None and locked_run["state"] == "succeeded":
                        locked_result = json.loads(
                            str(locked_run["result_json"] or "{}")
                        )
                        return (
                            run_id,
                            [int(item) for item in locked_result.get("node_ids", [])],
                            [int(item) for item in locked_result.get("edge_ids", [])],
                        )
                    node_ids: list[int] = []
                    edge_ids: list[int] = []
                    references: dict[str, int] = {}
                    for index, candidate in enumerate(candidates):
                        if candidate["kind"] != "node":
                            continue
                        quotes = cast(list[GraphQuote], candidate.get("quotes", []))
                        try:
                            statement = self._validate_content(
                                str(candidate["statement"]), quotes
                            )
                        except PermissionError:
                            _record_suppression("node")
                            continue
                        node_id = self._upsert_node_tx(
                            connection,
                            canonical_node_type(str(candidate["type"])),
                            statement,
                            source="conversation",
                            quotes=quotes,
                            session_id=session_id,
                            never_initiate=bool(candidate.get("never_initiate", False)),
                            extractor_version=extractor_version,
                            evidence_key=f"{run_id}:node:{index}",
                            direct=False,
                        )
                        if node_id is None:
                            continue
                        node_ids.append(node_id)
                        references[normalize_statement(statement)] = node_id
                        for alias in cast(
                            Sequence[Mapping[str, object]], candidate.get("aliases", [])
                        ):
                            alias_text = str(alias["text"]).strip()
                            if not alias_text or self.is_never_store(alias_text):
                                if alias_text:
                                    _record_suppression("node")
                                continue
                            connection.execute(
                                """
                                INSERT OR IGNORE INTO node_aliases (
                                    node_id, alias, alias_key, language, created_at
                                ) VALUES (?, ?, ?, ?, ?)
                                """,
                                (
                                    node_id,
                                    alias_text,
                                    normalize_statement(alias_text),
                                    str(alias.get("language", "und")),
                                    _utc_now(),
                                ),
                            )
                    for index, candidate in enumerate(candidates):
                        if candidate["kind"] != "edge":
                            continue
                        src = self._resolve_node_reference(
                            connection, str(candidate["src"]), references
                        )
                        dst = self._resolve_node_reference(
                            connection, str(candidate["dst"]), references
                        )
                        if src == dst:
                            raise ValueError("edge endpoints must be different")
                        quotes = cast(list[GraphQuote], candidate.get("quotes", []))
                        try:
                            edge_statement = self._validate_content(
                                str(candidate["statement"]), quotes
                            )
                        except PermissionError:
                            _record_suppression("edge")
                            continue
                        edge_id = self._upsert_edge_tx(
                            connection,
                            src,
                            dst,
                            canonical_edge_type(str(candidate["type"])),
                            edge_statement,
                            source="conversation",
                            quotes=quotes,
                            session_id=session_id,
                            extractor_version=extractor_version,
                            evidence_key=f"{run_id}:edge:{index}",
                            direct=False,
                        )
                        if edge_id is not None:
                            edge_ids.append(edge_id)
                    if inbox_ids:
                        placeholders = ",".join("?" for _ in inbox_ids)
                        matching = connection.execute(
                            f"""
                            SELECT count(*) AS count FROM observation_inbox
                            WHERE id IN ({placeholders}) AND session_id = ?
                              AND processed_at IS NULL
                            """,
                            (*inbox_ids, session_id),
                        ).fetchone()
                        if int(matching["count"]) != len(set(inbox_ids)):
                            raise ValueError(
                                "inbox rows must be unprocessed and belong to the session"
                            )
                        connection.execute(
                            f"""
                            UPDATE observation_inbox SET processed_at = ?,
                                distillation_run_id = ?
                            WHERE id IN ({placeholders})
                            """,
                            (_utc_now(), run_id, *inbox_ids),
                        )
                    result = json.dumps(
                        {"node_ids": node_ids, "edge_ids": edge_ids},
                        separators=(",", ":"),
                    )
                    connection.execute(
                        """
                        UPDATE distillation_runs SET state = 'succeeded',
                            finished_at = ?, error = NULL, result_json = ?
                        WHERE id = ?
                        """,
                        (_utc_now(), result, run_id),
                    )
            return run_id, node_ids, edge_ids
        except Exception as error:
            with self._connect() as connection:
                with connection:
                    connection.execute(
                        """
                        UPDATE distillation_runs SET state = 'failed',
                            finished_at = ?, error = ? WHERE id = ?
                        """,
                        (
                            _utc_now(),
                            f"{type(error).__name__}: {error}"[:1_000],
                            run_id,
                        ),
                    )
            raise

    def _begin_distillation_run(self, session_id: str, extractor_version: str) -> str:
        with self._connect() as connection:
            with connection:
                row = connection.execute(
                    """
                    SELECT id, state FROM distillation_runs
                    WHERE session_id = ? AND extractor_version = ?
                    """,
                    (session_id, extractor_version),
                ).fetchone()
                if row is not None:
                    run_id = str(row["id"])
                    if row["state"] == "succeeded":
                        return run_id
                    if row["state"] == "running":
                        return run_id
                    connection.execute(
                        """
                        UPDATE distillation_runs SET state = 'running',
                            attempt_count = attempt_count + 1, started_at = ?,
                            finished_at = NULL, error = NULL
                        WHERE id = ?
                        """,
                        (_utc_now(), run_id),
                    )
                    return run_id
                run_id = uuid4().hex
                connection.execute(
                    """
                    INSERT INTO distillation_runs (
                        id, session_id, extractor_version, state, started_at
                    ) VALUES (?, ?, ?, 'running', ?)
                    """,
                    (run_id, session_id, extractor_version, _utc_now()),
                )
                return run_id

    def start_distillation_run(self, session_id: str, extractor_version: str) -> str:
        """Start/retry idempotent run before invoking external extraction."""
        if not session_id or not extractor_version:
            raise ValueError("session_id and extractor_version are required")
        return self._begin_distillation_run(session_id, extractor_version)

    def distillation_run(self, run_id: str) -> dict[str, object] | None:
        """Return one run record for idempotent finalization."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM distillation_runs WHERE id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        if result.get("result_json"):
            result["result"] = json.loads(str(result["result_json"]))
        return result

    def fail_distillation_run(self, run_id: str, error: BaseException) -> None:
        """Persist bounded actionable failure without consuming source inbox rows."""
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE distillation_runs SET state = 'failed', finished_at = ?,
                        error = ? WHERE id = ? AND state != 'succeeded'
                    """,
                    (_utc_now(), f"{type(error).__name__}: {error}"[:1_000], run_id),
                )

    @staticmethod
    def _resolve_node_reference(
        connection: sqlite3.Connection,
        reference: str,
        batch: Mapping[str, int],
    ) -> int:
        key = normalize_statement(reference)
        if key in batch:
            return batch[key]
        matches = connection.execute(
            """
            SELECT DISTINCT node_id FROM node_aliases WHERE alias_key = ?
            UNION
            SELECT id FROM nodes WHERE statement_key = ?
            """,
            (key, key),
        ).fetchall()
        if len(matches) != 1:
            raise ValueError(
                f"edge endpoint reference is unresolved or ambiguous: {reference!r}"
            )
        return int(matches[0]["node_id"])

    # ------------------------------------------------------------------
    # Reads, edits, deletes
    # ------------------------------------------------------------------

    def evidence(self, claim_kind: ClaimKind, claim_id: int) -> list[ClaimEvidence]:
        """Return complete provenance for one claim."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM claim_evidence
                WHERE claim_kind = ? AND claim_id = ?
                ORDER BY observed_at, id
                """,
                (claim_kind, claim_id),
            ).fetchall()
        return [cast(ClaimEvidence, dict(row)) for row in rows]

    def _shape_node(self, row: sqlite3.Row) -> GraphNode:
        node = dict(row)
        evidence = self.evidence("node", int(row["id"]))
        node["quotes"] = [
            cast(
                GraphQuote,
                {
                    "text": item["quote_text"],
                    "language": item["language"] or "und",
                    **(
                        {"session_id": item["source_session_id"]}
                        if item["source_session_id"] is not None
                        else {}
                    ),
                    **(
                        {"turn_id": item["source_turn_id"]}
                        if item["source_turn_id"] is not None
                        else {}
                    ),
                    "observed_at": item["observed_at"],
                },
            )
            for item in evidence
            if item["quote_text"] is not None and item["source_state"] == "active"
        ]
        node["sessions"] = sorted(
            {
                item["source_session_id"]
                for item in evidence
                if item["source_session_id"] is not None
            }
        )
        node["never_initiate"] = bool(node["never_initiate"])
        node["user_edited"] = bool(node["user_edited"])
        return cast(GraphNode, node)

    def _shape_edge(self, row: sqlite3.Row) -> GraphEdge:
        edge = dict(row)
        evidence = self.evidence("edge", int(row["id"]))
        edge["quotes"] = [
            cast(
                GraphQuote,
                {
                    "text": item["quote_text"],
                    "language": item["language"] or "und",
                    **(
                        {"session_id": item["source_session_id"]}
                        if item["source_session_id"] is not None
                        else {}
                    ),
                    **(
                        {"turn_id": item["source_turn_id"]}
                        if item["source_turn_id"] is not None
                        else {}
                    ),
                    "observed_at": item["observed_at"],
                },
            )
            for item in evidence
            if item["quote_text"] is not None and item["source_state"] == "active"
        ]
        edge["sessions"] = sorted(
            {
                item["source_session_id"]
                for item in evidence
                if item["source_session_id"] is not None
            }
        )
        edge["user_edited"] = bool(edge["user_edited"])
        return cast(GraphEdge, edge)

    def get_node(self, node_id: int) -> GraphNode | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
        return self._shape_node(row) if row is not None else None

    def get_edge(self, edge_id: int) -> GraphEdge | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM edges WHERE id = ?", (edge_id,)
            ).fetchone()
        return self._shape_edge(row) if row is not None else None

    def nodes(
        self, *, type_: str | None = None, status: str | None = None
    ) -> list[GraphNode]:
        """Return graph nodes with canonical registry/status filters."""
        self.revalidate_stale()
        query = "SELECT * FROM nodes"
        clauses: list[str] = []
        params: list[object] = []
        if type_ is not None:
            clauses.append("type = ?")
            params.append(canonical_node_type(type_))
        if status is not None:
            mapped_status = "proposed" if status == "pattern" else status
            if mapped_status not in CLAIM_STATUSES:
                raise ValueError(f"Unknown claim status: {status!r}")
            clauses.append("status = ?")
            params.append(mapped_status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id"
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [self._shape_node(row) for row in rows]

    def edges(
        self, *, type_: str | None = None, status: str | None = None
    ) -> list[GraphEdge]:
        """Return graph edges with canonical registry/status filters."""
        self.revalidate_stale()
        query = "SELECT * FROM edges"
        clauses: list[str] = []
        params: list[object] = []
        if type_ is not None:
            clauses.append("type = ?")
            params.append(canonical_edge_type(type_))
        if status is not None:
            mapped_status = "proposed" if status == "pattern" else status
            if mapped_status not in CLAIM_STATUSES:
                raise ValueError(f"Unknown claim status: {status!r}")
            clauses.append("status = ?")
            params.append(mapped_status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id"
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [self._shape_edge(row) for row in rows]

    def edit_node(
        self,
        node_id: int,
        *,
        statement: str | None = None,
        never_initiate: bool | None = None,
    ) -> bool:
        """Apply an owner edit and record its bounded mutation outcome."""
        try:
            changed = self._edit_node(
                node_id,
                statement=statement,
                never_initiate=never_initiate,
            )
        except (PermissionError, ValueError, sqlite3.IntegrityError):
            _record_graph_mutation("edit", "rejected")
            raise
        except Exception as error:
            _record_graph_mutation("edit", _graph_error_outcome(error))
            raise
        _record_graph_mutation("edit", "success" if changed else "rejected")
        return changed

    def _edit_node(
        self,
        node_id: int,
        *,
        statement: str | None = None,
        never_initiate: bool | None = None,
    ) -> bool:
        """Apply authoritative owner edit, tombstoning prior identity."""
        with self._connect() as connection:
            with connection:
                row = connection.execute(
                    "SELECT * FROM nodes WHERE id = ?", (node_id,)
                ).fetchone()
                if row is None:
                    return False
                new_statement = str(row["statement"])
                if statement is not None:
                    try:
                        new_statement = self._validate_content(statement, [])
                    except PermissionError:
                        _record_suppression("node")
                        return False
                    new_key = normalize_statement(new_statement)
                    conflict = connection.execute(
                        "SELECT id FROM nodes WHERE type = ? AND statement_key = ? AND id != ?",
                        (row["type"], new_key, node_id),
                    ).fetchone()
                    if conflict is not None:
                        raise sqlite3.IntegrityError(
                            "edited statement conflicts with existing node"
                        )
                    self._tombstone(
                        connection,
                        "node",
                        self._node_signature(str(row["type"]), str(row["statement"])),
                    )
                now = _utc_now()
                connection.execute(
                    """
                    UPDATE nodes SET statement = ?, statement_key = ?,
                        never_initiate = ?, user_edited = 1, status = 'confirmed',
                        source = 'user-stated', last_seen = ? WHERE id = ?
                    """,
                    (
                        new_statement,
                        normalize_statement(new_statement),
                        int(
                            bool(row["never_initiate"])
                            if never_initiate is None
                            else never_initiate
                        ),
                        now,
                        node_id,
                    ),
                )
                self._insert_evidence(
                    connection,
                    claim_kind="node",
                    claim_id=node_id,
                    source_type="user-stated",
                    session_id=None,
                    quote=None,
                    extractor_version="owner-edit-v1",
                    evidence_key=f"owner-edit:{now}",
                )
                self._refresh_counts(connection, "node", node_id)
                self._record_event(
                    connection,
                    "node",
                    node_id,
                    str(row["status"]),
                    "confirmed",
                    "owner_edit",
                )
        return True

    def edit_edge(self, edge_id: int, *, statement: str) -> bool:
        """Apply an owner edge edit and record its bounded mutation outcome."""
        try:
            changed = self._edit_edge(edge_id, statement=statement)
        except (PermissionError, ValueError, sqlite3.IntegrityError):
            _record_graph_mutation("edit", "rejected")
            raise
        except Exception as error:
            _record_graph_mutation("edit", _graph_error_outcome(error))
            raise
        _record_graph_mutation("edit", "success" if changed else "rejected")
        return changed

    def _edit_edge(self, edge_id: int, *, statement: str) -> bool:
        """Apply authoritative owner edit to an edge claim."""
        try:
            new_statement = self._validate_content(statement, [])
        except PermissionError:
            _record_suppression("edge")
            return False
        with self._connect() as connection:
            with connection:
                row = connection.execute(
                    "SELECT * FROM edges WHERE id = ?", (edge_id,)
                ).fetchone()
                if row is None:
                    return False
                self._tombstone(
                    connection,
                    "edge",
                    self._edge_signature(
                        connection,
                        int(row["src"]),
                        int(row["dst"]),
                        str(row["type"]),
                        str(row["statement"]),
                    ),
                )
                now = _utc_now()
                connection.execute(
                    """
                    UPDATE edges SET statement = ?, statement_key = ?,
                        user_edited = 1, status = 'confirmed',
                        source = 'user-stated', last_seen = ? WHERE id = ?
                    """,
                    (new_statement, normalize_statement(new_statement), now, edge_id),
                )
                self._insert_evidence(
                    connection,
                    claim_kind="edge",
                    claim_id=edge_id,
                    source_type="user-stated",
                    session_id=None,
                    quote=None,
                    extractor_version="owner-edit-v1",
                    evidence_key=f"owner-edit:{now}",
                )
                self._refresh_counts(connection, "edge", edge_id)
                self._record_event(
                    connection,
                    "edge",
                    edge_id,
                    str(row["status"]),
                    "confirmed",
                    "owner_edit",
                )
        return True

    def delete_edge(self, edge_id: int) -> bool:
        """Delete an edge and record its bounded mutation outcome."""
        try:
            changed = self._delete_edge(edge_id)
        except Exception as error:
            _record_graph_mutation("delete", _graph_error_outcome(error))
            raise
        _record_graph_mutation("delete", "success" if changed else "rejected")
        return changed

    def _delete_edge(self, edge_id: int) -> bool:
        """Delete edge and write endpoint-stable keyed tombstone."""
        with self._connect() as connection:
            with connection:
                row = connection.execute(
                    "SELECT * FROM edges WHERE id = ?", (edge_id,)
                ).fetchone()
                if row is None:
                    return False
                self._tombstone(
                    connection,
                    "edge",
                    self._edge_signature(
                        connection,
                        int(row["src"]),
                        int(row["dst"]),
                        str(row["type"]),
                        str(row["statement"]),
                    ),
                )
                self._tombstone(
                    connection,
                    "edge",
                    self._edge_base_signature(
                        connection,
                        int(row["src"]),
                        int(row["dst"]),
                        str(row["type"]),
                    ),
                )
                connection.execute(
                    "DELETE FROM claim_evidence WHERE claim_kind = 'edge' AND claim_id = ?",
                    (edge_id,),
                )
                connection.execute(
                    "DELETE FROM pending_insights "
                    "WHERE claim_kind = 'edge' AND claim_id = ?",
                    (edge_id,),
                )
                connection.execute(
                    "DELETE FROM lifecycle_events "
                    "WHERE claim_kind = 'edge' AND claim_id = ?",
                    (edge_id,),
                )
                connection.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        return True

    def delete_node(self, node_id: int) -> bool:
        """Delete a node and record its bounded mutation outcome."""
        try:
            changed = self._delete_node(node_id)
        except Exception as error:
            _record_graph_mutation("delete", _graph_error_outcome(error))
            raise
        _record_graph_mutation("delete", "success" if changed else "rejected")
        return changed

    def _delete_node(self, node_id: int) -> bool:
        """Delete node/incident edges and write privacy-safe tombstones."""
        with self._connect() as connection:
            with connection:
                row = connection.execute(
                    "SELECT * FROM nodes WHERE id = ?", (node_id,)
                ).fetchone()
                if row is None:
                    return False
                incident = connection.execute(
                    "SELECT * FROM edges WHERE src = ? OR dst = ?", (node_id, node_id)
                ).fetchall()
                for edge in incident:
                    self._tombstone(
                        connection,
                        "edge",
                        self._edge_signature(
                            connection,
                            int(edge["src"]),
                            int(edge["dst"]),
                            str(edge["type"]),
                            str(edge["statement"]),
                        ),
                    )
                    self._tombstone(
                        connection,
                        "edge",
                        self._edge_base_signature(
                            connection,
                            int(edge["src"]),
                            int(edge["dst"]),
                            str(edge["type"]),
                        ),
                    )
                    connection.execute(
                        "DELETE FROM claim_evidence "
                        "WHERE claim_kind = 'edge' AND claim_id = ?",
                        (edge["id"],),
                    )
                    connection.execute(
                        "DELETE FROM pending_insights "
                        "WHERE claim_kind = 'edge' AND claim_id = ?",
                        (edge["id"],),
                    )
                    connection.execute(
                        "DELETE FROM lifecycle_events "
                        "WHERE claim_kind = 'edge' AND claim_id = ?",
                        (edge["id"],),
                    )
                self._tombstone(
                    connection,
                    "node",
                    self._node_signature(str(row["type"]), str(row["statement"])),
                )
                connection.execute(
                    "DELETE FROM claim_evidence WHERE claim_kind = 'node' AND claim_id = ?",
                    (node_id,),
                )
                connection.execute(
                    "DELETE FROM pending_insights "
                    "WHERE claim_kind = 'node' AND claim_id = ?",
                    (node_id,),
                )
                connection.execute(
                    "DELETE FROM lifecycle_events "
                    "WHERE claim_kind = 'node' AND claim_id = ?",
                    (node_id,),
                )
                connection.execute(
                    "DELETE FROM semantic_embeddings "
                    "WHERE entity_kind = 'node' AND entity_id = ?",
                    (str(node_id),),
                )
                connection.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _record_event(
        self,
        connection: sqlite3.Connection,
        claim_kind: ClaimKind,
        claim_id: int,
        from_status: str | None,
        to_status: str,
        event_type: str,
        *,
        session_id: str | None = None,
        delivery_id: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> int:
        cursor = connection.execute(
            """
            INSERT INTO lifecycle_events (
                claim_kind, claim_id, from_status, to_status, event_type,
                session_id, delivery_id, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim_kind,
                claim_id,
                from_status,
                to_status,
                event_type,
                session_id,
                delivery_id,
                json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                _utc_now(),
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return lifecycle event id")
        return int(cursor.lastrowid)

    @staticmethod
    def is_eligible(claim: Mapping[str, object]) -> bool:
        """Check mechanical floor using currently auditable evidence only."""
        occurrences = claim.get("n_auditable_occurrences", 0)
        sessions = claim.get("n_auditable_sessions", 0)
        if (
            isinstance(occurrences, bool)
            or not isinstance(occurrences, int)
            or isinstance(sessions, bool)
            or not isinstance(sessions, int)
        ):
            return False
        return (
            occurrences >= GRADUATION_MIN_OCCURRENCES
            and sessions >= GRADUATION_MIN_SESSIONS
        )

    def _propose(self, claim_kind: ClaimKind, claim_id: int) -> bool:
        table = "nodes" if claim_kind == "node" else "edges"
        with self._connect() as connection:
            with connection:
                row = connection.execute(
                    f"SELECT * FROM {table} WHERE id = ?", (claim_id,)
                ).fetchone()
                if row is None or row["status"] not in {
                    "observation",
                    "rejected",
                    "needs_revalidation",
                }:
                    return False
                claim = dict(row)
                if not self.is_eligible(claim):
                    return False
                if row["status"] == "rejected":
                    event = connection.execute(
                        """
                        SELECT details_json FROM lifecycle_events
                        WHERE claim_kind = ? AND claim_id = ?
                          AND event_type = 'user_rejected'
                        ORDER BY id DESC LIMIT 1
                        """,
                        (claim_kind, claim_id),
                    ).fetchone()
                    if event is not None:
                        details = json.loads(str(event["details_json"]))
                        if int(details.get("evidence_count", -1)) == int(
                            row["n_auditable_occurrences"]
                        ):
                            return False
                connection.execute(
                    f"UPDATE {table} SET status = 'proposed' WHERE id = ?", (claim_id,)
                )
                self._record_event(
                    connection,
                    claim_kind,
                    claim_id,
                    str(row["status"]),
                    "proposed",
                    "longitudinal_judgment",
                    details={
                        "evidence_count": int(row["n_auditable_occurrences"]),
                        "session_count": int(row["n_auditable_sessions"]),
                    },
                )
        return True

    def judgment_needed(self, claim_kind: ClaimKind, claim_id: int) -> bool:
        """Return false when unchanged evidence already failed proposal judgment."""
        table = "nodes" if claim_kind == "node" else "edges"
        with self._connect() as connection:
            claim = connection.execute(
                f"SELECT n_auditable_occurrences FROM {table} WHERE id = ?",
                (claim_id,),
            ).fetchone()
            if claim is None:
                return False
            event = connection.execute(
                """
                SELECT details_json FROM lifecycle_events
                WHERE claim_kind = ? AND claim_id = ?
                  AND event_type = 'judgment_deferred'
                ORDER BY id DESC LIMIT 1
                """,
                (claim_kind, claim_id),
            ).fetchone()
        if event is None:
            return True
        details = json.loads(str(event["details_json"]))
        return int(details.get("evidence_count", -1)) != int(
            claim["n_auditable_occurrences"]
        )

    def defer_proposal(self, claim_kind: ClaimKind, claim_id: int) -> bool:
        """Durably record negative longitudinal judgment for evidence snapshot."""
        table = "nodes" if claim_kind == "node" else "edges"
        with self._connect() as connection:
            with connection:
                row = connection.execute(
                    f"SELECT * FROM {table} WHERE id = ?", (claim_id,)
                ).fetchone()
                if row is None or row["status"] != "observation":
                    return False
                self._record_event(
                    connection,
                    claim_kind,
                    claim_id,
                    "observation",
                    "observation",
                    "judgment_deferred",
                    details={
                        "evidence_count": int(row["n_auditable_occurrences"]),
                        "session_count": int(row["n_auditable_sessions"]),
                    },
                )
        return True

    def propose(self, node_id: int) -> bool:
        """Compatibility alias for evidence-gated node proposal."""
        return self._propose("node", node_id)

    def propose_edge(self, edge_id: int) -> bool:
        """Propose eligible edge after explicit longitudinal judgment."""
        return self._propose("edge", edge_id)

    def _resolve(
        self,
        claim_kind: ClaimKind,
        claim_id: int,
        target: Literal["confirmed", "rejected"],
        *,
        session_id: str | None,
        delivery_id: str | None,
    ) -> bool:
        table = "nodes" if claim_kind == "node" else "edges"
        with self._connect() as connection:
            with connection:
                row = connection.execute(
                    f"SELECT * FROM {table} WHERE id = ?", (claim_id,)
                ).fetchone()
                if row is None or row["status"] != "proposed":
                    return False
                connection.execute(
                    f"UPDATE {table} SET status = ?, last_seen = ? WHERE id = ?",
                    (target, _utc_now(), claim_id),
                )
                event_id = self._record_event(
                    connection,
                    claim_kind,
                    claim_id,
                    "proposed",
                    target,
                    "user_confirmed" if target == "confirmed" else "user_rejected",
                    session_id=session_id,
                    delivery_id=delivery_id,
                    details={
                        "evidence_count": int(row["n_auditable_occurrences"]),
                        "session_count": int(row["n_auditable_sessions"]),
                    },
                )
                if target == "confirmed":
                    connection.execute(
                        """
                        INSERT INTO claim_evidence (
                            id, claim_kind, claim_id, source_type,
                            source_session_id, observed_at, source_state,
                            extractor_version, validation_event_id, evidence_key
                        ) VALUES (?, ?, ?, 'user-stated', ?, ?, 'active',
                                  'conversational-validation-v1', ?, ?)
                        """,
                        (
                            uuid4().hex,
                            claim_kind,
                            claim_id,
                            session_id,
                            _utc_now(),
                            event_id,
                            f"validation:{event_id}",
                        ),
                    )
                    self._refresh_counts(connection, claim_kind, claim_id)
        return True

    def confirm_node(
        self,
        node_id: int,
        *,
        session_id: str | None = None,
        delivery_id: str | None = None,
    ) -> bool:
        """Confirm only the proposed node explicitly raised to owner."""
        return self._resolve(
            "node", node_id, "confirmed", session_id=session_id, delivery_id=delivery_id
        )

    def reject_node(
        self,
        node_id: int,
        *,
        session_id: str | None = None,
        delivery_id: str | None = None,
    ) -> bool:
        """Durably reject proposed node at current evidence snapshot."""
        return self._resolve(
            "node", node_id, "rejected", session_id=session_id, delivery_id=delivery_id
        )

    def confirm_edge(
        self,
        edge_id: int,
        *,
        session_id: str | None = None,
        delivery_id: str | None = None,
    ) -> bool:
        """Confirm only the proposed edge explicitly raised to owner."""
        return self._resolve(
            "edge", edge_id, "confirmed", session_id=session_id, delivery_id=delivery_id
        )

    def reject_edge(
        self,
        edge_id: int,
        *,
        session_id: str | None = None,
        delivery_id: str | None = None,
    ) -> bool:
        """Durably reject proposed edge at current evidence snapshot."""
        return self._resolve(
            "edge", edge_id, "rejected", session_id=session_id, delivery_id=delivery_id
        )

    def pending_insights(self) -> list[GraphNode]:
        """Return node proposals awaiting owner response."""
        return self.nodes(status="proposed")

    def pending_edge_insights(self) -> list[GraphEdge]:
        """Return edge proposals awaiting owner response."""
        return self.edges(status="proposed")

    def lifecycle_events(
        self, claim_kind: ClaimKind | None = None, claim_id: int | None = None
    ) -> list[dict[str, object]]:
        """Return auditable claim transition history."""
        query = "SELECT * FROM lifecycle_events"
        clauses: list[str] = []
        params: list[object] = []
        if claim_kind is not None:
            clauses.append("claim_kind = ?")
            params.append(claim_kind)
        if claim_id is not None:
            clauses.append("claim_id = ?")
            params.append(claim_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id"
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(str(item.pop("details_json")))
            result.append(item)
        return result

    def revalidate_stale(self, now: datetime | None = None) -> int:
        """Revalidate stale claims and record only actual mutation work."""
        try:
            changed = self._revalidate_stale(now)
        except Exception as error:
            _record_graph_mutation("revalidate", _graph_error_outcome(error))
            raise
        if changed:
            _record_graph_mutation("revalidate", "success")
        return changed

    def _revalidate_stale(self, now: datetime | None = None) -> int:
        """Flag decayed thread/pattern/strategy and edge claims for review."""
        current = now or datetime.now(UTC)
        changed = 0
        with self._connect() as connection:
            with connection:
                for table, kind in (("nodes", "node"), ("edges", "edge")):
                    rows = connection.execute(
                        f"""
                        SELECT id, status FROM {table}
                        WHERE revalidation_due_at IS NOT NULL
                          AND revalidation_due_at <= ?
                          AND status IN ('observation','proposed','confirmed')
                        """,
                        (current.isoformat(),),
                    ).fetchall()
                    for row in rows:
                        connection.execute(
                            f"UPDATE {table} SET status = 'needs_revalidation' WHERE id = ?",
                            (row["id"],),
                        )
                        self._record_event(
                            connection,
                            cast(ClaimKind, kind),
                            int(row["id"]),
                            str(row["status"]),
                            "needs_revalidation",
                            "decay",
                        )
                        changed += 1
        return changed

    # ------------------------------------------------------------------
    # Session deletion and source availability
    # ------------------------------------------------------------------

    def delete_session_evidence(
        self, session_id: str, *, remove_derived: bool = False
    ) -> dict[str, int]:
        """Sanitize or remove knowledge provenance before transcript deletion."""
        now = _utc_now()
        affected: set[tuple[ClaimKind, int]] = set()
        with self._connect() as connection:
            with connection:
                rows = connection.execute(
                    """
                    SELECT id, claim_kind, claim_id FROM claim_evidence
                    WHERE source_session_id = ?
                    """,
                    (session_id,),
                ).fetchall()
                for row in rows:
                    affected.add(
                        (cast(ClaimKind, row["claim_kind"]), int(row["claim_id"]))
                    )
                    if remove_derived:
                        connection.execute(
                            "DELETE FROM claim_evidence WHERE id = ?", (row["id"],)
                        )
                    else:
                        connection.execute(
                            """
                            UPDATE claim_evidence SET quote_text = NULL,
                                source_state = 'deleted', source_deleted_at = ?,
                                source_marker = ? WHERE id = ?
                            """,
                            (
                                now,
                                self._digest(
                                    f"deleted-source:{session_id}:{row['id']}"
                                ),
                                row["id"],
                            ),
                        )
                removed_claims = 0
                revalidation = 0
                for kind, claim_id in affected:
                    table = "nodes" if kind == "node" else "edges"
                    self._refresh_counts(connection, kind, claim_id)
                    claim = connection.execute(
                        f"SELECT * FROM {table} WHERE id = ?", (claim_id,)
                    ).fetchone()
                    if claim is None:
                        continue
                    remaining = int(claim["n_occurrences"])
                    if remove_derived and remaining == 0:
                        connection.execute(
                            "DELETE FROM pending_insights "
                            "WHERE claim_kind = ? AND claim_id = ?",
                            (kind, claim_id),
                        )
                        connection.execute(
                            f"DELETE FROM {table} WHERE id = ?", (claim_id,)
                        )
                        removed_claims += 1
                        continue
                    if claim["status"] != "confirmed" and not self.is_eligible(
                        dict(claim)
                    ):
                        connection.execute(
                            f"UPDATE {table} SET status = 'needs_revalidation' WHERE id = ?",
                            (claim_id,),
                        )
                        self._record_event(
                            connection,
                            kind,
                            claim_id,
                            str(claim["status"]),
                            "needs_revalidation",
                            "source_deleted",
                            session_id=session_id,
                        )
                        revalidation += 1
                connection.execute(
                    "DELETE FROM observation_inbox WHERE session_id = ?", (session_id,)
                )
                connection.execute(
                    "DELETE FROM distillation_runs WHERE session_id = ?", (session_id,)
                )
                connection.execute(
                    "UPDATE lifecycle_events SET session_id = NULL WHERE session_id = ?",
                    (session_id,),
                )
                connection.execute(
                    "UPDATE insight_history SET session_id = NULL WHERE session_id = ?",
                    (session_id,),
                )
                connection.execute(
                    "DELETE FROM semantic_embeddings "
                    "WHERE entity_kind = 'episode' AND entity_id = ?",
                    (session_id,),
                )
                connection.execute(
                    "DELETE FROM claim_evidence WHERE claim_kind = 'node' "
                    "AND NOT EXISTS (SELECT 1 FROM nodes WHERE nodes.id = claim_id)"
                )
                connection.execute(
                    "DELETE FROM claim_evidence WHERE claim_kind = 'edge' "
                    "AND NOT EXISTS (SELECT 1 FROM edges WHERE edges.id = claim_id)"
                )
                connection.execute(
                    "DELETE FROM lifecycle_events WHERE claim_kind = 'node' "
                    "AND NOT EXISTS (SELECT 1 FROM nodes WHERE nodes.id = claim_id)"
                )
                connection.execute(
                    "DELETE FROM lifecycle_events WHERE claim_kind = 'edge' "
                    "AND NOT EXISTS (SELECT 1 FROM edges WHERE edges.id = claim_id)"
                )
        return {
            "evidence_affected": len(rows),
            "claims_removed": removed_claims,
            "claims_needing_revalidation": revalidation,
        }

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def _topic_explicitly_raises_boundary(self, topic: str) -> bool:
        lowered = topic.casefold()
        return any(
            value.casefold() in lowered for value in self.never_initiate_topics()
        )

    def graph_walk(
        self,
        topic: str,
        *,
        k: int = 8,
        now: datetime | None = None,
        allow_never_initiate: bool | None = None,
    ) -> GraphWalk:
        """Select relevant nodes then traverse confirmed outward edges under cap."""
        if not 1 <= k <= 50:
            raise ValueError("k must be between 1 and 50")
        self.revalidate_stale(now)
        topic_tokens = tokens(topic)
        allow_private = (
            self._topic_explicitly_raises_boundary(topic)
            if allow_never_initiate is None
            else allow_never_initiate
        )
        scored: list[tuple[float, GraphNode]] = []
        for node in self.nodes():
            if node["status"] in {"rejected", "superseded"}:
                continue
            boundary_match = any(
                value.casefold() in node["statement"].casefold()
                for value in self.never_initiate_topics()
            )
            if (node["never_initiate"] or boundary_match) and not allow_private:
                continue
            overlap = len(topic_tokens & tokens(node["statement"]))
            if overlap == 0:
                continue
            score = overlap * _STATUS_WEIGHT.get(node["status"], 0.0)
            scored.append((score, node))
        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        selected = [node for _, node in scored[:k]]
        selected_ids = {node["id"] for node in selected}
        candidate_edges = [
            edge
            for edge in self.edges(status="confirmed")
            if edge["src"] in selected_ids or edge["dst"] in selected_ids
        ]
        by_id = {node["id"]: node for node in self.nodes(status="confirmed")}
        for edge in candidate_edges:
            for endpoint in (edge["src"], edge["dst"]):
                if len(selected) >= k or endpoint in selected_ids:
                    continue
                node = by_id.get(endpoint)
                if node is not None and (allow_private or not node["never_initiate"]):
                    selected.append(node)
                    selected_ids.add(endpoint)
        edges = [
            edge
            for edge in candidate_edges
            if edge["src"] in selected_ids and edge["dst"] in selected_ids
        ][:k]
        return {
            "nodes": selected,
            "edges": edges,
            "never_initiate": self.never_initiate_topics(),
        }

    def assemble_context(
        self, topic: str = "", *, k: int = 8, now: datetime | None = None
    ) -> GraphContext:
        """Build bounded per-turn graph context with standing goals/threads."""
        allowed = {"confirmed", "proposed"}
        allow_private = self._topic_explicitly_raises_boundary(topic)
        private_patterns = [value.casefold() for value in self.never_initiate_topics()]

        def visible(node: GraphNode) -> bool:
            return allow_private or (
                not node["never_initiate"]
                and not any(
                    pattern in node["statement"].casefold()
                    for pattern in private_patterns
                )
            )

        identity = [
            node
            for node in self.nodes(type_="identity_fact")
            if node["status"] in allowed and visible(node)
        ][:4]
        preferences = [
            node
            for node in self.nodes(type_="preference")
            if node["status"] in allowed and visible(node)
        ][:4]
        goals = [
            node
            for node in self.nodes(type_="goal")
            if node["status"] in allowed and visible(node)
        ][:4]
        threads = [
            node
            for node in self.nodes(type_="thread")
            if node["status"] in allowed and visible(node)
        ][:4]
        walk = (
            self.graph_walk(topic, k=k, now=now)
            if topic.strip()
            else cast(
                GraphWalk,
                {
                    "nodes": [],
                    "edges": [],
                    "never_initiate": self.never_initiate_topics(),
                },
            )
        )
        return {
            "identity": identity,
            "preferences": preferences,
            "goals": goals,
            "threads": threads,
            "never_initiate": walk["never_initiate"],
            "walk_nodes": walk["nodes"],
            "walk_edges": walk["edges"],
        }

    # ------------------------------------------------------------------
    # Migration/export/delete
    # ------------------------------------------------------------------

    def migrate_v1_facts(self) -> int:
        """Import legacy flat facts once, then retire their table."""
        with self._connect() as connection:
            if not self._table_exists(connection, "facts"):
                return 0
            facts = connection.execute("SELECT * FROM facts ORDER BY id").fetchall()
        migrated = 0
        for fact in facts:
            statement = str(fact["statement"]).strip()
            if not statement or self.is_never_store(statement):
                continue
            count = max(int(fact["n_occurrences"]), 1)
            node_id: int | None = None
            for index in range(count):
                node_id = self.upsert_node(
                    _fact_kind_to_type(str(fact["kind"])),
                    statement,
                    extractor_version="flat-fact-migration-v1",
                    evidence_key=f"flat-fact:{fact['id']}:{index}",
                )
            if node_id is not None:
                with self._connect() as connection:
                    with connection:
                        connection.execute(
                            "UPDATE nodes SET first_seen = ?, last_seen = ? WHERE id = ?",
                            (fact["first_seen"], fact["last_seen"], node_id),
                        )
                migrated += 1
        with self._connect() as connection:
            with connection:
                connection.execute("DROP TABLE facts")
        return migrated

    def schema_version(self) -> int:
        """Return applied knowledge schema version."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT version FROM schema_migrations WHERE component = 'knowledge'"
            ).fetchone()
        return int(row["version"]) if row is not None else 0

    def export_all(self) -> dict[str, object]:
        """Export complete graph, provenance, inbox state, and deletion audit."""
        table_names = (
            "claim_evidence",
            "lifecycle_events",
            "observation_inbox",
            "distillation_runs",
            "privacy_purge_events",
            "tombstones",
            "pending_insights",
            "insight_history",
            "semantic_embeddings",
        )
        with self._connect() as connection:
            tables = {
                name: [dict(row) for row in connection.execute(f"SELECT * FROM {name}")]
                for name in table_names
            }
        return {
            "schema_version": self.schema_version(),
            "node_types": list(NODE_TYPES),
            "edge_types": list(EDGE_TYPES),
            "nodes": self.nodes(),
            "edges": self.edges(),
            "boundaries": self.boundaries(),
            **tables,
        }

    def delete_all(self) -> None:
        """Erase every graph/personal audit table and local tombstone key."""
        tables = (
            "insight_history",
            "pending_insights",
            "semantic_embeddings",
            "claim_evidence",
            "lifecycle_events",
            "observation_inbox",
            "distillation_runs",
            "privacy_purge_events",
            "edges",
            "nodes",
            "boundaries",
            "tombstones",
        )
        with self._connect() as connection:
            with connection:
                for table in tables:
                    connection.execute(f"DELETE FROM {table}")
        key_path = self.data_dir / TOMBSTONE_KEY_FILE
        key_path.unlink(missing_ok=True)
        self._tombstone_key = self._load_tombstone_key()


def render_context(assembled: GraphContext) -> str | None:
    """Render role-separated graph context without making episodic claims truth."""
    sections = (
        ("Identity", assembled["identity"]),
        ("Preferences", assembled["preferences"]),
        ("Active goals", assembled["goals"]),
        ("Active threads", assembled["threads"]),
    )
    if not any(nodes for _, nodes in sections) and not any(
        (assembled["never_initiate"], assembled["walk_nodes"])
    ):
        return None
    parts = ["# Confirmed and evidence-gated knowledge about the user"]
    seen: set[int] = set()
    for heading, nodes in sections:
        if not nodes:
            continue
        parts.append(
            heading
            + ":\n"
            + "\n".join(f"- ({node['status']}) {node['statement']}" for node in nodes)
        )
        seen.update(node["id"] for node in nodes)
    walk_nodes = [node for node in assembled["walk_nodes"] if node["id"] not in seen]
    if walk_nodes:
        by_id = {node["id"]: node for node in assembled["walk_nodes"]}
        lines = [f"- ({node['status']}) {node['statement']}" for node in walk_nodes]
        for edge in assembled["walk_edges"]:
            src = by_id.get(edge["src"])
            dst = by_id.get(edge["dst"])
            if src is not None and dst is not None:
                lines.append(
                    f"- {src['statement']} — {edge['statement']} → {dst['statement']}"
                )
        parts.append("Relevant now:\n" + "\n".join(lines))
    if assembled["never_initiate"]:
        parts.append(
            "Never initiate protected topics unless the current user text explicitly "
            "raises one. Application filtering has already omitted their private values."
        )
    parts.append(
        "Precedence: current explicit user statement > owner edit/direct statement > "
        "confirmed graph > proposed claim > observation. Never diagnose from patterns."
    )
    return "\n\n".join(parts)


__all__ = [
    "EDGE_TYPES",
    "GRADUATION_MIN_OCCURRENCES",
    "GRADUATION_MIN_SESSIONS",
    "LEGACY_EDGE_TYPE_MAP",
    "LEGACY_NODE_TYPE_MAP",
    "NODE_TYPES",
    "STATUSES",
    "ClaimEvidence",
    "GraphContext",
    "GraphContextProvider",
    "GraphEdge",
    "GraphNode",
    "GraphQuote",
    "GraphWalk",
    "UserModel",
    "render_context",
]
