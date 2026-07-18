"""Per-turn bounded multilingual graph + episodic context tests (Phase 4 C)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.type_contracts import MetricCall, metric_recorder
from therapy.knowledge.context import (
    MEMORY_MARKER,
    ContextAssembler,
    ContextBudget,
    replace_longitudinal_context,
)
from therapy.knowledge.embeddings import EmbeddingMetadata
from therapy.knowledge.user_model import UserModel
from therapy.memory import MemoryStore


class SemanticTestEmbedder:
    """Deterministic es/en/pt concept vectors at external model boundary."""

    @property
    def metadata(self) -> EmbeddingMetadata:
        return EmbeddingMetadata("test-multilingual", "v1", 4)

    @staticmethod
    def _embed(text: str) -> np.ndarray:
        lowered = text.casefold()
        groups = (
            ("sleep", "sono", "dormir", "sueño"),
            ("lunch", "almoço", "almoco", "comida"),
            ("father", "pai", "padre"),
            ("portfolio", "portfólio", "portafolio", "checklist", "planning", "lista"),
        )
        vector = np.asarray(
            [
                1.0 if any(word in lowered for word in group) else 0.0
                for group in groups
            ],
            dtype=np.float32,
        )
        if not vector.any():
            vector[:] = 0.01
        return vector / np.linalg.norm(vector)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


def _assembler(tmp_path: Path) -> tuple[ContextAssembler, UserModel, MemoryStore]:
    store = MemoryStore(tmp_path)
    model = UserModel(tmp_path)
    assembler = ContextAssembler(
        model,
        store,
        embedder=SemanticTestEmbedder(),
        budget=ContextBudget(
            max_nodes=6,
            max_edges=4,
            max_episodes=2,
            max_tokens=500,
            node_threshold=0.7,
            episode_threshold=0.7,
        ),
    )
    return assembler, model, store


def test_context_budget_clipping_records_bounded_source_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from therapy.observability import telemetry

    store = MemoryStore(tmp_path)
    model = UserModel(tmp_path)
    assembler = ContextAssembler(
        model,
        store,
        embedder=SemanticTestEmbedder(),
        budget=ContextBudget(max_tokens=100, node_threshold=0.0),
    )
    model.add_user_statement("pattern", "Sleep " + "planning " * 120)
    calls: list[MetricCall] = []
    monkeypatch.setattr(
        telemetry,
        "record_metric",
        metric_recorder(calls),
    )

    result = assembler.assemble("sleep planning", "synthetic-session")

    assert result["note"] is not None
    assert result["note"].endswith("[Context clipped to token budget]")
    truncations = [
        attrs
        for name, value, attrs in calls
        if name == "therapy_context_truncations_total" and value == 1
    ]
    assert truncations == [{"source": "graph"}]
    stages = {
        attrs["stage"]
        for name, _, attrs in calls
        if name == "therapy_context_stage_seconds" and attrs["outcome"] == "success"
    }
    assert stages == {"graph", "episode", "insight", "research", "embed"}
    selected_sources = {
        attrs["source"]
        for name, _, attrs in calls
        if name == "therapy_context_selected_items"
    }
    assert selected_sources == {"graph", "edge", "episode", "insight", "research"}
    assert any(name == "therapy_context_rendered_chars" for name, _, _ in calls)
    assert "planning" not in repr(truncations)


def test_context_stage_failure_is_bounded_and_content_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from therapy.observability import telemetry

    assembler, _, _ = _assembler(tmp_path)
    calls: list[MetricCall] = []
    monkeypatch.setattr(
        telemetry,
        "record_metric",
        metric_recorder(calls),
    )

    def fail_query(_text: str) -> np.ndarray:
        raise RuntimeError("private-context-stage-canary")

    monkeypatch.setattr(assembler.embedder, "embed_query", fail_query)
    with pytest.raises(RuntimeError, match="private-context-stage"):
        assembler.assemble("private-topic-canary", "private-session-canary")

    assert any(
        name == "therapy_context_stage_seconds"
        and attrs == {"stage": "embed", "outcome": "error"}
        for name, _, attrs in calls
    )
    assert any(
        name == "therapy_context_assembly_seconds"
        and attrs == {"outcome": "error"}
        for name, _, attrs in calls
    )
    assert "private-" not in repr(calls)


def test_portuguese_topic_retrieves_english_graph_and_relevant_episode(
    tmp_path: Path,
) -> None:
    assembler, model, store = _assembler(tmp_path)
    lunch = model.add_user_statement("pattern", "Skips lunch when work is busy.")
    goal = model.add_user_statement("goal", "Finish the portfolio.")
    assert lunch is not None
    assert goal is not None
    relevant = store.create_session()
    store.end_session(relevant, "The user skipped lunch during a deadline.")
    irrelevant = store.create_session()
    store.end_session(irrelevant, "The user slept well after a quiet evening.")

    turn = assembler.assemble("Quero falar sobre pular o almoço.", "current")

    graph_ids = {
        node["id"] for key in ("goals", "walk_nodes") for node in turn["graph"][key]
    }
    assert lunch in graph_ids
    assert goal in graph_ids
    assert [episode["session_id"] for episode in turn["episodes"]] == [relevant]
    assert turn["note"] is not None
    assert "not established truth" in turn["note"]


def test_never_initiate_episode_is_hidden_until_user_explicitly_raises_it(
    tmp_path: Path,
) -> None:
    assembler, model, store = _assembler(tmp_path)
    model.add_boundary("never_initiate", "pai")
    model.add_boundary("never_initiate", "salary")
    private = model.add_user_statement(
        "thread", "Conflict with father.", never_initiate=True
    )
    unrelated_private = model.add_user_statement(
        "thread", "Salary negotiation remains private.", never_initiate=True
    )
    unbound_private = model.add_user_statement(
        "thread", "Medical appointment planning.", never_initiate=True
    )
    assert private is not None
    assert unrelated_private is not None
    assert unbound_private is not None
    episode = store.create_session()
    store.end_session(episode, "The user described conflict with pai and family.")
    salary_episode = store.create_session()
    store.end_session(salary_episode, "The user discussed salary negotiation.")

    unsolicited = assembler.assemble("family stress", "current")
    explicit = assembler.assemble("Quero falar do meu pai", "current")
    unbound_explicit = assembler.assemble("Medical appointment planning", "other")

    unsolicited_ids = {node["id"] for node in unsolicited["graph"]["walk_nodes"]}
    explicit_ids = {
        node["id"]
        for key in ("threads", "walk_nodes")
        for node in explicit["graph"][key]
    }
    assert private not in unsolicited_ids
    assert unsolicited["episodes"] == []
    assert private in explicit_ids
    assert explicit["episodes"][0]["session_id"] == episode
    assert unrelated_private not in explicit_ids
    assert all(item["session_id"] != salary_episode for item in explicit["episodes"])
    assert unbound_private in {
        node["id"]
        for key in ("threads", "walk_nodes")
        for node in unbound_explicit["graph"][key]
    }


def test_semantic_adjacency_delivers_insight_then_confirmation_refreshes_graph(
    tmp_path: Path,
) -> None:
    assembler, model, _store = _assembler(tmp_path)
    node_id: int | None = None
    for session_id in ("s1", "s1", "s2"):
        node_id = model.upsert_node(
            "pattern", "Skips lunch when work is busy.", session_id=session_id
        )
    assert node_id is not None
    assert model.propose(node_id)

    proposed = assembler.assemble("Tenho problemas com o almoço.", "current")
    assert proposed["insight"] is not None
    proposed_note = proposed["note"]
    assert proposed_note is not None
    assert "Does that feel accurate" in proposed_note

    confirmed = assembler.assemble("Sim", "current")
    assert confirmed["insight"] is None
    node = model.get_node(node_id)
    assert node is not None
    assert node["status"] == "confirmed"


def test_context_replacement_keeps_one_bounded_memory_block() -> None:
    messages = [
        {"role": "system", "content": "base"},
        {"role": "system", "content": MEMORY_MARKER + "\nold"},
        {"role": "user", "content": "hello"},
    ]

    refreshed = replace_longitudinal_context(messages, MEMORY_MARKER + "\nnew")
    cleared = replace_longitudinal_context(refreshed, None)

    assert [message["content"] for message in refreshed].count(
        MEMORY_MARKER + "\nnew"
    ) == 1
    assert all(
        not str(message["content"]).startswith(MEMORY_MARKER) for message in cleared
    )


def test_per_turn_context_silently_includes_relevant_curated_technique(
    tmp_path: Path,
) -> None:
    assembler, _model, _store = _assembler(tmp_path)
    assembler.research.ingest(
        "Planning supports",
        "OT Review 2025",
        "A visible checklist can reduce planning load during task transitions.",
    )

    turn = assembler.assemble("Uma lista ajudaria no planejamento", "current")

    note = turn["note"]
    assert note is not None
    assert "visible checklist" in turn["research"]
    assert "UNTRUSTED REFERENCE MATERIAL" in turn["research"]
    assert "Silent technique grounding" in note
