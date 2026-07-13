"""Tests for distillation, promotion, and graduation (W2)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from therapy.knowledge import distill
from therapy.knowledge.user_model import UserModel


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_parse_candidates_tolerates_fences_and_prose() -> None:
    raw = (
        "Sure! Here you go:\n```json\n"
        '[{"kind":"node","type":"routine","statement":"Runs mornings."},'
        '{"nope":true}]\n```\nHope that helps.'
    )
    parsed = distill.parse_candidates(raw)
    assert parsed == [{"kind": "node", "type": "routine", "statement": "Runs mornings."}]
    assert distill.parse_candidates("no json here") == []


def test_promote_writes_nodes_and_resolves_edge_endpoints(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    candidates: list[dict[str, Any]] = [
        {"kind": "node", "type": "trigger", "statement": "Coffee after 4pm."},
        {"kind": "node", "type": "thread", "statement": "Poor sleep."},
        {
            "kind": "edge",
            "type": "causes",
            "src": "Coffee after 4pm.",
            "dst": "Poor sleep.",
            "statement": "caffeine hurts sleep",
        },
    ]
    node_ids, edge_ids = distill.promote(model, candidates, "s1")
    assert len(node_ids) == 2
    assert len(edge_ids) == 1
    edge = model.edges()[0]
    assert edge["type"] == "causes"


def test_distill_session_uses_inbox_and_runs_graduation(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    # Seed an inbox observation and pre-existing recurrence so the floor clears.
    model.add_observation("keeps skipping lunch", session_id="s1")
    nid = model.upsert_node("routine", "Skips lunch.", session_id="s0")
    model.upsert_node("routine", "Skips lunch.", session_id="s1")

    async def fake_extract(transcript: str, observations: list[str]) -> list[dict]:
        assert "skipping lunch" in " ".join(observations)
        return [{"kind": "node", "type": "routine", "statement": "Skips lunch."}]

    result = _run(
        distill.distill_session(model, [], "s2", extractor=fake_extract)
    )
    # Third occurrence across a third session -> eligible -> proposed to pattern.
    assert nid in result.proposed_patterns
    assert model.get_node(nid)["status"] == "pattern"
    # Inbox consumed, so it will not be reprocessed next run.
    assert model.pending_observations() == []
    assert result.processed_observations


def test_distill_respects_never_store(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    model.add_boundary("never_store", "affair")

    async def fake_extract(transcript: str, observations: list[str]) -> list[dict]:
        return [{"kind": "node", "type": "note", "statement": "Mentioned an affair."}]

    result = _run(distill.distill_session(model, [], "s1", extractor=fake_extract))
    assert result.promoted_nodes == []
    assert model.nodes() == []
