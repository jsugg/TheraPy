"""Tests for longitudinal insight and the pending-insights inbox (W4)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from therapy.knowledge import insight
from therapy.knowledge.user_model import UserModel


def _graduate_to_pattern(model: UserModel, type_: str, statement: str) -> int:
    nid = model.upsert_node(type_, statement, session_id="s1")
    model.upsert_node(type_, statement, session_id="s1")
    model.upsert_node(type_, statement, session_id="s2")
    assert model.propose(nid) is True
    return nid


def test_cross_session_patterns_ranks_recurring_and_hides_never_initiate(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)
    _graduate_to_pattern(model, "routine", "Skips lunch when busy.")
    once = model.upsert_node("note", "Watched a film once.", session_id="s1")
    assert once is not None
    private = model.upsert_node(
        "thread", "A private recurring worry.", never_initiate=True, session_id="s1"
    )
    model.upsert_node("thread", "A private recurring worry.", session_id="s2")
    assert private is not None

    recurring = insight.cross_session_patterns(model)
    statements = [n["statement"] for n in recurring]
    assert "Skips lunch when busy." in statements
    assert "Watched a film once." not in statements  # single session
    assert "A private recurring worry." not in statements  # never_initiate


def test_pending_insights_and_context_aware_raise(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    _graduate_to_pattern(model, "routine", "Skips lunch when busy.")

    pending = insight.pending_insights(model)
    assert len(pending) == 1
    # Raise only when the topic is adjacent; otherwise stay queued.
    assert insight.next_insight_for_topic(model, "talking about lunch and meals")
    assert insight.next_insight_for_topic(model, "the weather today") is None


def test_session_recap_uses_injected_recapper(tmp_path: Path) -> None:
    async def recapper(transcript: str) -> str:
        return f"recap of {len(transcript)} chars"

    turns = [{"role": "user", "text": "hi", "language": "en", "modality": "text"}]
    result = asyncio.run(insight.session_recap(turns, recapper=recapper))
    assert result.startswith("recap of")
    assert asyncio.run(insight.session_recap([], recapper=recapper)) == ""
