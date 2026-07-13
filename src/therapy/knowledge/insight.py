"""Longitudinal insight: cross-session patterns, recaps, pending inbox (W4).

The non-emotional half of the SPEC's P2 loop (emotional patterns wait for
`ser`). Framework-free: queries run over the property graph; the optional
recap uses the provider-agnostic `complete()` and is injectable so tests stay
deterministic.

The pending-insights inbox is the graph's proposed patterns awaiting the
user's confirm/reject. A proposed insight is raised *in conversation* only when
the current topic is adjacent — otherwise it is queued, so a reflection never
derails the moment (SPEC §3).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from therapy.knowledge.user_model import GRADUATION_MIN_SESSIONS, UserModel, _tokens
from therapy.memory.summarizer import complete, render_transcript

Recapper = Callable[[str], Awaitable[str]]

RECAP_PROMPT = """Write a brief end-of-session reflection for the user: two or
three sentences naming what came up and any thread left open. Address the user
as "you". Phrase observations as data, never as diagnosis. No advice unless the
user asked. Plain prose, no lists."""


def cross_session_patterns(model: UserModel) -> list[dict[str, Any]]:
    """Claims seen across multiple sessions, most-recurring first.

    The raw material for longitudinal reflection: an observation that keeps
    resurfacing across sessions is a candidate pattern even before it graduates.
    Ranked by session spread then occurrence count; `never_initiate` claims are
    excluded (they may be held, never surfaced).
    """
    recurring = [
        node
        for node in model.nodes()
        if not node["never_initiate"]
        and int(node["n_sessions"]) >= GRADUATION_MIN_SESSIONS
    ]
    recurring.sort(
        key=lambda n: (int(n["n_sessions"]), int(n["n_occurrences"])), reverse=True
    )
    return recurring


def pending_insights(model: UserModel) -> list[dict[str, Any]]:
    """Proposed patterns awaiting the user's confirmation (the W4 inbox)."""
    return model.pending_insights()


def should_raise_now(insight: dict[str, Any], topic: str) -> bool:
    """Whether a pending insight is adjacent enough to raise in the moment.

    Context-aware surfacing: only when the insight's statement shares
    vocabulary with what's being discussed. Otherwise the caller queues it.
    """
    if insight["never_initiate"]:
        return False
    return bool(_tokens(str(insight["statement"])) & _tokens(topic))


def next_insight_for_topic(
    model: UserModel, topic: str
) -> dict[str, Any] | None:
    """The first pending insight adjacent to `topic`, or None to stay queued."""
    for insight in pending_insights(model):
        if should_raise_now(insight, topic):
            return insight
    return None


async def session_recap(
    turns: list[dict[str, Any]], *, recapper: Recapper | None = None
) -> str:
    """End-of-session reflection text (text-derived; empty for no turns)."""
    if not turns:
        return ""
    if recapper is not None:
        return await recapper(render_transcript(turns))
    return await complete(RECAP_PROMPT, render_transcript(turns), max_tokens=160)
