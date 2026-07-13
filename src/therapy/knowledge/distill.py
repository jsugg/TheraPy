"""Distillation: observation inbox -> graph promotion; graduation.

Runs between sessions, off the realtime path. Freeform observations are
written to the inbox *during* conversation (zero schema pressure — see
`UserModel.add_observation`); this module turns the inbox plus the session
transcript into typed nodes/edges, attaching verbatim quotes as evidence, and
then applies the graduation floor.

Framework-free (SPEC dependency boundary): the only outside call is the
provider-agnostic `complete()` shared with the summarizer — no pipeline
services. The structured extractor is injectable so tests (and the
deterministic acceptance) can drive promotion without a live LLM; the default
uses schema-constrained JSON so a weak local model still yields parseable
output (SPEC §10 risk: gemma3:4b is poor at structured extraction).

Graduation (SPEC §3): the mechanical floor (>=3 occurrences across >=2
sessions) makes an observation *eligible*; LLM judgment then *proposes* it as a
pattern; only explicit user confirmation (`UserModel.confirm_node`) reaches
`confirmed`. `user-stated` claims start confirmed and skip the ladder.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from therapy.knowledge.user_model import EDGE_TYPES, NODE_TYPES, UserModel
from therapy.memory.summarizer import complete, render_transcript

# The extractor takes (transcript_text, observation_texts) and returns a list
# of candidate dicts. Injectable so tests can supply a deterministic function.
Extractor = Callable[[str, list[str]], Awaitable[list[dict[str, Any]]]]

EXTRACTION_PROMPT = """From the session transcript and the loose observations,
extract durable, distinctive facts about the user as a JSON array. Each element:
  {"kind": "node", "type": <one of TYPES>, "statement": <canonical English>,
   "quotes": [{"text": <verbatim user words>, "lang": <iso code>}],
   "never_initiate": <true only if the user asked you never to raise this>}
or a relationship between two node statements:
  {"kind": "edge", "type": <one of EDGE_TYPES>, "src": <statement>,
   "dst": <statement>, "statement": <short English gloss>}
Rules: phrase statements as observed data, never as diagnosis. Keep names and
specifics verbatim in quotes. Only include what the transcript supports. Output
ONLY the JSON array; no prose. If nothing is worth keeping, output [].
TYPES = {types}
EDGE_TYPES = {edge_types}"""


@dataclass
class DistillResult:
    """What one between-session distillation run changed in the graph."""

    promoted_nodes: list[int] = field(default_factory=list)
    promoted_edges: list[int] = field(default_factory=list)
    proposed_patterns: list[int] = field(default_factory=list)
    processed_observations: list[int] = field(default_factory=list)


def parse_candidates(raw: str) -> list[dict[str, Any]]:
    """Parse extractor output into candidate dicts (defensive by design).

    Tolerates a leading ```json fence and trailing prose: the first balanced
    JSON array in the string is used. Non-conforming elements are dropped
    rather than raising — a weak model must not be able to crash distillation.
    """
    text = raw.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    candidates: list[dict[str, Any]] = []
    for element in parsed:
        if isinstance(element, dict) and element.get("statement"):
            candidates.append(element)
    return candidates


async def extract_candidates(
    transcript: str, observations: list[str]
) -> list[dict[str, Any]]:
    """Default LLM extractor: schema-constrained node/edge candidates."""
    system = EXTRACTION_PROMPT.format(
        types=", ".join(NODE_TYPES), edge_types=", ".join(EDGE_TYPES)
    )
    body = transcript
    if observations:
        body += "\n\nLoose observations:\n" + "\n".join(f"- {o}" for o in observations)
    raw = await complete(system, body, max_tokens=800)
    return parse_candidates(raw)


def promote(
    model: UserModel, candidates: list[dict[str, Any]], session_id: str | None
) -> tuple[list[int], list[int]]:
    """Upsert candidate nodes/edges into the graph, attaching quotes.

    Nodes are promoted first so edges can resolve their endpoints by statement.
    `never_store` and tombstones are enforced inside `UserModel`, so a blocked
    candidate simply does not land (returns None) and is skipped here. Returns
    the (node_ids, edge_ids) actually written.
    """
    node_ids: list[int] = []
    statement_to_id: dict[str, int] = {}
    for candidate in candidates:
        if candidate.get("kind") == "edge":
            continue
        type_ = str(candidate.get("type", "note"))
        if type_ not in NODE_TYPES:
            type_ = "note"
        node_id = model.upsert_node(
            type_,
            str(candidate["statement"]),
            source=_source_of(candidate),
            quotes=_quotes_of(candidate),
            session_id=session_id,
            never_initiate=bool(candidate.get("never_initiate", False)),
        )
        if node_id is not None:
            node_ids.append(node_id)
            statement_to_id[str(candidate["statement"]).strip()] = node_id

    edge_ids: list[int] = []
    for candidate in candidates:
        if candidate.get("kind") != "edge":
            continue
        type_ = str(candidate.get("type", "relates_to"))
        if type_ not in EDGE_TYPES:
            type_ = "relates_to"
        src = statement_to_id.get(str(candidate.get("src", "")).strip())
        dst = statement_to_id.get(str(candidate.get("dst", "")).strip())
        if src is None or dst is None or src == dst:
            continue
        edge_id = model.upsert_edge(
            src,
            dst,
            type_,
            statement=str(candidate.get("statement", "")),
            source=_source_of(candidate),
            session_id=session_id,
        )
        if edge_id is not None:
            edge_ids.append(edge_id)
    return node_ids, edge_ids


def graduate(model: UserModel) -> list[int]:
    """Propose every eligible observation as a pattern (mechanical floor).

    This is the automatable half of graduation: the floor (>=3/>=2) plus the
    LLM's pattern judgment. It stops at `pattern` — the jump to `confirmed`
    stays a deliberate user act, so sparse data can never mint a confirmed
    claim (SPEC §3, R1). Returns the node ids newly proposed.
    """
    proposed: list[int] = []
    for node in model.nodes(status="observation"):
        if model.is_eligible(node) and model.propose(int(node["id"])):
            proposed.append(int(node["id"]))
    return proposed


async def distill_session(
    model: UserModel,
    turns: list[dict[str, Any]],
    session_id: str | None,
    *,
    extractor: Extractor | None = None,
) -> DistillResult:
    """Run one between-session distillation: inbox + transcript -> graph.

    Pulls the session's pending observations, extracts structured candidates
    (via `extractor`, defaulting to the LLM), promotes them, marks the inbox
    consumed, then runs the graduation floor. Replaces the Phase-2 disconnect-
    time flat-fact extraction.
    """
    extract = extractor or extract_candidates
    pending = model.pending_observations()
    observation_texts = [str(row["text"]) for row in pending]
    transcript = render_transcript(turns)
    candidates = await extract(transcript, observation_texts)
    node_ids, edge_ids = promote(model, candidates, session_id)
    observation_ids = [int(row["id"]) for row in pending]
    model.mark_observations_processed(observation_ids)
    proposed = graduate(model)
    return DistillResult(
        promoted_nodes=node_ids,
        promoted_edges=edge_ids,
        proposed_patterns=proposed,
        processed_observations=observation_ids,
    )


def _quotes_of(candidate: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize a candidate's quotes to `[{text, lang}]`, dropping junk."""
    quotes: list[dict[str, str]] = []
    for quote in candidate.get("quotes", []) or []:
        if isinstance(quote, dict) and quote.get("text"):
            quotes.append(
                {"text": str(quote["text"]), "lang": str(quote.get("lang", ""))}
            )
    return quotes


def _source_of(candidate: dict[str, Any]) -> str:
    """A candidate's source, defaulting to `conversation` (never trusts `ser`)."""
    source = str(candidate.get("source", "conversation"))
    return source if source in {"conversation", "user-stated"} else "conversation"
