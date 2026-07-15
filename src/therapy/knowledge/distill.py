"""Transactional session distillation and evidence-gated graduation (Phase 4 B).

LLM output is untrusted: complete schema validation runs before writes, source
authority is assigned here, and every stored quote is matched to an actual user
turn. Candidate/evidence writes and inbox consumption commit as one transaction.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypedDict, cast

from therapy.knowledge.schema import (
    EDGE_TYPES,
    NODE_TYPES,
    canonical_edge_type,
    canonical_node_type,
)
from therapy.knowledge.user_model import GraphQuote, UserModel
from therapy.memory.summarizer import complete, render_transcript

EXTRACTOR_VERSION = "phase4-distill-v2"
MAX_CANDIDATES = 100
MAX_QUOTES_PER_CANDIDATE = 10
MAX_EXTRACTION_ATTEMPTS = 3

type RawCandidate = Mapping[str, object]
type Extractor = Callable[[str, list[str]], Awaitable[list[RawCandidate]]]
type Judger = Callable[[Literal["node", "edge"], Mapping[str, object]], Awaitable[bool]]


class CandidateValidationError(ValueError):
    """Extractor output failed complete boundary validation."""


class ValidatedNode(TypedDict):
    """Normalized node candidate safe for transactional application."""

    kind: Literal["node"]
    type: str
    statement: str
    quotes: list[GraphQuote]
    aliases: list[dict[str, str]]
    never_initiate: bool


class ValidatedEdge(TypedDict):
    """Normalized edge candidate safe for transactional application."""

    kind: Literal["edge"]
    type: str
    src: str
    dst: str
    statement: str
    quotes: list[GraphQuote]


type ValidatedCandidate = ValidatedNode | ValidatedEdge


EXTRACTION_PROMPT = """Extract only durable, transcript-supported self-knowledge.
Return ONLY a JSON array. Node schema:
{{"kind":"node","type":TYPE,"statement":"canonical English claim",
 "quotes":[{{"text":"exact user substring"}}],"aliases":[],
 "never_initiate":false}}
Edge schema:
{{"kind":"edge","type":EDGE_TYPE,"src":"exact node statement or alias",
 "dst":"exact node statement or alias","statement":"non-empty English claim",
 "quotes":[{{"text":"exact user substring"}}]}}
Never emit source/status/confirmation. Observations are data, never diagnosis.
Node types: {types}
Edge types: {edge_types}"""

JUDGMENT_PROMPT = """Judge whether recurring evidence supports a distinctive,
useful longitudinal self-knowledge proposal. Reject diagnosis, generic advice,
one-off events, and claims not supported by counts. Reply only YES or NO."""


@dataclass(slots=True)
class DistillResult:
    """Auditable outcome of one idempotent session distillation."""

    run_id: str
    promoted_nodes: list[int] = field(default_factory=list)
    promoted_edges: list[int] = field(default_factory=list)
    proposed_nodes: list[int] = field(default_factory=list)
    proposed_edges: list[int] = field(default_factory=list)
    processed_observations: list[int] = field(default_factory=list)

    @property
    def proposed_patterns(self) -> list[int]:
        """Compatibility name for callers predating `proposed` status."""
        return self.proposed_nodes


def parse_candidates(raw: str) -> list[dict[str, object]]:
    """Parse strict JSON array, allowing only one surrounding Markdown fence."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise CandidateValidationError("unterminated JSON fence")
        text = "\n".join(lines[1:-1]).strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise CandidateValidationError("extractor must return only one JSON array")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise CandidateValidationError(f"invalid candidate JSON: {error.msg}") from error
    if not isinstance(parsed, list):
        raise CandidateValidationError("candidate root must be an array")
    if len(parsed) > MAX_CANDIDATES:
        raise CandidateValidationError(f"candidate count exceeds {MAX_CANDIDATES}")
    if not all(isinstance(item, dict) for item in parsed):
        raise CandidateValidationError("every candidate must be an object")
    return cast(list[dict[str, object]], parsed)


def _bounded_text(value: object, field: str, limit: int = 4_000) -> str:
    if not isinstance(value, str):
        raise CandidateValidationError(f"{field} must be a string")
    text = value.strip()
    if not text or len(text) > limit:
        raise CandidateValidationError(f"{field} must contain 1..{limit} characters")
    return text


def _raw_quotes(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_QUOTES_PER_CANDIDATE:
        raise CandidateValidationError(
            f"quotes must be an array with at most {MAX_QUOTES_PER_CANDIDATE} items"
        )
    quotes: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) - {"text", "language", "lang"}:
            raise CandidateValidationError("quote objects may contain text/language only")
        quotes.append(_bounded_text(item.get("text"), "quote.text", 8_000))
    return quotes


def _raw_aliases(value: object) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 20:
        raise CandidateValidationError("aliases must be an array with at most 20 items")
    aliases: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, str):
            aliases.append({"text": _bounded_text(item, "alias", 1_000), "language": "und"})
            continue
        if not isinstance(item, dict) or set(item) - {"text", "language"}:
            raise CandidateValidationError("alias must be string or text/language object")
        aliases.append(
            {
                "text": _bounded_text(item.get("text"), "alias.text", 1_000),
                "language": str(item.get("language", "und"))[:32],
            }
        )
    return aliases


def validate_candidates(candidates: Sequence[RawCandidate]) -> list[dict[str, object]]:
    """Validate complete candidate schema and canonicalize registry values."""
    if len(candidates) > MAX_CANDIDATES:
        raise CandidateValidationError(f"candidate count exceeds {MAX_CANDIDATES}")
    validated: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        kind = candidate.get("kind")
        if kind == "node":
            allowed = {
                "kind",
                "type",
                "statement",
                "quotes",
                "aliases",
                "never_initiate",
            }
            if set(candidate) - allowed:
                raise CandidateValidationError(
                    f"node candidate {index} has unknown fields: {sorted(set(candidate) - allowed)}"
                )
            never_initiate = candidate.get("never_initiate", False)
            if not isinstance(never_initiate, bool):
                raise CandidateValidationError("never_initiate must be boolean")
            try:
                node_type = canonical_node_type(_bounded_text(candidate.get("type"), "type", 100))
            except ValueError as error:
                raise CandidateValidationError(str(error)) from error
            validated.append(
                {
                    "kind": "node",
                    "type": node_type,
                    "statement": _bounded_text(candidate.get("statement"), "statement"),
                    "raw_quotes": _raw_quotes(candidate.get("quotes")),
                    "aliases": _raw_aliases(candidate.get("aliases")),
                    "never_initiate": never_initiate,
                }
            )
            continue
        if kind == "edge":
            allowed = {"kind", "type", "src", "dst", "statement", "quotes"}
            if set(candidate) - allowed:
                raise CandidateValidationError(
                    f"edge candidate {index} has unknown fields: {sorted(set(candidate) - allowed)}"
                )
            try:
                edge_type = canonical_edge_type(_bounded_text(candidate.get("type"), "type", 100))
            except ValueError as error:
                raise CandidateValidationError(str(error)) from error
            validated.append(
                {
                    "kind": "edge",
                    "type": edge_type,
                    "src": _bounded_text(candidate.get("src"), "src"),
                    "dst": _bounded_text(candidate.get("dst"), "dst"),
                    "statement": _bounded_text(candidate.get("statement"), "statement"),
                    "raw_quotes": _raw_quotes(candidate.get("quotes")),
                }
            )
            continue
        raise CandidateValidationError(f"candidate {index} kind must be node or edge")
    return validated


def verify_quotes(
    candidates: Sequence[Mapping[str, object]],
    turns: Sequence[Mapping[str, object]],
    session_id: str,
) -> list[ValidatedCandidate]:
    """Replace model quote metadata with exact user-turn provenance."""
    user_turns: list[Mapping[str, object]] = []
    for turn in turns:
        role = turn.get("role")
        text = turn.get("text")
        if role not in {"user", "assistant"} or not isinstance(text, str):
            raise CandidateValidationError("turns require valid role and text")
        if role == "user":
            user_turns.append(turn)
    verified: list[ValidatedCandidate] = []
    for candidate in candidates:
        quotes: list[GraphQuote] = []
        for quote_text in cast(list[str], candidate.get("raw_quotes", [])):
            match = next(
                (turn for turn in user_turns if quote_text in str(turn["text"])), None
            )
            if match is None:
                raise CandidateValidationError(
                    f"quote is not verbatim in a user turn: {quote_text[:120]!r}"
                )
            quote: GraphQuote = {
                "text": quote_text,
                "language": str(match.get("language", "und")),
                "session_id": session_id,
            }
            if isinstance(match.get("id"), int):
                quote["turn_id"] = cast(int, match["id"])
            if isinstance(match.get("ts"), str):
                quote["observed_at"] = cast(str, match["ts"])
            quotes.append(quote)
        if candidate["kind"] == "node":
            aliases = cast(list[dict[str, str]], candidate.get("aliases", []))
            aliases.extend(
                {"text": quote["text"], "language": quote["language"]}
                for quote in quotes
            )
            verified.append(
                {
                    "kind": "node",
                    "type": str(candidate["type"]),
                    "statement": str(candidate["statement"]),
                    "quotes": quotes,
                    "aliases": aliases,
                    "never_initiate": bool(candidate["never_initiate"]),
                }
            )
        else:
            verified.append(
                {
                    "kind": "edge",
                    "type": str(candidate["type"]),
                    "src": str(candidate["src"]),
                    "dst": str(candidate["dst"]),
                    "statement": str(candidate["statement"]),
                    "quotes": quotes,
                }
            )
    return verified


async def extract_candidates(transcript: str, observations: list[str]) -> list[RawCandidate]:
    """Call provider-agnostic extractor and parse strict JSON output."""
    system = EXTRACTION_PROMPT.format(
        types=", ".join(NODE_TYPES), edge_types=", ".join(EDGE_TYPES)
    )
    body = transcript
    if observations:
        body += "\n\nSession observations:\n" + "\n".join(
            f"- {item}" for item in observations
        )
    raw = await complete(system, body, max_tokens=1_500)
    return parse_candidates(raw)


async def judge_candidate(
    claim_kind: Literal["node", "edge"], claim: Mapping[str, object]
) -> bool:
    """Default longitudinal judgment step after mechanical eligibility."""
    body = json.dumps(
        {
            "kind": claim_kind,
            "statement": claim["statement"],
            "occurrences": claim["n_auditable_occurrences"],
            "sessions": claim["n_auditable_sessions"],
            "source": claim["source"],
        },
        ensure_ascii=False,
    )
    answer = (await complete(JUDGMENT_PROMPT, body, max_tokens=8)).strip().casefold()
    return answer == "yes"


async def _extract_with_retry(
    extractor: Extractor,
    transcript: str,
    observations: list[str],
    turns: Sequence[Mapping[str, object]],
    session_id: str,
) -> list[ValidatedCandidate]:
    last_error: BaseException | None = None
    for attempt in range(MAX_EXTRACTION_ATTEMPTS):
        try:
            validated = validate_candidates(await extractor(transcript, observations))
            return verify_quotes(validated, turns, session_id)
        except (
            CandidateValidationError,
            json.JSONDecodeError,
            TimeoutError,
            ValueError,
        ) as error:
            last_error = error
            if attempt + 1 == MAX_EXTRACTION_ATTEMPTS:
                break
            delay = (0.05 * (2**attempt)) + random.uniform(0.0, 0.03)
            await asyncio.sleep(delay)
    if last_error is None:
        raise RuntimeError("extractor retry loop ended without result")
    raise last_error


async def graduate(
    model: UserModel,
    *,
    judger: Judger | None = None,
) -> tuple[list[int], list[int]]:
    """Run explicit longitudinal judgment for eligible node and edge claims."""
    judge = judger or judge_candidate
    proposed_nodes: list[int] = []
    proposed_edges: list[int] = []
    for kind, claims in (("node", model.nodes()), ("edge", model.edges())):
        for claim in claims:
            claim_id = int(claim["id"])
            if claim["status"] != "observation" or not model.is_eligible(claim):
                continue
            claim_kind = cast(Literal["node", "edge"], kind)
            if not model.judgment_needed(claim_kind, claim_id):
                continue
            if await judge(claim_kind, claim):
                proposed = (
                    model.propose(claim_id)
                    if claim_kind == "node"
                    else model.propose_edge(claim_id)
                )
                if proposed:
                    (proposed_nodes if claim_kind == "node" else proposed_edges).append(
                        claim_id
                    )
            else:
                model.defer_proposal(claim_kind, claim_id)
    return proposed_nodes, proposed_edges


def promote(
    model: UserModel, candidates: list[RawCandidate], session_id: str | None
) -> tuple[list[int], list[int]]:
    """Compatibility helper; production finalization uses `distill_session`."""
    if session_id is None:
        raise ValueError("session_id is required")
    validated = validate_candidates(candidates)
    verified = verify_quotes(validated, [], session_id)
    _, nodes, edges = model.apply_distillation(
        session_id=session_id,
        extractor_version="compat-promote-v2",
        candidates=cast(list[Mapping[str, object]], verified),
        inbox_ids=[],
    )
    return nodes, edges


async def distill_session(
    model: UserModel,
    turns: list[dict[str, object]],
    session_id: str | None,
    *,
    extractor: Extractor | None = None,
    judger: Judger | None = None,
    extractor_version: str = EXTRACTOR_VERSION,
) -> DistillResult:
    """Distill one session idempotently; failed validation consumes nothing."""
    if session_id is None:
        raise ValueError("session_id is required")
    run_id = model.start_distillation_run(session_id, extractor_version)
    existing = model.distillation_run(run_id)
    if existing is not None and existing["state"] == "succeeded":
        result = cast(dict[str, list[int]], existing.get("result", {}))
        return DistillResult(
            run_id=run_id,
            promoted_nodes=[int(item) for item in result.get("node_ids", [])],
            promoted_edges=[int(item) for item in result.get("edge_ids", [])],
            processed_observations=[
                int(row["id"])
                for row in model.pending_observations(
                    session_id, include_processed=True
                )
                if row.get("distillation_run_id") == run_id
            ],
        )
    pending = model.pending_observations(session_id)
    observation_texts = [str(row["text"]) for row in pending]
    transcript = render_transcript(turns)
    try:
        verified = await _extract_with_retry(
            extractor or extract_candidates,
            transcript,
            observation_texts,
            turns,
            session_id,
        )
        inbox_ids = [int(row["id"]) for row in pending]
        run_id, node_ids, edge_ids = model.apply_distillation(
            session_id=session_id,
            extractor_version=extractor_version,
            candidates=cast(list[Mapping[str, object]], verified),
            inbox_ids=inbox_ids,
        )
        proposed_nodes, proposed_edges = await graduate(model, judger=judger)
    except Exception as error:
        model.fail_distillation_run(run_id, error)
        raise
    return DistillResult(
        run_id=run_id,
        promoted_nodes=node_ids,
        promoted_edges=edge_ids,
        proposed_nodes=proposed_nodes,
        proposed_edges=proposed_edges,
        processed_observations=inbox_ids,
    )


__all__ = [
    "CandidateValidationError",
    "DistillResult",
    "EXTRACTOR_VERSION",
    "extract_candidates",
    "graduate",
    "parse_candidates",
    "promote",
    "validate_candidates",
    "verify_quotes",
    "distill_session",
]
