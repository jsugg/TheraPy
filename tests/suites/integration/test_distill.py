"""Transactional, auditable distillation integration tests (Phase 4 B)."""

from __future__ import annotations

import asyncio
import json
import random
import string
from collections.abc import Coroutine, Mapping
from pathlib import Path
from typing import cast

import pytest

from therapy.knowledge import distill
from therapy.knowledge.user_model import UserModel
from therapy.observability.interactions import require_json_object

type MetricCall = tuple[str, float, dict[str, str]]


@pytest.fixture
def metric_calls(monkeypatch: pytest.MonkeyPatch) -> list[MetricCall]:
    from therapy.observability import telemetry

    calls: list[MetricCall] = []

    def capture(
        name: str, value: float, attrs: dict[str, str] | None = None
    ) -> None:
        calls.append((name, value, attrs or {}))

    monkeypatch.setattr(telemetry, "record_metric", capture)
    return calls


def _run[T](awaitable: Coroutine[object, object, T]) -> T:
    return asyncio.run(awaitable)


def _turns(session: str = "s1") -> list[dict[str, object]]:
    return [
        {
            "id": 1,
            "session_id": session,
            "ts": "2026-07-15T12:00:00+00:00",
            "role": "user",
            "modality": "text",
            "language": "pt",
            "text": "Eu pulo o almoço quando o trabalho fica corrido.",
        }
    ]


async def _accept(_kind: str, _claim: Mapping[str, object]) -> bool:
    return True


def test_candidate_parser_and_schema_reject_partial_or_authoritative_output() -> None:
    parsed = distill.parse_candidates(
        '```json\n[{"kind":"node","type":"pattern","statement":"A."}]\n```'
    )
    assert parsed[0]["statement"] == "A."
    with pytest.raises(distill.CandidateValidationError):
        distill.parse_candidates("prefix [] suffix")
    with pytest.raises(distill.CandidateValidationError):
        distill.validate_candidates(
            [
                {
                    "kind": "node",
                    "type": "pattern",
                    "statement": "A.",
                    "source": "user-stated",
                }
            ]
        )
    with pytest.raises(distill.CandidateValidationError):
        distill.validate_candidates(
            [
                {
                    "kind": "edge",
                    "type": "triggers",
                    "src": "A",
                    "dst": "B",
                    "statement": "",
                }
            ]
        )


def test_candidate_parser_fuzz_has_only_typed_failures_or_bounded_output() -> None:
    rng = random.Random(20260715)
    alphabet = string.ascii_letters + string.digits + "[]{}:,`\n\" "
    scalar_values: list[object] = [None, True, False, 0, 1.5, "text", [], {}]

    for _ in range(1_000):
        if rng.random() < 0.5:
            raw = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 300)))
        else:
            candidates = [
                {
                    "".join(rng.choice(string.ascii_lowercase) for _ in range(5)): rng.choice(
                        scalar_values
                    )
                    for _ in range(rng.randrange(0, 8))
                }
                for _ in range(rng.randrange(0, distill.MAX_CANDIDATES + 3))
            ]
            raw = json.dumps(candidates)
            if rng.random() < 0.3:
                raw = f"```json\n{raw}\n```"
        try:
            parsed = distill.parse_candidates(raw)
            validated = distill.validate_candidates(parsed)
        except distill.CandidateValidationError:
            continue
        assert len(parsed) <= distill.MAX_CANDIDATES
        assert all(candidate["kind"] in {"node", "edge"} for candidate in validated)


def test_quotes_must_match_user_turn_and_take_actual_provenance() -> None:
    raw = distill.validate_candidates(
        [
            {
                "kind": "node",
                "type": "pattern",
                "statement": "Skips lunch when busy.",
                "quotes": [{"text": "pulo o almoço", "language": "en"}],
            }
        ]
    )
    verified = distill.verify_quotes(raw, _turns(), "s1")
    quote = verified[0]["quotes"][0]
    assert quote == {
        "text": "pulo o almoço",
        "language": "pt",
        "session_id": "s1",
        "turn_id": 1,
        "observed_at": "2026-07-15T12:00:00+00:00",
    }
    raw[0]["raw_quotes"] = ["invented quote"]
    with pytest.raises(distill.CandidateValidationError, match="not verbatim"):
        distill.verify_quotes(raw, _turns(), "s1")


def test_distillation_is_session_scoped_atomic_and_idempotent(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    model = UserModel(tmp_path)
    own = model.add_observation("session one observation", session_id="s1")
    other = model.add_observation("session two observation", session_id="s2")
    assert own is not None
    assert other is not None
    calls = 0

    async def extractor(
        _transcript: str, observations: list[str]
    ) -> list[distill.RawCandidate]:
        nonlocal calls
        calls += 1
        assert observations == ["session one observation"]
        return [
            {
                "kind": "node",
                "type": "pattern",
                "statement": "Skips lunch when busy.",
                "quotes": [{"text": "pulo o almoço"}],
            }
        ]

    first = _run(
        distill.distill_session(
            model,
            _turns(),
            "s1",
            extractor=extractor,
            judger=_accept,
            extractor_version="test-v1",
        )
    )
    second = _run(
        distill.distill_session(
            model,
            _turns(),
            "s1",
            extractor=extractor,
            judger=_accept,
            extractor_version="test-v1",
        )
    )

    assert first.run_id == second.run_id
    assert first.promoted_nodes == second.promoted_nodes
    assert calls == 1
    first_node = model.get_node(first.promoted_nodes[0])
    assert first_node is not None
    assert first_node["n_occurrences"] == 1
    assert model.pending_observations("s1") == []
    assert [row["id"] for row in model.pending_observations("s2")] == [other]
    attempts = [
        attrs["outcome"]
        for name, _, attrs in metric_calls
        if name == "therapy_distillation_attempts_total"
    ]
    assert attempts == ["success"]
    runs = [
        attrs
        for name, _, attrs in metric_calls
        if name == "therapy_distillation_runs_total"
    ]
    assert runs == [
        {"outcome": "success", "idempotent": "false"},
        {"outcome": "success", "idempotent": "true"},
    ]
    assert all(set(attrs) == {"outcome", "idempotent"} for attrs in runs)
    assert [
        attrs["outcome"]
        for name, _, attrs in metric_calls
        if name == "therapy_distillation_seconds"
    ] == ["success", "success"]


def test_overlapping_finalizers_commit_one_idempotent_run(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    assert model.add_observation("overlapping observation", session_id="s1") is not None
    both_extractors_started = asyncio.Event()
    extractor_calls = 0

    async def extractor(
        _transcript: str, _observations: list[str]
    ) -> list[distill.RawCandidate]:
        nonlocal extractor_calls
        extractor_calls += 1
        if extractor_calls == 2:
            both_extractors_started.set()
        await asyncio.wait_for(both_extractors_started.wait(), timeout=2)
        return [
            {
                "kind": "node",
                "type": "pattern",
                "statement": "Skips lunch when busy.",
                "quotes": [{"text": "pulo o almoço"}],
            }
        ]

    async def finalize_twice():
        return await asyncio.gather(
            *(
                distill.distill_session(
                    model,
                    _turns(),
                    "s1",
                    extractor=extractor,
                    judger=_accept,
                    extractor_version="overlap-v1",
                )
                for _ in range(2)
            )
        )

    first, second = _run(finalize_twice())

    assert extractor_calls == 2
    assert first.run_id == second.run_id
    assert first.promoted_nodes == second.promoted_nodes
    node = model.get_node(first.promoted_nodes[0])
    assert node is not None
    assert node["n_occurrences"] == 1
    assert model.pending_observations("s1") == []
    runs = model.export_all()["distillation_runs"]
    assert isinstance(runs, list)
    typed_runs = cast(list[object], runs)
    assert len(typed_runs) == 1
    run_value = typed_runs[0]
    run = require_json_object(run_value, "distillation_runs[0]")
    assert run["state"] == "succeeded"


def test_validation_failure_retries_then_keeps_inbox_unconsumed(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    model = UserModel(tmp_path)
    inbox_id = model.add_observation("keep me", session_id="s1")
    attempts = 0

    async def invalid(
        _transcript: str, _observations: list[str]
    ) -> list[distill.RawCandidate]:
        nonlocal attempts
        attempts += 1
        return [{"kind": "node", "type": "bogus", "statement": "No."}]

    with pytest.raises(distill.CandidateValidationError):
        _run(
            distill.distill_session(
                model,
                _turns(),
                "s1",
                extractor=invalid,
                extractor_version="invalid-v1",
            )
        )

    assert attempts == distill.MAX_EXTRACTION_ATTEMPTS
    assert [row["id"] for row in model.pending_observations("s1")] == [inbox_id]
    runs = model.export_all()["distillation_runs"]
    assert isinstance(runs, list)
    typed_runs = cast(list[object], runs)
    run_value = typed_runs[0]
    run = require_json_object(run_value, "distillation_runs[0]")
    assert run["state"] == "failed"
    error = run.get("error")
    assert isinstance(error, str)
    assert "CandidateValidationError" in error
    assert [
        attrs["outcome"]
        for name, _, attrs in metric_calls
        if name == "therapy_distillation_attempts_total"
    ] == ["invalid"] * distill.MAX_EXTRACTION_ATTEMPTS
    assert (
        "therapy_distillation_runs_total",
        1,
        {"outcome": "error", "idempotent": "false"},
    ) in metric_calls
    assert [
        attrs["category"]
        for name, _, attrs in metric_calls
        if name == "therapy_distillation_validation_failures_total"
    ] == ["schema"] * distill.MAX_EXTRACTION_ATTEMPTS
    assert any(
        name == "therapy_distillation_seconds" and attrs == {"outcome": "error"}
        for name, _, attrs in metric_calls
    )


def test_unresolved_edge_rolls_back_nodes_evidence_and_inbox(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    inbox_id = model.add_observation("edge observation", session_id="s1")

    async def extractor(
        _transcript: str, _observations: list[str]
    ) -> list[distill.RawCandidate]:
        return [
            {
                "kind": "node",
                "type": "pattern",
                "statement": "Skips lunch.",
            },
            {
                "kind": "edge",
                "type": "triggers",
                "src": "Unknown endpoint",
                "dst": "Skips lunch.",
                "statement": "Pressure triggers skipped lunch.",
            },
        ]

    with pytest.raises(ValueError, match="unresolved"):
        _run(
            distill.distill_session(
                model,
                _turns(),
                "s1",
                extractor=extractor,
                extractor_version="rollback-v1",
            )
        )

    assert model.nodes() == []
    assert model.edges() == []
    assert [row["id"] for row in model.pending_observations("s1")] == [inbox_id]


def test_edge_resolves_existing_multilingual_alias_and_gets_evidence(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)

    async def first_extractor(
        _transcript: str, _observations: list[str]
    ) -> list[distill.RawCandidate]:
        return [
            {
                "kind": "node",
                "type": "thread",
                "statement": "Work becomes busy.",
                "quotes": [{"text": "trabalho fica corrido"}],
            },
            {
                "kind": "node",
                "type": "pattern",
                "statement": "Skips lunch.",
                "quotes": [{"text": "pulo o almoço"}],
            },
        ]

    _run(
        distill.distill_session(
            model,
            _turns(),
            "s1",
            extractor=first_extractor,
            extractor_version="alias-nodes",
        )
    )

    async def edge_extractor(
        _transcript: str, _observations: list[str]
    ) -> list[distill.RawCandidate]:
        return [
            {
                "kind": "edge",
                "type": "triggers",
                "src": "trabalho fica corrido",
                "dst": "pulo o almoço",
                "statement": "Busy work triggers skipped lunch.",
                "quotes": [{"text": "pulo o almoço quando o trabalho fica corrido"}],
            }
        ]

    result = _run(
        distill.distill_session(
            model,
            _turns("s2"),
            "s2",
            extractor=edge_extractor,
            extractor_version="alias-edge",
        )
    )

    assert len(result.promoted_edges) == 1
    edge = model.get_edge(result.promoted_edges[0])
    assert edge is not None
    assert edge["type"] == "triggers"
    evidence = model.evidence("edge", edge["id"])
    assert evidence[0]["language"] == "pt"
    assert evidence[0]["quote_text"] == "pulo o almoço quando o trabalho fica corrido"


def test_judgment_is_separate_and_negative_snapshot_is_not_reasked(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)
    node_id: int | None = None
    for session_id in ("s1", "s1", "s2"):
        node_id = model.upsert_node("pattern", "Checks email early.", session_id=session_id)
    assert node_id is not None
    judgments = 0

    async def reject(_kind: str, _claim: Mapping[str, object]) -> bool:
        nonlocal judgments
        judgments += 1
        return False

    assert _run(distill.graduate(model, judger=reject)) == ([], [])
    assert _run(distill.graduate(model, judger=reject)) == ([], [])
    assert judgments == 1
    node = model.get_node(node_id)
    assert node is not None
    assert node["status"] == "observation"

    model.upsert_node("pattern", "Checks email early.", session_id="s3")
    assert _run(distill.graduate(model, judger=reject)) == ([], [])
    assert judgments == 2


def test_node_and_edge_both_graduate_only_after_judgment(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    src = model.add_user_statement("thread", "Busy workdays.")
    dst = model.add_user_statement("pattern", "Skips lunch.")
    assert src is not None
    assert dst is not None
    edge_id: int | None = None
    for session_id in ("s1", "s1", "s2"):
        edge_id = model.upsert_edge(
            src,
            dst,
            "triggers",
            statement="Busy workdays trigger skipped lunch.",
            session_id=session_id,
        )
    assert edge_id is not None

    nodes, edges = _run(distill.graduate(model, judger=_accept))

    assert nodes == []
    assert edges == [edge_id]
    edge = model.get_edge(edge_id)
    assert edge is not None
    assert edge["status"] == "proposed"


def test_distillation_candidate_dispositions_are_bounded_and_content_free(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    model = UserModel(tmp_path)
    for session_id in ("s1", "s1", "s2"):
        model.upsert_node(
            "pattern", "Checks private mail early.", session_id=session_id
        )
    metric_calls.clear()

    async def extractor(
        _transcript: str, _observations: list[str]
    ) -> list[distill.RawCandidate]:
        return [
            {
                "kind": "node",
                "type": "pattern",
                "statement": "Checks private mail early.",
            }
        ]

    async def reject(_kind: str, _claim: Mapping[str, object]) -> bool:
        return False

    _run(
        distill.distill_session(
            model,
            _turns("s3"),
            "s3",
            extractor=extractor,
            judger=reject,
            extractor_version="disposition-deferred",
        )
    )
    _run(
        distill.distill_session(
            model,
            _turns("s4"),
            "s4",
            extractor=extractor,
            judger=_accept,
            extractor_version="disposition-proposed",
        )
    )

    dispositions = [
        attrs
        for name, _, attrs in metric_calls
        if name == "therapy_distillation_candidates_total"
    ]
    assert {attrs["disposition"] for attrs in dispositions} == {
        "candidate",
        "promoted",
        "proposed",
        "deferred",
    }
    assert all(set(attrs) == {"disposition"} for attrs in dispositions)
    candidate_histograms = [
        attrs["disposition"]
        for name, _, attrs in metric_calls
        if name == "therapy_distillation_candidate_count"
    ]
    assert set(candidate_histograms) == {
        "candidate",
        "promoted",
        "proposed",
        "deferred",
    }
    assert all(
        attrs["disposition"]
        in {"candidate", "promoted", "proposed", "deferred"}
        for attrs in dispositions
    )
    assert "private mail" not in json.dumps(metric_calls).casefold()
    assert "disposition-deferred" not in json.dumps(metric_calls)


def test_distillation_retryable_attempt_and_unexpected_error_outcomes(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    model = UserModel(tmp_path)
    attempts = 0

    async def retry_once(
        _transcript: str, _observations: list[str]
    ) -> list[distill.RawCandidate]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("restricted timeout detail")
        return []

    _run(
        distill.distill_session(
            model,
            _turns(),
            "retry-session",
            extractor=retry_once,
            extractor_version="retry-v1",
        )
    )
    assert [
        attrs["outcome"]
        for name, _, attrs in metric_calls
        if name == "therapy_distillation_attempts_total"
    ] == ["retry", "success"]

    metric_calls.clear()

    async def fail_hard(
        _transcript: str, _observations: list[str]
    ) -> list[distill.RawCandidate]:
        raise RuntimeError("private completion and session 42")

    with pytest.raises(RuntimeError, match="private completion"):
        _run(
            distill.distill_session(
                model,
                _turns("hard-error-session"),
                "hard-error-session",
                extractor=fail_hard,
                extractor_version="hard-error-v1",
            )
        )

    assert (
        "therapy_distillation_attempts_total",
        1,
        {"outcome": "error"},
    ) in metric_calls
    assert (
        "therapy_distillation_runs_total",
        1,
        {"outcome": "error", "idempotent": "false"},
    ) in metric_calls
    attempt_attrs = [
        attrs
        for name, _, attrs in metric_calls
        if name == "therapy_distillation_attempts_total"
    ]
    assert all(
        attrs["outcome"] in {"success", "invalid", "retry", "error"}
        for attrs in attempt_attrs
    )
    encoded = json.dumps(metric_calls).casefold()
    assert "private completion" not in encoded
    assert "hard-error-session" not in encoded
