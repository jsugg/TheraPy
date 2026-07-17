"""Run deterministic and optional sampled-judge behavior experiments (O3.4).

Manifests are restricted, self-contained replay records. The CLI is offline by
default and contacts Ollama only when ``--judge ollama`` is explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

import httpx

REPO_ROOT = next(
    path
    for path in Path(__file__).resolve().parents
    if (path / "pyproject.toml").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from scripts.observability.evaluate_behavior import (  # noqa: E402
    BEHAVIOR_FIXTURE_PATH,
    EVALUATOR_VERSION,
    REVIEW_VERDICT,
    BehaviorCase,
    behavior_case_verdicts,
    evaluate_behavior_cases,
    load_behavior_cases,
    load_responses,
)
from scripts.observability.judge import (  # noqa: E402
    JUDGE_TEMPERATURE,
    JUDGE_VERSION,
    Completion,
    JudgeResult,
    judge_case,
)
from scripts.observability.report_io import write_restricted_report  # noqa: E402

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "pedrolucas/smollm3:3b-q4_k_m"
OLLAMA_TIMEOUT_SECONDS = 120.0
MANIFEST_SCHEMA_VERSION = 1
BEHAVIOR_DATASET_NAME = "therapy-behavior-cases"

type JudgeProvider = Literal["none", "ollama"]
type OwnerVerdict = Literal["pass", "fail", "uncertain"]


class OwnerAnnotation(TypedDict):
    """Versioned owner review attached to an exact case response."""

    verdict: OwnerVerdict
    reviewed_at: str
    response_sha256: str
    notes: str


class OllamaCompletion:
    """Synchronous OpenAI-compatible Ollama completion adapter."""

    def __init__(self, *, base_url: str, model: str) -> None:
        """Configure the explicit local judge endpoint and model."""
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must use http:// or https://")
        if not model.strip():
            raise ValueError("THERAPY_LLM_MODEL must not be empty")
        self._base_url = base_url.rstrip("/")
        self._model = model

    def __call__(self, prompt: str, *, temperature: float) -> str:
        """Return one non-streaming judge completion."""
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "system", "content": prompt}],
                "stream": False,
                "temperature": temperature,
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload: object = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Ollama response must be a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Ollama response must contain at least one choice")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError("Ollama response choice must be an object")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama response choice must contain a message")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama response message content must be a string")
        return content.strip()


def _fixture_metadata(fixture_path: Path) -> tuple[int, str, dict[str, object]]:
    """Return the fixture version, byte-exact hash, and replay payload."""
    fixture_bytes = fixture_path.read_bytes()
    try:
        payload: object = json.loads(fixture_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"{fixture_path}: fixture must contain valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{fixture_path}: fixture must be a JSON object")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError(f"{fixture_path}: schema_version must be an integer")
    return (
        schema_version,
        hashlib.sha256(fixture_bytes).hexdigest(),
        cast(dict[str, object], payload),
    )


def _judge_manifest_config(
    provider: JudgeProvider, *, model: str, samples: int
) -> str | dict[str, object]:
    """Build the exact judge configuration recorded in the manifest."""
    if provider == "none":
        return "none"
    return {
        "provider": provider,
        "judge_version": JUDGE_VERSION,
        "model": model,
        "samples": samples,
        "temperature": JUDGE_TEMPERATURE,
    }


def _owner_annotation(value: object, label: str) -> OwnerAnnotation:
    """Validate one bounded owner annotation object."""
    if not isinstance(value, dict) or set(value) != {
        "verdict",
        "reviewed_at",
        "response_sha256",
        "notes",
    }:
        raise ValueError(
            f"{label} must contain exactly verdict, reviewed_at, "
            "response_sha256, and notes"
        )
    verdict = value.get("verdict")
    reviewed_at = value.get("reviewed_at")
    response_sha256 = value.get("response_sha256")
    notes = value.get("notes")
    if verdict not in {"pass", "fail", "uncertain"}:
        raise ValueError(f"{label}.verdict must be pass, fail, or uncertain")
    if not isinstance(reviewed_at, str):
        raise ValueError(f"{label}.reviewed_at must be an ISO-8601 string")
    try:
        parsed_timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label}.reviewed_at must be valid ISO-8601") from error
    if parsed_timestamp.tzinfo is None:
        raise ValueError(f"{label}.reviewed_at must include a UTC offset")
    if (
        not isinstance(response_sha256, str)
        or len(response_sha256) != 64
        or any(character not in "0123456789abcdef" for character in response_sha256)
    ):
        raise ValueError(f"{label}.response_sha256 must be a lowercase SHA-256")
    if not isinstance(notes, str) or len(notes) > 10_000:
        raise ValueError(f"{label}.notes must be a string of at most 10000 characters")
    return OwnerAnnotation(
        verdict=cast(OwnerVerdict, verdict),
        reviewed_at=reviewed_at,
        response_sha256=response_sha256,
        notes=notes,
    )


def load_owner_annotations(path: Path) -> dict[str, OwnerAnnotation]:
    """Load a versioned case-ID-to-owner-annotation mapping."""
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: annotations must contain valid JSON") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError(f"{path}: annotations schema_version must be 1")
    raw_annotations = raw.get("annotations")
    if not isinstance(raw_annotations, dict) or not all(
        isinstance(case_id, str) for case_id in raw_annotations
    ):
        raise ValueError(f"{path}: annotations must be a case-ID object")
    return {
        case_id: _owner_annotation(value, f"{path}: annotations[{case_id!r}]")
        for case_id, value in raw_annotations.items()
    }


def _effective_verdict(
    deterministic_verdict: str, judge_result: JudgeResult | None
) -> str:
    """Resolve the experiment verdict without weakening deterministic results."""
    if deterministic_verdict in {"pass", "fail"}:
        return deterministic_verdict
    if judge_result is None:
        return REVIEW_VERDICT
    return judge_result["verdict"]


def _manifest_case_key(case: Mapping[str, object]) -> tuple[str, str, str]:
    """Validate comparison fields and return case ID, response hash, verdict."""
    case_id = case.get("case_id")
    response_sha256 = case.get("response_sha256")
    effective_verdict = case.get("effective_verdict")
    if not all(
        isinstance(value, str)
        for value in (case_id, response_sha256, effective_verdict)
    ):
        raise ValueError(
            "manifest cases require string case_id, response_sha256, and "
            "effective_verdict"
        )
    return cast(str, case_id), cast(str, response_sha256), cast(str, effective_verdict)


def _manifest_cases(manifest: Mapping[str, object], label: str) -> list[dict[str, object]]:
    """Validate and return comparable case records from a manifest."""
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError(f"{label}: cases must be a list")
    cases: list[dict[str, object]] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"{label}: cases[{index}] must be an object")
        _manifest_case_key(raw_case)
        cases.append(cast(dict[str, object], raw_case))
    return cases


def compare_manifests(
    current: Mapping[str, object],
    baseline: Mapping[str, object],
    *,
    baseline_path: str | None = None,
) -> dict[str, object]:
    """Compare verdicts matched strictly by case ID and response hash."""
    current_cases = _manifest_cases(current, "current manifest")
    baseline_cases = _manifest_cases(baseline, "baseline manifest")

    baseline_by_key: dict[tuple[str, str], str] = {}
    for case in baseline_cases:
        case_id, response_sha256, verdict = _manifest_case_key(case)
        key = (case_id, response_sha256)
        if key in baseline_by_key:
            raise ValueError(f"baseline manifest contains duplicate case key {key!r}")
        baseline_by_key[key] = verdict

    changes: list[dict[str, str]] = []
    matched_keys: set[tuple[str, str]] = set()
    unmatched_current: list[dict[str, str]] = []
    unchanged_count = 0
    for case in current_cases:
        case_id, response_sha256, current_verdict = _manifest_case_key(case)
        key = (case_id, response_sha256)
        baseline_verdict = baseline_by_key.get(key)
        if baseline_verdict is None:
            unmatched_current.append(
                {"case_id": case_id, "response_sha256": response_sha256}
            )
            continue
        matched_keys.add(key)
        if baseline_verdict == current_verdict:
            unchanged_count += 1
        else:
            changes.append(
                {
                    "case_id": case_id,
                    "response_sha256": response_sha256,
                    "baseline_verdict": baseline_verdict,
                    "current_verdict": current_verdict,
                }
            )

    unmatched_baseline = [
        {"case_id": case_id, "response_sha256": response_sha256}
        for case_id, response_sha256 in sorted(set(baseline_by_key) - matched_keys)
    ]
    baseline_id = baseline.get("experiment_id")
    if not isinstance(baseline_id, str):
        raise ValueError("baseline manifest experiment_id must be a string")
    return {
        "baseline_experiment_id": baseline_id,
        "baseline_path": baseline_path,
        "match_key": ["case_id", "response_sha256"],
        "matched_case_count": len(matched_keys),
        "unchanged_case_count": unchanged_count,
        "verdict_changes": sorted(changes, key=lambda change: change["case_id"]),
        "unmatched_current": unmatched_current,
        "unmatched_baseline": unmatched_baseline,
    }


def _validate_high_risk_review(manifest: Mapping[str, object]) -> None:
    """Enforce the invariant that every high-risk case remains human-reviewed."""
    for case in _manifest_cases(manifest, "experiment manifest"):
        if case.get("high_risk") is True and case.get("human_review_required") is not True:
            raise ValueError(
                f"high-risk case {case.get('case_id')!r} lacks required human review"
            )


def build_experiment_manifest(
    cases: list[BehaviorCase],
    responses: Mapping[str, str],
    *,
    judge: JudgeProvider = "none",
    completion: Completion | None = None,
    samples: int = 3,
    model: str = DEFAULT_OLLAMA_MODEL,
    fixture_path: Path = BEHAVIOR_FIXTURE_PATH,
    responses_sha256: str | None = None,
    owner_annotations: Mapping[str, OwnerAnnotation] | None = None,
    annotations_sha256: str | None = None,
    baseline: Mapping[str, object] | None = None,
    baseline_path: str | None = None,
) -> dict[str, object]:
    """Build one self-contained deterministic and judge experiment manifest.

    Args:
        cases: Validated versioned behavior cases.
        responses: Exact case-ID-to-response mapping.
        judge: ``none`` for offline deterministic evaluation or ``ollama``.
        completion: Injectable completion callable required for Ollama judging.
        samples: Positive number of judge samples per review verdict.
        model: Judge model identifier recorded for replay.
        fixture_path: Versioned fixture used to produce ``cases``.
        responses_sha256: Optional byte-exact source response-file hash.
        owner_annotations: Optional versioned owner reviews keyed by case ID.
        annotations_sha256: Optional byte-exact source annotation-file hash.
        baseline: Optional prior experiment manifest.
        baseline_path: Printable provenance for ``baseline``.

    Returns:
        JSON-serializable, exact-replay experiment manifest.

    Raises:
        ValueError: If configuration, inputs, or the review invariant are invalid.
    """
    if judge not in {"none", "ollama"}:
        raise ValueError(f"unknown judge provider: {judge!r}")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if judge == "ollama" and completion is None:
        raise ValueError("Ollama judge requires a completion callable")
    if judge == "none" and completion is not None:
        raise ValueError("completion must be omitted when judge is none")

    fixture_schema_version, fixture_sha256, fixture_payload = _fixture_metadata(
        fixture_path
    )
    evaluation = evaluate_behavior_cases(cases, responses)
    deterministic_verdicts = behavior_case_verdicts(evaluation)
    case_by_id = {case["id"]: case for case in cases}
    annotations = {
        case_id: _owner_annotation(annotation, f"owner annotation {case_id!r}")
        for case_id, annotation in (owner_annotations or {}).items()
    }
    unexpected_annotations = sorted(set(annotations) - set(case_by_id))
    if unexpected_annotations:
        raise ValueError(
            "owner annotations contain unknown case IDs: "
            + ", ".join(unexpected_annotations)
        )

    experiment_cases: list[dict[str, object]] = []
    deterministic_counts = {"pass": 0, "fail": 0, REVIEW_VERDICT: 0}
    effective_counts = {
        "pass": 0,
        "fail": 0,
        "uncertain": 0,
        REVIEW_VERDICT: 0,
    }
    judge_evaluated_count = 0
    human_review_required_count = 0
    owner_annotated_count = 0
    for deterministic in deterministic_verdicts:
        case_id = deterministic["case_id"]
        case = case_by_id[case_id]
        deterministic_verdict = deterministic["verdict"]
        deterministic_counts[deterministic_verdict] += 1
        judge_result: JudgeResult | None = None
        if deterministic_verdict == REVIEW_VERDICT and judge == "ollama":
            assert completion is not None
            judge_result = judge_case(
                case,
                responses[case_id],
                completion=completion,
                samples=samples,
            )
            judge_evaluated_count += 1
        effective_verdict = _effective_verdict(deterministic_verdict, judge_result)
        effective_counts[effective_verdict] += 1
        human_review_required = (
            deterministic["human_review_required"]
            if judge_result is None
            else judge_result["human_review_required"]
        )
        if human_review_required:
            human_review_required_count += 1
        owner_annotation = annotations.get(case_id)
        if owner_annotation is not None:
            if owner_annotation["response_sha256"] != deterministic["response_sha256"]:
                raise ValueError(
                    f"owner annotation {case_id!r} does not match the response hash"
                )
            owner_annotated_count += 1
        experiment_cases.append(
            {
                "case_id": case_id,
                "dimension": deterministic["dimension"],
                "high_risk": deterministic["high_risk"],
                "deterministic_verdict": deterministic_verdict,
                "deterministic_reason": deterministic["reason"],
                "judge": dict(judge_result) if judge_result is not None else None,
                "judge_verdict": (
                    judge_result["verdict"] if judge_result is not None else None
                ),
                "judge_uncertainty": (
                    judge_result["uncertainty"] if judge_result is not None else None
                ),
                "effective_verdict": effective_verdict,
                "human_review_required": human_review_required,
                "owner_annotation": (
                    dict(owner_annotation) if owner_annotation is not None else None
                ),
                "response_sha256": deterministic["response_sha256"],
            }
        )

    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "report_type": "therapy-behavior-experiment",
        "experiment_id": str(uuid.uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "evaluator_version": EVALUATOR_VERSION,
        "fixture_sha256": fixture_sha256,
        "dataset": {
            "name": BEHAVIOR_DATASET_NAME,
            "fixture_path": str(fixture_path),
            "fixture_schema_version": fixture_schema_version,
            "fixture_sha256": fixture_sha256,
        },
        "responses_sha256": responses_sha256,
        "owner_annotations": {
            "schema_version": 1,
            "source_sha256": annotations_sha256,
            "annotated_cases": owner_annotated_count,
        },
        "judge_config": _judge_manifest_config(
            judge, model=model, samples=samples
        ),
        "therapeutic_benefit_claim": False,
        "caveat": (
            "Evaluator and judge verdicts are regression evidence only and never "
            "evidence of therapeutic benefit."
        ),
        "cases": experiment_cases,
        "summary": {
            "total_cases": len(experiment_cases),
            "deterministic_verdicts": deterministic_counts,
            "effective_verdicts": effective_counts,
            "judge_evaluated": judge_evaluated_count,
            "human_review_required": human_review_required_count,
            "owner_annotated": owner_annotated_count,
        },
        "replay": {
            "fixture": fixture_payload,
            "evaluated_cases": [dict(case) for case in cases],
            "responses": dict(responses),
            "owner_annotations": {
                case_id: dict(annotation)
                for case_id, annotation in annotations.items()
            },
        },
    }
    if baseline is not None:
        manifest["comparison"] = compare_manifests(
            manifest, baseline, baseline_path=baseline_path
        )
    _validate_high_risk_review(manifest)
    return manifest


def load_manifest(path: Path) -> dict[str, object]:
    """Load and validate the top-level shape of a prior manifest."""
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: baseline must contain valid JSON") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: baseline must be a JSON object")
    manifest = cast(dict[str, object], raw)
    _manifest_cases(manifest, str(path))
    if manifest.get("report_type") != "therapy-behavior-experiment":
        raise ValueError(f"{path}: baseline has an unexpected report_type")
    return manifest


def write_experiment_manifest(
    manifest: Mapping[str, object],
    output: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> str:
    """Write one owner-only experiment manifest beneath ``.local``."""
    _validate_high_risk_review(manifest)
    return write_restricted_report(manifest, output, repo_root=repo_root)


def _parser() -> argparse.ArgumentParser:
    """Build the experiment command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--judge", choices=("ollama", "none"), default="none")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one behavior experiment and write exactly one restricted manifest."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        cases = load_behavior_cases()
        responses = load_responses(args.responses)
        annotations = (
            load_owner_annotations(args.annotations)
            if args.annotations is not None
            else None
        )
        baseline = load_manifest(args.baseline) if args.baseline is not None else None
        model = os.environ.get("THERAPY_LLM_MODEL", DEFAULT_OLLAMA_MODEL)
        completion: Completion | None = None
        if args.judge == "ollama":
            completion = OllamaCompletion(
                base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
                model=model,
            )
        manifest = build_experiment_manifest(
            cases,
            responses,
            judge=args.judge,
            completion=completion,
            samples=args.samples,
            model=model,
            responses_sha256=hashlib.sha256(args.responses.read_bytes()).hexdigest(),
            owner_annotations=annotations,
            annotations_sha256=(
                hashlib.sha256(args.annotations.read_bytes()).hexdigest()
                if args.annotations is not None
                else None
            ),
            baseline=baseline,
            baseline_path=str(args.baseline) if args.baseline is not None else None,
        )
        label = write_experiment_manifest(manifest, args.output)
    except (OSError, httpx.HTTPError, ValueError) as error:
        parser.error(str(error))
    print(f"wrote behavior experiment manifest: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
