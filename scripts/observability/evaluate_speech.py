"""Evaluate fixture STT hypotheses with reviewed-only JiWER metrics (plan O3.4).

Normalization is explicit and versioned: lowercase Unicode text, replace every
Unicode punctuation character with a space, then collapse whitespace. Accents
and other letters are preserved. Silence fixtures are detection cases and are
never included in WER, CER, MER, or WIL.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, TypedDict, cast

REPO_ROOT = next(
    path
    for path in Path(__file__).resolve().parents
    if (path / "pyproject.toml").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from scripts.observability.fixture_hash import fixture_hash  # noqa: E402
from scripts.observability.report_io import write_restricted_report  # noqa: E402

SPEECH_FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/observability/speech"
SPEECH_FIXTURE_PATH = SPEECH_FIXTURE_ROOT / "cases.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / ".local/obs-eval/speech-report.json"
NORMALIZATION_ID = "lowercase-strip-unicode-punctuation-v1"
NORMALIZATION_DESCRIPTION = (
    "Lowercase with str.lower(); replace Unicode punctuation characters with spaces; "
    "collapse all whitespace; preserve accents and other letters."
)
INVALID_CLAIMS_CAVEAT = (
    "WER/CER claims are NOT valid: one or more fixture references are seeded and "
    "have not been approved by the owner."
)
VALID_CLAIMS_CAVEAT = (
    "WER/CER claims apply only to this owner-approved synthetic fixture corpus and "
    "are not evidence of therapeutic benefit."
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]


class Review(TypedDict):
    """Validated review fields used by the evaluator."""

    status: str


class SpeechCase(TypedDict):
    """Validated speech fixture fields used by the evaluator."""

    id: str
    kind: str
    language_group: str
    reference_transcript: str
    review: Review


class UnreviewedFixtureError(ValueError):
    """Raised when metric computation is attempted on unapproved references."""


class _WordMetrics(Protocol):
    wer: float
    mer: float
    wil: float


class _CharacterMetrics(Protocol):
    cer: float


class _ProcessWords(Protocol):
    def __call__(self, reference: list[str], hypothesis: list[str]) -> _WordMetrics: ...


class _ProcessCharacters(Protocol):
    def __call__(
        self, reference: list[str], hypothesis: list[str]
    ) -> _CharacterMetrics: ...


def normalize_transcript(text: str) -> str:
    """Normalize text for JiWER without removing letters or accents."""
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in text.lower()
    )
    return " ".join(without_punctuation.split())


def _compute_normalized_error_rates(
    references: list[str], hypotheses: list[str]
) -> dict[str, float]:
    try:
        jiwer = import_module("jiwer")
    except ImportError as error:
        raise RuntimeError(
            "JiWER is required for non-silence scoring; run `uv sync` first."
        ) from error
    raw_process_words = getattr(jiwer, "process_words", None)
    raw_process_characters = getattr(jiwer, "process_characters", None)
    if not callable(raw_process_words) or not callable(raw_process_characters):
        raise RuntimeError("JiWER does not expose the expected version 4 API.")
    process_words = cast(_ProcessWords, raw_process_words)
    process_characters = cast(_ProcessCharacters, raw_process_characters)

    word_result = process_words(references, hypotheses)
    character_result = process_characters(references, hypotheses)
    return {
        "wer": float(word_result.wer),
        "cer": float(character_result.cer),
        "mer": float(word_result.mer),
        "wil": float(word_result.wil),
    }


def compute_error_rates(reference: str, hypothesis: str) -> dict[str, float]:
    """Compute WER, CER, MER, and WIL for one normalized transcript pair.

    Args:
        reference: Reviewed reference transcript.
        hypothesis: Transcript emitted by a separate STT run.

    Returns:
        JiWER error rates after the module's documented normalization.

    Raises:
        RuntimeError: If the evaluation-only JiWER dependency is unavailable.
    """
    return _compute_normalized_error_rates(
        [normalize_transcript(reference)], [normalize_transcript(hypothesis)]
    )


def evaluate_silence_case(hypothesis: str) -> JsonObject:
    """Score a silence fixture as false-speech detection, never as WER."""
    false_speech_detected = hypothesis != ""
    return {
        "evaluation": "silence_detection",
        "hypothesis_empty": not false_speech_detected,
        "false_speech_detected": false_speech_detected,
    }


def _review_label(cases: list[SpeechCase], allow_seeded: bool) -> str:
    unapproved = [
        f"{case['id']} ({case['review']['status']})"
        for case in cases
        if case["review"]["status"] != "approved"
    ]
    if unapproved and not allow_seeded:
        case_list = ", ".join(unapproved)
        raise UnreviewedFixtureError(
            "refusing to score unapproved speech fixtures: "
            f"{case_list}; obtain owner approval or pass --allow-seeded for a "
            "non-claim exploratory report"
        )
    return "seeded-not-reviewed" if unapproved else "approved-reviewed"


def evaluate_speech_cases(
    cases: list[SpeechCase],
    hypotheses: Mapping[str, str],
    *,
    allow_seeded: bool = False,
) -> JsonObject:
    """Evaluate validated speech cases and keep silence out of error rates.

    Args:
        cases: Validated fixture cases.
        hypotheses: STT hypotheses keyed by fixture case ID.
        allow_seeded: Permit an explicitly invalid-for-claims exploratory run.

    Returns:
        Per-case results, aggregate rates, and the review caveat.

    Raises:
        UnreviewedFixtureError: If any reference is unapproved and override is off.
        ValueError: If hypothesis IDs do not exactly match fixture case IDs.
        RuntimeError: If JiWER is unavailable for non-silence cases.
    """
    review_label = _review_label(cases, allow_seeded)
    expected_ids = {case["id"] for case in cases}
    actual_ids = set(hypotheses)
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing case IDs: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected case IDs: {', '.join(unexpected)}")
        raise ValueError(
            "hypotheses must exactly match fixtures (" + "; ".join(details) + ")"
        )

    results: list[JsonValue] = []
    corpus_references: list[str] = []
    corpus_hypotheses: list[str] = []
    false_speech_count = 0
    silence_count = 0

    for case in cases:
        hypothesis = hypotheses[case["id"]]
        result: JsonObject = {
            "case_id": case["id"],
            "kind": case["kind"],
            "language_group": case["language_group"],
            "review_status": case["review"]["status"],
            "result_label": review_label,
        }
        if case["reference_transcript"] == "":
            silence_result = evaluate_silence_case(hypothesis)
            result.update(silence_result)
            silence_count += 1
            false_speech_count += int(bool(silence_result["false_speech_detected"]))
        else:
            normalized_reference = normalize_transcript(case["reference_transcript"])
            normalized_hypothesis = normalize_transcript(hypothesis)
            result.update(
                {
                    "evaluation": "transcript_error_rates",
                    "normalized_reference": normalized_reference,
                    "normalized_hypothesis": normalized_hypothesis,
                    "metrics": _compute_normalized_error_rates(
                        [normalized_reference], [normalized_hypothesis]
                    ),
                }
            )
            corpus_references.append(normalized_reference)
            corpus_hypotheses.append(normalized_hypothesis)
        results.append(result)

    corpus_metrics: JsonValue = None
    if corpus_references:
        corpus_metrics = _compute_normalized_error_rates(
            corpus_references, corpus_hypotheses
        )
    silence_rate = false_speech_count / silence_count if silence_count else None
    caveat = (
        INVALID_CLAIMS_CAVEAT
        if review_label == "seeded-not-reviewed"
        else VALID_CLAIMS_CAVEAT
    )
    return {
        "result_label": review_label,
        "review_status_caveat": caveat,
        "therapeutic_benefit_claim": False,
        "cases": results,
        "aggregate": {
            "speech_case_count": len(corpus_references),
            "speech_metrics": corpus_metrics,
            "silence_detection": {
                "case_count": silence_count,
                "false_speech_detection_count": false_speech_count,
                "false_speech_detection_rate": silence_rate,
            },
        },
    }


def _json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def load_speech_cases(path: Path = SPEECH_FIXTURE_PATH) -> list[SpeechCase]:
    """Load and validate the speech fixture fields used for evaluation."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    payload = _json_object(raw, str(path))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError(f"{path}: cases must be a JSON list")

    cases: list[SpeechCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        case = _json_object(raw_case, f"{path}: cases[{index}]")
        review = _json_object(case.get("review"), f"{path}: cases[{index}].review")
        case_id = case.get("id")
        kind = case.get("kind")
        language_group = case.get("language_group")
        reference = case.get("reference_transcript")
        review_status = review.get("status")
        values = (case_id, kind, language_group, reference, review_status)
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"{path}: cases[{index}] has invalid required fields")
        case_id = cast(str, case_id)
        if case_id in seen_ids:
            raise ValueError(f"{path}: duplicate case ID {case_id!r}")
        seen_ids.add(case_id)
        cases.append(
            SpeechCase(
                id=case_id,
                kind=cast(str, kind),
                language_group=cast(str, language_group),
                reference_transcript=cast(str, reference),
                review=Review(status=cast(str, review_status)),
            )
        )
    if not cases:
        raise ValueError(f"{path}: cases must not be empty")
    return cases


def load_hypotheses(path: Path) -> dict[str, str]:
    """Load the strict case-ID-to-transcript hypothesis mapping."""
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: hypotheses must contain valid JSON") from error
    payload = _json_object(raw, str(path))
    if not all(isinstance(value, str) for value in payload.values()):
        raise ValueError(f"{path}: every hypothesis value must be a string")
    return {key: cast(str, value) for key, value in payload.items()}


def _jiwer_version() -> str:
    try:
        return version("jiwer")
    except PackageNotFoundError as error:
        raise RuntimeError("JiWER is not installed; run `uv sync` first.") from error


def write_report(
    result: JsonObject, output: Path, *, allow_unrestricted: bool = False
) -> str:
    """Write a versioned speech evaluation report to a restricted path."""
    report: JsonObject = {
        "schema_version": 1,
        "report_type": "therapy-speech-evaluation",
        "generated_at": datetime.now(UTC).isoformat(),
        "fixture_path": str(SPEECH_FIXTURE_PATH.relative_to(REPO_ROOT)),
        "fixture_corpus_sha256": fixture_hash(SPEECH_FIXTURE_ROOT),
        "jiwer_version": _jiwer_version(),
        "normalization": {
            "id": NORMALIZATION_ID,
            "description": NORMALIZATION_DESCRIPTION,
        },
        **result,
    }
    return write_restricted_report(
        report, output, repo_root=REPO_ROOT, allow_unrestricted=allow_unrestricted
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score separate STT hypotheses against reviewed speech fixtures."
    )
    parser.add_argument(
        "--hypotheses",
        type=Path,
        required=True,
        help="JSON object mapping every speech fixture case ID to a transcript",
    )
    parser.add_argument(
        "--allow-seeded",
        action="store_true",
        help="allow a seeded exploratory report whose WER/CER claims are invalid",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--unrestricted-output",
        action="store_true",
        help=(
            "deliberately allow a report destination outside the restricted "
            ".local directory (reports contain exact transcript content)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the speech evaluation CLI with traceback-free input errors."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        cases = load_speech_cases()
        hypotheses = load_hypotheses(args.hypotheses)
        result = evaluate_speech_cases(
            cases, hypotheses, allow_seeded=args.allow_seeded
        )
        label = write_report(
            result, args.output, allow_unrestricted=args.unrestricted_output
        )
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"wrote speech evaluation report: {label}")
    if result["result_label"] == "seeded-not-reviewed":
        print(INVALID_CLAIMS_CAVEAT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
