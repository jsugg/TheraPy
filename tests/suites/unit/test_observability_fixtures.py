"""O0 gate over the committed fixture corpus (plan O0.2 and the O0 gate).

- Required content canaries occur in restricted records only.
- Content is absent from every broad surface.
- Credential/infrastructure canaries occur nowhere.
- The golden scenarios and classification matrix stay complete.
"""

import json
from pathlib import Path

import pytest

# scripts/ is not bind-mounted into the container image; the canary gate
# runs on the host (and in any tree that carries scripts/observability/).
from therapy.observability.interactions import JsonValue, require_json_object
from therapy.observability.model import FieldClassification

canary_scan = pytest.importorskip("scripts.observability.canary_scan")
FIXTURE_ROOT = canary_scan.FIXTURE_ROOT
load_canaries = canary_scan.load_canaries
scan_fixture_tree = canary_scan.scan_fixture_tree

REQUIRED_SCENARIOS = {
    "provider_success",
    "empty_output",
    "truncation",
    "structured_output",
    "tools",
    "retry_fallback",
    "http_error",
    "in_stream_error_after_200",
    "timeout",
    "cancellation",
    "partial_stream",
}

REQUIRED_PROVIDERS = {"anthropic", "openrouter", "ollama"}

# The only keys a broad twin may carry: bounded enums, counts, durations,
# sizes, statuses, and correlation IDs (plan §1, O1.3 item 4).
ALLOWED_BROAD_TWIN_KEYS = {
    "event.name",
    "component",
    "operation",
    "provider",
    "requested_model_policy",
    "trace_id",
    "span_id",
    "input_size_bucket",
    "output_size_bucket",
    "input_tokens",
    "output_tokens",
    "finish_class",
    "retry_count",
    "status_class",
    "ttft_ms",
    "duration_ms",
    "outcome",
}


def _load_object(path: Path) -> dict[str, JsonValue]:
    """Load a fixture path as a validated JSON object."""
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    return require_json_object(decoded, str(path))


def _object_list(value: JsonValue, where: str) -> list[dict[str, JsonValue]]:
    """Validate a fixture field as a list of JSON objects."""
    assert isinstance(value, list), f"{where} must be a list"
    return [require_json_object(item, f"{where}[{index}]") for index, item in enumerate(value)]


def _interaction_fixtures() -> list[dict[str, JsonValue]]:
    return [
        _load_object(path)
        for path in sorted((FIXTURE_ROOT / "interactions").glob("*.json"))
    ]


def test_canary_gate_passes() -> None:
    report = scan_fixture_tree()
    assert report.ok, "\n".join(report.violations)
    # every content canary is present in at least one restricted record
    assert all(count > 0 for count in report.restricted_hits.values())


def test_canaries_do_not_resemble_real_secrets() -> None:
    content, forbidden = load_canaries()
    for value in {**content, **forbidden}.values():
        assert value.startswith("OBS-CANARY-"), value
        assert not value.lower().startswith(("sk-", "bearer ", "aka", "ghp_")), value


def test_golden_scenarios_and_providers_are_covered() -> None:
    fixtures = _interaction_fixtures()
    scenarios: set[str] = set()
    providers: set[str] = set()
    for fixture in fixtures:
        scenario = fixture.get("scenario")
        assert isinstance(scenario, str)
        scenarios.add(scenario)
        record = require_json_object(fixture.get("record"), "fixture.record")
        provider = record.get("provider")
        assert isinstance(provider, str)
        providers.add(provider)
    assert REQUIRED_SCENARIOS <= scenarios, REQUIRED_SCENARIOS - scenarios
    assert REQUIRED_PROVIDERS <= providers, REQUIRED_PROVIDERS - providers


def test_broad_twins_carry_only_bounded_fields() -> None:
    for fixture in _interaction_fixtures():
        fixture_case = fixture.get("case")
        assert isinstance(fixture_case, str)
        twin = require_json_object(fixture.get("broad_twin"), "fixture.broad_twin")
        unexpected = set(twin) - ALLOWED_BROAD_TWIN_KEYS
        assert not unexpected, f"{fixture_case}: {unexpected}"
        for key, value in twin.items():
            assert isinstance(value, str | int | float) or value is None, (
                f"{fixture_case}.{key} is unbounded: {type(value)}"
            )
            if isinstance(value, str):
                assert len(value) <= 64, f"{fixture_case}.{key} too long"


def test_classification_matrix_is_complete_and_bounded() -> None:
    matrix = _load_object(FIXTURE_ROOT / "classification.json")
    valid = {member.value for member in FieldClassification}
    classifications: set[str] = set()
    for value in matrix.values():
        assert isinstance(value, str)
        classifications.add(value)
    assert classifications <= valid

    # every top-level canonical field in every golden record is classified
    classified_roots = {key.split(".")[0] for key in matrix}
    for fixture in _interaction_fixtures():
        fixture_case = fixture.get("case")
        assert isinstance(fixture_case, str)
        record = require_json_object(fixture.get("record"), "fixture.record")
        unclassified = set(record) - classified_roots
        assert not unclassified, f"{fixture_case}: {unclassified}"

    # the §1 forbidden list stays forbidden
    for field in (
        "sdp",
        "raw_audio",
        "api_key",
        "authorization_header",
        "provider_request_headers",
        "provider_response_headers",
        "sql_statement",
        "url_query",
        "exception_repr",
        "vapid_private_key",
        "turn_secret",
        "tls_key",
    ):
        assert matrix[field] == "forbidden", field


def test_terminal_states_are_never_silent_success() -> None:
    """HTTP 200 plus in-band stream error is failure, never success (§5.2)."""
    for fixture in _interaction_fixtures():
        fixture_case = fixture.get("case")
        assert isinstance(fixture_case, str)
        record = require_json_object(fixture.get("record"), "fixture.record")
        native = require_json_object(
            record.get("provider_native"), "fixture.record.provider_native"
        )
        if native.get("terminal_error") is not None:
            assert record["status"] in {"failed", "incomplete"}, fixture_case
        if record["status"] == "succeeded":
            assert native.get("terminal_error") is None, fixture_case
            assert native.get("terminal_response") is not None, fixture_case


def test_speech_research_behavior_fixtures_are_well_formed() -> None:
    speech = _load_object(FIXTURE_ROOT / "speech/cases.json")
    speech_cases = _object_list(speech["cases"], "speech.cases")
    groups: set[str] = set()
    kinds: set[str] = set()
    for case in speech_cases:
        language_group = case.get("language_group")
        kind = case.get("kind")
        assert isinstance(language_group, str)
        assert isinstance(kind, str)
        groups.add(language_group)
        kinds.add(kind)
    assert {"en", "es", "pt", "code_switch", "none"} <= groups
    assert {"speech", "code_switch", "silence"} <= kinds
    for case in speech_cases:
        review = require_json_object(case["review"], "speech.case.review")
        assert review["status"] in {"seeded", "approved", "rejected"}

    research = _load_object(FIXTURE_ROOT / "research/cases.json")
    documents = _object_list(research["documents"], "research.documents")
    for document in documents:
        source_file = document.get("source_file")
        assert isinstance(source_file, str)
        assert (FIXTURE_ROOT / "research" / source_file).is_file()
    retrieval_cases = _object_list(
        research["retrieval_cases"], "research.retrieval_cases"
    )
    assert any(
        require_json_object(case.get("expected"), "research.case.expected").get(
            "no_hit"
        )
        for case in retrieval_cases
    )

    behavior = _load_object(FIXTURE_ROOT / "behavior/cases.json")
    behavior_cases = _object_list(behavior["cases"], "behavior.cases")
    dimensions: set[str] = set()
    for case in behavior_cases:
        dimension = case.get("dimension")
        assert isinstance(dimension, str)
        dimensions.add(dimension)
    assert {
        "crisis_safety",
        "non_diagnostic_boundary",
        "medication_boundary",
        "retrieval_grounding",
        "memory_attribution",
        "tool_authorization",
        "longitudinal_claim_support",
        "language_adherence",
        "structured_output_validity",
    } <= dimensions
    for case in behavior_cases:
        assert case["expected_behavior"]
        assert case["forbidden_behavior"]
        assert isinstance(case["high_risk"], bool)
