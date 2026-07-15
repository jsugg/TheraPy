"""O0 gate over the committed fixture corpus (plan O0.2 and the O0 gate).

- Required content canaries occur in restricted records only.
- Content is absent from every broad surface.
- Credential/infrastructure canaries occur nowhere.
- The golden scenarios and classification matrix stay complete.
"""

import json

from scripts.observability.canary_scan import (
    FIXTURE_ROOT,
    load_canaries,
    scan_fixture_tree,
)
from therapy.observability.model import FieldClassification

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


def _interaction_fixtures() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
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
    scenarios = {fixture["scenario"] for fixture in fixtures}
    providers = {fixture["record"]["provider"] for fixture in fixtures}
    assert REQUIRED_SCENARIOS <= scenarios, REQUIRED_SCENARIOS - scenarios
    assert REQUIRED_PROVIDERS <= providers, REQUIRED_PROVIDERS - providers


def test_broad_twins_carry_only_bounded_fields() -> None:
    for fixture in _interaction_fixtures():
        twin = fixture["broad_twin"]
        unexpected = set(twin) - ALLOWED_BROAD_TWIN_KEYS
        assert not unexpected, f"{fixture['case']}: {unexpected}"
        for key, value in twin.items():
            assert isinstance(value, str | int | float) or value is None, (
                f"{fixture['case']}.{key} is unbounded: {type(value)}"
            )
            if isinstance(value, str):
                assert len(value) <= 64, f"{fixture['case']}.{key} too long"


def test_classification_matrix_is_complete_and_bounded() -> None:
    matrix = json.loads(
        (FIXTURE_ROOT / "classification.json").read_text(encoding="utf-8")
    )
    valid = {member.value for member in FieldClassification}
    assert set(matrix.values()) <= valid

    # every top-level canonical field in every golden record is classified
    classified_roots = {key.split(".")[0] for key in matrix}
    for fixture in _interaction_fixtures():
        unclassified = set(fixture["record"]) - classified_roots
        assert not unclassified, f"{fixture['case']}: {unclassified}"

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
        record = fixture["record"]
        native = record["provider_native"]
        if native.get("terminal_error") is not None:
            assert record["status"] in {"failed", "incomplete"}, fixture["case"]
        if record["status"] == "succeeded":
            assert native.get("terminal_error") is None, fixture["case"]
            assert native.get("terminal_response") is not None, fixture["case"]


def test_speech_research_behavior_fixtures_are_well_formed() -> None:
    speech = json.loads(
        (FIXTURE_ROOT / "speech/cases.json").read_text(encoding="utf-8")
    )
    groups = {case["language_group"] for case in speech["cases"]}
    assert {"en", "es", "pt", "code_switch", "none"} <= groups
    kinds = {case["kind"] for case in speech["cases"]}
    assert {"speech", "code_switch", "silence"} <= kinds
    for case in speech["cases"]:
        assert case["review"]["status"] in {"seeded", "approved", "rejected"}

    research = json.loads(
        (FIXTURE_ROOT / "research/cases.json").read_text(encoding="utf-8")
    )
    for document in research["documents"]:
        assert (FIXTURE_ROOT / "research" / document["source_file"]).is_file()
    assert any(case.get("expected", {}).get("no_hit") for case in research["retrieval_cases"])

    behavior = json.loads(
        (FIXTURE_ROOT / "behavior/cases.json").read_text(encoding="utf-8")
    )
    dimensions = {case["dimension"] for case in behavior["cases"]}
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
    for case in behavior["cases"]:
        assert case["expected_behavior"]
        assert case["forbidden_behavior"]
        assert isinstance(case["high_risk"], bool)
