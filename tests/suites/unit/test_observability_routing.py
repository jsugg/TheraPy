"""Plane classifier and denylist contract (plan §5.5, O1 test list)."""

import json
from pathlib import Path

import pytest

from therapy.observability.model import TelemetryPlane
from therapy.observability.routing import (
    BROAD_SCOPES,
    CONTENT_ATTRIBUTE_KEYS,
    RESTRICTED_SCOPES,
    classify_scope,
    find_canaries,
    is_forbidden_broad_key,
    scrub_broad_attributes,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/observability"


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("pipecat", TelemetryPlane.RESTRICTED),
        ("pipecat.turn", TelemetryPlane.RESTRICTED),
        ("pipecat.future.scope", TelemetryPlane.RESTRICTED),
        ("therapy.interactions", TelemetryPlane.RESTRICTED),
        ("therapy.broad", TelemetryPlane.BROAD),
        ("opentelemetry.instrumentation.fastapi", TelemetryPlane.BROAD),
        ("opentelemetry.instrumentation.httpx", TelemetryPlane.BROAD),
    ],
)
def test_known_scopes_route_to_exactly_one_plane(scope, expected) -> None:
    assert classify_scope(scope) is expected


@pytest.mark.parametrize(
    "scope",
    [None, "", "random.sdk", "opentelemetry.instrumentation.sqlite3",
     "therapy.unknown", "langchain", "grpc"],
)
def test_unknown_scopes_are_dropped_not_fanned_out(scope) -> None:
    assert classify_scope(scope) is None


def test_scope_sets_are_disjoint() -> None:
    assert not (RESTRICTED_SCOPES & BROAD_SCOPES)


def test_pinned_pipecat_scopes_are_covered() -> None:
    """Every tracer scope recorded in the O0 snapshot routes restricted."""
    snapshot = json.loads(
        (FIXTURES / "pipecat/snapshot-1.5.0.json").read_text(encoding="utf-8")
    )
    for module in snapshot["tracing"].values():
        for scope in module.get("tracer_scopes", []):
            assert classify_scope(scope) is TelemetryPlane.RESTRICTED, scope


def test_pipecat_content_attribute_keys_are_denylisted() -> None:
    """Content-bearing keys from the pinned snapshot cannot survive a broad
    scrub even if a routing bug sent such a span to the broad plane."""
    denylisted_content = {
        "transcript", "text", "messages", "context_messages",
        "context_system_instruction", "instructions",
        "gen_ai.system_instructions", "tools.definitions", "arguments",
        "function_calls",
    }
    assert denylisted_content <= CONTENT_ATTRIBUTE_KEYS
    for key in denylisted_content:
        assert is_forbidden_broad_key(key), key


@pytest.mark.parametrize(
    "key",
    [
        "url.query", "url.full", "http.target", "client.address",
        "http.request.header.authorization", "http.response.header.set-cookie",
        "db.statement", "exception.message", "session.id", "user_agent.original",
    ],
)
def test_forbidden_broad_keys(key: str) -> None:
    assert is_forbidden_broad_key(key)


@pytest.mark.parametrize(
    "key",
    ["http.route", "http.request.method", "http.response.status_code",
     "duration_ms", "component", "outcome", "retry_count"],
)
def test_allowed_broad_keys(key: str) -> None:
    assert not is_forbidden_broad_key(key)


def test_scrub_drops_and_reports() -> None:
    result = scrub_broad_attributes(
        {
            "http.route": "/api/sessions/{session_id}",
            "url.query": "token=abc",
            "transcript": "secret words",
            "outcome": "success",
        }
    )
    assert result.attributes == {
        "http.route": "/api/sessions/{session_id}",
        "outcome": "success",
    }
    assert set(result.dropped_keys) == {"url.query", "transcript"}


def test_runtime_canary_scanner_matches_fixture_corpus() -> None:
    canaries = json.loads((FIXTURES / "canaries.json").read_text(encoding="utf-8"))
    merged = {**canaries["content"], **canaries["forbidden"]}
    sample = f"prefix {canaries['content']['completion']} suffix"
    assert find_canaries(sample, merged) == ["completion"]
    assert find_canaries("clean text", merged) == []
