"""Broad JSON logging contract (plan §5.4, §4; O1 test list).

Proves: fixed-schema single-line JSON, no exception payloads, filtered
package-relative stacks for owned failures only, third-party policy that a
global DEBUG setting cannot override, and rate-limited repeat events.
"""

import io
import json
import logging
from collections.abc import Iterator

import pytest

from therapy.observability.logging import (
    THIRD_PARTY_LOGGER_POLICY,
    BroadJsonFormatter,
    RateLimitFilter,
    apply_third_party_policy,
    emit_event,
)

ALLOWED_KEYS = {
    "timestamp", "severity", "event.name", "service.name", "service.version",
    "service.instance.id", "deployment.environment", "runtime.python",
    "trace_id", "span_id", "component", "operation", "outcome",
    "duration_ms", "error.type", "retry_count", "count", "stack",
}


@pytest.fixture
def capture() -> Iterator[tuple[io.StringIO, logging.Logger]]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        BroadJsonFormatter(service_version="0.1.0", environment="test")
    )
    logger = logging.getLogger("therapy.broad")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield stream, logger
    logger.handlers = []


def _lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_events_are_single_line_fixed_schema_json(capture) -> None:
    stream, _ = capture
    emit_event(
        "voice.connect",
        component="voice",
        operation="negotiate",
        outcome="success",
        duration_ms=12.3456,
    )
    lines = _lines(stream)
    assert len(lines) == 1
    event = lines[0]
    assert event["event.name"] == "voice.connect"
    assert event["component"] == "voice"
    assert event["outcome"] == "success"
    assert event["duration_ms"] == 12.346
    assert set(event) <= ALLOWED_KEYS
    assert event["deployment.environment"] == "test"
    assert len(event["service.instance.id"]) == 32


def test_provider_exception_message_never_reaches_broad_output(capture) -> None:
    stream, _ = capture
    secret = "sk-super-secret-token in the provider error"
    exc = RuntimeError(secret)
    emit_event(
        "llm.attempt",
        severity=logging.WARNING,
        component="llm",
        operation="summary",
        outcome="error",
        error_type=type(exc).__name__,
        exc=exc,
        owned_failure=False,  # provider failure: classified event only
    )
    text = stream.getvalue()
    assert "sk-super-secret" not in text
    event = _lines(stream)[0]
    assert event["error.type"] == "RuntimeError"
    assert "stack" not in event  # stacks are for unexpected owned failures


def test_owned_failure_gets_filtered_package_relative_stack(capture) -> None:
    stream, _ = capture

    def _boom() -> None:
        raise ValueError("contains /Users/jsugg/private/path and secrets")

    try:
        _boom()
    except ValueError as exc:
        emit_event(
            "journal.write_failed",
            severity=logging.ERROR,
            component="journal",
            operation="append",
            outcome="error",
            exc=exc,
            owned_failure=True,
        )
    event = _lines(stream)[0]
    assert event["error.type"] == "ValueError"
    assert event["stack"], "owned failure must carry a stack"
    joined = " ".join(event["stack"])
    assert "secrets" not in joined  # no message text
    assert "/Users/" not in joined  # package-relative paths only
    assert "_boom" in joined


def test_global_debug_cannot_reenable_third_party_loggers() -> None:
    apply_third_party_policy()
    logging.getLogger().setLevel(logging.DEBUG)  # the "global DEBUG" attempt

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    for name, policy in THIRD_PARTY_LOGGER_POLICY.items():
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        logger.debug("third-party debug payload with content")
        if policy == "disabled":
            logger.critical("even critical is dropped for disabled loggers")
        logger.removeHandler(handler)
    assert stream.getvalue() == ""

    # WARNING passes for fixed-level (non-disabled) loggers
    stream2 = io.StringIO()
    handler2 = logging.StreamHandler(stream2)
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.addHandler(handler2)
    httpx_logger.warning("warn-level passes")
    httpx_logger.removeHandler(handler2)
    assert "warn-level passes" in stream2.getvalue()

    # uvicorn.access stays disabled outright
    stream3 = io.StringIO()
    handler3 = logging.StreamHandler(stream3)
    access = logging.getLogger("uvicorn.access")
    access.addHandler(handler3)
    access.warning('GET /api/sessions?token=abc 200')
    access.removeHandler(handler3)
    assert stream3.getvalue() == ""

    logging.getLogger().setLevel(logging.WARNING)


def test_rate_limit_filter_collapses_repeats() -> None:
    filt = RateLimitFilter(interval_seconds=60.0)

    def record(name: str) -> logging.LogRecord:
        rec = logging.LogRecord("therapy.broad", logging.WARNING, "", 0, name, (), None)
        rec.event_name = name
        rec.rate_limited = True
        return rec

    first = record("capture_degraded")
    assert filt.filter(first) is True
    suppressed = [filt.filter(record("capture_degraded")) for _ in range(5)]
    assert suppressed == [False] * 5
    # unrelated events are unaffected
    other = record("capture_recovered")
    assert filt.filter(other) is True
