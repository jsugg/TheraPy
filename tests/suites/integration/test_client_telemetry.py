"""Strict client telemetry endpoint (plan O4.1; O4 gate schema tests)."""

import random
from collections.abc import Iterator

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture
def enabled(client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("THERAPY_CLIENT_TELEMETRY", "1")
    # Same-origin is mandatory (O4 audit F-03): the shared test client
    # presents its own origin so accept-path tests exercise the happy path.
    client.headers["Origin"] = "http://testserver"
    from therapy.server import app as app_mod

    app_mod._client_bucket["tokens"] = 60.0
    return monkeypatch


def _batch(events: list[dict]) -> dict:
    return {"schema_version": 1, "events": events}


@pytest.fixture
def otel_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    """Capture owned and instrumented spans without changing the global provider."""
    from therapy.observability import telemetry

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry.state(), "enabled", True)
    try:
        yield provider, exporter
    finally:
        provider.shutdown()


def test_disabled_by_default_is_404(client, monkeypatch) -> None:
    monkeypatch.delenv("THERAPY_CLIENT_TELEMETRY", raising=False)
    response = client.post(
        "/api/telemetry/client",
        json=_batch([{"name": "sw_lifecycle", "outcome": "installed"}]),
    )
    assert response.status_code == 404


def test_valid_batch_accepted(client, enabled) -> None:
    response = client.post(
        "/api/telemetry/client",
        json=_batch(
            [
                {
                    "name": "webrtc_sample",
                    "rtt_ms": 42.5,
                    "jitter_ms": 3.1,
                    "packet_loss_ratio": 0.01,
                    "candidate_type": "relay",
                },
                {"name": "data_channel_state", "outcome": "connected",
                 "duration_ms": 350},
                {"name": "cache_fallback", "outcome": "fallback"},
            ]
        ),
    )
    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}


def test_valid_and_malformed_traceparents_control_owned_broad_span(
    client,
    enabled,
    monkeypatch: pytest.MonkeyPatch,
    otel_spans: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, exporter = otel_spans
    monkeypatch.setattr(trace_api, "get_tracer", provider.get_tracer)
    trace_id_hex = "0123456789abcdef0123456789abcdef"
    traceparent = f"00-{trace_id_hex}-0123456789abcdef-01"

    response = client.post(
        "/api/telemetry/client",
        json=_batch([{"name": "shell_fetch", "outcome": "success"}]),
        headers={"traceparent": traceparent},
    )
    assert response.status_code == 200
    spans = [
        span
        for span in exporter.get_finished_spans()
        if span.name == "client_telemetry"
        and span.instrumentation_scope.name == "therapy.broad"
    ]
    assert len(spans) == 1
    assert spans[0].context.trace_id == int(trace_id_hex, 16)
    assert spans[0].parent is not None
    assert spans[0].parent.is_remote
    assert dict(spans[0].attributes or {}) == {
        "component": "server",
        "operation": "client_telemetry",
    }

    exporter.clear()
    malformed = client.post(
        "/api/telemetry/client",
        json=_batch([{"name": "shell_fetch", "outcome": "success"}]),
        headers={"traceparent": "not-a-traceparent"},
    )
    assert malformed.status_code == 200
    malformed_spans = [
        span
        for span in exporter.get_finished_spans()
        if span.name == "client_telemetry"
        and span.instrumentation_scope.name == "therapy.broad"
    ]
    assert len(malformed_spans) == 1
    assert malformed_spans[0].context.trace_id != int(trace_id_hex, 16)
    assert malformed_spans[0].context.trace_id != 0
    assert malformed_spans[0].parent is None
    assert dict(malformed_spans[0].attributes or {}) == {
        "component": "server",
        "operation": "client_telemetry",
    }


def test_offer_fastapi_instrumentation_extracts_traceparent(
    data_dir,
    monkeypatch: pytest.MonkeyPatch,
    otel_spans: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.trace import SpanKind

    from therapy.server import app as app_mod
    from therapy.voice.contracts import (
        SessionTarget,
        WebRTCAnswer,
        WebRTCOffer,
    )

    class FakeGateway:
        async def negotiate(
            self, offer: WebRTCOffer, target: SessionTarget
        ) -> WebRTCAnswer:
            return WebRTCAnswer(sdp="answer-sdp", type="answer", pc_id="peer-1")

    provider, exporter = otel_spans
    traced_app = FastAPI()
    traced_app.add_api_route("/api/offer", app_mod.offer, methods=["POST"])
    traced_app.dependency_overrides[app_mod.get_voice_gateway] = FakeGateway
    FastAPIInstrumentor.instrument_app(
        traced_app,
        tracer_provider=provider,
        exclude_spans=["receive", "send"],
    )
    monkeypatch.setattr(trace_api, "get_tracer", provider.get_tracer)
    trace_id_hex = "fedcba9876543210fedcba9876543210"
    traceparent = f"00-{trace_id_hex}-fedcba9876543210-01"
    app_mod._store.cache_clear()
    try:
        with TestClient(traced_app) as traced_client:
            response = traced_client.post(
                "/api/offer?new_session=1",
                json={"sdp": "offer-sdp", "type": "offer"},
                headers={"traceparent": traceparent},
            )
    finally:
        FastAPIInstrumentor.uninstrument_app(traced_app)
        app_mod._store.cache_clear()

    assert response.status_code == 200
    trace_id = int(trace_id_hex, 16)
    server_spans = [
        span for span in exporter.get_finished_spans() if span.kind is SpanKind.SERVER
    ]
    assert len(server_spans) == 1
    assert server_spans[0].context.trace_id == trace_id
    assert server_spans[0].parent is not None
    assert server_spans[0].parent.is_remote
    owned_spans = [
        span
        for span in exporter.get_finished_spans()
        if span.instrumentation_scope.name == "therapy.broad"
    ]
    assert owned_spans
    assert {span.context.trace_id for span in owned_spans} == {trace_id}


@pytest.mark.parametrize(
    "event",
    [
        {"name": "free text about the user"},  # unlisted name
        {"name": "webrtc_sample", "url": "https://leak.example"},  # unknown key
        {"name": "webrtc_sample", "rtt_ms": -1},  # out of range
        {"name": "webrtc_sample", "rtt_ms": 999_999_999},  # out of range
        {"name": "webrtc_sample", "candidate_type": "prflx-with-ip"},  # unlisted
        {"name": "disconnect", "outcome": "because my mic broke"},  # free text
        {"name": "webrtc_sample", "packet_loss_ratio": 1.5},
    ],
)
def test_free_text_identifiers_and_ranges_rejected(client, enabled, event) -> None:
    response = client.post("/api/telemetry/client", json=_batch([event]))
    assert response.status_code == 422


def test_batch_bounds(client, enabled) -> None:
    too_many = [{"name": "shell_fetch"}] * 21
    assert (
        client.post("/api/telemetry/client", json=_batch(too_many)).status_code
        == 422
    )
    assert (
        client.post("/api/telemetry/client", json=_batch([])).status_code == 422
    )
    wrong_version = {"schema_version": 2, "events": [{"name": "shell_fetch"}]}
    assert (
        client.post("/api/telemetry/client", json=wrong_version).status_code == 422
    )


def test_oversize_body_rejected(client, enabled) -> None:
    response = client.post(
        "/api/telemetry/client",
        content=b"x" * 100,
        headers={
            "Content-Length": str(20 * 1024),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 413


def test_cross_origin_rejected(client, enabled) -> None:
    response = client.post(
        "/api/telemetry/client",
        json=_batch([{"name": "shell_fetch"}]),
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "headers",
    [
        {},  # Origin is required, not optional (O4 audit probe)
        {"Origin": "null"},  # opaque origin
        {"Origin": "::::"},  # malformed origin
        {"Origin": "https://testserver"},  # same netloc, wrong scheme
        {"Origin": "http://testserver/path"},  # an origin has no path
    ],
)
def test_missing_null_malformed_or_wrong_scheme_origin_rejected(
    client, enabled, headers
) -> None:
    request_headers = dict(headers) or {"Origin": ""}
    response = client.post(
        "/api/telemetry/client",
        json=_batch([{"name": "shell_fetch"}]),
        headers=request_headers,
    )
    assert response.status_code == 403


def test_forwarded_proto_defines_the_expected_scheme(client, enabled) -> None:
    """Behind the HTTPS proxy the browser origin is https; the forwarded
    scheme must win over the socket scheme."""
    response = client.post(
        "/api/telemetry/client",
        json=_batch([{"name": "shell_fetch"}]),
        headers={
            "Origin": "https://testserver",
            "X-Forwarded-Proto": "https",
        },
    )
    assert response.status_code == 200
    mismatched = client.post(
        "/api/telemetry/client",
        json=_batch([{"name": "shell_fetch"}]),
        headers={
            "Origin": "http://testserver",
            "X-Forwarded-Proto": "https",
        },
    )
    assert mismatched.status_code == 403


@pytest.mark.parametrize(
    "event_override",
    [
        {"rtt_ms": "12.5"},  # numeric string
        {"bytes_delta": True},  # boolean as integer
        {"bytes_delta": 1.0},  # integral float as integer
        {"jitter_ms": False},  # boolean as float
    ],
)
def test_lax_type_coercions_rejected(client, enabled, event_override) -> None:
    event = {"name": "webrtc_sample", **event_override}
    response = client.post("/api/telemetry/client", json=_batch([event]))
    assert response.status_code == 422


def test_boolean_schema_version_rejected(client, enabled) -> None:
    payload = {"schema_version": True, "events": [{"name": "shell_fetch"}]}
    response = client.post("/api/telemetry/client", json=payload)
    assert response.status_code == 422


def test_malformed_content_length_is_controlled_client_error(
    client, enabled
) -> None:
    response = client.post(
        "/api/telemetry/client",
        content=b"{}",
        headers={
            "Content-Length": "invalid",
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 400


def test_rejections_emit_fixed_schema_event_without_payload_values(
    client, enabled
) -> None:
    import io
    import json
    import logging

    from therapy.observability.logging import BroadJsonFormatter

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        BroadJsonFormatter(service_version="0.1.0", environment="test")
    )
    logger = logging.getLogger("therapy.broad")
    previous_handlers = logger.handlers
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        marker = "sk-secret-payload-value"
        response = client.post(
            "/api/telemetry/client",
            json=_batch([{"name": "webrtc_sample", "url": marker}]),
        )
    finally:
        logger.handlers = previous_handlers
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid telemetry batch"}
    lines = stream.getvalue().splitlines()
    events = [json.loads(line) for line in lines]
    rejected = [e for e in events if e["event.name"] == "client_telemetry_rejected"]
    assert [e["outcome"] for e in rejected] == ["schema_error"]
    assert marker not in stream.getvalue()


def test_all_accepted_webrtc_fields_feed_metrics(
    client, enabled, monkeypatch
) -> None:
    """O4 audit F-06: accepted diagnostics must never be silently discarded."""
    from therapy.observability import telemetry

    recorded: list[tuple[str, float, dict]] = []
    monkeypatch.setattr(
        telemetry,
        "record_metric",
        lambda name, value, attrs=None: recorded.append((name, value, attrs or {})),
    )
    response = client.post(
        "/api/telemetry/client",
        json=_batch(
            [
                {
                    "name": "webrtc_sample",
                    "rtt_ms": 42.5,
                    "jitter_ms": 3.1,
                    "packet_loss_ratio": 0.01,
                    "bitrate_kbps": 128.0,
                    "bytes_delta": 20_000,
                    "concealed_samples": 12,
                    "candidate_type": "relay",
                    "dropped_events": 3,
                }
            ]
        ),
    )
    assert response.status_code == 200
    names = {name for name, _, _ in recorded}
    assert {
        "therapy_webrtc_rtt_seconds",
        "therapy_webrtc_jitter_seconds",
        "therapy_webrtc_packet_loss_ratio",
        "therapy_webrtc_bitrate_kbps",
        "therapy_webrtc_bytes_total",
        "therapy_webrtc_concealed_samples_total",
        "therapy_client_dropped_events_total",
        "therapy_client_events_total",
    } <= names


def test_rate_limit_is_process_wide_not_ip_keyed(client, enabled) -> None:
    from therapy.server import app as app_mod

    app_mod._client_bucket["tokens"] = 1.0
    ok = client.post(
        "/api/telemetry/client", json=_batch([{"name": "shell_fetch"}])
    )
    assert ok.status_code == 200
    limited = client.post(
        "/api/telemetry/client", json=_batch([{"name": "shell_fetch"}])
    )
    assert limited.status_code == 429


def test_fuzz_random_payloads_never_pass_schema(client, enabled) -> None:
    """Deterministic fuzz loop (O4 gate): randomly structured junk is always
    422, never accepted or a 500."""
    rng = random.Random(20260716)
    fragments = [
        "session-1234", "https://x.test/?q=1", "v=0\\no=- 46117", "sk-abc",
        {"nested": {"deep": True}}, ["a", 1], 1e308, -5, "shell_fetch",
    ]
    from therapy.server import app as app_mod

    for _ in range(100):
        app_mod._client_bucket["tokens"] = 60.0  # isolate schema behavior
        event = {
            rng.choice(["name", "outcome", "rtt_ms", "extra", "url", "id"]):
                rng.choice(fragments)
            for _ in range(rng.randint(1, 4))
        }
        response = client.post("/api/telemetry/client", json=_batch([event]))
        assert response.status_code in (200, 422), response.text
        if response.status_code == 200:
            # only possible when the random draw formed a fully valid event
            assert event.get("name") == "shell_fetch"
            assert set(event) <= {"name", "outcome", "rtt_ms"}
