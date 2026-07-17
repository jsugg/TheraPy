"""O3.1 server-route instrumentation: readiness model, offer stages, finite
signal outcomes, read-result buckets, audio-serve evidence, crisis config
event. Broad surfaces carry enums/buckets only — never IDs, paths, or content.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def metric_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, float, dict]]:
    from therapy.observability import telemetry

    calls: list[tuple[str, float, dict]] = []
    monkeypatch.setattr(
        telemetry,
        "record_metric",
        lambda name, value, attrs=None: calls.append((name, value, attrs or {})),
    )
    return calls


def test_ready_reports_bounded_component_enums(client) -> None:
    payload = client.get("/ready").json()
    checks = payload["checks"]

    assert checks["db"] in ("ready", "degraded")
    assert checks["schema"].startswith("v") or checks["schema"] == "unknown"
    assert checks["db_size"] in ("ok", "large", "unknown")
    assert checks["wal_size"] in ("ok", "large", "unknown")
    assert checks["scheduler"] in ("starting", "ready", "degraded")
    assert checks["turn"] in ("ready", "unreachable")
    assert checks["telemetry"] in ("ready", "disabled")
    assert checks["voice"] in ("ready", "starting")
    # Enums only: no value may look like a path, error text, or identifier.
    for value in checks.values():
        assert " " not in value
        assert "/" not in value


def test_ready_scheduler_check_follows_heartbeat(client, monkeypatch) -> None:
    import time

    from therapy.dialogue import outreach

    monkeypatch.setitem(outreach._scheduler_heartbeat, "last_tick", time.time())
    assert client.get("/ready").json()["checks"]["scheduler"] == "ready"

    monkeypatch.setitem(
        outreach._scheduler_heartbeat, "last_tick", time.time() - 3_600
    )
    assert client.get("/ready").json()["checks"]["scheduler"] == "degraded"


def test_rejected_offer_counts_bounded_outcome(client, metric_calls) -> None:
    from therapy.server import app as app_module

    app_module.app.dependency_overrides[app_module.get_voice_gateway] = (
        lambda: object()
    )
    try:
        response = client.post(
            "/api/offer",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
    finally:
        app_module.app.dependency_overrides.clear()
    assert response.status_code == 400
    assert (
        "therapy_offers_total",
        1,
        {"outcome": "rejected"},
    ) in metric_calls


def test_resumable_and_ice_config_emit_finite_signal_outcomes(
    client, metric_calls
) -> None:
    assert client.get("/api/resumable").status_code == 200
    assert client.get("/api/ice-config").status_code == 200

    signals = [
        attrs for name, _, attrs in metric_calls
        if name == "therapy_voice_signal_total"
    ]
    assert {"operation": "resumable", "outcome": "none"} in signals
    assert {"operation": "ice_config", "outcome": "success"} in signals


def test_read_routes_record_count_buckets_only(client, metric_calls) -> None:
    assert client.get("/api/sessions").status_code == 200
    assert client.get("/api/graph").status_code == 200
    assert client.get("/api/research").status_code == 200

    reads = {
        attrs["route_class"]: attrs["bucket"]
        for name, _, attrs in metric_calls
        if name == "therapy_http_read_rows_total"
    }
    assert reads["sessions"] == "0"
    assert reads["graph"] == "0"
    assert reads["research_documents"] == "0"
    for bucket in reads.values():
        assert bucket in ("0", "1-9", "10-99", "100-999", "1000+")


def test_missing_turn_audio_counts_missing_outcome(client, metric_calls) -> None:
    response = client.get("/api/sessions/nope/turns/1/audio")
    assert response.status_code == 404
    assert (
        "therapy_audio_serve_total",
        1,
        {"outcome": "missing"},
    ) in metric_calls


def test_turn_audio_records_range_full_and_bytes(
    client, metric_calls, monkeypatch, tmp_path
) -> None:
    from therapy.server import app as app_mod

    wav = tmp_path / "turn.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 100)
    monkeypatch.setattr(
        type(app_mod._store()), "turn_audio_path", lambda self, s, t: str(wav)
    )

    assert client.get("/api/sessions/s/turns/1/audio").status_code == 200
    assert (
        client.get(
            "/api/sessions/s/turns/1/audio", headers={"Range": "bytes=0-3"}
        ).status_code
        in (200, 206)
    )

    outcomes = [
        attrs["outcome"] for name, _, attrs in metric_calls
        if name == "therapy_audio_serve_total"
    ]
    assert outcomes == ["full", "range"]
    byte_values = [
        value for name, value, _ in metric_calls
        if name == "therapy_audio_serve_bytes"
    ]
    assert byte_values == [104, 104]


def test_invalid_crisis_config_emits_fixed_event(client, monkeypatch) -> None:
    import io
    import json
    import logging

    from therapy.observability.logging import BroadJsonFormatter

    monkeypatch.setenv("THERAPY_CRISIS_CONTACTS", "not-json")
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
        assert client.get("/api/crisis-resources").status_code == 503
    finally:
        logger.handlers = previous_handlers

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    invalid = [e for e in events if e["event.name"] == "crisis_config_invalid"]
    assert [e["outcome"] for e in invalid] == ["error"]
    # Config validity only: the event must not carry the env value.
    assert "not-json" not in stream.getvalue()
