"""O3.1 server-route instrumentation: readiness model, offer stages, finite
signal outcomes, read-result buckets, audio-serve evidence, crisis config
event. Broad surfaces carry enums/buckets only — never IDs, paths, or content.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from tests.type_contracts import HttpTestClient
from therapy.memory import MemoryStore

type MetricCall = tuple[str, float, dict[str, str]]
type MetricCalls = list[MetricCall]


@pytest.fixture
def metric_calls(monkeypatch: pytest.MonkeyPatch) -> MetricCalls:
    from therapy.observability import telemetry

    calls: MetricCalls = []

    def _record_metric(
        name: str, value: float, attrs: dict[str, str] | None = None
    ) -> None:
        calls.append((name, value, attrs or {}))

    monkeypatch.setattr(telemetry, "record_metric", _record_metric)
    return calls


def test_ready_reports_bounded_component_enums(client: HttpTestClient) -> None:
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


def test_ready_scheduler_check_follows_heartbeat(
    client: HttpTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from therapy.dialogue import outreach

    heartbeat = {"value": time.time()}
    def _last_tick() -> float:
        return heartbeat["value"]

    monkeypatch.setattr(outreach, "last_scheduler_tick", _last_tick)
    assert client.get("/ready").json()["checks"]["scheduler"] == "ready"

    heartbeat["value"] = time.time() - 3_600
    assert client.get("/ready").json()["checks"]["scheduler"] == "degraded"


def test_rejected_offer_counts_bounded_outcome(
    client: HttpTestClient, metric_calls: MetricCalls
) -> None:
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
    client: HttpTestClient, metric_calls: MetricCalls
) -> None:
    assert client.get("/api/resumable").status_code == 200
    assert client.get("/api/ice-config").status_code == 200

    signals = [
        attrs for name, _, attrs in metric_calls
        if name == "therapy_voice_signal_total"
    ]
    assert {"operation": "resumable", "outcome": "none"} in signals
    assert {"operation": "ice_config", "outcome": "success"} in signals


def test_read_routes_record_count_buckets_only(
    client: HttpTestClient, metric_calls: MetricCalls
) -> None:
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


def test_missing_turn_audio_counts_missing_outcome(
    client: HttpTestClient, metric_calls: MetricCalls
) -> None:
    response = client.get("/api/sessions/nope/turns/1/audio")
    assert response.status_code == 404
    assert (
        "therapy_audio_serve_total",
        1,
        {"outcome": "missing"},
    ) in metric_calls


def test_turn_audio_records_range_full_and_bytes(
    client: HttpTestClient,
    metric_calls: MetricCalls,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from therapy.server import app as app_mod

    wav = tmp_path / "turn.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 100)
    store_factory_value = getattr(app_mod, "_store", None)
    if not callable(store_factory_value):
        raise TypeError("server store factory is unavailable")
    store_factory = cast(Callable[[], MemoryStore], store_factory_value)

    def _turn_audio_path(
        store: MemoryStore, session_id: str, turn_id: int
    ) -> str:
        del store, session_id, turn_id
        return str(wav)

    monkeypatch.setattr(
        type(store_factory()), "turn_audio_path", _turn_audio_path
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


def test_invalid_crisis_config_emits_fixed_event(
    client: HttpTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    from therapy.observability.interactions import require_json_object

    events = [
        require_json_object(json.loads(line), "test.log.event")
        for line in stream.getvalue().splitlines()
    ]
    invalid = [e for e in events if e["event.name"] == "crisis_config_invalid"]
    assert [e["outcome"] for e in invalid] == ["error"]
    # Config validity only: the event must not carry the env value.
    assert "not-json" not in stream.getvalue()
