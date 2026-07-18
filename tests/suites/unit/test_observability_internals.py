"""Focused coverage for correlation scopes and readiness state internals."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import pytest
from opentelemetry.sdk.resources import Resource

from therapy.observability import capture, telemetry
from therapy.observability.config import ObservabilityConfig
from therapy.observability.context import (
    TraceContext,
    active_trace_context,
    current_interaction_id,
    interaction_scope,
    trace_scope,
)
from therapy.observability.health import ComponentState, HealthRegistry
from therapy.observability.journal import JournalHealth, JournalStore
from therapy.observability.model import CaptureMode, Component


class _HealthMonitor(Protocol):
    async def __call__(
        self,
        store: JournalStore | None,
        interval_s: float = 15.0,
        *,
        retention_days: int = 30,
        ack_backend: str | None = None,
    ) -> None: ...


class _HealthyStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.maintenance: list[str] = []

    def health(self) -> JournalHealth:
        return JournalHealth(0, 0, 0, 0, "2026-07-17T00:00:00+00:00", 7, None, 11)

    def checkpoint(self) -> None:
        self.maintenance.append("checkpoint")

    def apply_retention(
        self, retention_days: int, *, require_ack_backend: str | None = None
    ) -> int:
        assert retention_days == 7
        assert require_ack_backend == "phoenix"
        self.maintenance.append("retention")
        return 0

    def integrity_check(self) -> bool:
        self.maintenance.append("integrity")
        return True


def _config(
    tmp_path: Path,
    *,
    interaction_backend: str = "journal",
    otlp_broad_endpoint: str | None = None,
    otlp_restricted_endpoint: str | None = None,
) -> ObservabilityConfig:
    return ObservabilityConfig(
        capture_mode=CaptureMode.RUNTIME,
        journal_path=tmp_path / "journal.sqlite3",
        interaction_backend=interaction_backend,
        remote_export_enabled=False,
        retention_days=30,
        queue_size=32,
        group_commit_ms=10,
        otel_enabled=True,
        otlp_broad_endpoint=otlp_broad_endpoint,
        otlp_restricted_endpoint=otlp_restricted_endpoint,
        otel_export_timeout_secs=1.0,
        environment="test",
        log_level="INFO",
        client_telemetry_enabled=False,
    )


def test_nested_trace_and_interaction_scopes_restore_outer_values() -> None:
    outer = TraceContext("1" * 32, "2" * 16)
    inner = outer.child()
    assert inner.trace_id == outer.trace_id
    assert inner.span_id != outer.span_id

    before = active_trace_context()
    with trace_scope(outer) as bound:
        assert bound is outer
        assert active_trace_context() is outer
        with trace_scope(inner):
            assert active_trace_context() is inner
        assert active_trace_context() is outer
    assert active_trace_context() is before

    assert current_interaction_id() is None
    with interaction_scope("itx-explicit") as interaction_id:
        assert interaction_id == "itx-explicit"
        assert current_interaction_id() == "itx-explicit"
        with interaction_scope() as generated:
            assert generated.startswith("itx-")
        assert current_interaction_id() == "itx-explicit"
    assert current_interaction_id() is None


def test_health_registry_snapshot_dedupes_and_lists_degraded() -> None:
    health = HealthRegistry()
    health.set_state(Component.SERVER, ComponentState.STARTING, "boot")
    first = health.snapshot()["server"]
    health.set_state(Component.SERVER, ComponentState.STARTING, "ignored")
    assert health.snapshot()["server"] == first

    health.set_state(Component.SERVER, ComponentState.READY)
    health.set_state(Component.JOURNAL, ComponentState.DEGRADED, "backlog")
    snapshot = health.snapshot()
    assert snapshot["server"]["state"] == "ready"
    assert snapshot["journal"]["reason"] == "backlog"
    assert health.degraded_components() == [Component.JOURNAL]


def test_telemetry_initialize_builds_owned_provider_without_exporters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def noop() -> None:
        return None

    monkeypatch.setattr(telemetry, "_instrument_fastapi", noop)
    monkeypatch.setattr(telemetry, "_instrument_system_metrics", noop)
    current = telemetry.state()
    current.enabled = False
    current.provider = None

    assert telemetry.initialize(_config(tmp_path), service_version="test") is True
    assert current.enabled is True
    assert current.provider is not None
    assert telemetry.active_span_ids() is None


def test_interaction_exporter_selection_and_remote_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert telemetry.make_interaction_exporter(_config(tmp_path)) is None
    phoenix_without_endpoint = _config(tmp_path, interaction_backend="phoenix")
    assert telemetry.make_interaction_exporter(phoenix_without_endpoint) is None

    remote = _config(
        tmp_path,
        interaction_backend="phoenix",
        otlp_restricted_endpoint="https://telemetry.example",
    )
    assert telemetry.make_interaction_exporter(remote) is None

    sentinel = object()

    def fake_exporter(endpoint: str, *, timeout: float) -> object:
        assert endpoint == "http://phoenix:6006"
        assert timeout == 1.0
        return sentinel

    monkeypatch.setattr(telemetry, "PhoenixInteractionExporter", fake_exporter)
    local = _config(
        tmp_path,
        interaction_backend="phoenix",
        otlp_restricted_endpoint="http://phoenix:6006",
    )
    assert telemetry.make_interaction_exporter(local) is sentinel


def test_metric_initializer_builds_full_manifest(tmp_path: Path) -> None:
    value = vars(telemetry).get("_initialize_metrics")
    assert callable(value)
    initialize_metrics = cast(
        Callable[[ObservabilityConfig, Resource], None], value
    )
    config = _config(tmp_path, otlp_broad_endpoint="http://collector:4318")

    initialize_metrics(config, Resource({"service.name": "therapy-test"}))

    current = telemetry.state()
    assert current.meter_provider is not None
    assert "therapy_conversation_turns_total" in current.instruments


def test_capture_health_monitor_samples_storage_and_maintenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "journal.sqlite3"
    path.write_bytes(b"journal")
    store = _HealthyStore(path)
    sleeps = 0

    async def one_tick(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps > 1:
            raise asyncio.CancelledError

    metrics: list[str] = []

    def record_metric(
        name: str, value: float, attributes: dict[str, str] | None = None
    ) -> None:
        del value, attributes
        metrics.append(name)

    def inspect_storage() -> bool:
        return True

    monkeypatch.setattr(asyncio, "sleep", one_tick)
    monkeypatch.setattr(telemetry, "record_metric", record_metric)
    monkeypatch.setattr(capture, "inspect_product_storage", inspect_storage)
    value = vars(capture).get("_health_monitor")
    assert callable(value)
    monitor = cast(_HealthMonitor, value)

    asyncio.run(
        monitor(
            cast(JournalStore, store),
            interval_s=3_600.0,
            retention_days=7,
            ack_backend="phoenix",
        )
    )

    assert store.maintenance == ["checkpoint", "retention", "integrity"]
    assert "therapy_event_loop_lag_seconds" in metrics
    assert "therapy_llm_capture_journal_bytes" in metrics
    assert "therapy_sqlite_integrity_last_success_unixtime" in metrics


def test_capture_health_monitor_contains_store_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class BrokenStore(_HealthyStore):
        def health(self) -> JournalHealth:
            raise OSError("journal unavailable")

    store = BrokenStore(tmp_path / "missing.sqlite3")
    sleeps = 0

    async def one_tick(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps > 1:
            raise asyncio.CancelledError

    events: list[str] = []

    def emit_event(name: str, **_fields: object) -> None:
        events.append(name)

    def inspect_storage() -> bool:
        return True

    monkeypatch.setattr(asyncio, "sleep", one_tick)
    monkeypatch.setattr(capture, "emit_event", emit_event)
    monkeypatch.setattr(capture, "inspect_product_storage", inspect_storage)
    value = vars(capture).get("_health_monitor")
    assert callable(value)
    monitor = cast(_HealthMonitor, value)

    asyncio.run(
        monitor(
            cast(JournalStore, store),
            interval_s=3_600.0,
            retention_days=7,
            ack_backend="phoenix",
        )
    )

    assert events == ["journal.health_failed"]


def test_capture_health_monitor_contains_size_inspection_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class InvalidTimestampStore(_HealthyStore):
        def health(self) -> JournalHealth:
            return JournalHealth(0, 0, 0, 0, "not-a-timestamp", 7, None, 11)

    store = InvalidTimestampStore(tmp_path / "missing.sqlite3")
    sleeps = 0

    async def one_tick(_seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps > 1:
            raise asyncio.CancelledError

    events: list[str] = []

    def emit_event(name: str, **_fields: object) -> None:
        events.append(name)

    def inspect_storage() -> bool:
        return True

    monkeypatch.setattr(asyncio, "sleep", one_tick)
    monkeypatch.setattr(capture, "emit_event", emit_event)
    monkeypatch.setattr(capture, "inspect_product_storage", inspect_storage)
    value = vars(capture).get("_health_monitor")
    assert callable(value)
    monitor = cast(_HealthMonitor, value)

    asyncio.run(
        monitor(
            cast(JournalStore, store),
            interval_s=3_600.0,
            retention_days=7,
            ack_backend="phoenix",
        )
    )

    assert events == [
        "journal.size_inspection_failed",
        "journal.health_invalid_timestamp",
    ]
