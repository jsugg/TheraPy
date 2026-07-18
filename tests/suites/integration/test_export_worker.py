"""Export worker outage/retry/ACK contract (plan O1.2, O1 gate)."""

import asyncio
import sqlite3
from pathlib import Path

import pytest

from tests.type_contracts import MetricCall, metric_recorder
from therapy.observability.exporters import (
    ExportError,
    ExportWorker,
    ExportWorkerConfig,
    PermanentExportError,
)
from therapy.observability.interactions import (
    InteractionRecord,
    InteractionRequest,
    InteractionResponse,
    Message,
    ProviderNative,
)
from therapy.observability.journal import JournalStore, LoadedInteraction
from therapy.observability.model import (
    InteractionOperation,
    InteractionStatus,
    Provider,
)


def _connection(store: JournalStore) -> sqlite3.Connection:
    """Return the journal connection for export-state assertions."""
    connection = getattr(store, "_conn", None)
    assert isinstance(connection, sqlite3.Connection)
    return connection


def _record(interaction_id: str) -> InteractionRecord:
    return InteractionRecord(
        interaction_id=interaction_id,
        trace_id="e" * 32,
        span_id="f" * 16,
        operation=InteractionOperation.SUMMARY,
        provider=Provider.OLLAMA,
        requested_model="m",
        actual_model="m",
        prompt_template_version="v1",
        request=InteractionRequest(
            system_instructions="s", messages=(Message("user", "u"),)
        ),
        response=InteractionResponse(),
        stream=(),
        error=None,
        provider_native=ProviderNative(request={"model": "m"}),
        language="en",
        modality="text",
        build_version="t",
        policy_version="p",
        config_version="c",
        started_at="2026-07-16T00:00:00+00:00",
        completed_at=None,
        status=InteractionStatus.STARTED,
    )


class FlakyExporter:
    """Fails `failures` times per record, then acknowledges."""

    backend_name = "phoenix"

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.attempts: dict[str, int] = {}
        self.acked: list[str] = []

    async def export(self, interaction: LoadedInteraction) -> str | None:
        interaction_id = interaction["interaction"]["interaction_id"]
        count = self.attempts.get(interaction_id, 0) + 1
        self.attempts[interaction_id] = count
        if count <= self.failures:
            raise ExportError("backend down")
        self.acked.append(interaction_id)
        return f"backend-{interaction_id}"


class RejectingExporter:
    backend_name = "phoenix"

    async def export(self, interaction: LoadedInteraction) -> str | None:
        raise PermanentExportError("payload rejected")


def test_backend_outage_keeps_records_until_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from therapy.observability import telemetry

    metrics: list[MetricCall] = []
    monkeypatch.setattr(
        telemetry,
        "record_metric",
        metric_recorder(metrics),
    )

    async def scenario() -> None:
        store = JournalStore(tmp_path / "journal.sqlite3")
        for index in range(2):
            store.start_attempt(_record(f"itx-w-{index}"))
            store.finish_success(f"itx-w-{index}", {"completion": "x"})

        exporter = FlakyExporter(failures=1)
        worker = ExportWorker(
            store, exporter, ExportWorkerConfig(base_backoff_s=0.0, max_backoff_s=0.0)
        )
        # cycle 1: both fail; records stay pending with backoff recorded
        await worker.drain_once()
        assert exporter.acked == []
        assert store.health().unexported_records == 2

        # cycle 2: retries succeed; ACK recorded idempotently
        await worker.drain_once()
        assert sorted(exporter.acked) == ["itx-w-0", "itx-w-1"]
        assert store.pending_exports("phoenix") == []

        # replaying a drain never re-exports acknowledged records
        await worker.drain_once()
        assert len(exporter.acked) == 2
        store.close()

    asyncio.run(scenario())
    capture = [item for item in metrics if item[0].startswith("therapy_llm_capture")]
    durations = [attrs["outcome"] for name, _, attrs in capture if name.endswith("seconds")]
    assert durations == ["error", "error", "success", "success"]
    records = [
        attrs
        for name, _, attrs in capture
        if name == "therapy_llm_capture_records_total"
    ]
    assert records == [
        {"operation": "summary", "status": "failed"},
        {"operation": "summary", "status": "failed"},
        {"operation": "summary", "status": "exported"},
        {"operation": "summary", "status": "exported"},
    ]
    export_outcomes = [
        attrs["outcome"]
        for name, _, attrs in capture
        if name == "therapy_llm_capture_exports_total"
    ]
    assert export_outcomes == ["failure", "failure", "accepted", "accepted"]
    assert "completion" not in repr(capture)


def test_permanent_rejection_is_parked_not_looped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from therapy.observability import telemetry

    metrics: list[MetricCall] = []
    monkeypatch.setattr(
        telemetry,
        "record_metric",
        metric_recorder(metrics),
    )

    async def scenario() -> None:
        store = JournalStore(tmp_path / "journal.sqlite3")
        store.start_attempt(_record("itx-perm"))
        store.finish_error("itx-perm", {"provider_type": "boom"})
        worker = ExportWorker(store, RejectingExporter(), ExportWorkerConfig())
        await worker.drain_once()
        # parked far in the future; the record itself is preserved
        assert store.pending_exports("phoenix") == []
        assert store.load("itx-perm") is not None
        row = _connection(store).execute(
            "SELECT state, last_error_type FROM interaction_exports "
            "WHERE interaction_id='itx-perm'"
        ).fetchone()
        assert row["state"] == "pending"
        assert row["last_error_type"] == "PermanentExportError"
        store.close()

    asyncio.run(scenario())
    assert (
        "therapy_llm_capture_exports_total",
        1,
        {"backend": "phoenix", "outcome": "rejected"},
    ) in metrics
    assert "payload rejected" not in repr(metrics)


def test_nonterminal_records_are_never_exported(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = JournalStore(tmp_path / "journal.sqlite3")
        store.start_attempt(_record("itx-open"))
        exporter = FlakyExporter(failures=0)
        worker = ExportWorker(store, exporter, ExportWorkerConfig())
        await worker.drain_once()
        assert exporter.acked == []  # started-but-open attempts stay local
        store.close()

    asyncio.run(scenario())
