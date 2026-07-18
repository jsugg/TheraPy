"""Backend-neutral interaction export (plan §3, O1.2).

`InteractionExporter` is the only seam a backend adapter implements; no
product or domain module imports a backend SDK. The Phoenix OTLP adapter is
constructed by `telemetry.py` (the sole OTel importer) and injected here.

The worker drains the journal's replay cursor asynchronously: backend ACK is
recorded idempotently, only retryable failures retry (exponential backoff
plus jitter, bounded attempts per cycle), and records are kept until ACK.
Backend availability is never synchronous with the product path (§5.3).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

from therapy.observability.journal import JournalStore, LoadedInteraction
from therapy.observability.logging import emit_event


class ExportError(RuntimeError):
    """Base export failure."""

    retryable = True


class PermanentExportError(ExportError):
    """Rejected payloads and misconfiguration: retrying cannot help."""

    retryable = False


class InteractionExporter(Protocol):
    """One selected backend destination (plan: journal + ONE backend)."""

    @property
    def backend_name(self) -> str: ...

    async def export(self, interaction: LoadedInteraction) -> str | None:
        """Deliver one journaled interaction (`{"interaction":…,"events":…}`).

        Returns the backend record ID when available. Raises `ExportError`
        (retryable) or `PermanentExportError`. Success means the backend
        acknowledged receipt — HTTP 200 without acknowledgment semantics
        must not be reported as ACK by an adapter unless the spike evidence
        for that backend justified it.
        """
        ...


@dataclass(frozen=True, slots=True)
class ExportWorkerConfig:
    batch_limit: int = 16
    poll_interval_s: float = 5.0
    base_backoff_s: float = 2.0
    max_backoff_s: float = 300.0
    max_attempts_per_cycle: int = 3


class ExportWorker:
    """Single async task draining pending exports to the one backend."""

    def __init__(
        self,
        store: JournalStore,
        exporter: InteractionExporter,
        config: ExportWorkerConfig | None = None,
    ) -> None:
        self._store = store
        self._exporter = exporter
        self._config = config or ExportWorkerConfig()
        self._task: asyncio.Task[None] | None = None
        self._halt = asyncio.Event()
        self._degraded = False

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="interaction-export")

    async def _run(self) -> None:
        while not self._halt.is_set():
            try:
                worked = await self.drain_once()
            except Exception as exc:  # never let the worker die silently
                self._note_degraded(type(exc).__name__)
                worked = 0
            if worked == 0:
                try:
                    await asyncio.wait_for(
                        self._halt.wait(), self._config.poll_interval_s
                    )
                except TimeoutError:
                    pass

    async def drain_once(self) -> int:
        """One bounded cycle; returns how many records were attempted."""
        from therapy.observability.model import WorkloadClass
        from therapy.observability.telemetry import run_in_thread

        backend = self._exporter.backend_name
        backend_label = "phoenix" if backend == "phoenix" else "unknown"
        pending = await run_in_thread(
            WorkloadClass.BACKGROUND,
            self._store.pending_exports,
            backend,
            self._config.batch_limit,
        )
        attempted = 0
        for interaction_id in pending:
            if attempted >= self._config.max_attempts_per_cycle * self._config.batch_limit:
                break
            payload = await run_in_thread(
                WorkloadClass.BACKGROUND, self._store.load, interaction_id
            )
            if payload is None:
                continue
            operation = self._operation(payload)
            attempted += 1
            export_started = time.monotonic()
            outcome = "error"
            export_outcome = "failure"
            try:
                backend_record_id = await self._exporter.export(payload)
            except PermanentExportError as exc:
                export_outcome = "rejected"
                await run_in_thread(
                    WorkloadClass.BACKGROUND,
                    self._store.record_export_attempt,
                    interaction_id,
                    backend,
                    acknowledged=False,
                    error_type=type(exc).__name__,
                    next_attempt_at=self._backoff_at(attempts=8),  # park far out
                )
                self._note_degraded(type(exc).__name__)
            except Exception as exc:
                attempts = await run_in_thread(
                    WorkloadClass.BACKGROUND,
                    self._attempts_so_far,
                    interaction_id,
                    backend,
                )
                await run_in_thread(
                    WorkloadClass.BACKGROUND,
                    self._store.record_export_attempt,
                    interaction_id,
                    backend,
                    acknowledged=False,
                    error_type=type(exc).__name__,
                    next_attempt_at=self._backoff_at(attempts=attempts + 1),
                )
                self._note_degraded(type(exc).__name__)
            else:
                await run_in_thread(
                    WorkloadClass.BACKGROUND,
                    self._store.record_export_attempt,
                    interaction_id,
                    backend,
                    acknowledged=True,
                    backend_record_id=backend_record_id,
                )
                outcome = "success"
                export_outcome = "accepted"
                self._observe_export_success(operation)
                self._note_recovered()
            finally:
                from therapy.observability.telemetry import record_metric

                record_metric(
                    "therapy_llm_capture_export_seconds",
                    time.monotonic() - export_started,
                    {"outcome": outcome},
                )
                record_metric(
                    "therapy_llm_capture_exports_total",
                    1,
                    {"backend": backend_label, "outcome": export_outcome},
                )
                if outcome == "error":
                    record_metric(
                        "therapy_llm_capture_records_total",
                        1,
                        {"operation": operation, "status": "failed"},
                    )
        return attempted

    @staticmethod
    def _operation(payload: Mapping[str, object]) -> str:
        """Return the finite journal operation without exposing payload data."""
        from therapy.observability.model import InteractionOperation

        interaction_value = payload.get("interaction")
        interaction: Mapping[str, object] | None = None
        if isinstance(interaction_value, Mapping):
            raw_interaction = cast(Mapping[object, object], interaction_value)
            if all(isinstance(key, str) for key in raw_interaction):
                interaction = cast(Mapping[str, object], interaction_value)
        raw = interaction.get("operation") if interaction is not None else None
        try:
            return InteractionOperation(raw).value if isinstance(raw, str) else "unknown"
        except ValueError:
            return "unknown"

    @staticmethod
    def _observe_export_success(operation: str) -> None:
        import time as _time

        from therapy.observability.telemetry import record_metric

        record_metric(
            "therapy_llm_capture_last_export_success_unixtime", _time.time()
        )
        record_metric(
            "therapy_llm_capture_records_total",
            1,
            {"operation": operation, "status": "exported"},
        )

    def _attempts_so_far(self, interaction_id: str, backend: str) -> int:
        return self._store.export_attempts(interaction_id, backend)

    def _backoff_at(self, attempts: int) -> str:
        delay = min(
            self._config.max_backoff_s,
            self._config.base_backoff_s * (2 ** max(0, attempts - 1)),
        )
        delay *= 0.5 + random.random()  # jitter in [0.5x, 1.5x)
        return (datetime.now(UTC) + timedelta(seconds=delay)).isoformat()

    def _note_degraded(self, error_type: str) -> None:
        if not self._degraded:
            self._degraded = True
            emit_event(
                "capture_degraded",
                severity=logging.WARNING,
                component="telemetry",
                operation="export",
                outcome="error",
                error_type=error_type,
                rate_limited=True,
            )

    def _note_recovered(self) -> None:
        if self._degraded:
            self._degraded = False
            emit_event(
                "capture_recovered",
                component="telemetry",
                operation="export",
                outcome="success",
                rate_limited=True,
            )

    async def close(self, timeout: float = 5.0) -> None:
        self._halt.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout)
            except TimeoutError:
                self._task.cancel()
