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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from therapy.observability.journal import JournalStore
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

    async def export(self, interaction: dict) -> str | None:
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
        backend = self._exporter.backend_name
        pending = await asyncio.to_thread(
            self._store.pending_exports, backend, self._config.batch_limit
        )
        attempted = 0
        for interaction_id in pending:
            if attempted >= self._config.max_attempts_per_cycle * self._config.batch_limit:
                break
            payload = await asyncio.to_thread(self._store.load, interaction_id)
            if payload is None:
                continue
            attempted += 1
            try:
                backend_record_id = await self._exporter.export(payload)
            except PermanentExportError as exc:
                await asyncio.to_thread(
                    self._store.record_export_attempt,
                    interaction_id,
                    backend,
                    acknowledged=False,
                    error_type=type(exc).__name__,
                    next_attempt_at=self._backoff_at(attempts=8),  # park far out
                )
                self._note_degraded(type(exc).__name__)
            except Exception as exc:
                attempts = await asyncio.to_thread(
                    self._attempts_so_far, interaction_id, backend
                )
                await asyncio.to_thread(
                    self._store.record_export_attempt,
                    interaction_id,
                    backend,
                    acknowledged=False,
                    error_type=type(exc).__name__,
                    next_attempt_at=self._backoff_at(attempts=attempts + 1),
                )
                self._note_degraded(type(exc).__name__)
            else:
                await asyncio.to_thread(
                    self._store.record_export_attempt,
                    interaction_id,
                    backend,
                    acknowledged=True,
                    backend_record_id=backend_record_id,
                )
                self._observe_export_success()
                self._note_recovered()
        return attempted

    @staticmethod
    def _observe_export_success() -> None:
        import time as _time

        from therapy.observability.telemetry import record_metric

        record_metric(
            "therapy_llm_capture_last_export_success_unixtime", _time.time()
        )
        record_metric(
            "therapy_llm_capture_records_total",
            1,
            {"operation": "reply", "status": "exported"},
        )

    def _attempts_so_far(self, interaction_id: str, backend: str) -> int:
        row = self._store._conn.execute(
            "SELECT attempts FROM interaction_exports "
            "WHERE interaction_id=? AND backend=?",
            (interaction_id, backend),
        ).fetchone()
        return int(row["attempts"]) if row is not None else 0

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
