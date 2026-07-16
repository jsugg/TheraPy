"""Interaction capture service (plan §5.3 failure policy, O1.3).

One `CaptureService` instance is installed at bootstrap. Every LLM boundary
opens an attempt BEFORE dispatching to the provider:

- the pre-dispatch canonical/native request is durably journaled first;
- stream/native events append in order; terminals commit immediately;
- if the pre-dispatch append fails, `runtime` mode raises
  `CaptureUnavailable` (a distinct result the boundary surfaces), and
  `evaluation` mode always stops; `disabled` skips capture entirely but is
  forbidden for acceptance routes (enforced at those routes).

The service also emits the content-free broad twin for each attempt
(provider, operation, sizes, tokens, finish class, duration, outcome) —
never prompts, bodies, or exception messages.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from therapy.observability.context import (
    current_trace_context,
    new_interaction_id,
)
from therapy.observability.interactions import (
    InteractionError,
    InteractionRecord,
    InteractionRequest,
    InteractionResponse,
    JsonValue,
    ProviderNative,
    StreamEvent,
)
from therapy.observability.journal import AsyncJournalWriter, JournalError
from therapy.observability.logging import emit_event
from therapy.observability.model import (
    CaptureMode,
    InteractionEventKind,
    InteractionOperation,
    InteractionStatus,
    Outcome,
    Provider,
)


class CaptureUnavailable(RuntimeError):
    """Pre-dispatch capture failed; the boundary decides how to surface it."""


def _utcnow_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


@dataclass
class AttemptHandle:
    """Journal-backed handle for one provider attempt."""

    service: CaptureService
    record: InteractionRecord
    _sequence: int = 0
    _started_monotonic: float = 0.0
    _first_token_monotonic: float | None = None
    _closed: bool = False

    @property
    def interaction_id(self) -> str:
        return self.record.interaction_id

    async def record_event(
        self, kind: InteractionEventKind, payload: dict[str, JsonValue]
    ) -> None:
        if self.service.writer is None or self._closed:
            return
        if (
            kind in (InteractionEventKind.STREAM_DELTA, InteractionEventKind.TOOL_DELTA)
            and self._first_token_monotonic is None
        ):
            self._first_token_monotonic = time.monotonic()
        sequence = self._sequence
        self._sequence += 1
        try:
            await self.service.writer.append_stream_event(
                self.record.interaction_id, sequence, kind, _utcnow_iso(), payload
            )
        except JournalError:
            raise
        except Exception as exc:
            raise JournalError(f"stream append failed: {type(exc).__name__}") from exc

    async def succeed(
        self,
        response: InteractionResponse,
        native_terminal: dict[str, JsonValue],
        *,
        finish_class: str = "stop",
    ) -> None:
        await self._finish(
            InteractionStatus.SUCCEEDED,
            {"response": _response_json(response), "provider_terminal": native_terminal},
            outcome=Outcome.SUCCESS,
            finish_class=finish_class,
            response=response,
        )

    async def fail(
        self,
        error: InteractionError,
        native_terminal: dict[str, JsonValue] | None = None,
        *,
        finish_class: str = "provider_error",
        outcome: Outcome = Outcome.ERROR,
    ) -> None:
        payload: dict[str, JsonValue] = {
            "error": {
                "http_status": error.http_status,
                "provider_type": error.provider_type,
                "provider_code": error.provider_code,
                "provider_error_body": error.provider_error_body,
                "retry_attempt": error.retry_attempt,
                "provider_request_id": error.provider_request_id,
            }
        }
        if native_terminal is not None:
            payload["provider_terminal"] = native_terminal
        await self._finish(
            InteractionStatus.FAILED,
            payload,
            outcome=outcome,
            finish_class=finish_class,
            error=error,
        )

    async def incomplete(self, reason: str) -> None:
        if self.service.writer is None or self._closed:
            return
        self._closed = True
        try:
            await self.service.writer.mark_incomplete(
                self.record.interaction_id, reason
            )
        finally:
            self.service.emit_twin(
                self.record,
                outcome=Outcome.INCOMPLETE,
                finish_class="incomplete",
                duration_ms=(time.monotonic() - self._started_monotonic) * 1000,
                ttft_ms=self._ttft_ms(),
            )

    async def _finish(
        self,
        status: InteractionStatus,
        terminal: dict[str, JsonValue],
        *,
        outcome: Outcome,
        finish_class: str,
        response: InteractionResponse | None = None,
        error: InteractionError | None = None,
    ) -> None:
        if self.service.writer is None or self._closed:
            return
        self._closed = True
        try:
            if status is InteractionStatus.SUCCEEDED:
                await self.service.writer.finish_success(
                    self.record.interaction_id, terminal
                )
            else:
                await self.service.writer.finish_error(
                    self.record.interaction_id, terminal
                )
        finally:
            usage = (response.usage if response else None) or {}
            self.service.emit_twin(
                self.record,
                outcome=outcome,
                finish_class=finish_class,
                duration_ms=(time.monotonic() - self._started_monotonic) * 1000,
                ttft_ms=self._ttft_ms(),
                output_chars=len(response.completion or "") if response else 0,
                usage=usage,
                retry_count=error.retry_attempt if error else 0,
                status_class=_status_class(error.http_status if error else 200),
            )

    def _ttft_ms(self) -> float | None:
        if self._first_token_monotonic is None:
            return None
        return (self._first_token_monotonic - self._started_monotonic) * 1000


def _response_json(response: InteractionResponse) -> dict[str, JsonValue]:
    from dataclasses import asdict

    return asdict(response)


def _status_class(status: int | None) -> str:
    if status is None:
        return "none"
    return f"{status // 100}xx"


class CaptureService:
    """Owns the journal writer and the capture-mode policy."""

    def __init__(
        self,
        writer: AsyncJournalWriter | None,
        *,
        mode: CaptureMode,
        build_version: str = "0.1.0",
        policy_version: str = "p1",
        config_version: str = "c1",
    ) -> None:
        self.writer = writer
        self.mode = mode
        self._versions = (build_version, policy_version, config_version)

    async def start_attempt(
        self,
        *,
        operation: InteractionOperation,
        provider: Provider,
        requested_model: str,
        request: InteractionRequest,
        provider_request: dict[str, JsonValue],
        prompt_template_version: str = "v1",
        language: str = "unknown",
        modality: str = "text",
        session_id: str | None = None,
        turn_id: int | None = None,
    ) -> AttemptHandle:
        """Durable pre-dispatch commit; must succeed before provider I/O."""
        context = current_trace_context()
        record = InteractionRecord(
            interaction_id=new_interaction_id(),
            trace_id=context.trace_id,
            span_id=context.span_id,
            operation=operation,
            provider=provider,
            requested_model=requested_model,
            actual_model="",
            prompt_template_version=prompt_template_version,
            request=request,
            response=InteractionResponse(),
            stream=(),
            error=None,
            provider_native=ProviderNative(request=provider_request),
            language=language,
            modality=modality,
            build_version=self._versions[0],
            policy_version=self._versions[1],
            config_version=self._versions[2],
            started_at=_utcnow_iso(),
            completed_at=None,
            status=InteractionStatus.STARTED,
            session_id=session_id,
            turn_id=turn_id,
        )
        handle = AttemptHandle(
            service=self, record=record, _started_monotonic=time.monotonic()
        )
        if self.mode is CaptureMode.EVALUATION and self.writer is None:
            raise CaptureUnavailable("evaluation capture requires a journal")
        if self.mode is CaptureMode.DISABLED:
            return handle
        if self.writer is None:
            # Documented availability exception (§5.3): the journal failed at
            # startup; product stays available but every attempt leaves a
            # visible, rate-limited gap instead of a silent hole.
            emit_event(
                "capture_degraded",
                severity=logging.ERROR,
                component="journal",
                operation="start_attempt",
                outcome="error",
                error_type="JournalUnavailable",
                rate_limited=True,
            )
            return handle
        try:
            await self.writer.start_attempt(record)
        except Exception as exc:
            emit_event(
                "capture_degraded",
                severity=logging.ERROR,
                component="journal",
                operation="start_attempt",
                outcome="error",
                error_type=type(exc).__name__,
                rate_limited=True,
            )
            # §5.3: evaluation stops immediately; runtime returns a distinct
            # capture-unavailable result — continuing is the boundary's
            # explicitly documented safety/availability exception.
            raise CaptureUnavailable(
                f"pre-dispatch journal append failed: {type(exc).__name__}"
            ) from exc
        return handle

    def emit_twin(
        self,
        record: InteractionRecord,
        *,
        outcome: Outcome,
        finish_class: str,
        duration_ms: float,
        ttft_ms: float | None = None,
        output_chars: int = 0,
        usage: dict[str, JsonValue] | None = None,
        retry_count: int = 0,
        status_class: str = "2xx",
    ) -> None:
        """Content-free broad twin (O1.3 item 4) + O2 instruments."""
        emit_event(
            "llm.attempt",
            severity=logging.INFO,
            component="llm",
            operation=record.operation.value,
            outcome=outcome.value,
            duration_ms=duration_ms,
            retry_count=retry_count,
        )
        from therapy.observability.telemetry import record_metric

        dims = {
            "provider": record.provider.value,
            "operation": record.operation.value,
            "outcome": outcome.value,
        }
        record_metric("therapy_llm_requests_total", 1, dims)
        record_metric(
            "therapy_llm_capture_records_total",
            1,
            {
                "operation": record.operation.value,
                "status": "journaled" if self.writer is not None else "failed",
            },
        )
        if ttft_ms is not None:
            record_metric(
                "therapy_llm_time_to_first_token_seconds", ttft_ms / 1000, dims
            )
        for key, instrument in (
            ("input_tokens", "therapy_llm_input_tokens_total"),
            ("prompt_tokens", "therapy_llm_input_tokens_total"),
            ("output_tokens", "therapy_llm_output_tokens_total"),
            ("completion_tokens", "therapy_llm_output_tokens_total"),
        ):
            value = (usage or {}).get(key)
            if isinstance(value, int | float) and value > 0:
                record_metric(
                    instrument,
                    value,
                    {"provider": dims["provider"], "operation": dims["operation"]},
                )
        if status_class == "4xx" and finish_class == "rate_limit":
            record_metric(
                "therapy_llm_rate_limits_total",
                1,
                {"provider": dims["provider"], "operation": dims["operation"]},
            )
        result = "ok" if output_chars else "empty"
        if outcome is Outcome.SUCCESS:
            record_metric(
                "therapy_llm_output_total",
                1,
                {
                    "provider": dims["provider"],
                    "operation": dims["operation"],
                    "result": result,
                },
            )


_service: CaptureService | None = None


def set_capture_service(service: CaptureService | None) -> None:
    global _service
    _service = service


def capture_service() -> CaptureService | None:
    return _service


async def _health_monitor(store, interval_s: float = 15.0) -> None:
    """Owned event-loop lag + capture-health gauges (plan O1.2 item 3, O2.3).

    Loop lag is measured as sleep drift — the authoritative signal (plan
    O2.1 item 5); journal gauges publish last-success timestamps, never
    precomputed ages (§8)."""
    import asyncio as _asyncio
    from datetime import datetime

    from therapy.observability.telemetry import record_metric

    while True:
        started = time.monotonic()
        try:
            await _asyncio.sleep(interval_s)
        except _asyncio.CancelledError:
            return
        lag = max(0.0, (time.monotonic() - started) - interval_s)
        record_metric("therapy_event_loop_lag_seconds", lag)
        if store is None:
            continue
        try:
            health = await _asyncio.to_thread(store.health)
        except Exception:
            continue
        if health.oldest_unexported_at:
            try:
                oldest = datetime.fromisoformat(health.oldest_unexported_at)
                record_metric(
                    "therapy_llm_capture_oldest_unexported_unixtime",
                    oldest.timestamp(),
                )
            except ValueError:
                pass


@dataclass
class CaptureRuntime:
    """Everything the app lifespan owns: journal, writer, service, worker."""

    store: object | None
    writer: AsyncJournalWriter | None
    service: CaptureService
    worker: object | None = None  # ExportWorker when a backend is selected
    monitor: object | None = None  # health/loop-lag task

    async def close(self, timeout: float = 5.0) -> None:
        """Bounded flush + close; never blocks product shutdown (O1.1)."""
        import asyncio as _asyncio
        import contextlib

        set_capture_service(None)
        if self.monitor is not None:
            self.monitor.cancel()  # type: ignore[attr-defined]
            with contextlib.suppress(Exception):
                await _asyncio.wait_for(self.monitor, 2.0)  # type: ignore[arg-type]
        if self.worker is not None:
            await self.worker.close(timeout)  # type: ignore[attr-defined]
        if self.writer is not None:
            try:
                await self.writer.flush()
            except Exception:
                pass
            await self.writer.close(timeout)

    def health(self):
        from therapy.observability.journal import JournalStore

        if isinstance(self.store, JournalStore):
            return self.store.health()
        return None


async def start_capture(config, *, build_version: str = "0.1.0") -> CaptureRuntime:
    """Open the journal, recover stale attempts, install the service.

    Journal failure degrades capture visibly (rate-limited `capture_degraded`
    per attempt) but never blocks product startup in runtime mode.
    """
    from therapy.observability.journal import JournalStore

    store: JournalStore | None = None
    writer: AsyncJournalWriter | None = None
    try:
        store = JournalStore(config.journal_path)
        recovered = store.recover()
        if recovered:
            emit_event(
                "capture_recovered_incomplete",
                severity=logging.WARNING,
                component="journal",
                operation="recover",
                outcome="success",
                count=recovered,
            )
        writer = AsyncJournalWriter(
            store,
            queue_size=config.queue_size,
            group_commit_ms=config.group_commit_ms,
        )
        await writer.start()
    except Exception as exc:
        emit_event(
            "capture_degraded",
            severity=logging.CRITICAL,
            component="journal",
            operation="open",
            outcome="error",
            error_type=type(exc).__name__,
        )
        store, writer = None, None
        if config.capture_mode is CaptureMode.EVALUATION:
            raise CaptureUnavailable("evaluation mode requires the journal") from exc

    service = CaptureService(
        writer,
        mode=config.capture_mode,
        build_version=build_version,
    )
    set_capture_service(service)

    # Selected-backend export (O1.2): asynchronous, ACK-gated, retryable.
    # A backend outage rolls back to journal-only, never capture-off.
    worker = None
    if store is not None and config.interaction_backend != "journal":
        from therapy.observability.exporters import ExportWorker
        from therapy.observability.telemetry import make_interaction_exporter

        exporter = make_interaction_exporter(config)
        if exporter is not None:
            worker = ExportWorker(store, exporter)
            await worker.start()

    monitor = None
    from therapy.observability.telemetry import state as telemetry_state

    if telemetry_state().enabled:
        import asyncio as _asyncio

        monitor = _asyncio.create_task(
            _health_monitor(store), name="observability-health"
        )

    return CaptureRuntime(
        store=store, writer=writer, service=service, worker=worker, monitor=monitor
    )


def stream_event_from_delta(delta: str) -> tuple[InteractionEventKind, dict[str, JsonValue]]:
    return InteractionEventKind.STREAM_DELTA, {"delta": delta}


def build_stream_tuple(events: list[dict[str, JsonValue]]) -> tuple[StreamEvent, ...]:
    return tuple(
        StreamEvent(
            sequence=index,
            observed_at=str(event.get("observed_at", "")),
            delta=event.get("delta"),  # type: ignore[arg-type]
            tool_delta=event.get("tool_delta"),  # type: ignore[arg-type]
        )
        for index, event in enumerate(events)
    )
