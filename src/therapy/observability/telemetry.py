"""Owned OTel bootstrap: the ONLY module importing the OTel SDK (plan §3, §5.5).

Safe no-op when `THERAPY_OTEL_ENABLED=0` or the SDK isn't installed —
journal correlation uses locally generated W3C IDs (`context.py`) and never
depends on an exporter. Pipecat's `setup_tracing()` is never called; the
provider installed here is global before the app imports, so Pipecat obtains
tracers from it (plan §2).

Routing is default-deny: a span whose instrumentation scope is not
explicitly audited is DROPPED and counted, never fanned out to both planes.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from therapy.observability.config import ObservabilityConfig
from therapy.observability.logging import emit_event
from therapy.observability.model import TelemetryPlane, WorkloadClass
from therapy.observability.routing import classify_scope, scrub_broad_attributes

if TYPE_CHECKING:
    import httpx
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import SpanProcessor
    from opentelemetry.trace import StatusCode

    from therapy.observability.journal import LoadedInteraction


class SpanLike(Protocol):
    """The small span surface exposed to workflow modules."""

    def set_attribute(self, key: str, value: str | bool | int | float) -> None:
        """Set one already-scrubbed finite attribute."""
        ...


@runtime_checkable
class _SpanProcessorLike(Protocol):
    """Runtime-validated processor surface used by the optional OTel SDK."""

    def on_start(self, span: object, parent_context: object | None = None) -> None: ...

    def on_end(self, span: object) -> None: ...

    def shutdown(self) -> None: ...

    def force_flush(self, timeout_millis: int = 30000) -> bool: ...


@runtime_checkable
class _InstrumentationScopeLike(Protocol):
    """Minimal instrumentation-scope surface used for default-deny routing."""

    name: str


@runtime_checkable
class _ReadableSpanLike(Protocol):
    """Runtime span envelope required before broad-plane scrubbing."""

    instrumentation_scope: object
    attributes: Mapping[str, object] | None
    status: object


@runtime_checkable
class _StatusLike(Protocol):
    """Status fields retained while dropping free-form descriptions."""

    status_code: object
    description: str | None


def _processor(value: object | None) -> _SpanProcessorLike | None:
    """Validate an optional SDK processor at the integration boundary."""
    if value is None:
        return None
    if not isinstance(value, _SpanProcessorLike):
        raise TypeError("OTel processor does not satisfy the required contract")
    return value


@dataclass
class TelemetryState:
    enabled: bool = False
    provider: object | None = None
    meter_provider: object | None = None
    instruments: dict[str, object] = field(default_factory=lambda: {})
    dropped_unknown_scopes: int = 0
    dropped_forbidden_keys: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


_state = TelemetryState()
_executor_lock = threading.Lock()
_executor_active: dict[WorkloadClass, int] = {}


def state() -> TelemetryState:
    return _state


def active_span_ids() -> tuple[str, str] | None:
    """The active OTel span's (trace_id, span_id) hex pair, if recording."""
    if not _state.enabled:
        return None
    try:
        from opentelemetry import trace

        span_context = trace.get_current_span().get_span_context()
        if not span_context.is_valid:
            return None
        return (
            f"{span_context.trace_id:032x}",
            f"{span_context.span_id:016x}",
        )
    except Exception:
        return None


def _service_instance_id() -> str:
    from therapy.observability.logging import SERVICE_INSTANCE_ID

    return SERVICE_INSTANCE_ID


def initialize(config: ObservabilityConfig, *, service_version: str) -> bool:
    """Install the one owned global TracerProvider. Returns enabled state."""
    if not config.otel_enabled:
        return False
    import os

    # Stable HTTP semconv: metrics/spans use `http.route` templates instead
    # of legacy `http.target` concrete paths (plan O2.1 item 2). Pinned by
    # the routing tests; re-audited on every instrumentation upgrade.
    os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "http")
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        emit_event(
            "telemetry.sdk_unavailable",
            severity=logging.WARNING,
            component="telemetry",
            operation="bootstrap",
            outcome="rejected",
        )
        return False

    import platform

    # Fixed resource ONLY (audit C-02): the direct constructor skips the
    # OTEL_RESOURCE_ATTRIBUTES env detector, so arbitrary env-injected
    # key/values can never ride the broad resource. Belt and braces: the
    # variable is also cleared for any later SDK path that calls create().
    os.environ.pop("OTEL_RESOURCE_ATTRIBUTES", None)
    resource = Resource(
        {
            "service.name": "therapy",
            "service.version": service_version,
            "service.instance.id": _service_instance_id(),
            "deployment.environment": config.environment,
            "process.runtime.version": platform.python_version(),
        }
    )
    provider = TracerProvider(resource=resource)

    broad_processor = None
    if config.otlp_broad_endpoint:
        broad_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{config.otlp_broad_endpoint}/v1/traces",
                timeout=config.otel_export_timeout_secs,
            ),
            max_queue_size=2048,
            max_export_batch_size=256,
        )
    restricted_processor = None
    if config.otlp_restricted_endpoint:
        restricted_processor = BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{config.otlp_restricted_endpoint}/v1/traces",
                timeout=config.otel_export_timeout_secs,
            ),
            max_queue_size=2048,
            max_export_batch_size=256,
        )

    routing = PlaneRoutingSpanProcessor(
        broad=broad_processor, restricted=restricted_processor
    )
    provider.add_span_processor(cast("SpanProcessor", routing))
    trace.set_tracer_provider(provider)
    _state.enabled = True
    _state.provider = provider
    _initialize_metrics(config, resource)
    _instrument_fastapi()
    _instrument_system_metrics()
    return True


def _initialize_metrics(config: ObservabilityConfig, resource: Resource) -> None:
    """Meter provider + every §8 instrument from the frozen manifest."""
    if not config.otlp_broad_endpoint:
        return
    try:
        from opentelemetry import metrics as metrics_api
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    except ImportError:
        return
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            endpoint=f"{config.otlp_broad_endpoint}/v1/metrics",
            timeout=config.otel_export_timeout_secs,
        ),
        export_interval_millis=15_000,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics_api.set_meter_provider(meter_provider)
    meter = meter_provider.get_meter("therapy.broad")

    from therapy.observability.metrics import INSTRUMENTS, InstrumentKind

    built: dict[str, object] = {}
    for spec in INSTRUMENTS:
        if spec.kind is InstrumentKind.COUNTER:
            built[spec.name] = meter.create_counter(spec.name, unit=spec.unit)
        elif spec.kind is InstrumentKind.HISTOGRAM:
            built[spec.name] = meter.create_histogram(spec.name, unit=spec.unit)
        else:
            built[spec.name] = meter.create_gauge(spec.name, unit=spec.unit)
    _state.meter_provider = meter_provider
    _state.instruments = built


def record_metric(name: str, value: float, attributes: dict[str, str] | None = None) -> None:
    """Record on a manifest instrument with bounded-attribute enforcement.

    Unknown instrument names and undeclared attribute keys are dropped and
    counted — arbitrary values never become labels (plan §8).
    """
    instrument = _state.instruments.get(name)
    if instrument is None:
        return
    from therapy.observability.metrics import INSTRUMENT_INDEX

    spec = INSTRUMENT_INDEX.get(name)
    if spec is None:
        return
    supplied = attributes or {}
    clean: dict[str, str] = {}
    undeclared_count = len(supplied.keys() - spec.attributes.keys())
    if undeclared_count:
        with _state.lock:
            _state.dropped_forbidden_keys += undeclared_count
    for key, allowed in spec.attributes.items():
        raw = supplied.get(key)
        if raw is None:
            with _state.lock:
                _state.dropped_forbidden_keys += 1
            return
        text = str(raw)
        if text in allowed:
            clean[key] = text
        elif "unknown" in allowed:
            clean[key] = "unknown"
        else:
            with _state.lock:
                _state.dropped_forbidden_keys += 1
            return
    add = getattr(instrument, "add", None)
    if callable(add):
        add(value, clean)
        return
    record = getattr(instrument, "record", None)
    if callable(record):
        record(value, clean)
        return
    set_value = getattr(instrument, "set", None)
    if callable(set_value):
        set_value(value, clean)


async def run_in_thread[**P, R](
    workload_class: WorkloadClass,
    func: Callable[P, R],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Run blocking work with bounded executor queue/active/completion evidence."""
    import asyncio

    queued_at = time.monotonic()

    def invoke() -> R:
        started_at = time.monotonic()
        dimensions = {"workload_class": workload_class.value}
        record_metric(
            "therapy_executor_queue_wait_seconds", started_at - queued_at, dimensions
        )
        with _executor_lock:
            active = _executor_active.get(workload_class, 0) + 1
            _executor_active[workload_class] = active
        record_metric("therapy_executor_active", active, dimensions)
        outcome = "success"
        try:
            return func(*args, **kwargs)
        except BaseException:
            outcome = "error"
            raise
        finally:
            elapsed = time.monotonic() - started_at
            with _executor_lock:
                remaining = max(0, _executor_active.get(workload_class, 1) - 1)
                _executor_active[workload_class] = remaining
            record_metric("therapy_executor_active", remaining, dimensions)
            record_metric(
                "therapy_executor_completed_total",
                1,
                {**dimensions, "outcome": outcome},
            )
            record_metric(
                "therapy_executor_task_seconds",
                elapsed,
                {**dimensions, "outcome": outcome},
            )

    return await asyncio.to_thread(invoke)


def _instrument_fastapi() -> None:
    """Explicit FastAPI/ASGI instrumentation (plan O2.1 item 2).

    Route templates only; liveness/readiness/static/metrics excluded; the
    plane router's denylist scrub removes any query/header/address
    attribute the instrumentation might still attach.
    """
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        from therapy.server.app import app
    except ImportError:
        return

    def _server_request_hook(span: SpanLike, scope: dict[str, object]) -> None:
        # strip concrete-target attributes at the source; the routing scrub
        # is the second, default-deny line of defense
        del scope
        for key in ("url.query", "url.full", "http.target", "client.address",
                    "client.port", "user_agent.original"):
            try:
                span.set_attribute(key, "")
            except Exception:
                pass

    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="health,ready,metrics,static/.*,favicon.ico,^/$",
        server_request_hook=_server_request_hook,
        exclude_spans=["receive", "send"],
    )


def _instrument_system_metrics() -> None:
    """Process-only allowlist (plan O2.1 item 5); broad host set disabled."""
    try:
        from opentelemetry.instrumentation.system_metrics import (
            SystemMetricsInstrumentor,
        )
    except ImportError:
        return
    # exactly the plan's allowlist: CPU time/utilization, RSS/virtual,
    # open FDs, threads, GC — no host/disk/network/context-switch set,
    # no legacy `process.runtime.*` duplicate names
    configuration = {
        "process.cpu.time": ["user", "system"],
        "process.cpu.utilization": ["user", "system"],
        "process.memory.usage": None,
        "process.memory.virtual": None,
        "process.open_file_descriptor.count": None,
        "process.thread.count": None,
        "process.runtime.gc_count": None,
    }
    try:
        SystemMetricsInstrumentor(config=configuration).instrument()
    except Exception:
        emit_event(
            "telemetry.system_metrics_unavailable",
            severity=logging.WARNING,
            component="telemetry",
            operation="bootstrap",
            outcome="rejected",
            rate_limited=True,
        )


def instrumented_async_client(
    destination: str, *, timeout: httpx.Timeout | float = 5.0
) -> httpx.AsyncClient:
    """An httpx.AsyncClient for ONE audited outbound destination.

    Never global instrumentation (plan O2.1 item 3): each owned wrapper
    declares its finite destination; spans carry the destination enum and
    the scrubbed standard HTTP client attributes only.
    """
    import httpx

    from therapy.observability.model import Destination, normalize_enum

    label = normalize_enum(destination, Destination, Destination.UNKNOWN)
    if label is Destination.UNKNOWN:
        emit_event(
            "telemetry.unknown_destination",
            severity=logging.WARNING,
            component="telemetry",
            operation="outbound",
            outcome="rejected",
            rate_limited=True,
        )
    client = httpx.AsyncClient(timeout=timeout)

    async def _owned_request(request: httpx.Request) -> None:
        request.extensions["therapy.started_at"] = time.monotonic()

    async def _owned_response(response: httpx.Response) -> None:
        started = response.request.extensions.pop("therapy.started_at", None)
        elapsed = (
            time.monotonic() - started if isinstance(started, float) else 0.0
        )
        status_class = (
            f"{response.status_code // 100}xx"
            if 200 <= response.status_code < 600
            else "none"
        )
        outcome = (
            "success" if 200 <= response.status_code < 400 else "error"
        )
        _record_outbound_result(
            destination=label.value,
            operation=response.request.method.casefold(),
            status_class=status_class,
            tls=response.request.url.scheme.casefold() == "https",
            outcome=outcome,
            elapsed=elapsed,
            request_bytes=_content_length(response.request.headers),
            response_bytes=_content_length(response.headers),
        )

    client.event_hooks["request"].append(_owned_request)
    client.event_hooks["response"].append(_owned_response)
    if not _state.enabled:
        return client
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        def _request_hook(span: SpanLike, request: object) -> None:
            del request
            span.set_attribute("therapy.destination", label.value)
            for key in ("url.full", "url.query", "http.url"):
                try:
                    span.set_attribute(key, "")
                except Exception:
                    pass

        HTTPXClientInstrumentor.instrument_client(
            client, request_hook=_request_hook
        )
    except ImportError:
        pass
    return client


def _content_length(headers: object) -> int | None:
    """Return one bounded HTTP content length without retaining headers."""
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw = getter("content-length")
    if not isinstance(raw, str | int):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if 0 <= value <= 1_000_000_000_000 else None


def _record_outbound_result(
    *,
    destination: str,
    operation: str,
    status_class: str,
    tls: bool,
    outcome: str,
    elapsed: float,
    request_bytes: int | None = None,
    response_bytes: int | None = None,
) -> None:
    """Record one owned outbound request using only finite content-free fields."""
    finite_operation = (
        operation
        if operation in {"get", "post", "put", "patch", "delete", "head", "options"}
        else "other"
    )
    dimensions = {
        "destination": destination,
        "operation": finite_operation,
        "outcome": outcome,
    }
    record_metric(
        "therapy_outbound_requests_total",
        1,
        {
            **dimensions,
            "status_class": status_class,
            "tls": str(tls).lower(),
        },
    )
    record_metric("therapy_outbound_request_seconds", elapsed, dimensions)
    for direction, byte_count in (
        ("request", request_bytes),
        ("response", response_bytes),
    ):
        if byte_count is not None:
            record_metric(
                "therapy_outbound_bytes",
                byte_count,
                {"destination": destination, "direction": direction},
            )


def record_outbound_retry_count(destination: str, count: int) -> None:
    """Record retries for one logical owned outbound request."""
    from therapy.observability.model import Destination, normalize_enum

    label = normalize_enum(destination, Destination, Destination.UNKNOWN)
    record_metric(
        "therapy_outbound_retry_count",
        max(0, count),
        {"destination": label.value},
    )


def record_outbound_failure(
    destination: str,
    operation: str,
    *,
    tls: bool,
    started_at: float,
    timed_out: bool,
) -> None:
    """Record a transport failure that has no HTTP response hook."""
    from therapy.observability.model import Destination, normalize_enum

    label = normalize_enum(destination, Destination, Destination.UNKNOWN)
    _record_outbound_result(
        destination=label.value,
        operation=operation,
        status_class="none",
        tls=tls,
        outcome="timeout" if timed_out else "error",
        elapsed=max(0.0, time.monotonic() - started_at),
    )


def storage_operation(component: str, operation: str):
    """Finite db.operation instrumentation for owned stores (plan O3.3).

    Records operation count/duration and SQLite busy events; never SQL, DB
    paths, or row content. Safe no-op when telemetry is off.
    """
    import sqlite3 as _sqlite3
    import time as _time
    from contextlib import contextmanager

    @contextmanager
    def _instrumented():
        started = _time.monotonic()
        outcome = "success"
        try:
            with broad_span(
                "db.operation", component=component, operation=operation
            ):
                yield
        except _sqlite3.OperationalError as exc:
            outcome = "error"
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                record_metric(
                    "therapy_sqlite_busy_total", 1, {"component": component}
                )
            raise
        except Exception:
            outcome = "error"
            raise
        finally:
            dims = {
                "component": component,
                "operation": operation,
                "outcome": outcome,
            }
            record_metric("therapy_storage_operations_total", 1, dims)
            record_metric(
                "therapy_storage_operation_seconds",
                _time.monotonic() - started,
                dims,
            )

    return _instrumented()


def record_storage_result(component: str, operation: str, result: object) -> None:
    """Record only a result-size bucket for one successful storage operation."""
    from collections.abc import Sized

    from therapy.observability.model import count_bucket

    if result is None:
        count = 0
    elif isinstance(result, bool):
        count = int(result)
    elif isinstance(result, Sized) and not isinstance(result, (str, bytes, bytearray)):
        count = len(result)
    else:
        count = 1
    record_metric(
        "therapy_storage_result_rows_total",
        1,
        {
            "component": component,
            "operation": operation,
            "bucket": count_bucket(count),
        },
    )


def broad_span(
    name: str, *, component: str, operation: str
) -> AbstractContextManager[SpanLike | None]:
    """Owned broad span helper: records status once, scope `therapy.broad`."""
    @contextmanager
    def _noop() -> Generator[SpanLike | None, None, None]:
        yield None

    if not _state.enabled:
        return _noop()
    from opentelemetry import trace

    tracer = trace.get_tracer("therapy.broad")
    span_cm = tracer.start_as_current_span(
        name,
        attributes={"component": component, "operation": operation},
    )
    return span_cm


def link_root(
    name: str,
    *,
    component: str,
    operation: str,
    parent_trace_id: str,
    parent_span_id: str,
) -> AbstractContextManager[SpanLike | None]:
    """A NEW trace root LINKED to its trigger (detached finalizers/batches);
    never a multi-hour parent span (plan O2.1 item 4, O2.2)."""
    @contextmanager
    def _noop() -> Generator[SpanLike | None, None, None]:
        yield None

    if not _state.enabled:
        return _noop()
    from opentelemetry import context as context_api
    from opentelemetry import trace
    from opentelemetry.trace import Link, SpanContext, TraceFlags

    tracer = trace.get_tracer("therapy.broad")
    link = Link(
        SpanContext(
            trace_id=int(parent_trace_id, 16),
            span_id=int(parent_span_id, 16),
            is_remote=False,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    )
    fresh = context_api.Context()  # detach from any active span
    return tracer.start_as_current_span(
        name,
        context=fresh,
        links=[link],
        attributes={"component": component, "operation": operation},
    )


def shutdown(timeout_millis: int = 5000) -> None:
    """Bounded flush; product shutdown never waits indefinitely (O1.1).

    Provider shutdowns run in a worker thread joined with the given budget —
    a wedged exporter can leak a daemon thread but can never block exit."""
    import concurrent.futures

    owners = [o for o in (_state.provider, _state.meter_provider) if o is not None]
    _state.enabled = False
    _state.provider = None
    _state.meter_provider = None
    _state.instruments = {}
    if not owners:
        return

    def _shutdown_all() -> None:
        for owner in owners:
            try:
                getattr(owner, "shutdown", lambda: None)()
            except Exception:
                pass

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_shutdown_all)
    try:
        future.result(timeout=timeout_millis / 1000)
    except concurrent.futures.TimeoutError:
        pass
    finally:
        executor.shutdown(wait=False)


class PlaneRoutingSpanProcessor:
    """Default-deny scope router (§5.5).

    - audited `pipecat.*` / `therapy.interactions` scopes -> restricted;
    - allowlisted broad scopes -> broad, after a denylist scrub;
    - unknown scopes -> dropped and counted (rate-limited diagnostic).

    Implements the OTel SpanProcessor duck interface so this module stays
    importable without the SDK.
    """

    def __init__(self, broad: object | None, restricted: object | None) -> None:
        self._broad = _processor(broad)
        self._restricted = _processor(restricted)

    def on_start(self, span: object, parent_context: object | None = None) -> None:
        for processor in (self._broad, self._restricted):
            if processor is not None:
                processor.on_start(span, parent_context)

    def on_ending(self, span: object) -> None:
        """SDK >=1.4x pre-end hook; routing happens in on_end."""
        del span

    # the SDK's synchronous multiplexer calls the private name directly
    def _on_ending(self, span: object) -> None:
        self.on_ending(span)

    def on_end(self, span: object) -> None:
        readable = span if isinstance(span, _ReadableSpanLike) else None
        scope = readable.instrumentation_scope if readable is not None else None
        scope_name = scope.name if isinstance(scope, _InstrumentationScopeLike) else None
        plane = classify_scope(scope_name)
        if plane is TelemetryPlane.RESTRICTED:
            if self._restricted is not None:
                self._restricted.on_end(span)
            return
        if plane is TelemetryPlane.BROAD:
            if self._broad is not None:
                if readable is None:
                    raise TypeError("broad OTel span does not satisfy the read contract")
                attributes = dict(readable.attributes or {})
                result = scrub_broad_attributes(attributes)
                if result.dropped_keys:
                    with _state.lock:
                        _state.dropped_forbidden_keys += len(result.dropped_keys)
                    record_metric(
                        "therapy_broad_span_drops_total",
                        len(result.dropped_keys),
                        {"reason": "forbidden_attribute"},
                    )
                    emit_event(
                        "telemetry.broad_attribute_dropped",
                        severity=logging.WARNING,
                        component="telemetry",
                        operation="routing",
                        outcome="rejected",
                        count=len(result.dropped_keys),
                        rate_limited=True,
                    )
                # EVERY broad span goes out as a strict envelope (audit
                # C-01): denylisted attributes gone, span events (exception
                # message/stack carriers) dropped, status description
                # stripped to its code.
                self._broad.on_end(_ScrubbedSpanView(readable, result.attributes))
            return
        with _state.lock:
            _state.dropped_unknown_scopes += 1
        record_metric(
            "therapy_broad_span_drops_total", 1, {"reason": "unknown_scope"}
        )
        emit_event(
            "telemetry.unknown_scope_dropped",
            severity=logging.WARNING,
            component="telemetry",
            operation="routing",
            outcome="rejected",
            rate_limited=True,
        )

    def shutdown(self) -> None:
        for processor in (self._broad, self._restricted):
            if processor is not None:
                processor.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        ok = True
        for processor in (self._broad, self._restricted):
            if processor is not None:
                ok = processor.force_flush(timeout_millis) and ok
        return ok


class PhoenixInteractionExporter:
    """O0-ADR-selected backend adapter (Phoenix 18.0.0 over OTLP HTTP).

    Maps one journaled interaction to one OpenInference LLM span carrying the
    lossless canonical envelope, preserving the journaled trace/span IDs. ACK
    is the OTLP exporter's SUCCESS result — Phoenix returns HTTP success even
    for ignored duplicates (spike evidence), which is exactly the idempotent
    replay semantic the journal needs.
    """

    backend_name = "phoenix"

    def __init__(self, endpoint: str, timeout: float = 3.0) -> None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.id_generator import IdGenerator

        class _NextIdGenerator(IdGenerator):
            """Serial exports set the next (trace, span) pair explicitly."""

            def __init__(self) -> None:
                self.next_trace_id = 0
                self.next_span_id = 0

            def generate_trace_id(self) -> int:
                return self.next_trace_id

            def generate_span_id(self) -> int:
                return self.next_span_id

        self._id_generator = _NextIdGenerator()
        self._provider = TracerProvider(
            # direct constructor: no env resource detector (audit C-02)
            resource=Resource({"service.name": "therapy-interactions"}),
            id_generator=self._id_generator,
        )
        self._tracer = self._provider.get_tracer("therapy.interactions")
        self._exporter = OTLPSpanExporter(
            endpoint=f"{endpoint}/v1/traces", timeout=timeout
        )

    async def export(self, interaction: LoadedInteraction) -> str | None:
        return await run_in_thread(
            WorkloadClass.BACKGROUND, self._export_sync, interaction
        )

    def _export_sync(self, payload: LoadedInteraction) -> str:
        import json as _json
        from datetime import datetime

        from opentelemetry.sdk.trace import ReadableSpan
        from opentelemetry.sdk.trace.export import SpanExportResult
        from opentelemetry.trace import SpanKind

        from therapy.observability.exporters import ExportError

        row = payload["interaction"]
        events = payload["events"]

        def _ns(iso: str | None) -> int:
            if not iso:
                return 0
            return int(datetime.fromisoformat(iso).timestamp() * 1_000_000_000)

        self._id_generator.next_trace_id = int(row["trace_id"], 16)
        self._id_generator.next_span_id = int(row["span_id"], 16)

        envelope: dict[str, object] = {
            "interaction": row,
            "events": events,
        }
        attributes = {
            "openinference.span.kind": "LLM",
            "llm.provider": row["provider"],
            "therapy.operation": row["operation"],
            "therapy.status": row["status"],
            "session.id": row["interaction_id"],
            "input.value": row["canonical_request_json"],
            "input.mime_type": "application/json",
            "output.value": row["terminal_json"] or "",
            "output.mime_type": "application/json",
            "therapy.canonical_record": _json.dumps(
                envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        }
        span = self._tracer.start_span(
            name=f"llm.{row['operation']}",
            kind=SpanKind.CLIENT,
            attributes=attributes,
            start_time=_ns(row["started_at"]) or None,
        )
        span.end(_ns(row["completed_at"] or row["updated_at"]) or None)
        if not isinstance(span, ReadableSpan):
            raise ExportError("owned tracer did not return an SDK readable span")
        result = self._exporter.export([span])
        if result is not SpanExportResult.SUCCESS:
            raise ExportError("phoenix OTLP export failed")
        return row["span_id"]


_LOCAL_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "::1", "phoenix", "collector", "lgtm"}
)


def make_interaction_exporter(
    config: ObservabilityConfig,
) -> PhoenixInteractionExporter | None:
    """The single selected backend adapter, or None for journal-only."""
    if config.interaction_backend != "phoenix":
        return None
    if not config.otlp_restricted_endpoint:
        return None
    # Owner remote-export gate (§4): any destination beyond loopback/compose-
    # internal names requires THERAPY_INTERACTION_REMOTE_EXPORT=1 explicitly.
    from urllib.parse import urlsplit

    host = urlsplit(config.otlp_restricted_endpoint).hostname or ""
    if host not in _LOCAL_HOSTS and not config.remote_export_enabled:
        emit_event(
            "telemetry.remote_export_gated",
            severity=logging.WARNING,
            component="telemetry",
            operation="backend_adapter",
            outcome="rejected",
        )
        return None
    try:
        return PhoenixInteractionExporter(
            config.otlp_restricted_endpoint,
            timeout=config.otel_export_timeout_secs,
        )
    except ImportError:
        emit_event(
            "telemetry.sdk_unavailable",
            severity=logging.WARNING,
            component="telemetry",
            operation="backend_adapter",
            outcome="rejected",
        )
        return None


class _ScrubbedSpanView:
    """Strict broad-span envelope (audit C-01).

    Presents scrubbed attributes, NO span events (FastAPI records exception
    message/stacktrace as events), and a status stripped of its description
    (FastAPI sets it to `ExcType: message`). Everything else proxies.
    """

    __slots__ = ("_span", "_attributes", "_status")

    def __init__(
        self, span: _ReadableSpanLike, attributes: dict[str, object]
    ) -> None:
        self._span = span
        self._attributes = attributes
        status = span.status
        self._status = status
        if isinstance(status, _StatusLike) and status.description:
            try:
                from opentelemetry.trace.status import Status

                self._status = Status(
                    status_code=cast("StatusCode", status.status_code)
                )
            except Exception:
                pass

    @property
    def attributes(self) -> dict[str, object]:
        return self._attributes

    @property
    def events(self) -> tuple[()]:
        return ()

    @property
    def status(self) -> object:
        return self._status

    def __getattr__(self, name: str) -> object:
        return getattr(self._span, name)
