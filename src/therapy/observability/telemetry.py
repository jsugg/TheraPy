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
from dataclasses import dataclass, field
from typing import Any

from therapy.observability.config import ObservabilityConfig
from therapy.observability.logging import emit_event
from therapy.observability.model import TelemetryPlane
from therapy.observability.routing import classify_scope, scrub_broad_attributes


@dataclass
class TelemetryState:
    enabled: bool = False
    provider: object | None = None
    meter_provider: object | None = None
    instruments: dict[str, object] = field(default_factory=dict)
    dropped_unknown_scopes: int = 0
    dropped_forbidden_keys: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)


_state = TelemetryState()


def state() -> TelemetryState:
    return _state


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

    resource = Resource.create(
        {
            "service.name": "therapy",
            "service.version": service_version,
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

    provider.add_span_processor(
        PlaneRoutingSpanProcessor(
            broad=broad_processor, restricted=restricted_processor
        )
    )
    trace.set_tracer_provider(provider)
    _state.enabled = True
    _state.provider = provider
    _initialize_metrics(config, resource)
    _instrument_fastapi()
    _instrument_system_metrics()
    return True


def _initialize_metrics(config: ObservabilityConfig, resource) -> None:
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
    clean: dict[str, str] = {}
    for key, raw in (attributes or {}).items():
        if spec is None or key not in spec.attributes:
            with _state._lock:
                _state.dropped_forbidden_keys += 1
            continue
        allowed = spec.attributes[key]
        text = str(raw)
        clean[key] = text if allowed is None or text in allowed else "unknown"
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

    def _server_request_hook(span, scope: dict) -> None:
        # strip concrete-target attributes at the source; the routing scrub
        # is the second, default-deny line of defense
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


def instrumented_async_client(destination: str, **kwargs):
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
    client = httpx.AsyncClient(**kwargs)
    if not _state.enabled:
        return client
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        def _request_hook(span, request) -> None:
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


def broad_span(name: str, *, component: str, operation: str):
    """Owned broad span helper: records status once, scope `therapy.broad`."""
    from contextlib import contextmanager

    @contextmanager
    def _noop():
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


def link_root(name: str, *, component: str, operation: str, parent_trace_id: str,
              parent_span_id: str):
    """A NEW trace root LINKED to its trigger (detached finalizers/batches);
    never a multi-hour parent span (plan O2.1 item 4, O2.2)."""
    from contextlib import contextmanager

    @contextmanager
    def _noop():
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

    def __init__(self, broad: Any | None, restricted: Any | None) -> None:
        self._broad = broad
        self._restricted = restricted

    def on_start(self, span, parent_context=None) -> None:
        for processor in (self._broad, self._restricted):
            if processor is not None:
                processor.on_start(span, parent_context)

    def on_ending(self, span) -> None:
        """SDK >=1.4x pre-end hook; routing happens in on_end."""

    # the SDK's synchronous multiplexer calls the private name directly
    _on_ending = on_ending

    def on_end(self, span) -> None:
        scope = getattr(span, "instrumentation_scope", None)
        plane = classify_scope(getattr(scope, "name", None))
        if plane is TelemetryPlane.RESTRICTED:
            if self._restricted is not None:
                self._restricted.on_end(span)
            return
        if plane is TelemetryPlane.BROAD:
            if self._broad is not None:
                attributes = dict(span.attributes or {})
                result = scrub_broad_attributes(attributes)
                if result.dropped_keys:
                    with _state._lock:
                        _state.dropped_forbidden_keys += len(result.dropped_keys)
                    emit_event(
                        "telemetry.broad_attribute_dropped",
                        severity=logging.WARNING,
                        component="telemetry",
                        operation="routing",
                        outcome="rejected",
                        count=len(result.dropped_keys),
                        rate_limited=True,
                    )
                    span = _ScrubbedSpanView(span, result.attributes)
                self._broad.on_end(span)
            return
        with _state._lock:
            _state.dropped_unknown_scopes += 1
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
            resource=Resource.create({"service.name": "therapy-interactions"}),
            id_generator=self._id_generator,
        )
        self._tracer = self._provider.get_tracer("therapy.interactions")
        self._exporter = OTLPSpanExporter(
            endpoint=f"{endpoint}/v1/traces", timeout=timeout
        )

    async def export(self, interaction: dict) -> str | None:
        import asyncio

        return await asyncio.to_thread(self._export_sync, interaction)

    def _export_sync(self, payload: dict) -> str:
        import json as _json
        from datetime import datetime

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

        envelope = {
            "interaction": {k: row[k] for k in row.keys()},
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
        result = self._exporter.export([span])  # type: ignore[list-item]
        if result is not SpanExportResult.SUCCESS:
            raise ExportError("phoenix OTLP export failed")
        return row["span_id"]


_LOCAL_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "::1", "phoenix", "collector", "lgtm"}
)


def make_interaction_exporter(config: ObservabilityConfig):
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
    """Read-only span proxy presenting scrubbed attributes to the exporter."""

    __slots__ = ("_span", "_attributes")

    def __init__(self, span, attributes: dict[str, object]) -> None:
        self._span = span
        self._attributes = attributes

    @property
    def attributes(self) -> dict[str, object]:
        return self._attributes

    def __getattr__(self, name: str):
        return getattr(self._span, name)
