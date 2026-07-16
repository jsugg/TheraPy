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
    return True


def shutdown(timeout_millis: int = 5000) -> None:
    """Bounded flush; product shutdown never waits indefinitely (O1.1)."""
    provider = _state.provider
    if provider is None:
        return
    try:
        # BatchSpanProcessor flushes with its own bounded timeout
        getattr(provider, "shutdown", lambda: None)()
    except Exception:
        pass
    _state.enabled = False
    _state.provider = None


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
