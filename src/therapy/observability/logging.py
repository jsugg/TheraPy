"""Broad-plane JSON logging and the third-party logger policy (plan §5.4, §4).

Every broad log line is one newline of JSON carrying ONLY the fixed §5.4
fields. Free-form interpolated messages are not part of this surface: owned
code emits fixed events through `emit_event()`; third-party loggers are
constrained by `THIRD_PARTY_LOGGER_POLICY`, and a global DEBUG setting can
never re-enable them (each policy logger gets an explicit level plus a
policy filter that ignores the root level).

Stack traces appear only for unexpected owned-code failures, with paths
normalized to package-relative form and exception messages/arguments
filtered out — only the exception class name survives (`error.type`).
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
import traceback
import uuid
from typing import Final

BROAD_LOGGER_NAME: Final = "therapy.broad"

#: Random per process (plan §5.4).
SERVICE_INSTANCE_ID: Final = uuid.uuid4().hex

#: Explicit policy per third-party logger (plan §4): `disabled` drops all
#: records; an int is a fixed minimum level that DEBUG cannot lower.
#: Pipecat's Loguru output is disabled separately at `load_pipecat()`.
THIRD_PARTY_LOGGER_POLICY: Final[dict[str, str | int]] = {
    "uvicorn.access": "disabled",  # concrete paths/queries; replaced in O2
    "uvicorn": logging.WARNING,
    "uvicorn.error": logging.WARNING,
    "fastapi": logging.WARNING,
    "starlette": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "anthropic": logging.WARNING,
    "openai": logging.WARNING,
    "multipart": logging.WARNING,
    "PIL": logging.WARNING,
    "pymupdf": logging.WARNING,
    "fitz": logging.WARNING,
    "pytesseract": logging.WARNING,
    "cryptography": logging.WARNING,
    "opentelemetry": logging.WARNING,
    "opentelemetry.sdk": logging.WARNING,
    "opentelemetry.exporter": logging.WARNING,
    "aiortc": logging.WARNING,
    "aioice": "disabled",  # logs ICE candidates/addresses
}

_ALLOWED_EVENT_FIELDS: Final = frozenset(
    {
        "component",
        "operation",
        "outcome",
        "duration_ms",
        "error.type",
        "retry_count",
        "count",
        "trace_id",
        "span_id",
    }
)


class _PolicyFilter(logging.Filter):
    """Pins a third-party logger to its policy regardless of global level."""

    def __init__(self, minimum: int | None) -> None:
        super().__init__()
        self._minimum = minimum  # None = disabled

    def filter(self, record: logging.LogRecord) -> bool:
        if self._minimum is None:
            return False
        return record.levelno >= self._minimum


def apply_third_party_policy() -> None:
    """Idempotent; called by the bootstrap before the app imports."""
    for name, policy in THIRD_PARTY_LOGGER_POLICY.items():
        logger = logging.getLogger(name)
        for existing in list(logger.filters):
            if isinstance(existing, _PolicyFilter):
                logger.removeFilter(existing)
        if policy == "disabled":
            logger.addFilter(_PolicyFilter(None))
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
        else:
            level = int(policy)
            logger.addFilter(_PolicyFilter(level))
            logger.setLevel(level)


def _relative_path(path: str) -> str:
    """Normalize traceback paths to package-relative form (§5.4)."""
    for marker in ("/site-packages/", "/src/", "/lib/python"):
        index = path.rfind(marker)
        if index != -1:
            return path[index + len(marker) :].lstrip("/")
    return path.rsplit("/", 1)[-1]


def _filtered_stack(exc: BaseException) -> list[str]:
    """Frames only — no exception message, no locals, no arguments."""
    frames: list[str] = []
    for frame in traceback.extract_tb(exc.__traceback__):
        frames.append(f"{_relative_path(frame.filename)}:{frame.lineno}:{frame.name}")
    return frames


class BroadJsonFormatter(logging.Formatter):
    """One fixed-schema JSON object per line (§5.4)."""

    def __init__(
        self,
        *,
        service_name: str = "therapy",
        service_version: str = "0.0.0",
        environment: str = "development",
        resource: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._static = {
            "service.name": service_name,
            "service.version": service_version,
            "service.instance.id": SERVICE_INSTANCE_ID,
            "deployment.environment": environment,
            "runtime.python": platform.python_version(),
            # extra safe process-constant resource fields (§5.4/O1.1 item 4):
            # build revision, pipecat/vendor version, capture mode/backend,
            # pinned schema version, secret-free config fingerprint.
            **(resource or {}),
        }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            )
            + f".{int(record.msecs):03d}Z",
            "severity": record.levelname,
            "event.name": getattr(record, "event_name", record.getMessage()),
            **self._static,
        }
        for field in _ALLOWED_EVENT_FIELDS:
            value = getattr(record, field.replace(".", "_"), None)
            if value is not None:
                payload[field] = value
        if record.exc_info and record.exc_info[1] is not None:
            exc = record.exc_info[1]
            payload["error.type"] = type(exc).__name__
            if getattr(record, "owned_failure", False):
                payload["stack"] = _filtered_stack(exc)
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)


class RateLimitFilter(logging.Filter):
    """At most one record per (event.name, interval); repeats are counted."""

    def __init__(self, interval_seconds: float = 30.0) -> None:
        super().__init__()
        self._interval = interval_seconds
        self._last: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "rate_limited", False):
            return True
        name = getattr(record, "event_name", record.getMessage())
        now = time.monotonic()
        last = self._last.get(name)
        if last is not None and now - last < self._interval:
            self._suppressed[name] = self._suppressed.get(name, 0) + 1
            return False
        if self._suppressed.get(name):
            record.count = self._suppressed.pop(name) + 1
        self._last[name] = now
        return True


def broad_logger() -> logging.Logger:
    return logging.getLogger(BROAD_LOGGER_NAME)


def emit_event(
    event_name: str,
    *,
    severity: int = logging.INFO,
    component: str,
    operation: str,
    outcome: str,
    duration_ms: float | None = None,
    error_type: str | None = None,
    retry_count: int | None = None,
    count: int | None = None,
    rate_limited: bool = False,
    exc: BaseException | None = None,
    owned_failure: bool = False,
) -> None:
    """The only owned broad-log entry point: fixed name, bounded fields."""
    from therapy.observability.context import active_trace_context

    extra: dict[str, object] = {
        "event_name": event_name,
        "component": component,
        "operation": operation,
        "outcome": outcome,
        "rate_limited": rate_limited,
        "owned_failure": owned_failure,
    }
    if duration_ms is not None:
        extra["duration_ms"] = round(duration_ms, 3)
    if error_type is not None:
        extra["error_type"] = error_type
    if retry_count is not None:
        extra["retry_count"] = retry_count
    if count is not None:
        extra["count"] = count
    context = active_trace_context()
    if context is not None:
        extra["trace_id"] = context.trace_id
        extra["span_id"] = context.span_id
    broad_logger().log(
        severity,
        event_name,
        extra=extra,
        exc_info=(type(exc), exc, exc.__traceback__) if exc is not None else None,
    )


class RootPolicyFilter(logging.Filter):
    """Root-handler enforcement (audit C-03).

    Python does not run ancestor-logger filters for records emitted by
    descendants (`httpx._client`, `opentelemetry.exporter.*`, …), so the
    named-logger policy alone cannot stop a child logger's free-form message
    from becoming a broad `event.name`. This filter runs on the ROOT handler:

    - owned `therapy.*`/`scripts` records pass (their messages are fixed,
      audited event names);
    - third-party records are matched by longest logger-name prefix against
      the policy: disabled prefixes drop entirely; others pass only at or
      above their fixed level AND with the message REPLACED by a bounded
      classified name — the original text/args are discarded;
    - unknown third-party loggers get the same sanitization at WARNING+.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        if name == "root" or name.startswith(("therapy", "scripts", "__main__")):
            return True
        matched: str | int | None = None
        matched_len = -1
        for prefix, policy in THIRD_PARTY_LOGGER_POLICY.items():
            if (name == prefix or name.startswith(prefix + ".")) and len(
                prefix
            ) > matched_len:
                matched, matched_len = policy, len(prefix)
        if matched == "disabled":
            return False
        minimum = int(matched) if matched is not None else logging.WARNING
        if record.levelno < minimum:
            return False
        # classified, content-free replacement: package + exception class only
        top = name.split(".", 1)[0]
        event_name = f"third_party.{top}"
        record.__dict__["event_name"] = event_name
        record.msg = event_name
        record.args = ()
        if record.exc_info and record.exc_info[1] is not None:
            record.error_type = type(record.exc_info[1]).__name__
            record.exc_info = None
        return True


def configure_stdout_json_logging(
    *,
    level: str = "INFO",
    service_version: str = "0.0.0",
    environment: str = "development",
    resource: dict[str, str] | None = None,
) -> None:
    """Root JSON handler on stdout + third-party policy (bootstrap only)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        BroadJsonFormatter(
            service_version=service_version,
            environment=environment,
            resource=resource,
        )
    )
    handler.addFilter(RootPolicyFilter())
    handler.addFilter(RateLimitFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    apply_third_party_policy()
