"""Trace/interaction correlation context (plan §3, O1.1 item 4).

Framework-free: W3C-shaped trace/span IDs are generated locally with
`secrets` so journal correlation NEVER depends on an exporter or remote
backend being up. When the OTel SDK is active, `telemetry.py` seeds these
contextvars from the real span context instead; the shape is identical.
"""

from __future__ import annotations

import secrets
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


def new_trace_id() -> str:
    """128-bit lowercase hex, never all-zero (W3C trace-id shape)."""
    while True:
        value = secrets.token_hex(16)
        if value != "0" * 32:
            return value


def new_span_id() -> str:
    """64-bit lowercase hex, never all-zero (W3C span-id shape)."""
    while True:
        value = secrets.token_hex(8)
        if value != "0" * 16:
            return value


def new_interaction_id() -> str:
    """Unique per provider attempt (plan §1)."""
    return f"itx-{secrets.token_hex(12)}"


@dataclass(frozen=True, slots=True)
class TraceContext:
    """One correlation scope shared by both planes."""

    trace_id: str
    span_id: str

    def child(self) -> TraceContext:
        return TraceContext(trace_id=self.trace_id, span_id=new_span_id())


_current: ContextVar[TraceContext | None] = ContextVar(
    "therapy_trace_context", default=None
)
_current_interaction: ContextVar[str | None] = ContextVar(
    "therapy_interaction_id", default=None
)


def current_trace_context() -> TraceContext:
    """The active context, minting a fresh root when none exists."""
    context = _current.get()
    if context is None:
        context = TraceContext(trace_id=new_trace_id(), span_id=new_span_id())
        _current.set(context)
    return context


def current_interaction_id() -> str | None:
    return _current_interaction.get()


@contextmanager
def trace_scope(context: TraceContext | None = None):
    """Bind a trace context for the duration of a logical operation.

    Detached work (finalizers, scheduler batches) passes an explicit fresh
    root and records the parent as a LINK, never as a multi-hour parent
    span (plan O2.1 item 4).
    """
    bound = context or TraceContext(trace_id=new_trace_id(), span_id=new_span_id())
    token = _current.set(bound)
    try:
        yield bound
    finally:
        _current.reset(token)


@contextmanager
def interaction_scope(interaction_id: str | None = None):
    """Bind the per-attempt interaction ID (one per provider attempt)."""
    bound = interaction_id or new_interaction_id()
    token = _current_interaction.set(bound)
    try:
        yield bound
    finally:
        _current_interaction.reset(token)
