"""Pipecat-only telemetry adapter (plan §3, O1.4).

The ONLY module that converts Pipecat types into owned capture records.
Pipecat 1.5.0 exposes no documented hook for raw provider wire events
(SSE/NDJSON) from its LLM services, so the realtime attempt is captured at
the frame boundary — exact context/messages in, ordered text deltas, and
the aggregated completion — which O0's ADR records as the supported seam;
no undocumented internals are patched (plan O1.4 item 2). Wire-level
`provider_native.ordered_events` for the realtime path therefore contains
frame-level evidence; the non-realtime path captures true native bodies.

`build_task_telemetry()` wires `PipelineTask` per the pinned 1.5.0 snapshot:
`enable_tracing`/`enable_turn_tracking` only when the owned global provider
is installed (Pipecat then obtains tracers from it; its `pipecat`/
`pipecat.turn` scopes route restricted by default-deny), plus a telemetry
conversation ID that is never the product session ID.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from therapy.observability.capture import capture_service
from therapy.observability.interactions import (
    InteractionRequest,
    InteractionResponse,
    Message,
)
from therapy.observability.logging import emit_event
from therapy.observability.model import (
    InteractionEventKind,
    InteractionOperation,
    Provider,
    normalize_enum,
)

logger = logging.getLogger(__name__)


def _context_messages(frame: LLMContextFrame) -> tuple[str, list[dict[str, Any]]]:
    """Exact system instructions + ordered messages from a context frame."""
    context = frame.context
    messages: list[dict[str, Any]] = []
    getter = getattr(context, "get_messages", None)
    raw = getter() if callable(getter) else getattr(context, "messages", [])
    system = ""
    for item in raw or []:
        if isinstance(item, dict):
            entry = dict(item)
        else:  # provider-specific message objects
            entry = {
                "role": str(getattr(item, "role", "unknown")),
                "content": str(getattr(item, "content", item)),
            }
        if entry.get("role") == "system" and not system:
            content = entry.get("content", "")
            system = content if isinstance(content, str) else str(content)
        messages.append(entry)
    explicit_system = getattr(context, "system", None)
    if isinstance(explicit_system, str) and explicit_system:
        system = explicit_system
    return system, messages


class InteractionCaptureObserver(BaseObserver):
    """Owned provider-attempt/stream capture for the realtime reply boundary.

    Journals BEFORE the LLM service processes the context frame (observers
    are awaited on push), appends ordered deltas, and always reaches an
    explicit terminal: success, or `incomplete` on interruption/cancel/end.
    """

    def __init__(
        self,
        llm_service: object,
        *,
        session_id: str | None,
        provider: str,
        requested_model: str,
        language: str = "unknown",
    ) -> None:
        super().__init__()
        self._llm = llm_service
        self._session_id = session_id
        self._provider = normalize_enum(provider, Provider, Provider.UNKNOWN)
        self._requested_model = requested_model
        self._language = language
        self._pending_request: tuple[str, list[dict[str, Any]]] | None = None
        self._handle = None
        self._completion_parts: list[str] = []

    async def on_push_frame(self, data: FramePushed) -> None:
        # A frame traverses many hops, but exactly one hop has the LLM
        # service as source (its output push) or destination (its input),
        # so these filters see each logical event once — no dedupe state.
        frame = data.frame
        try:
            if isinstance(frame, LLMContextFrame) and data.destination is self._llm:
                self._pending_request = _context_messages(frame)
            elif (
                isinstance(frame, LLMFullResponseStartFrame)
                and data.source is self._llm
            ):
                await self._start_attempt()
            elif isinstance(frame, LLMTextFrame) and data.source is self._llm:
                self._completion_parts.append(frame.text)
                if self._handle is not None:
                    await self._handle.record_event(
                        InteractionEventKind.STREAM_DELTA, {"delta": frame.text}
                    )
            elif (
                isinstance(frame, LLMFullResponseEndFrame)
                and data.source is self._llm
            ):
                await self._finish_success()
            elif isinstance(frame, InterruptionFrame):
                await self._finish_incomplete("barge_in")
            elif isinstance(frame, CancelFrame | EndFrame):
                await self._finish_incomplete("pipeline_stopped")
        except Exception as exc:
            # Capture failure never breaks the voice path (§5.3 runtime
            # policy) — it degrades visibly instead.
            emit_event(
                "capture_degraded",
                severity=logging.ERROR,
                component="llm",
                operation="reply",
                outcome="error",
                error_type=type(exc).__name__,
                rate_limited=True,
            )

    async def _start_attempt(self) -> None:
        service = capture_service()
        if service is None or self._handle is not None:
            return
        system, raw_messages = self._pending_request or ("", [])
        request = InteractionRequest(
            system_instructions=system,
            messages=tuple(
                Message(
                    role=str(m.get("role", "unknown")),
                    content=(
                        m.get("content")
                        if isinstance(m.get("content"), str)
                        else str(m.get("content"))
                    ),
                )
                for m in raw_messages
            ),
        )
        self._completion_parts = []
        self._handle = await service.start_attempt(
            operation=InteractionOperation.REPLY,
            provider=self._provider,
            requested_model=self._requested_model,
            request=request,
            provider_request={
                "model": self._requested_model,
                "messages": raw_messages,  # exact context as supplied
                "stream": True,
                "capture_seam": "pipecat-frame-boundary",
            },
            language=self._language,
            modality="voice",
            session_id=self._session_id,
        )

    async def _finish_success(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        completion = "".join(self._completion_parts)
        await handle.succeed(
            InteractionResponse(
                messages=(Message(role="assistant", content=completion),),
                completion=completion,
                finish_reason="end_turn",
            ),
            native_terminal={
                "capture_seam": "pipecat-frame-boundary",
                "delta_count": len(self._completion_parts),
            },
        )

    async def _finish_incomplete(self, reason: str) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        await handle.incomplete(reason)


def build_task_telemetry(
    llm_service: object,
    *,
    session_id: str | None,
) -> dict[str, Any]:
    """`PipelineTask` telemetry kwargs (pinned 1.5.0 parameter surface).

    Tracing/turn tracking activate only when the owned global provider is
    installed (`therapy.observability.telemetry.initialize`); the audited
    `pipecat`/`pipecat.turn` scopes then route to the restricted exporter
    via the default-deny processor. Pipecat's own `setup_tracing()` is
    never called. The conversation ID is a fresh telemetry ID, never the
    product session ID.
    """
    import os

    from therapy.observability.telemetry import state

    observer = InteractionCaptureObserver(
        llm_service,
        session_id=session_id,
        provider=os.environ.get("THERAPY_LLM", "anthropic"),
        requested_model=os.environ.get("THERAPY_LLM_MODEL")
        or {"anthropic": "claude-opus-4-8", "openrouter": "openrouter/free"}.get(
            os.environ.get("THERAPY_LLM", "anthropic"),
            "pedrolucas/smollm3:3b-q4_k_m",
        ),
    )
    tracing = state().enabled
    return {
        "enable_tracing": tracing,
        "enable_turn_tracking": tracing,
        "conversation_id": f"tele-{uuid.uuid4().hex}",
        "observers": [observer],
    }
