"""Pipecat-only telemetry adapter (plan §3, O1.4).

The ONLY module that converts Pipecat types into owned capture records.
Pipecat 1.5.0 exposes no documented hook for raw provider wire events
(SSE/NDJSON) from its LLM services, so the realtime attempt is captured at
the frame boundary — exact context/messages in, ordered text deltas, and
the aggregated completion — which O0's ADR records as the supported seam;
no undocumented internals are patched (plan O1.4 item 2). Wire-level
`provider_native.ordered_events` for the realtime path therefore contains
frame-level evidence; the non-realtime path captures true native bodies.

`build_task_telemetry()` wires `PipelineWorker` per the pinned 1.5.0 snapshot:
`enable_tracing`/`enable_turn_tracking` only when the owned global provider
is installed (Pipecat then obtains tracers from it; its `pipecat`/
`pipecat.turn` scopes route restricted by default-deny), plus a telemetry
conversation ID that is never the product session ID.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from typing import Protocol, TypedDict, cast

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFAMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from therapy.observability.capture import AttemptHandle, capture_service
from therapy.observability.interactions import (
    InteractionRequest,
    InteractionResponse,
    JsonValue,
    Message,
    require_json_object,
)
from therapy.observability.logging import emit_event
from therapy.observability.model import (
    InteractionEventKind,
    InteractionOperation,
    Provider,
    normalize_enum,
)

logger = logging.getLogger(__name__)


class _ObserverInitializer(Protocol):
    """Typed constructor seam for Pipecat's partially annotated observer."""

    def __call__(self, observer: BaseObserver) -> None: ...


class TaskTelemetry(TypedDict):
    """Telemetry options passed to the pinned Pipecat worker surface."""

    enable_tracing: bool
    enable_turn_tracking: bool
    conversation_id: str
    observers: list[BaseObserver]


def _runtime_value(value: object) -> object:
    """Erase incomplete third-party hints before runtime boundary checks."""
    return value


def _initialize_observer(observer: BaseObserver) -> None:
    """Invoke the validated no-argument BaseObserver constructor."""
    initializer: object | None = None
    for base in BaseObserver.__mro__:
        candidate = _runtime_value(vars(base).get("__init__"))
        if callable(candidate):
            initializer = candidate
            break
    if not callable(initializer):
        raise TypeError("Pipecat BaseObserver constructor is not callable")
    cast(_ObserverInitializer, initializer)(observer)


def _context_messages(
    frame: LLMContextFrame,
) -> tuple[str, list[dict[str, JsonValue]]]:
    """Exact system instructions + ordered messages from a context frame."""
    context = frame.context
    messages: list[dict[str, JsonValue]] = []
    getter = getattr(context, "get_messages", None)
    raw_value = _runtime_value(
        getter() if callable(getter) else getattr(context, "messages", [])
    )
    system = ""
    raw = (
        cast(Sequence[object], raw_value)
        if isinstance(raw_value, list | tuple)
        else ()
    )
    for item in raw:
        item_value = _runtime_value(item)
        entry: dict[str, JsonValue]
        if isinstance(item_value, dict):
            raw_item = cast(dict[object, object], item_value)
            if not all(isinstance(key, str) for key in raw_item):
                raise TypeError("Pipecat message field names must be strings")
            entry = require_json_object(
                cast(dict[str, object], raw_item), "pipecat.context.message"
            )
        else:  # provider-specific message objects
            entry = {
                "role": str(getattr(item_value, "role", "unknown")),
                "content": str(getattr(item_value, "content", item_value)),
            }
        if entry.get("role") == "system" and not system:
            content = entry.get("content", "")
            system = content if isinstance(content, str) else str(content)
        messages.append(entry)
    explicit_system = getattr(context, "system", None)
    if isinstance(explicit_system, str) and explicit_system:
        system = explicit_system
    return system, messages


def _message_content(message: dict[str, JsonValue]) -> str:
    """Render one canonical text field without weakening native capture."""
    content = message.get("content")
    return content if isinstance(content, str) else str(content)


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
        _initialize_observer(self)
        self._llm = llm_service
        self._session_id = session_id
        self._provider = normalize_enum(provider, Provider, Provider.UNKNOWN)
        self._requested_model = requested_model
        self._language = language
        self._pending_request: tuple[str, list[dict[str, JsonValue]]] | None = None
        self._handle: AttemptHandle | None = None
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
            # Runtime mode: capture failure never breaks the voice path
            # (§5.3) — it degrades visibly. Evaluation mode fails CLOSED:
            # the capture error propagates and stops the pipeline run.
            from therapy.observability.capture import (
                CaptureUnavailable,
                capture_service,
            )
            from therapy.observability.model import CaptureMode

            service = capture_service()
            if isinstance(exc, CaptureUnavailable) or (
                service is not None and service.mode is CaptureMode.EVALUATION
            ):
                raise
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
        provider_messages: list[JsonValue] = [
            message for message in raw_messages
        ]
        request = InteractionRequest(
            system_instructions=system,
            messages=tuple(
                Message(
                    role=str(m.get("role", "unknown")),
                    content=_message_content(m),
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
                "messages": provider_messages,  # exact context as supplied
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


class MetricsFrameAdapter(BaseObserver):
    """Translate Pipecat `MetricsFrame` payloads to owned instruments (O2.3).

    Consumes the signal `PipelineParams(enable_metrics=True)` already emits —
    no parallel Pipecat metric system. Values that owned code already
    represents (the TTFAMonitor broad event) are recorded here exactly once
    as histograms; processor class names map to a bounded component set and
    are never used as raw labels.
    """

    _COMPONENT_HINTS = (
        ("llm", "llm"),
        ("tts", "tts"),
        ("stt", "stt"),
        ("whisper", "stt"),
        ("kokoro", "tts"),
    )

    def __init__(self) -> None:
        _initialize_observer(self)
        self._tts_audio_seconds = 0.0
        self._tts_processing_seconds = 0.0
        self._tts_started_at: float | None = None

    def _component(self, processor: str) -> str:
        lowered = processor.lower()
        for hint, component in self._COMPONENT_HINTS:
            if hint in lowered:
                return component
        return "unknown"

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        source_component = self._component(type(data.source).__name__)
        if isinstance(frame, TTSStartedFrame) and source_component == "tts":
            self._tts_started_at = time.monotonic()
        elif isinstance(frame, TTSStoppedFrame) and source_component == "tts":
            self._tts_started_at = None
        elif isinstance(frame, ErrorFrame):
            processor = getattr(frame, "processor", None)
            error_component = self._component(type(processor).__name__)
            if error_component == "unknown":
                error_component = source_component
            if error_component == "tts":
                from therapy.observability.telemetry import record_metric

                elapsed = (
                    time.monotonic() - self._tts_started_at
                    if self._tts_started_at is not None
                    else 0.0
                )
                record_metric(
                    "therapy_tts_requests_total",
                    1,
                    {"language_group": "unknown", "outcome": "error"},
                )
                record_metric(
                    "therapy_tts_synthesis_seconds",
                    elapsed,
                    {"language_group": "unknown", "outcome": "error"},
                )
                self._tts_started_at = None
        elif isinstance(frame, TTSAudioRawFrame) and source_component == "tts":
            sample_rate = max(1, int(getattr(frame, "sample_rate", 0) or 0))
            channels = max(1, int(getattr(frame, "num_channels", 1) or 1))
            self._tts_audio_seconds += len(frame.audio) / (sample_rate * channels * 2)
        elif isinstance(frame, BotStoppedSpeakingFrame) and self._tts_audio_seconds > 0:
            from therapy.observability.telemetry import record_metric

            record_metric(
                "therapy_tts_realtime_factor",
                self._tts_processing_seconds / self._tts_audio_seconds,
                {"language_group": "unknown", "outcome": "success"},
            )
            self._tts_audio_seconds = 0.0
            self._tts_processing_seconds = 0.0
        if not isinstance(frame, MetricsFrame):
            return
        from therapy.observability.telemetry import record_metric

        for item in frame.data:
            processor = getattr(item, "processor", "") or ""
            component = self._component(processor)
            try:
                if isinstance(item, TTFBMetricsData):
                    if component == "llm":
                        record_metric(
                            "therapy_llm_time_to_first_token_seconds",
                            item.value,
                            {
                                "provider": _env_provider(),
                                "operation": "reply",
                                "outcome": "success",
                            },
                        )
                    elif component == "tts":
                        record_metric(
                            "therapy_tts_time_to_first_audio_seconds",
                            item.value,
                            {"language_group": "unknown", "outcome": "success"},
                        )
                elif isinstance(item, TTFAMetricsData):
                    # TTFAMonitor is the single TTFA producer (honest
                    # cold/warm labeling); recording here double-counted.
                    pass
                elif isinstance(item, LLMUsageMetricsData):
                    usage = item.value
                    record_metric(
                        "therapy_llm_input_tokens_total",
                        getattr(usage, "prompt_tokens", 0) or 0,
                        {"provider": _env_provider(), "operation": "reply"},
                    )
                    record_metric(
                        "therapy_llm_output_tokens_total",
                        getattr(usage, "completion_tokens", 0) or 0,
                        {"provider": _env_provider(), "operation": "reply"},
                    )
                elif isinstance(item, TTSUsageMetricsData):
                    record_metric(
                        "therapy_tts_requests_total",
                        1,
                        {"language_group": "unknown", "outcome": "success"},
                    )
                    record_metric(
                        "therapy_tts_characters_total",
                        item.value,
                        {"language_group": "unknown"},
                    )
                elif isinstance(item, ProcessingMetricsData) and component != "unknown":
                    record_metric(
                        "therapy_turn_stage_duration_seconds",
                        item.value,
                        {"stage": component, "outcome": "success"},
                    )
                    if component == "tts":
                        self._tts_processing_seconds += item.value
                        record_metric(
                            "therapy_tts_synthesis_seconds",
                            item.value,
                            {"language_group": "unknown", "outcome": "success"},
                        )
            except Exception:
                continue  # metric translation must never disturb the pipeline


def _env_provider() -> str:
    import os

    # bounded: raw env input never becomes a label (audit H-02)
    return normalize_enum(
        os.environ.get("THERAPY_LLM", "anthropic"), Provider, Provider.UNKNOWN
    ).value


def build_task_telemetry(
    llm_service: object,
    *,
    session_id: str | None,
) -> TaskTelemetry:
    """`PipelineWorker` telemetry kwargs (pinned 1.5.0 parameter surface).

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
        "observers": [observer, MetricsFrameAdapter()],
    }
