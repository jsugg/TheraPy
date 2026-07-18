"""Pipecat pipeline assembly inside the framework integration boundary.

Framework: Pipecat with SmallWebRTCTransport (docs/framework-spike.md).
Everything domain-shaped (prompts, language tables, voice maps) lives in
framework-free modules; a framework swap is contained within
`therapy.integrations.pipecat` and the client transport layer (SPEC §5).

Pipeline per connection:

    transport.input (WebRTC)
      → VADProcessor (Silero)
      → MultilingualWhisperSTT (per-utterance language detection)
      → TurnRelay (voice-switch TTS to the turn's language; transcript → client)
      → context.user aggregator
      → LLM (provider-agnostic factory: anthropic | openrouter | ollama)
      → BotTextRelay (full reply text → client data channel)
      → Kokoro TTS
      → TTFAMonitor (time-to-first-audio, risk R1)
      → transport.output
      → context.assistant aggregator

Typed turns arrive on the data channel (`on_app_message`) and are appended
to the same context — voice and text are one conversation (SPEC §5).

Reply modality mirrors the input server-side (SPEC §5): each user turn
pushes `LLMConfigureOutputFrame(skip_tts=…)` so typed turns get silent
replies — no wasted synthesis — while spoken turns get voice. The client's
speaker toggle sends a `voice_replies` override on the same data channel.

Reply language (SPEC §7): per turn, `ReplyLanguage` picks the reply
language — word-level dominant language of the phrase in auto mode, or the
user's pin sent as a `reply_language` data-channel override (`null` = auto).
The pin constrains replies and TTS voice only; STT stays auto.

Memory (SPEC §8): each connection is one session in the local store — every
turn is recorded (language, modality, timestamps), raw utterance audio is
archived on the host, and new sessions open with prior summaries plus the
user model in context (older history never verbatim). On disconnect the
session is summarized and distilled in the background.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMConfigureOutputFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesTransformFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSUpdateSettingsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextMessage
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.kokoro.tts import KokoroTTSService, KokoroTTSSettings
from pipecat.services.whisper.stt import WhisperSTTService, WhisperSTTSettings
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.utils.time import time_now_iso8601

from therapy.dialogue.language_choice import (
    ReplyLanguage,
    dominant_language,
    label_language,
    reply_language_override_effect,
)
from therapy.dialogue.modality import TEXT, VOICE, ReplyModality
from therapy.dialogue.policy import (
    build_system_prompt,
    language_pin_note,
    language_switch_note,
    rehydrate_messages,
    reply_language_reminder,
    resume_note,
)
from therapy.knowledge import distill as knowledge_distill
from therapy.knowledge.context import ContextAssembler, replace_longitudinal_context
from therapy.knowledge.insight import session_recap
from therapy.knowledge.user_model import UserModel
from therapy.memory import MemoryStore, make_summarizer
from therapy.memory.store import RowDict, resume_window_secs
from therapy.memory.summarizer import entitle
from therapy.observability.interactions import JsonValue, require_json_object
from therapy.observability.logging import emit_event
from therapy.observability.model import WorkloadClass
from therapy.observability.telemetry import run_in_thread
from therapy.perception.stt import (
    DEFAULT_LANGUAGE,
    clamp_language,
    genuine_foreign_speech,
    is_supported,
    plausible_segment,
    whisper_model,
)
from therapy.server import live
from therapy.server.protocol import presence_message, session_state_message
from therapy.speech.tts import voice_for

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

LANGUAGE_ENUM = {"en": Language.EN, "es": Language.ES, "pt": Language.PT}

# Shared across connections — see MultilingualWhisperSTTService._load.
_whisper_model: WhisperModel | None = None

# Reconnect-resume coordination (SPEC §8). A dropped connection schedules
# finalization, but the same client usually reconnects seconds later — the
# new pipeline must be able to cancel an in-flight finalize (_finalizers)
# and, for one scheduled after it already resumed, make it a no-op
# (live.py ownership tokens; only the current owner may close a session).
_finalizers: dict[str, asyncio.Task[None]] = {}

type SessionTurns = Sequence[Mapping[str, object]]
type SummaryGenerator = Callable[[SessionTurns], Awaitable[str]]
type DistillationGenerator = Callable[
    [UserModel, SessionTurns, str], Awaitable[knowledge_distill.DistillResult]
]
type TextArtifactGenerator = Callable[[SessionTurns], Awaitable[str | None]]


class _FrameProcessorInitializer(Protocol):
    """Typed constructor seam for Pipecat's partially annotated processor."""

    def __call__(self, processor: FrameProcessor) -> None: ...


class _WhisperInitializer(Protocol):
    """Typed constructor seam for the pinned Whisper service."""

    def __call__(
        self,
        service: WhisperSTTService,
        *,
        settings: WhisperSTTSettings,
    ) -> None: ...


def _runtime_value(value: object) -> object:
    """Erase incomplete third-party hints before runtime boundary checks."""
    return value


def _initialize_frame_processor(processor: FrameProcessor) -> None:
    """Invoke the validated no-argument FrameProcessor constructor."""
    initializer = vars(FrameProcessor).get("__init__")
    if not callable(initializer):
        raise TypeError("Pipecat FrameProcessor constructor is not callable")
    cast(_FrameProcessorInitializer, initializer)(processor)


def _initialize_whisper_service(service: WhisperSTTService) -> None:
    """Initialize the pinned Whisper service through a typed boundary."""
    initializer = vars(WhisperSTTService).get("__init__")
    if not callable(initializer):
        raise TypeError("Pipecat Whisper constructor is not callable")
    cast(_WhisperInitializer, initializer)(
        service,
        settings=WhisperSTTSettings(model=whisper_model(), language=Language.EN),
    )


def _object_mapping(value: object) -> dict[str, object] | None:
    """Narrow a third-party mapping after validating every key type."""
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw):
        return None
    return cast(dict[str, object], raw)


def _replace_pipecat_context(
    messages: list[LLMContextMessage], note: str | None
) -> list[LLMContextMessage]:
    """Adapt owned message dictionaries back to Pipecat's typed union."""
    owned_messages: list[dict[str, JsonValue]] = [
        require_json_object(message, "pipecat.context.message")
        for message in messages
    ]
    replaced = replace_longitudinal_context(owned_messages, note)
    if not all(
        isinstance(message.get("role"), str) and "content" in message
        for message in replaced
    ):
        raise TypeError("longitudinal context produced an invalid message")
    return cast(list[LLMContextMessage], replaced)


@dataclass(frozen=True, slots=True)
class _SessionArtifacts:
    summary: str | None
    distillation: knowledge_distill.DistillResult
    recap: str | None
    title: str | None


async def _default_summary(turns: SessionTurns) -> str:
    return await make_summarizer().summarize(turns)


async def _default_distillation(
    model: UserModel, turns: SessionTurns, session_id: str
) -> knowledge_distill.DistillResult:
    return await knowledge_distill.distill_session(model, turns, session_id)


async def generate_session_artifacts(
    turns: SessionTurns,
    model: UserModel,
    session_id: str,
    *,
    summarize: SummaryGenerator = _default_summary,
    distill: DistillationGenerator = _default_distillation,
    recap: TextArtifactGenerator = session_recap,
    title: TextArtifactGenerator = entitle,
) -> _SessionArtifacts:
    """Generate independent finalization artifacts without failure coupling."""

    def _artifact_outcome(artifact: str, outcome: str) -> None:
        """Per-artifact finalization evidence (plan O3.2) — success AND
        failure, so a silently skipped artifact is visible."""
        from therapy.observability.telemetry import record_metric

        record_metric(
            "therapy_session_finalizations_total",
            1,
            {"artifact": artifact, "outcome": outcome},
        )

    def _artifact_failed(artifact: str, exc: Exception) -> None:
        _artifact_outcome(artifact, "error")
        emit_event(
            "finalizer.artifact_failed",
            severity=logging.WARNING,
            component="voice",
            operation=artifact,
            outcome="error",
            error_type=type(exc).__name__,
        )

    summary_value: str | None = None
    distilled = knowledge_distill.DistillResult(run_id="not-run")
    recap_value: str | None = None
    title_value: str | None = None
    try:
        summary_value = (await summarize(turns)).strip() or None
        _artifact_outcome("summary", "success")
    except Exception as exc:
        _artifact_failed("summary", exc)
    try:
        distilled = await distill(model, turns, session_id)
        _artifact_outcome("distill", "success")
    except Exception as exc:
        _artifact_failed("distill", exc)
    try:
        recap_result = await recap(turns)
        recap_value = recap_result.strip() or None if recap_result else None
        _artifact_outcome("recap", "success")
    except Exception as exc:
        _artifact_failed("recap", exc)
    try:
        title_result = await title(turns)
        title_value = title_result.strip() or None if title_result else None
        _artifact_outcome("title", "success")
    except Exception as exc:
        _artifact_failed("title", exc)
    return _SessionArtifacts(
        summary=summary_value,
        distillation=distilled,
        recap=recap_value,
        title=title_value,
    )


def _record_finalizers_pending() -> None:
    from therapy.observability.telemetry import record_metric

    record_metric("therapy_session_finalizers_pending", len(_finalizers))


def _count_data_channel(type_class: str, outcome: str) -> None:
    """Bounded data-channel send evidence (plan O3.2): message class and
    outcome only — never the protocol payload."""
    from therapy.observability.telemetry import record_metric

    record_metric(
        "therapy_data_channel_messages_total",
        1,
        {"type_class": type_class, "outcome": outcome},
    )


async def drain_session_finalizers(timeout: float) -> None:
    """Wait boundedly for all session-finalization writes to finish."""
    tasks = tuple(_finalizers.values())
    if not tasks:
        return
    _, pending = await asyncio.wait(tasks, timeout=timeout)
    if pending:
        emit_event(
            "finalizer.drain_timeout",
            severity=logging.ERROR,
            component="voice",
            operation="finalize",
            outcome="timeout",
            count=len(pending),
        )
        raise TimeoutError("Session finalization did not finish in time")
    await asyncio.sleep(0)


def vad_params() -> VADParams:
    """Speech-detection thresholds, env-tunable without a rebuild.

    Pipecat's defaults (confidence .7, min_volume .6) were tuned for close,
    clean microphones; real phone speech over opus sits quieter, and the
    field test lost most of it. stop_secs=1.0 also keeps a phrase with a
    natural mid-sentence pause in one utterance instead of three.
    """
    return VADParams(
        confidence=float(os.environ.get("THERAPY_VAD_CONFIDENCE", "0.6")),
        start_secs=0.2,
        # 1.2 s: thinking pauses mid-sentence kept splitting real speech at
        # 1.0 s (field test); the cost is turn-end latency, tune per taste.
        stop_secs=float(os.environ.get("THERAPY_VAD_STOP_SECS", "1.2")),
        min_volume=float(os.environ.get("THERAPY_VAD_MIN_VOLUME", "0.3")),
    )


def tts_settings_for(language: str) -> TTSUpdateSettingsFrame:
    """TTS voice+language switch frame for a supported language code."""
    from therapy.observability.telemetry import record_metric

    try:
        frame = TTSUpdateSettingsFrame(
            delta=KokoroTTSSettings(
                voice=voice_for(language),
                language=LANGUAGE_ENUM[language],
            )
        )
    except (KeyError, ValueError):
        # Output-config failure is a finite transition (plan O3.2) — the
        # offending code never becomes a label.
        record_metric(
            "therapy_language_transitions_total",
            1,
            {"kind": "tts_config", "outcome": "invalid"},
        )
        raise
    record_metric(
        "therapy_language_transitions_total",
        1,
        {"kind": "tts_config", "outcome": "changed"},
    )
    return frame


def _transcribe_utterance(
    model: WhisperModel,
    audio_float: NDArray[np.float32],
    language: str | None,
    no_speech_threshold: float,
) -> tuple[str, str | None, float]:
    """Decode one utterance fully inside the calling (worker) thread.

    faster-whisper's transcribe() is lazy — the heavy decode runs where the
    segments generator is consumed. Consuming it on the event loop blocked
    the loop for the length of the decode; on a long utterance that starved
    WebRTC keepalives and dropped the connection mid-turn (field test
    2026-07-10, confirmed by a watchdog health-probe failure at that
    moment). Everything whisper-bound stays in this function.
    """
    segments, info = model.transcribe(
        audio_float,
        language=language,
        # Carrying decoder context across utterances compounds
        # hallucinations on degraded audio.
        condition_on_previous_text=False,
    )
    text = " ".join(
        segment.text.strip()
        for segment in segments
        if plausible_segment(
            segment.no_speech_prob,
            segment.avg_logprob,
            segment.compression_ratio,
            no_speech_threshold,
        )
    ).strip()
    return text, info.language, info.language_probability


class MultilingualWhisperSTTService(WhisperSTTService):
    """faster-whisper with per-utterance language auto-detection (SPEC §7).

    The stock service transcribes with a fixed language; here each utterance
    is transcribed with `language=None` so Whisper detects it, clamped to
    TheraPy's supported set. The detected language rides on the
    TranscriptionFrame for downstream voice switching.
    """

    def __init__(
        self,
        recorder: Callable[[bytes, str, str], None] | None = None,
    ) -> None:
        _initialize_whisper_service(self)
        self.current_language: str = DEFAULT_LANGUAGE
        # (audio, text, language) → persisted user turn; this is the one spot
        # where the raw utterance and its transcript exist together (SPEC §8).
        self._recorder = recorder

    def _load(self) -> None:
        # One faster-whisper instance per process, not per connection: loading
        # takes tens of seconds on CPU (it blocked handshakes of connections
        # racing the load) and each copy costs ~1 GB. Only one pipeline is
        # live at a time (connection preemption), so sharing is safe.
        global _whisper_model
        if _whisper_model is None:
            import time as time_module

            from therapy.observability.telemetry import record_metric

            load_started = time_module.monotonic()
            try:
                super()._load()
            except Exception:
                record_metric(
                    "therapy_stt_model_load_seconds",
                    time_module.monotonic() - load_started,
                    {"outcome": "error"},
                )
                raise
            record_metric(
                "therapy_stt_model_load_seconds",
                time_module.monotonic() - load_started,
                {"outcome": "success"},
            )
            _whisper_model = self._model
        else:
            self._model = _whisper_model

    async def run_stt(self, audio: bytes):
        if not self._model:
            return
        import time as time_module

        from therapy.observability.telemetry import record_metric

        await self.start_processing_metrics()
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        # 16 kHz mono int16 — duration from sample count, never the audio.
        audio_seconds = len(audio_float) / 16_000.0
        record_metric("therapy_utterance_audio_seconds", audio_seconds)
        threshold = self._settings.no_speech_prob
        no_speech = threshold if isinstance(threshold, float) else 0.6

        decode_started = time_module.monotonic()
        text, detected, probability = await run_in_thread(
            WorkloadClass.REALTIME,
            _transcribe_utterance,
            self._model,
            audio_float,
            None,
            no_speech,
        )
        if audio_seconds > 0:
            record_metric(
                "therapy_stt_realtime_factor",
                (time_module.monotonic() - decode_started) / audio_seconds,
                {"outcome": "success"},
            )
        record_metric(
            "therapy_stt_language_probability",
            probability,
            {
                "language_group": detected
                if detected in ("es", "en", "pt")
                else "other"
            },
        )
        if probability <= 0.5:
            detected = None
        # Whisper's coarse language ID hears out-of-scope speech (e.g. German)
        # as es/en/pt inconsistently. A text detector names it correctly for
        # the tag; supported detections are left to Whisper. An out-of-scope
        # tag then keeps the reply in the conversation's language (below).
        label = label_language(text, fallback="")
        if label and not is_supported(label):
            detected = label
        if genuine_foreign_speech(detected, text):
            # The user really is speaking a language TheraPy doesn't (the
            # decode survived the plausibility filter) — transcribe it
            # honestly under its own tag. The conversation language and
            # reply choice stay put (TurnRelay skips unsupported turns).
            language = str(detected).lower().split("-")[0]
        else:
            if detected and not is_supported(detected):
                # Unsupported detection whose decode died in the filter —
                # a hallucination signature (repeated Korean stock phrases
                # in field testing), not a user switching languages.
                # Re-decode anchored to the conversation to recover it.
                logger.debug("stt_unsupported_detection_redecode")
                record_metric(
                    "therapy_stt_redecode_total",
                    1,
                    {"reason": "unsupported_detection"},
                )
                text, _, _ = await run_in_thread(
                    WorkloadClass.REALTIME,
                    _transcribe_utterance,
                    self._model,
                    audio_float,
                    self.current_language,
                    no_speech,
                )
                detected = self.current_language
            language = clamp_language(detected, self.current_language)
            self.current_language = language
        await self.stop_processing_metrics()

        record_metric(
            "therapy_vad_utterances_total",
            1,
            {"outcome": "speech" if text else "silence"},
        )
        if not text:
            record_metric("therapy_stt_empty_total", 1, {"reason": "no_speech"})
        if text:
            if self._recorder:
                await run_in_thread(
                    WorkloadClass.REALTIME, self._recorder, audio, text, language
                )
            try:
                frame_language = Language(language)
            except ValueError:
                frame_language = None
            yield TranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=frame_language,
            )


class TurnRelay(FrameProcessor):
    """Per-turn language + modality handling; user transcript relay.

    On each voice turn: choose the reply language (word-level dominant
    language of the phrase, or the user's pin — SPEC §7) and, if it changed,
    re-voice the TTS (voice + language) before the reply is synthesized;
    tell the LLM whether the reply should be spoken (modality mirroring,
    SPEC §5); always forward the transcript over the data channel so the
    client can render it. The transcript keeps Whisper's detected language —
    a pin constrains replies only, STT stays auto.
    """

    def __init__(
        self,
        modality: ReplyModality,
        reply_language: ReplyLanguage,
        memory_refresher: Callable[[str], Awaitable[str | None]] | None = None,
    ) -> None:
        _initialize_frame_processor(self)
        self._language: str | None = None
        self._modality = modality
        self._reply_language = reply_language
        self._memory_refresher = memory_refresher
        self._last_note_at: float = 0.0

    async def _note(self, content: str) -> None:
        self._last_note_at = time.monotonic()
        await self.push_frame(
            LLMMessagesAppendFrame(messages=[{"role": "system", "content": content}])
        )

    @property
    def language(self) -> str | None:
        return self._language

    def note_language(self, language: str) -> None:
        """Record a language change applied elsewhere (e.g. a typed turn)."""
        self._language = language

    async def _maybe_switch(self, language: str) -> bool:
        if language == self._language:
            return False
        self._language = language
        await self.push_frame(tts_settings_for(language))
        await self._note(language_switch_note(language))
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            if self._memory_refresher is not None:
                memory_note = await self._memory_refresher(frame.text)
                await self.push_frame(
                    LLMMessagesTransformFrame(
                        transform=lambda messages: _replace_pipecat_context(
                            messages, memory_note
                        ),
                        run_llm=False,
                    )
                )
            raw = frame.language.value if frame.language else None
            base = raw.lower().split("-")[0] if raw else None
            if base and not is_supported(base):
                # Genuine speech in a language TheraPy doesn't speak: the
                # transcript keeps its honest tag, but it must not steer
                # the reply language — lingua only knows es/en/pt and
                # would misread German words as one of them.
                language = base
                reply = self._reply_language.language
            else:
                language = clamp_language(base, self._language or DEFAULT_LANGUAGE)
                reply = self._reply_language.note_phrase(frame.text)
            switched = await self._maybe_switch(reply)
            # Anchor EVERY turn, not just switches: small local models drift
            # back to English mid-conversation (field test 2026-07-10) while
            # the tag and voice stay correct. Bursty fragments of one
            # aggregated turn share a single note (4 s dedupe).
            if not switched and time.monotonic() - self._last_note_at > 4.0:
                if self._reply_language.pinned and language != reply:
                    await self._note(language_pin_note(reply))
                else:
                    await self._note(reply_language_reminder(reply))
            # Spoken turn → spoken reply unless the user overrode the speaker.
            speak = self._modality.note_turn(VOICE)
            await self.push_frame(LLMConfigureOutputFrame(skip_tts=not speak))
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={
                        "type": "transcript",
                        "role": "user",
                        "modality": "voice",
                        "language": language,
                        "text": frame.text,
                    }
                )
            )
            _count_data_channel("transcript", "sent")
        await self.push_frame(frame, direction)


class BotTextRelay(FrameProcessor):
    """Accumulates the LLM reply; ships the full text to client and store."""

    def __init__(
        self,
        get_language: Callable[[], str],
        get_modality: Callable[[], str] = lambda: "unknown",
        recorder: Callable[[str], None] | None = None,
    ) -> None:
        _initialize_frame_processor(self)
        self._parts: list[str] = []
        self._get_language = get_language
        self._get_modality = get_modality
        self._recorder = recorder

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMTextFrame):
                self._parts.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame):
                text = "".join(self._parts).strip()
                self._parts = []
                from therapy.observability.model import (
                    LanguageGroup,
                    Modality,
                    normalize_enum,
                )
                from therapy.observability.telemetry import record_metric

                record_metric(
                    "therapy_conversation_turns_total",
                    1,
                    {
                        "modality": normalize_enum(
                            self._get_modality(), Modality, Modality.UNKNOWN
                        ).value,
                        "language_group": normalize_enum(
                            self._get_language(),
                            LanguageGroup,
                            LanguageGroup.UNKNOWN,
                        ).value,
                        "outcome": "success" if text else "incomplete",
                    },
                )
                if text:
                    if self._recorder:
                        await run_in_thread(
                            WorkloadClass.REALTIME, self._recorder, text
                        )
                    await self.push_frame(
                        OutputTransportMessageUrgentFrame(
                            message={
                                "type": "transcript",
                                "role": "assistant",
                                "language": self._get_language(),
                                "text": text,
                            }
                        )
                    )
                    _count_data_channel("transcript", "sent")
        await self.push_frame(frame, direction)


class DeterministicTestLLM(FrameProcessor):
    """Test-only Pipecat LLM boundary; exercises framing without a provider."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(
                LLMTextFrame(text="I’m with you. What feels most useful right now?")
            )
            await self.push_frame(LLMFullResponseEndFrame())
            return
        await self.push_frame(frame, direction)


class _DeterministicTestAudioInput(FrameProcessor):
    """Test-only STT/VAD boundary that consumes fake-browser microphone audio."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not isinstance(frame, InputAudioRawFrame):
            await self.push_frame(frame, direction)


class _DeterministicTestPassthrough(FrameProcessor):
    """Test-only lightweight processor preserving real Pipecat frame flow."""

    def __init__(self, language: str = DEFAULT_LANGUAGE) -> None:
        _initialize_frame_processor(self)
        self.current_language = language

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class InputAudioProbe(FrameProcessor):
    """Diagnostics (THERAPY_DEBUG_AUDIO=1): logs input level once per second.

    Confirms what the pipeline actually hears — separates transport problems
    from VAD/STT problems when a client claims to be speaking.
    """

    def __init__(self) -> None:
        _initialize_frame_processor(self)
        self._samples = 0
        self._sumsq = 0.0
        self._rate = 16_000

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            samples = np.frombuffer(frame.audio, dtype=np.int16)
            self._rate = frame.sample_rate
            self._samples += samples.size
            self._sumsq += float(np.sum(samples.astype(np.float64) ** 2))
            if self._samples >= self._rate:
                rms = (self._sumsq / self._samples) ** 0.5
                logger.debug(
                    "input_audio_rms",
                    extra={"rms": round(rms), "samples": self._samples},
                )
                self._samples = 0
                self._sumsq = 0.0
        await self.push_frame(frame, direction)


class TTFAMonitor(FrameProcessor):
    """Logs time from end-of-user-speech to first synthesized audio (R1)."""

    def __init__(self) -> None:
        _initialize_frame_processor(self)
        self._turn_started_at: float | None = None
        # First reply of a pipeline pays model warm-up; label it honestly
        # (plan O3.2/O3 audit: mode was hard-coded "warm").
        self._cold = True

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_started_at = time.monotonic()
        elif isinstance(frame, TTSAudioRawFrame) and self._turn_started_at is not None:
            ttfa = time.monotonic() - self._turn_started_at
            self._turn_started_at = None
            from therapy.observability.telemetry import record_metric

            record_metric(
                "therapy_turn_ttfa_seconds",
                ttfa,
                {
                    "provider": _env_provider_label(),
                    "mode": "cold" if self._cold else "warm",
                },
            )
            self._cold = False
            emit_event(
                "turn.ttfa",
                component="voice",
                operation="turn",
                outcome="success",
                duration_ms=ttfa * 1000,
            )
        await self.push_frame(frame, direction)


def _env_provider_label() -> str:
    from therapy.observability.model import Provider, normalize_enum

    return normalize_enum(
        os.environ.get("THERAPY_LLM", "anthropic"), Provider, Provider.UNKNOWN
    ).value


class PresenceRelay(FrameProcessor):
    """Pushes authoritative companion presence to the client (SPEC phase C).

    companion.js can *infer* presence, but the pipeline witnesses it exactly:
    the VAD opens and closes each user turn, and the output transport reports
    when the bot's audio starts and stops. Emitting it here makes the avatar
    track the real machine — above all 'thinking', the user-stopped→reply gap
    the client could only guess at.

    Placed just above ``transport.output()``: user speaking frames pass through
    on their way down, and the bot speaking frames the output emits travel back
    up through here. Emissions are deduped, and a dropped message self-heals on
    the next transition (the client falls back to inference if none arrive).
    """

    def __init__(self, modality: ReplyModality) -> None:
        _initialize_frame_processor(self)
        self._modality = modality
        self._state: str | None = None
        self._barge_started_at: float | None = None

    async def _emit(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        await self.push_frame(
            OutputTransportMessageUrgentFrame(message=presence_message(state))
        )
        _count_data_channel("presence", "sent")

    def _finish_barge_in(self, outcome: str) -> None:
        """Record a detected interruption once its audio-stop outcome is known."""
        if self._barge_started_at is None:
            return
        from therapy.observability.telemetry import record_metric

        record_metric("therapy_barge_ins_total", 1, {"outcome": outcome})
        record_metric(
            "therapy_barge_in_stop_seconds",
            time.monotonic() - self._barge_started_at,
            {"outcome": outcome},
        )
        self._barge_started_at = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            # The user (re)took the floor — barge-in included: back to listening.
            if self._state == "speaking":
                self._barge_started_at = time.monotonic()
            await self._emit("listening")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._emit("thinking")
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._finish_barge_in("timeout")
            await self._emit("speaking")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._finish_barge_in("success")
            await self._emit("listening")
        elif isinstance(frame, (CancelFrame, EndFrame)):
            self._finish_barge_in("error")
        elif isinstance(frame, LLMFullResponseEndFrame) and not self._modality.speak:
            # A silent (typed→text) reply produces no bot audio, so no
            # BotStopped will bring us back — return to listening here.
            await self._emit("listening")
        elif isinstance(frame, OutputTransportMessageUrgentFrame):
            message = _object_mapping(_runtime_value(frame.message))
            if (
                message is not None
                and message.get("type") == "transcript"
                and message.get("role") == "user"
                and message.get("modality") == "text"
            ):
                # A typed turn has no VAD frames; its echoed transcript marks thinking.
                await self._emit("thinking")
        await self.push_frame(frame, direction)


def make_llm_service():
    """Provider-agnostic LLM factory (SPEC §5): claude first, swappable.

    THERAPY_LLM selects the provider; THERAPY_LLM_MODEL overrides the model.
    `openrouter` and `ollama` share an OpenAI-compatible surface — the
    former gives hosted free models for dev, the latter fully-local models.
    """
    if os.environ.get("THERAPY_TEST_MODE") == "1":
        return DeterministicTestLLM()
    provider = os.environ.get("THERAPY_LLM", "anthropic")
    model = os.environ.get("THERAPY_LLM_MODEL")
    if provider == "anthropic":
        from pipecat.services.anthropic.llm import AnthropicLLMService

        return AnthropicLLMService(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            settings=AnthropicLLMService.Settings(model=model or "claude-opus-4-8"),
        )
    if provider == "openrouter":
        from pipecat.services.openrouter.llm import OpenRouterLLMService

        return OpenRouterLLMService(
            api_key=os.environ["OPENROUTER_API_KEY"],
            # Meta-route to whichever free model is currently available.
            settings=OpenRouterLLMService.Settings(model=model or "openrouter/free"),
        )
    if provider == "ollama":
        from pipecat.services.ollama.llm import OLLamaLLMService

        return OLLamaLLMService(
            # pedrolucas/smollm3:3b-q4_k_m — best es/en/pt quality that still streams fast enough
            # for voice on a CPU-only host; override via THERAPY_LLM_MODEL.
            settings=OLLamaLLMService.Settings(model=model or "pedrolucas/smollm3:3b-q4_k_m"),
            # In Docker, Ollama runs on the host: host.docker.internal.
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    raise ValueError(f"Unknown THERAPY_LLM provider: {provider!r}")


async def run_bot(
    webrtc_connection: SmallWebRTCConnection,
    *,
    new_session: bool = False,
    resume_session_id: str | None = None,
) -> None:
    """Build and run one conversation pipeline for a WebRTC connection.

    Args:
        webrtc_connection: The signaling-complete SmallWebRTC connection.
        new_session: Skip reconnect-resume and always open a fresh session
            (test scripts need isolated sessions; real clients resume).
        resume_session_id: Continue this specific session (history browser's
            explicit choice — no freshness window applies). Unknown ids fall
            back to the default behavior. Wins over new_session.
    """
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )
    test_mode = os.environ.get("THERAPY_TEST_MODE") == "1"
    # Pipecat ≥1.x: VAD is an explicit pipeline stage. `TransportParams`
    # still accepts a `vad_analyzer` but non-Daily transports ignore it —
    # without this processor no speech is ever detected.
    vad: FrameProcessor = (
        _DeterministicTestAudioInput()
        if test_mode
        else VADProcessor(vad_analyzer=SileroVADAnalyzer(params=vad_params()))
    )

    store = MemoryStore()
    # The property-graph user model shares the store's data dir (SPEC Appendix
    # A). v1 facts are migrated to observation nodes on init.
    user_model = UserModel()

    # A dropped connection is not a session boundary: reconnecting within
    # the resume window reopens the interrupted session instead of starting
    # an amnesiac new one (field test: a 4-minute WebRTC drop made the bot
    # deny the whole prior conversation).
    resumed_turns: list[RowDict] = []
    session_id = None
    if resume_session_id:
        if store.has_session(resume_session_id):
            session_id = resume_session_id
        else:
            emit_event(
                "voice.resume_unknown",
                severity=logging.WARNING,
                component="voice",
                operation="resume",
                outcome="rejected",
            )
    if session_id is None and not new_session:
        session_id = store.resume_candidate(resume_window_secs())
    if session_id:
        pending = _finalizers.pop(session_id, None)
        if pending is not None:
            pending.cancel()
        store.reopen_session(session_id)
        resumed_turns = store.session_turns(session_id)
        emit_event(
            "voice.session_resumed",
            component="voice",
            operation="resume",
            outcome="success",
            count=len(resumed_turns),
        )
    else:
        session_id = store.create_session()
    owner = live.claim(session_id)

    # Language state survives the reconnect too — pick up where the user
    # left off instead of snapping back to English.
    initial_language = DEFAULT_LANGUAGE
    for turn in reversed(resumed_turns):
        # Skip out-of-scope turns (e.g. a German aside tagged "de"): the reply
        # language resumes from the last turn actually in a supported language.
        if turn["role"] == "user" and is_supported(str(turn["language"])):
            initial_language = clamp_language(str(turn["language"]), DEFAULT_LANGUAGE)
            break

    modality = ReplyModality()
    # established: only a resumed session has real history behind `initial`
    # — a fresh one must adopt the user's first phrase (greeting) outright.
    reply_language = ReplyLanguage(
        initial=initial_language, established=bool(resumed_turns)
    )
    if test_mode:
        from therapy.acceptance import AcceptanceEmbedding

        context_assembler = ContextAssembler(
            user_model, store, embedder=AcceptanceEmbedding()
        )
    else:
        context_assembler = ContextAssembler(user_model, store)

    async def refresh_memory(text: str) -> str | None:
        turn_context = await run_in_thread(
            WorkloadClass.REALTIME, context_assembler.assemble, text, session_id
        )
        return turn_context["note"]

    def _persisted(artifact: str, outcome: str, size: int | None = None) -> None:
        """Bounded persistence-boundary evidence (plan O3.2): artifact class,
        outcome, and byte size only — never text, audio, or IDs."""
        from therapy.observability.telemetry import record_metric

        record_metric(
            "therapy_persist_total", 1, {"artifact": artifact, "outcome": outcome}
        )
        if size is not None:
            record_metric("therapy_persist_bytes", size, {"artifact": artifact})

    def record_user_voice(audio: bytes, text: str, language: str) -> None:
        try:
            store.add_turn(
                session_id, "user", VOICE, language, text,
                audio=audio, sample_rate=16_000,
            )
        except Exception:
            _persisted("audio", "error")
            raise
        _persisted("audio", "success", len(audio))
        _persisted("turn", "success", len(text.encode("utf-8")))
        # Freeform observation inbox (W2): distillation promotes it between
        # sessions. never_store is enforced inside add_observation.
        try:
            user_model.add_observation(text, session_id=session_id, language=language)
        except Exception:
            _persisted("observation", "error")
            raise
        _persisted("observation", "success")

    def record_assistant(text: str) -> None:
        try:
            store.add_turn(
                session_id,
                "assistant",
                VOICE if modality.speak else TEXT,
                reply_language.language,
                text,
            )
        except Exception:
            _persisted("turn", "error")
            raise
        _persisted("turn", "success", len(text.encode("utf-8")))

    stt = (
        _DeterministicTestPassthrough(initial_language)
        if test_mode
        else MultilingualWhisperSTTService(recorder=record_user_voice)
    )
    stt.current_language = initial_language
    turn_relay = TurnRelay(modality, reply_language, refresh_memory)
    if test_mode:
        tts: FrameProcessor = _DeterministicTestPassthrough()
    else:
        tts_load_started = time.monotonic()
        from therapy.observability.telemetry import record_metric

        try:
            tts = KokoroTTSService(
                settings=KokoroTTSSettings(
                    voice=voice_for(initial_language),
                    language=LANGUAGE_ENUM[initial_language],
                )
            )
        except Exception:
            record_metric(
                "therapy_tts_model_load_seconds",
                time.monotonic() - tts_load_started,
                {"outcome": "error"},
            )
            raise
        record_metric(
            "therapy_tts_model_load_seconds",
            time.monotonic() - tts_load_started,
            {"outcome": "success"},
        )
    llm = make_llm_service()

    # Continuity (SPEC §8): current conversation stays verbatim; past context
    # is bounded, semantic, role-separated, and refreshed on every turn. A resumed
    # session IS the current conversation — its turns are rehydrated
    # verbatim after the reconnect marker. (recent_summaries already
    # excludes it: reopen_session cleared its ended_at.)
    messages: list[LLMContextMessage] = [
        {"role": "system", "content": build_system_prompt()}
    ]
    # Graph-walk context assembly (W3): identity + preferences + the
    # never_initiate list, plus relevant confirmed/pattern claims. The opening
    # topic is the resumed conversation's last user turn, if any.
    opening_topic = next(
        (
            str(turn["text"])
            for turn in reversed(resumed_turns)
            if turn["role"] == "user" and turn.get("text")
        ),
        "",
    )
    memory_note = context_assembler.assemble(opening_topic, session_id)["note"]
    if memory_note:
        messages.append({"role": "system", "content": memory_note})
    if resumed_turns:
        messages.append({"role": "system", "content": resume_note()})
        messages.extend(
            cast(list[LLMContextMessage], rehydrate_messages(resumed_turns))
        )
    context = LLMContext(messages=messages)
    aggregators = LLMContextAggregatorPair(context)

    stages: list[FrameProcessor] = [transport.input()]
    if os.environ.get("THERAPY_DEBUG_AUDIO"):
        stages.append(InputAudioProbe())
    pipeline = Pipeline(
        stages
        + [
            vad,
            stt,
            turn_relay,
            aggregators.user(),
            llm,
            BotTextRelay(
                get_language=lambda: reply_language.language,
                get_modality=lambda: modality.last_input,
                recorder=record_assistant,
            ),
            tts,
            TTFAMonitor(),
            PresenceRelay(modality),
            transport.output(),
            aggregators.assistant(),
        ]
    )

    from therapy.integrations.pipecat.observability import build_task_telemetry

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
        ),
        **build_task_telemetry(llm, session_id=session_id),
    )

    def _count_app_message(type_class: str, outcome: str) -> None:
        """Finite accepted/rejected/queued evidence per message class (O3.2)."""
        from therapy.observability.telemetry import record_metric

        record_metric(
            "therapy_app_messages_total",
            1,
            {"type_class": type_class, "outcome": outcome},
        )

    @transport.event_handler("on_app_message")
    async def _on_app_message(
        transport_instance: SmallWebRTCTransport,
        message_value: object,
        sender: str,
    ) -> None:
        del transport_instance, sender
        message = _object_mapping(_runtime_value(message_value))
        if message is None:
            _count_app_message("invalid", "rejected")
            return

        # Client's channel just opened — send the server-truth chat state so
        # a reconnect or page reload renders the resumed transcript.
        if message.get("type") == "client_ready":
            state = session_state_message(
                session_id, bool(resumed_turns), resumed_turns
            )
            await worker.queue_frames([OutputTransportMessageUrgentFrame(message=state)])
            _count_data_channel("session", "replayed")
            _count_app_message("client_ready", "accepted")
            return

        # Speaker override from the client: true/false, or null for auto.
        if message.get("type") == "voice_replies":
            enabled = message.get("enabled")
            speak = modality.set_override(enabled if isinstance(enabled, bool) else None)
            await worker.queue_frames([LLMConfigureOutputFrame(skip_tts=not speak)])
            _count_app_message("voice_reply", "accepted")
            return

        # Reply-language pin from the client: es/en/pt, or null for auto
        # (SPEC §7). Re-sent on every connect from persisted client state.
        if message.get("type") == "reply_language":
            code = message.get("language")
            try:
                reply = reply_language.set_pin(code if isinstance(code, str) else None)
            except ValueError:
                emit_event(
                    "voice.reply_language_unsupported",
                    severity=logging.WARNING,
                    component="voice",
                    operation="reply_language",
                    outcome="rejected",
                )
                _count_app_message("reply_language", "rejected")
                return
            # Auto asserts nothing on replay; only a pin anchors (the effect
            # helper carries the reasoning). Otherwise the fresh-connect auto
            # default (en) primes the model to English before the user speaks.
            voice_language, note = reply_language_override_effect(
                code if isinstance(code, str) else None, reply, turn_relay.language
            )
            frames: list[Frame] = []
            if voice_language:
                turn_relay.note_language(voice_language)
                frames.append(tts_settings_for(voice_language))
            if note:
                frames.append(
                    LLMMessagesAppendFrame(
                        messages=[{"role": "system", "content": note}]
                    )
                )
            if frames:
                await worker.queue_frames(frames)
            _count_app_message("reply_language", "accepted")
            return

        # Typed turn from the data channel — same conversation, text modality.
        if message.get("type") != "user_text":
            _count_app_message("unknown", "rejected")
            return
        text = str(message.get("text", "")).strip()
        if not text:
            _count_app_message("text", "rejected")
            return
        # No audio to detect from: lingua's word-level majority picks the reply
        # language (es/en/pt); label_language names the turn honestly for the
        # shown/stored tag (German stays "de", not forced onto Portuguese).
        language = dominant_language(text, current=stt.current_language)
        label = label_language(text, fallback=language)
        reply = reply_language.note_phrase(text)
        # Echo the typed turn as a transcript so it renders with its label the
        # same way live as on replay — the client no longer draws it locally
        # (which left typed turns unlabeled).
        frames: list[Frame] = [
            OutputTransportMessageUrgentFrame(
                message={
                    "type": "transcript",
                    "role": "user",
                    "modality": "text",
                    "language": label,
                    "text": text,
                }
            )
        ]
        if reply != turn_relay.language:
            frames.append(tts_settings_for(reply))
            note = language_switch_note(reply)
        elif reply_language.pinned and language != reply:
            note = language_pin_note(reply)
        else:
            # Per-turn anchor, mirroring TurnRelay (small-model drift).
            note = reply_language_reminder(reply)
        frames.append(
            LLMMessagesAppendFrame(messages=[{"role": "system", "content": note}])
        )
        stt.current_language = language
        turn_relay.note_language(reply)
        await run_in_thread(
            WorkloadClass.REALTIME,
            store.add_turn,
            session_id,
            "user",
            TEXT,
            label,
            text,
        )
        await run_in_thread(
            WorkloadClass.REALTIME,
            user_model.add_observation, text, session_id=session_id, language=label
        )
        turn_memory = await refresh_memory(text)
        frames.append(
            LLMMessagesTransformFrame(
                transform=lambda messages: _replace_pipecat_context(
                    messages, turn_memory
                ),
                run_llm=False,
            )
        )
        # Typed turn → silent reply unless the user overrode the speaker.
        speak = modality.note_turn(TEXT)
        frames.append(LLMConfigureOutputFrame(skip_tts=not speak))
        frames.append(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": text}], run_llm=True
            )
        )
        # Queued at the pipeline source — frames must not be pushed from the
        # transport's event task directly.
        await worker.queue_frames(frames)
        _count_data_channel("transcript", "queued")
        _count_app_message("text", "queued")

    _ = _on_app_message

    finalized = False
    # Detached finalizers get a NEW linked trace root, never a multi-hour
    # parent span (obs plan O2.2); the link joins it to the voice connection.
    from therapy.observability.context import current_trace_context

    _finalize_parent = current_trace_context()

    async def finalize_session() -> None:
        """Summarize, distill, and close the session (SPEC §8) off the pipeline path."""
        nonlocal finalized
        if finalized:
            return
        from therapy.observability.telemetry import link_root

        with link_root(
            "session.finalize",
            component="voice",
            operation="finalize",
            parent_trace_id=_finalize_parent.trace_id,
            parent_span_id=_finalize_parent.span_id,
        ):
            await _finalize_session_inner()

    async def _finalize_session_inner() -> None:
        nonlocal finalized
        import time as time_module

        from therapy.observability.telemetry import record_metric

        # A reconnect may have resumed this session before this task ran
        # (preemption schedules finalize during cancellation, possibly after
        # the new pipeline already took ownership) — then it is no longer
        # ours to close.
        if not live.owns(session_id, owner):
            record_metric(
                "therapy_session_finalizations_total",
                1,
                {"artifact": "session", "outcome": "ownership_skip"},
            )
            return
        finalized = True
        finalize_started = time_module.monotonic()
        turns = await run_in_thread(
            WorkloadClass.BACKGROUND, store.session_turns, session_id
        )
        summary: str | None = None
        title: str | None = None
        recap: str | None = None
        if turns:
            if test_mode:
                from therapy.acceptance import ScriptedLLM

                scripted = ScriptedLLM()
                last_user_text = next(
                    (
                        str(turn["text"])
                        for turn in reversed(turns)
                        if turn["role"] == "user"
                    ),
                    "Conversation",
                )
                summary = f"The user said: {last_user_text}"
                await knowledge_distill.distill_session(
                    user_model,
                    turns,
                    session_id,
                    extractor=scripted.extract,
                    judger=scripted.judge,
                    extractor_version="pipecat-acceptance-v1",
                )
                recap = f"You reflected on: {last_user_text}"
                title = "Acceptance conversation"
            else:
                artifacts = await generate_session_artifacts(
                    turns, user_model, session_id
                )
                summary = artifacts.summary
                recap = artifacts.recap
                title = artifacts.title
        if title:
            # Fill-only: a user's rename (or an earlier generation) wins.
            await run_in_thread(
                WorkloadClass.BACKGROUND, store.ensure_title, session_id, title
            )
        if recap:
            await run_in_thread(
                WorkloadClass.BACKGROUND, store.ensure_recap, session_id, recap
            )
        await run_in_thread(
            WorkloadClass.BACKGROUND, store.end_session, session_id, summary
        )
        live.release(session_id, owner)
        record_metric(
            "therapy_session_finalization_seconds",
            time_module.monotonic() - finalize_started,
            {"outcome": "success"},
        )
        emit_event(
            "voice.session_closed",
            component="voice",
            operation="finalize",
            outcome="success",
            count=len(turns),
        )

    finalize_task: asyncio.Task[None] | None = None

    def schedule_finalize() -> None:
        """Start finalization, registered so a resuming reconnect can cancel it.

        Scheduled at most once: a normal disconnect reaches both call sites
        (the event handler, then the runner's finally), and a duplicate
        registration would shadow the real task in _finalizers — a resume
        would then cancel the no-op while the original closed the session
        out from under the new pipeline.
        """
        nonlocal finalize_task
        if finalize_task is not None:
            return
        finalize_task = asyncio.create_task(finalize_session())
        _finalizers[session_id] = finalize_task
        _record_finalizers_pending()

        def _forget(done: asyncio.Task[None]) -> None:
            from therapy.observability.telemetry import record_metric

            cancelled = False
            try:
                failure = done.exception()
            except asyncio.CancelledError:
                failure = None
                cancelled = True
            if cancelled:
                # Cancellation-on-resume is a normal, visible outcome (O3.2).
                record_metric(
                    "therapy_session_finalizations_total",
                    1,
                    {"artifact": "session", "outcome": "cancelled"},
                )
            if failure is not None:
                emit_event(
                    "finalizer.failed",
                    severity=logging.ERROR,
                    component="voice",
                    operation="finalize",
                    outcome="error",
                    error_type=type(failure).__name__,
                )
            live.release(session_id, owner)
            if _finalizers.get(session_id) is done:
                del _finalizers[session_id]
            _record_finalizers_pending()

        finalize_task.add_done_callback(_forget)

    @transport.event_handler("on_client_disconnected")
    async def _on_client_disconnected(
        transport_instance: SmallWebRTCTransport,
        connection: SmallWebRTCConnection,
    ) -> None:
        del transport_instance, connection
        schedule_finalize()
        await worker.cancel()

    _ = _on_client_disconnected

    runner = WorkerRunner(handle_sigint=False)
    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        # A preempted pipeline (new connection cancelled this one) never sees
        # on_client_disconnected — its session must still close and summarize.
        schedule_finalize()
