"""Pipeline assembly — the only module that imports the voice-agent framework.

Framework: Pipecat with SmallWebRTCTransport (docs/framework-spike.md).
Everything domain-shaped (prompts, language tables, voice maps) lives in
framework-free modules; a framework swap — or a reversal of the spike
verdict — touches this file and the client transport layer only (SPEC §5).

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

import asyncio
import os
import time
from typing import Any

import numpy as np
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMConfigureOutputFrame,
    LLMFullResponseEndFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSUpdateSettingsFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.kokoro.tts import KokoroTTSService, KokoroTTSSettings
from pipecat.services.whisper.stt import WhisperSTTService, WhisperSTTSettings
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
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
    continuity_note,
    language_pin_note,
    language_switch_note,
    rehydrate_messages,
    reply_language_reminder,
    resume_note,
)
from therapy.memory import MemoryStore, make_summarizer
from therapy.memory.distill import distill_facts
from therapy.memory.store import resume_window_secs
from therapy.memory.summarizer import entitle
from therapy.perception.stt import (
    DEFAULT_LANGUAGE,
    clamp_language,
    genuine_foreign_speech,
    is_supported,
    plausible_segment,
    whisper_model,
)
from therapy.server import live
from therapy.server.protocol import session_state_message
from therapy.speech.tts import voice_for

LANGUAGE_ENUM = {"en": Language.EN, "es": Language.ES, "pt": Language.PT}

# Shared across connections — see MultilingualWhisperSTTService._load.
_whisper_model: Any = None

# Reconnect-resume coordination (SPEC §8). A dropped connection schedules
# finalization, but the same client usually reconnects seconds later — the
# new pipeline must be able to cancel an in-flight finalize (_finalizers)
# and, for one scheduled after it already resumed, make it a no-op
# (live.py ownership tokens; only the current owner may close a session).
_finalizers: dict[str, asyncio.Task[None]] = {}


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
    return TTSUpdateSettingsFrame(
        delta=KokoroTTSSettings(
            voice=voice_for(language),
            language=LANGUAGE_ENUM[language],
        )
    )


def _transcribe_utterance(
    model: Any,
    audio_float: Any,
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

    def __init__(self, recorder: Any = None, **kwargs: Any) -> None:
        super().__init__(
            settings=WhisperSTTSettings(model=whisper_model(), language=Language.EN),
            **kwargs,
        )
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
            super()._load()
            _whisper_model = self._model
        else:
            self._model = _whisper_model

    async def run_stt(self, audio: bytes):
        if not self._model:
            return
        await self.start_processing_metrics()
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        threshold = self._settings.no_speech_prob
        no_speech = threshold if isinstance(threshold, float) else 0.6

        text, detected, probability = await asyncio.to_thread(
            _transcribe_utterance, self._model, audio_float, None, no_speech
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
                logger.debug(
                    f"Unsupported detection {detected!r}; re-decoding as "
                    f"{self.current_language}"
                )
                text, _, _ = await asyncio.to_thread(
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

        if text:
            logger.debug(f"Transcription [{language}]: {text}")
            if self._recorder:
                await asyncio.to_thread(self._recorder, audio, text, language)
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

    def __init__(self, modality: ReplyModality, reply_language: ReplyLanguage) -> None:
        super().__init__()
        self._language: str | None = None
        self._modality = modality
        self._reply_language = reply_language
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
        await self.push_frame(frame, direction)


class BotTextRelay(FrameProcessor):
    """Accumulates the LLM reply; ships the full text to client and store."""

    def __init__(self, get_language, recorder: Any = None) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._get_language = get_language
        self._recorder = recorder

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMTextFrame):
                self._parts.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame) and self._parts:
                text = "".join(self._parts).strip()
                self._parts = []
                if text:
                    if self._recorder:
                        await asyncio.to_thread(self._recorder, text)
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
        await self.push_frame(frame, direction)


class InputAudioProbe(FrameProcessor):
    """Diagnostics (THERAPY_DEBUG_AUDIO=1): logs input level once per second.

    Confirms what the pipeline actually hears — separates transport problems
    from VAD/STT problems when a client claims to be speaking.
    """

    def __init__(self) -> None:
        super().__init__()
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
                logger.info(f"Input audio: rms={rms:.0f} over {self._samples} samples @{self._rate}Hz")
                self._samples = 0
                self._sumsq = 0.0
        await self.push_frame(frame, direction)


class TTFAMonitor(FrameProcessor):
    """Logs time from end-of-user-speech to first synthesized audio (R1)."""

    def __init__(self) -> None:
        super().__init__()
        self._turn_started_at: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_started_at = time.monotonic()
        elif isinstance(frame, TTSAudioRawFrame) and self._turn_started_at is not None:
            ttfa = time.monotonic() - self._turn_started_at
            self._turn_started_at = None
            logger.info(f"TTFA: {ttfa:.2f}s (user stopped speaking → first audio)")
        await self.push_frame(frame, direction)


def make_llm_service():
    """Provider-agnostic LLM factory (SPEC §5): claude first, swappable.

    THERAPY_LLM selects the provider; THERAPY_LLM_MODEL overrides the model.
    `openrouter` and `ollama` share an OpenAI-compatible surface — the
    former gives hosted free models for dev, the latter fully-local models.
    """
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
            # gemma3:4b — best es/en/pt quality that still streams fast enough
            # for voice on a CPU-only host; override via THERAPY_LLM_MODEL.
            settings=OLLamaLLMService.Settings(model=model or "gemma3:4b"),
            # In Docker, Ollama runs on the host: host.docker.internal.
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    raise ValueError(f"Unknown THERAPY_LLM provider: {provider!r}")


async def run_bot(
    webrtc_connection: Any,
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
    # Pipecat ≥1.x: VAD is an explicit pipeline stage. `TransportParams`
    # still accepts a `vad_analyzer` but non-Daily transports ignore it —
    # without this processor no speech is ever detected.
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=vad_params()))

    store = MemoryStore()

    # A dropped connection is not a session boundary: reconnecting within
    # the resume window reopens the interrupted session instead of starting
    # an amnesiac new one (field test: a 4-minute WebRTC drop made the bot
    # deny the whole prior conversation).
    resumed_turns: list[dict] = []
    session_id = None
    if resume_session_id:
        if store.has_session(resume_session_id):
            session_id = resume_session_id
        else:
            logger.warning(f"Cannot resume unknown session {resume_session_id!r}")
    if session_id is None and not new_session:
        session_id = store.resume_candidate(resume_window_secs())
    if session_id:
        pending = _finalizers.pop(session_id, None)
        if pending is not None:
            pending.cancel()
        store.reopen_session(session_id)
        resumed_turns = store.session_turns(session_id)
        logger.info(f"Resuming session {session_id} ({len(resumed_turns)} turns)")
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

    def record_user_voice(audio: bytes, text: str, language: str) -> None:
        store.add_turn(
            session_id, "user", VOICE, language, text, audio=audio, sample_rate=16_000
        )

    def record_assistant(text: str) -> None:
        store.add_turn(
            session_id,
            "assistant",
            VOICE if modality.speak else TEXT,
            reply_language.language,
            text,
        )

    stt = MultilingualWhisperSTTService(recorder=record_user_voice)
    stt.current_language = initial_language
    turn_relay = TurnRelay(modality, reply_language)
    tts = KokoroTTSService(
        settings=KokoroTTSSettings(
            voice=voice_for(initial_language),
            language=LANGUAGE_ENUM[initial_language],
        )
    )
    llm = make_llm_service()

    # Continuity (SPEC §8): the current conversation goes verbatim; the past
    # arrives only as distilled summaries + the user model. A resumed
    # session IS the current conversation — its turns are rehydrated
    # verbatim after the reconnect marker. (recent_summaries already
    # excludes it: reopen_session cleared its ended_at.)
    messages = [{"role": "system", "content": build_system_prompt()}]
    memory_note = continuity_note(store.recent_summaries(), store.facts())
    if memory_note:
        messages.append({"role": "system", "content": memory_note})
    if resumed_turns:
        messages.append({"role": "system", "content": resume_note()})
        messages.extend(rehydrate_messages(resumed_turns))
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
                recorder=record_assistant,
            ),
            tts,
            TTFAMonitor(),
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,  # barge-in
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_app_message")
    async def on_app_message(transport: Any, message: Any, sender: str) -> None:
        if not isinstance(message, dict):
            return

        # Client's channel just opened — send the server-truth chat state so
        # a reconnect or page reload renders the resumed transcript.
        if message.get("type") == "client_ready":
            state = session_state_message(
                session_id, bool(resumed_turns), resumed_turns
            )
            await task.queue_frames([OutputTransportMessageUrgentFrame(message=state)])
            return

        # Speaker override from the client: true/false, or null for auto.
        if message.get("type") == "voice_replies":
            enabled = message.get("enabled")
            speak = modality.set_override(enabled if isinstance(enabled, bool) else None)
            await task.queue_frames([LLMConfigureOutputFrame(skip_tts=not speak)])
            return

        # Reply-language pin from the client: es/en/pt, or null for auto
        # (SPEC §7). Re-sent on every connect from persisted client state.
        if message.get("type") == "reply_language":
            code = message.get("language")
            try:
                reply = reply_language.set_pin(code if isinstance(code, str) else None)
            except ValueError:
                logger.warning(f"Ignoring unsupported reply_language: {code!r}")
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
                await task.queue_frames(frames)
            return

        # Typed turn from the data channel — same conversation, text modality.
        if message.get("type") != "user_text":
            return
        text = str(message.get("text", "")).strip()
        if not text:
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
        await asyncio.to_thread(store.add_turn, session_id, "user", TEXT, label, text)
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
        await task.queue_frames(frames)

    finalized = False

    async def finalize_session() -> None:
        """Summarize, distill, and close the session (SPEC §8) off the pipeline path."""
        nonlocal finalized
        if finalized:
            return
        # A reconnect may have resumed this session before this task ran
        # (preemption schedules finalize during cancellation, possibly after
        # the new pipeline already took ownership) — then it is no longer
        # ours to close.
        if not live.owns(session_id, owner):
            return
        finalized = True
        turns = await asyncio.to_thread(store.session_turns, session_id)
        summary: str | None = None
        facts: list[str] = []
        title: str | None = None
        if turns:
            try:
                summary = (await make_summarizer().summarize(turns)).strip() or None
                facts = await distill_facts(turns)
                title = await entitle(turns)
            except Exception as exc:  # LLM down ≠ lost session; keep the turns.
                logger.warning(f"Session distillation failed for {session_id}: {exc}")
        for statement in facts:
            await asyncio.to_thread(store.upsert_fact, statement)
        if title:
            # Fill-only: a user's rename (or an earlier generation) wins.
            await asyncio.to_thread(store.ensure_title, session_id, title)
        await asyncio.to_thread(store.end_session, session_id, summary)
        live.release(session_id, owner)
        logger.info(
            f"Session {session_id} closed "
            f"({len(turns)} turns, summary={bool(summary)}, facts={len(facts)})"
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

        def _forget(done: asyncio.Task[None]) -> None:
            if _finalizers.get(session_id) is done:
                del _finalizers[session_id]

        finalize_task.add_done_callback(_forget)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport: Any, connection: Any) -> None:
        schedule_finalize()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    finally:
        # A preempted pipeline (new connection cancelled this one) never sees
        # on_client_disconnected — its session must still close and summarize.
        schedule_finalize()
