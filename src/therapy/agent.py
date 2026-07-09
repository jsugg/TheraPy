"""Pipeline assembly — the only module that imports the voice-agent framework.

Framework: Pipecat with SmallWebRTCTransport (docs/framework-spike.md).
Everything domain-shaped (prompts, language tables, voice maps) lives in
framework-free modules; a framework swap — or a reversal of the spike
verdict — touches this file and the client transport layer only (SPEC §5).

Pipeline per connection:

    transport.input (WebRTC + Silero VAD)
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
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.kokoro.tts import KokoroTTSService, KokoroTTSSettings
from pipecat.services.whisper.stt import WhisperSTTService, WhisperSTTSettings
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.utils.time import time_now_iso8601

from therapy.dialogue.policy import build_system_prompt
from therapy.perception.stt import (
    DEFAULT_LANGUAGE,
    clamp_language,
    guess_language,
    whisper_model,
)
from therapy.speech.tts import voice_for

LANGUAGE_ENUM = {"en": Language.EN, "es": Language.ES, "pt": Language.PT}


def tts_settings_for(language: str) -> TTSUpdateSettingsFrame:
    """TTS voice+language switch frame for a supported language code."""
    return TTSUpdateSettingsFrame(
        settings={
            "voice": voice_for(language),
            "language": LANGUAGE_ENUM[language],
        }
    )


class MultilingualWhisperSTTService(WhisperSTTService):
    """faster-whisper with per-utterance language auto-detection (SPEC §7).

    The stock service transcribes with a fixed language; here each utterance
    is transcribed with `language=None` so Whisper detects it, clamped to
    TheraPy's supported set. The detected language rides on the
    TranscriptionFrame for downstream voice switching.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            settings=WhisperSTTSettings(model=whisper_model(), language=Language.EN),
            **kwargs,
        )
        self.current_language: str = DEFAULT_LANGUAGE

    async def run_stt(self, audio: bytes):
        if not self._model:
            return
        await self.start_processing_metrics()
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = await asyncio.to_thread(
            self._model.transcribe, audio_float, language=None
        )
        detected = info.language if info.language_probability > 0.5 else None
        language = clamp_language(detected, self.current_language)
        self.current_language = language

        threshold = self._settings.no_speech_prob
        text = " ".join(
            segment.text.strip()
            for segment in segments
            if not isinstance(threshold, float) or segment.no_speech_prob < threshold
        ).strip()
        await self.stop_processing_metrics()

        if text:
            logger.debug(f"Transcription [{language}]: {text}")
            yield TranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=LANGUAGE_ENUM[language],
            )


class TurnRelay(FrameProcessor):
    """Per-turn language handling + user transcript relay to the client.

    On each user turn: if the utterance language changed, re-voice the TTS
    (voice + language) before the reply is synthesized; always forward the
    transcript over the data channel so the client can render it.
    """

    def __init__(self) -> None:
        super().__init__()
        self._language: str | None = None

    @property
    def language(self) -> str | None:
        return self._language

    def note_language(self, language: str) -> None:
        """Record a language change applied elsewhere (e.g. a typed turn)."""
        self._language = language

    async def _maybe_switch(self, language: str) -> None:
        if language != self._language:
            self._language = language
            await self.push_frame(tts_settings_for(language))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
            language = clamp_language(
                frame.language.value if frame.language else None,
                self._language or DEFAULT_LANGUAGE,
            )
            await self._maybe_switch(language)
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
    """Accumulates the LLM reply and ships the full text to the client."""

    def __init__(self, get_language) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._get_language = get_language

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMTextFrame):
                self._parts.append(frame.text)
            elif isinstance(frame, LLMFullResponseEndFrame) and self._parts:
                text = "".join(self._parts).strip()
                self._parts = []
                if text:
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
            model=model or "claude-opus-4-8",
        )
    if provider == "openrouter":
        from pipecat.services.openrouter.llm import OpenRouterLLMService

        return OpenRouterLLMService(
            api_key=os.environ["OPENROUTER_API_KEY"],
            # Meta-route to whichever free model is currently available.
            model=model or "openrouter/free",
        )
    if provider == "ollama":
        from pipecat.services.ollama.llm import OLLamaLLMService

        return OLLamaLLMService(
            model=model or "llama3.2",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    raise ValueError(f"Unknown THERAPY_LLM provider: {provider!r}")


async def run_bot(webrtc_connection: Any) -> None:
    """Build and run one conversation pipeline for a WebRTC connection."""
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.7)),
        ),
    )

    stt = MultilingualWhisperSTTService()
    turn_relay = TurnRelay()
    tts = KokoroTTSService(
        settings=KokoroTTSSettings(
            voice=voice_for(DEFAULT_LANGUAGE),
            language=LANGUAGE_ENUM[DEFAULT_LANGUAGE],
        )
    )
    llm = make_llm_service()

    context = LLMContext(
        messages=[{"role": "system", "content": build_system_prompt()}]
    )
    aggregators = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            turn_relay,
            aggregators.user(),
            llm,
            BotTextRelay(get_language=lambda: stt.current_language),
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
        # Typed turn from the data channel — same conversation, text modality.
        if not (isinstance(message, dict) and message.get("type") == "user_text"):
            return
        text = str(message.get("text", "")).strip()
        if not text:
            return
        language = guess_language(text, fallback=stt.current_language)
        frames: list[Frame] = []
        if language != turn_relay.language:
            frames.append(tts_settings_for(language))
        stt.current_language = language
        turn_relay.note_language(language)
        frames.append(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": text}], run_llm=True
            )
        )
        # Queued at the pipeline source — frames must not be pushed from the
        # transport's event task directly.
        await task.queue_frames(frames)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport: Any, connection: Any) -> None:
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
