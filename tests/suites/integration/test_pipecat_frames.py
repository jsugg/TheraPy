"""Frame-processor harness for the live voice pipeline (P0 wave 1).

The pipeline's own processors are pure and injectable: each `process_frame`
is driven directly with `push_frame` monkeypatched to a collector, so the
field-hardened logic (4 s language-anchor dedupe, unsupported-language honesty,
modality mirroring, barge-in, time-to-first-audio) is pinned in-suite instead
of only by out-of-band scripts and manual phone tests. Container-only: the
`pipecat` realtime stack has no macOS x86_64 wheels, so this skips on the host.
"""

from __future__ import annotations

import asyncio
import threading
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, cast

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

pytest.importorskip("pipecat")

from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMConfigureOutputFrame,
    LLMFullResponseEndFrame,
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
from pipecat.pipeline.runner import WorkerRunner  # noqa: E402
from pipecat.pipeline.worker import PipelineWorker  # noqa: E402
from pipecat.processors.frame_processor import (  # noqa: E402
    FrameDirection,
    FrameProcessor,
)
from pipecat.services.kokoro.tts import KokoroTTSSettings  # noqa: E402
from pipecat.services.whisper.stt import WhisperSTTService  # noqa: E402
from pipecat.transcriptions.language import Language  # noqa: E402
from pipecat.transports.smallwebrtc.connection import (
    SmallWebRTCConnection,  # noqa: E402
)

from therapy.dialogue.language_choice import ReplyLanguage  # noqa: E402
from therapy.dialogue.modality import ReplyModality  # noqa: E402
from therapy.dialogue.policy import (  # noqa: E402
    language_pin_note,
    language_switch_note,
    reply_language_reminder,
)
from therapy.integrations.pipecat import pipeline  # noqa: E402
from therapy.integrations.pipecat.pipeline import (  # noqa: E402
    BotTextRelay,
    InputAudioProbe,
    MultilingualWhisperSTTService,
    PresenceRelay,
    TTFAMonitor,
    TurnRelay,
    tts_settings_for,
    vad_params,
)
from therapy.server.protocol import presence_message  # noqa: E402

DOWN = FrameDirection.DOWNSTREAM

# --- harness ---------------------------------------------------------------

type Pushed = list[tuple[Frame, FrameDirection]]


def _capture(monkeypatch: pytest.MonkeyPatch, processor: FrameProcessor) -> Pushed:
    """Replace `push_frame` with a collector so `process_frame` can be driven
    in isolation (the base processor's own `process_frame` stays real)."""
    pushed: Pushed = []

    async def _push(frame: Frame, direction: FrameDirection = DOWN) -> None:
        pushed.append((frame, direction))

    monkeypatch.setattr(processor, "push_frame", _push)
    return pushed


def _of_type[T: Frame](pushed: Pushed, cls: type[T]) -> list[T]:
    return [frame for frame, _ in pushed if isinstance(frame, cls)]


def _system_notes(pushed: Pushed) -> list[str]:
    """Content strings of every `LLMMessagesAppendFrame` pushed (system notes)."""
    notes: list[str] = []
    for frame in _of_type(pushed, LLMMessagesAppendFrame):
        for message in frame.messages:
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    notes.append(content)
    return notes


def _messages(pushed: Pushed) -> list[object]:
    """`.message` payload of every urgent data-channel frame pushed."""
    return [frame.message for frame in _of_type(pushed, OutputTransportMessageUrgentFrame)]


def _transcription(text: str, language: Language) -> TranscriptionFrame:
    return TranscriptionFrame(
        text=text, user_id="user", timestamp="2026-07-17T00:00:00Z", language=language
    )


class _Clock:
    """Deterministic monotonic clock for the 4 s dedupe window."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


type MetricCall = tuple[str, float, dict[str, str]]


@pytest.fixture
def metrics(monkeypatch: pytest.MonkeyPatch) -> list[MetricCall]:
    """Spy every `record_metric` call. Processors import it lazily inside the
    call, so patching the telemetry attribute is observed."""
    calls: list[MetricCall] = []

    def _spy(name: str, value: float, attributes: dict[str, str] | None = None) -> None:
        calls.append((name, value, attributes or {}))

    monkeypatch.setattr("therapy.observability.telemetry.record_metric", _spy)
    return calls


def _named(calls: list[MetricCall], name: str) -> list[dict[str, str]]:
    return [attributes for metric_name, _, attributes in calls if metric_name == name]


# --- T3.1 TurnRelay --------------------------------------------------------


def test_turnrelay_supported_switch_emits_tts_and_switch_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = TurnRelay(ReplyModality(), ReplyLanguage())
    pushed = _capture(monkeypatch, relay)
    asyncio.run(
        relay.process_frame(
            _transcription("Hola, hoy me siento cansado y con poca energía.", Language.ES),
            DOWN,
        )
    )
    assert _of_type(pushed, TTSUpdateSettingsFrame), "supported switch re-voices TTS"
    assert language_switch_note("es") in _system_notes(pushed)


def test_turnrelay_unsupported_language_keeps_tag_without_steering_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply_language = ReplyLanguage()  # auto, defaults to "en"
    relay = TurnRelay(ReplyModality(), reply_language)
    relay.note_language("en")  # already established at en → no switch on this turn
    pushed = _capture(monkeypatch, relay)
    asyncio.run(
        relay.process_frame(_transcription("Ich fühle mich heute sehr müde.", Language.DE), DOWN)
    )
    transcript = _of_type(pushed, OutputTransportMessageUrgentFrame)[0].message
    assert isinstance(transcript, Mapping)
    assert transcript["language"] == "de"  # honest tag kept
    assert reply_language.language == "en"  # 'de' did not steer the reply
    assert not _of_type(pushed, TTSUpdateSettingsFrame)  # no re-voice to unsupported lang


def test_turnrelay_pinned_reply_language_emits_pin_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply_language = ReplyLanguage()
    reply_language.set_pin("en")
    relay = TurnRelay(ReplyModality(), reply_language)
    relay.note_language("en")  # reply pins en and we are at en → no switch
    pushed = _capture(monkeypatch, relay)
    asyncio.run(
        relay.process_frame(_transcription("Hola, me siento cansado hoy.", Language.ES), DOWN)
    )
    assert language_pin_note("en") in _system_notes(pushed)


def test_turnrelay_reply_reminder_anchors_after_4s_and_dedupes_within(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock(100.0)
    monkeypatch.setattr(pipeline, "time", types.SimpleNamespace(monotonic=clock))
    relay = TurnRelay(ReplyModality(), ReplyLanguage())
    relay.note_language("en")  # established → English input causes no switch
    pushed = _capture(monkeypatch, relay)
    frame = _transcription("I feel pretty tired today and low on energy.", Language.EN)

    # last_note_at=0, clock=100 → 100 s elapsed > 4 s → anchor emitted.
    asyncio.run(relay.process_frame(frame, DOWN))
    assert reply_language_reminder("en") in _system_notes(pushed)

    # +2 s (within the 4 s window) → suppressed (the field-test dedupe).
    clock.value = 102.0
    pushed.clear()
    asyncio.run(relay.process_frame(frame, DOWN))
    assert not _system_notes(pushed)

    # +7 s from the last note → anchored again.
    clock.value = 107.0
    pushed.clear()
    asyncio.run(relay.process_frame(frame, DOWN))
    assert reply_language_reminder("en") in _system_notes(pushed)


@pytest.mark.parametrize(("override", "expected_skip"), [(None, False), (False, True)])
def test_turnrelay_mirrors_modality_into_skip_tts(
    monkeypatch: pytest.MonkeyPatch, override: bool | None, expected_skip: bool
) -> None:
    modality = ReplyModality()
    if override is not None:
        modality.set_override(override)
    relay = TurnRelay(modality, ReplyLanguage())
    relay.note_language("en")
    pushed = _capture(monkeypatch, relay)
    asyncio.run(relay.process_frame(_transcription("I feel tired today.", Language.EN), DOWN))
    config = _of_type(pushed, LLMConfigureOutputFrame)[0]
    assert config.skip_tts is expected_skip


def test_turnrelay_relays_transcript_and_counts_data_channel(
    monkeypatch: pytest.MonkeyPatch, metrics: list[MetricCall]
) -> None:
    relay = TurnRelay(ReplyModality(), ReplyLanguage())
    relay.note_language("en")
    pushed = _capture(monkeypatch, relay)
    asyncio.run(relay.process_frame(_transcription("I feel tired today.", Language.EN), DOWN))
    transcript = _of_type(pushed, OutputTransportMessageUrgentFrame)[0].message
    assert transcript == {
        "type": "transcript",
        "role": "user",
        "modality": "voice",
        "language": "en",
        "text": "I feel tired today.",
    }
    assert {"type_class": "transcript", "outcome": "sent"} in _named(
        metrics, "therapy_data_channel_messages_total"
    )


def test_turnrelay_memory_refresher_pushes_transform_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    async def refresher(text: str) -> str | None:
        seen.append(text)
        return "a bounded memory note"

    relay = TurnRelay(ReplyModality(), ReplyLanguage(), memory_refresher=refresher)
    relay.note_language("en")
    pushed = _capture(monkeypatch, relay)
    asyncio.run(relay.process_frame(_transcription("I feel tired today.", Language.EN), DOWN))
    assert seen == ["I feel tired today."]
    assert _of_type(pushed, LLMMessagesTransformFrame)


# --- T3.2 PresenceRelay ----------------------------------------------------


def test_presencerelay_emits_listening_thinking_speaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = PresenceRelay(ReplyModality())
    pushed = _capture(monkeypatch, relay)
    asyncio.run(relay.process_frame(UserStartedSpeakingFrame(), DOWN))
    asyncio.run(relay.process_frame(UserStoppedSpeakingFrame(), DOWN))
    asyncio.run(relay.process_frame(BotStartedSpeakingFrame(), DOWN))
    states = _messages(pushed)
    for expected in ("listening", "thinking", "speaking"):
        assert presence_message(expected) in states


def test_presencerelay_barge_in_records_success_outcome(
    monkeypatch: pytest.MonkeyPatch, metrics: list[MetricCall]
) -> None:
    relay = PresenceRelay(ReplyModality())
    _capture(monkeypatch, relay)
    asyncio.run(relay.process_frame(BotStartedSpeakingFrame(), DOWN))  # state → speaking
    asyncio.run(relay.process_frame(UserStartedSpeakingFrame(), DOWN))  # barge begins
    asyncio.run(relay.process_frame(BotStoppedSpeakingFrame(), DOWN))  # barge resolved
    assert {"outcome": "success"} in _named(metrics, "therapy_barge_ins_total")


def test_presencerelay_barge_in_records_error_on_cancel(
    monkeypatch: pytest.MonkeyPatch, metrics: list[MetricCall]
) -> None:
    relay = PresenceRelay(ReplyModality())
    _capture(monkeypatch, relay)
    asyncio.run(relay.process_frame(BotStartedSpeakingFrame(), DOWN))
    asyncio.run(relay.process_frame(UserStartedSpeakingFrame(), DOWN))
    asyncio.run(relay.process_frame(CancelFrame(), DOWN))
    assert _named(metrics, "therapy_barge_ins_total") == [{"outcome": "error"}]


def test_presencerelay_silent_reply_returns_to_listening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modality = ReplyModality()
    modality.set_override(False)  # typed → silent reply, no bot audio
    relay = PresenceRelay(modality)
    pushed = _capture(monkeypatch, relay)
    asyncio.run(relay.process_frame(UserStoppedSpeakingFrame(), DOWN))  # thinking
    pushed.clear()
    asyncio.run(relay.process_frame(LLMFullResponseEndFrame(), DOWN))
    assert presence_message("listening") in _messages(pushed)


def test_presencerelay_typed_turn_transcript_marks_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay = PresenceRelay(ReplyModality())
    pushed = _capture(monkeypatch, relay)
    typed = OutputTransportMessageUrgentFrame(
        message={"type": "transcript", "role": "user", "modality": "text", "text": "hi"}
    )
    asyncio.run(relay.process_frame(typed, DOWN))
    assert presence_message("thinking") in _messages(pushed)


# --- T3.3 TTFAMonitor / BotTextRelay / InputAudioProbe ---------------------


def test_ttfa_monitor_records_ttfa_exactly_once_per_turn(
    monkeypatch: pytest.MonkeyPatch, metrics: list[MetricCall]
) -> None:
    clock = _Clock(10.0)
    monkeypatch.setattr(pipeline, "time", types.SimpleNamespace(monotonic=clock))
    monitor = TTFAMonitor()
    _capture(monkeypatch, monitor)
    asyncio.run(monitor.process_frame(UserStoppedSpeakingFrame(), DOWN))  # turn starts
    clock.value = 10.5
    audio = TTSAudioRawFrame(audio=b"\x00\x00", sample_rate=16_000, num_channels=1)
    asyncio.run(monitor.process_frame(audio, DOWN))  # first audio → record once
    clock.value = 11.0
    asyncio.run(monitor.process_frame(audio, DOWN))  # second audio → no re-record
    ttfa = _named(metrics, "therapy_turn_ttfa_seconds")
    assert len(ttfa) == 1
    assert ttfa[0]["mode"] == "cold"  # first reply pays warm-up honestly


def test_bottextrelay_accumulates_and_ships_full_reply(
    monkeypatch: pytest.MonkeyPatch, metrics: list[MetricCall]
) -> None:
    recorded: list[str] = []
    relay = BotTextRelay(
        get_language=lambda: "es", get_modality=lambda: "voice", recorder=recorded.append
    )
    pushed = _capture(monkeypatch, relay)
    asyncio.run(relay.process_frame(LLMTextFrame(text="Hola "), DOWN))
    asyncio.run(relay.process_frame(LLMTextFrame(text="mundo"), DOWN))
    asyncio.run(relay.process_frame(LLMFullResponseEndFrame(), DOWN))
    assert recorded == ["Hola mundo"]
    assert _messages(pushed) == [
        {"type": "transcript", "role": "assistant", "language": "es", "text": "Hola mundo"}
    ]
    turns = _named(metrics, "therapy_conversation_turns_total")
    assert turns
    assert turns[0]["outcome"] == "success"


def test_bottextrelay_empty_reply_is_incomplete_and_unsent(
    monkeypatch: pytest.MonkeyPatch, metrics: list[MetricCall]
) -> None:
    recorded: list[str] = []
    relay = BotTextRelay(get_language=lambda: "en", recorder=recorded.append)
    pushed = _capture(monkeypatch, relay)
    asyncio.run(relay.process_frame(LLMFullResponseEndFrame(), DOWN))
    assert recorded == []
    assert not _of_type(pushed, OutputTransportMessageUrgentFrame)
    turns = _named(metrics, "therapy_conversation_turns_total")
    assert turns
    assert turns[0]["outcome"] == "incomplete"


def test_input_audio_probe_accumulates_and_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = InputAudioProbe()
    pushed = _capture(monkeypatch, probe)
    half_second = np.zeros(8_000, dtype=np.int16).tobytes()
    one_second = np.full(16_000, 500, dtype=np.int16).tobytes()
    asyncio.run(
        probe.process_frame(
            InputAudioRawFrame(audio=half_second, sample_rate=16_000, num_channels=1), DOWN
        )
    )
    asyncio.run(
        probe.process_frame(
            InputAudioRawFrame(audio=one_second, sample_rate=16_000, num_channels=1), DOWN
        )
    )
    # Both frames pass through untouched; the 1 s RMS-report branch is crossed.
    assert len(_of_type(pushed, InputAudioRawFrame)) == 2


# --- T3.4 run_bot assembly -------------------------------------------------


class _AppMessageHandler(Protocol):
    async def __call__(
        self, transport: object, message: object, sender: str
    ) -> None: ...


class _Passthrough(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _FakeTransport:
    """Minimal transport boundary retaining run_bot's real pipeline stages."""

    current: ClassVar[_FakeTransport | None] = None

    def __init__(self, *, webrtc_connection: object, params: object) -> None:
        del webrtc_connection, params
        self.handlers: dict[str, object] = {}
        self._input = _Passthrough()
        self._output = _Passthrough()
        type(self).current = self

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output

    def event_handler[T](self, event: str) -> Callable[[T], T]:
        def register(handler: T) -> T:
            self.handlers[event] = handler
            return handler

        return register


def test_run_bot_drives_typed_turn_without_raw_audio_reaching_llm(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path
) -> None:
    del data_dir
    monkeypatch.setenv("THERAPY_TEST_MODE", "1")
    monkeypatch.setattr(pipeline, "SmallWebRTCTransport", _FakeTransport)

    real_runner = WorkerRunner

    class _DrivingRunner:
        """Queue one app turn, microphone frame, and terminal frame."""

        def __init__(self, *, handle_sigint: bool) -> None:
            self._runner = real_runner(handle_sigint=handle_sigint)
            self._worker: PipelineWorker | None = None

        async def add_workers(self, worker: PipelineWorker) -> None:
            self._worker = worker
            await self._runner.add_workers(worker)

        async def run(self) -> None:
            worker = self._worker
            transport = _FakeTransport.current
            assert worker is not None
            assert transport is not None
            handler_value = transport.handlers["on_app_message"]
            assert callable(handler_value)
            handler = cast(_AppMessageHandler, handler_value)
            await handler(transport, {"type": "client_ready"}, "test-client")
            await handler(
                transport,
                {"type": "voice_replies", "enabled": False},
                "test-client",
            )
            await handler(
                transport,
                {"type": "reply_language", "language": "es"},
                "test-client",
            )
            await handler(
                transport,
                {"type": "user_text", "text": "Hoy me siento cansado y sin energía."},
                "test-client",
            )
            await worker.queue_frames(
                [
                    InputAudioRawFrame(
                        audio=b"\x00\x00", sample_rate=16_000, num_channels=1
                    ),
                    EndFrame(),
                ]
            )
            await self._runner.run()

    monkeypatch.setattr(pipeline, "WorkerRunner", _DrivingRunner)

    add_turn_calls: list[tuple[str, str, str]] = []
    real_add_turn = pipeline.MemoryStore.add_turn

    def add_turn_spy(
        store: pipeline.MemoryStore,
        session_id: str,
        role: str,
        modality: str,
        language: str,
        text: str,
        *,
        audio: bytes | None = None,
        sample_rate: int = 16_000,
    ) -> int:
        add_turn_calls.append((role, modality, language))
        return real_add_turn(
            store,
            session_id,
            role,
            modality,
            language,
            text,
            audio=audio,
            sample_rate=sample_rate,
        )

    monkeypatch.setattr(pipeline.MemoryStore, "add_turn", add_turn_spy)

    llm_frames: list[type[Frame]] = []
    real_process = pipeline.DeterministicTestLLM.process_frame

    async def process_spy(
        llm: pipeline.DeterministicTestLLM,
        frame: Frame,
        direction: FrameDirection,
    ) -> None:
        llm_frames.append(type(frame))
        await real_process(llm, frame, direction)

    monkeypatch.setattr(pipeline.DeterministicTestLLM, "process_frame", process_spy)

    connection = cast(SmallWebRTCConnection, object())
    asyncio.run(pipeline.run_bot(connection, new_session=True))
    asyncio.run(pipeline.drain_session_finalizers(5.0))

    assert ("user", "text", "es") in add_turn_calls
    assert ("assistant", "text", "es") in add_turn_calls
    assert llm_frames
    assert InputAudioRawFrame not in llm_frames


# --- T4 STT boundary -------------------------------------------------------


@dataclass(frozen=True)
class _Segment:
    text: str
    no_speech_prob: float = 0.05
    avg_logprob: float = -0.1
    compression_ratio: float = 1.0


@dataclass(frozen=True)
class _TranscriptionInfo:
    language: str | None
    language_probability: float


class _FakeWhisper:
    """Small faster-whisper behavioral fake; decoding stays lazy-compatible."""

    def __init__(
        self, segments: list[_Segment], info: _TranscriptionInfo
    ) -> None:
        self._segments = segments
        self._info = info
        self.calls: list[tuple[str | None, bool]] = []
        self.thread_ids: list[int] = []

    def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        language: str | None,
        condition_on_previous_text: bool,
    ) -> tuple[list[_Segment], _TranscriptionInfo]:
        assert audio.dtype == np.float32
        self.calls.append((language, condition_on_previous_text))
        self.thread_ids.append(threading.get_ident())
        return self._segments, self._info


class _TranscribeUtterance(Protocol):
    def __call__(
        self,
        model: WhisperModel,
        audio: NDArray[np.float32],
        language: str | None,
        no_speech_threshold: float,
    ) -> tuple[str, str | None, float]: ...


class _LoadWhisper(Protocol):
    def __call__(self, service: MultilingualWhisperSTTService) -> None: ...


def _transcribe_boundary() -> _TranscribeUtterance:
    value = vars(pipeline).get("_transcribe_utterance")
    assert callable(value)
    return cast(_TranscribeUtterance, value)


def _load_boundary() -> _LoadWhisper:
    value = vars(MultilingualWhisperSTTService).get("_load")
    assert callable(value)
    return cast(_LoadWhisper, value)


def test_transcribe_utterance_filters_segments_and_preserves_language() -> None:
    fake = _FakeWhisper(
        [
            _Segment(" Hola "),
            _Segment("discard", no_speech_prob=0.99, avg_logprob=-4.0),
            _Segment(" mundo"),
        ],
        _TranscriptionInfo("es", 0.94),
    )
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)

    text, language, probability = _transcribe_boundary()(
        cast("WhisperModel", fake), audio, None, 0.6
    )

    assert text == "Hola mundo"
    assert language == "es"
    assert probability == 0.94
    assert fake.calls == [(None, False)]


def test_run_stt_decodes_in_worker_thread_and_records_turn(
    monkeypatch: pytest.MonkeyPatch, metrics: list[MetricCall]
) -> None:
    fake = _FakeWhisper(
        [_Segment("Hoy me siento cansado.")],
        _TranscriptionInfo("es", 0.97),
    )

    def fake_load(service: MultilingualWhisperSTTService) -> None:
        monkeypatch.setattr(service, "_model", cast("WhisperModel", fake))

    monkeypatch.setattr(MultilingualWhisperSTTService, "_load", fake_load)
    recorded: list[tuple[bytes, str, str]] = []
    service = MultilingualWhisperSTTService(
        recorder=lambda audio, text, language: recorded.append((audio, text, language))
    )
    audio = np.full(1_600, 500, dtype=np.int16).tobytes()
    event_loop_thread = threading.get_ident()

    async def collect() -> list[TranscriptionFrame]:
        return [frame async for frame in service.run_stt(audio)]

    frames = asyncio.run(collect())

    assert len(frames) == 1
    assert frames[0].text == "Hoy me siento cansado."
    assert frames[0].language is Language.ES
    assert recorded == [(audio, "Hoy me siento cansado.", "es")]
    assert fake.calls == [(None, False)]
    assert fake.thread_ids
    assert fake.thread_ids[0] != event_loop_thread
    assert _named(metrics, "therapy_stt_realtime_factor") == [
        {"outcome": "success"}
    ]


def test_stt_load_caches_one_model_for_later_services(
    monkeypatch: pytest.MonkeyPatch, metrics: list[MetricCall]
) -> None:
    fake = _FakeWhisper([], _TranscriptionInfo("en", 1.0))
    load = _load_boundary()

    def skip_load(_service: MultilingualWhisperSTTService) -> None:
        return None

    monkeypatch.setattr(MultilingualWhisperSTTService, "_load", skip_load)
    first = MultilingualWhisperSTTService()
    second = MultilingualWhisperSTTService()
    monkeypatch.setattr(pipeline, "_whisper_model", None)

    def base_load(service: WhisperSTTService) -> None:
        monkeypatch.setattr(service, "_model", cast("WhisperModel", fake))

    monkeypatch.setattr(WhisperSTTService, "_load", base_load)

    load(first)
    load(second)

    assert vars(first).get("_model") is fake
    assert vars(second).get("_model") is fake
    assert _named(metrics, "therapy_stt_model_load_seconds") == [
        {"outcome": "success"}
    ]


@pytest.mark.parametrize(
    ("language", "voice", "enum"),
    [
        ("en", "af_heart", Language.EN),
        ("es", "ef_dora", Language.ES),
        ("pt", "pf_dora", Language.PT),
    ],
)
def test_tts_settings_revoice_each_supported_language(
    language: str, voice: str, enum: Language
) -> None:
    settings = tts_settings_for(language).delta
    assert isinstance(settings, KokoroTTSSettings)
    assert settings.voice == voice
    assert settings.language is enum


def test_vad_params_use_defaults_and_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults = vad_params()
    assert defaults.confidence == 0.6
    assert defaults.start_secs == 0.2
    assert defaults.stop_secs == 1.2
    assert defaults.min_volume == 0.3

    monkeypatch.setenv("THERAPY_VAD_CONFIDENCE", "0.72")
    monkeypatch.setenv("THERAPY_VAD_STOP_SECS", "0.85")
    monkeypatch.setenv("THERAPY_VAD_MIN_VOLUME", "0.15")
    overridden = vad_params()
    assert overridden.confidence == 0.72
    assert overridden.stop_secs == 0.85
    assert overridden.min_volume == 0.15
