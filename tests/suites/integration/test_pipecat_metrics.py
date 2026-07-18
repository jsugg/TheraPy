"""Pipecat frame-to-owned-metric regression coverage (plan O2.3/O3.2)."""

import asyncio

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTSUsageMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from therapy.dialogue.modality import ReplyModality
from therapy.integrations.pipecat.observability import MetricsFrameAdapter
from therapy.integrations.pipecat.pipeline import BotTextRelay, PresenceRelay

type MetricCall = tuple[str, float, dict[str, str]]


class FakeTTS(FrameProcessor):
    """Processor whose finite component classifier resolves to TTS."""


def _capture_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> list[MetricCall]:
    from therapy.observability import telemetry

    calls: list[MetricCall] = []

    def capture(
        name: str, value: float, attributes: dict[str, str] | None = None
    ) -> None:
        calls.append((name, value, attributes or {}))

    monkeypatch.setattr(telemetry, "record_metric", capture)
    return calls


def _pushed(processor: FrameProcessor, frame: Frame) -> FramePushed:
    return FramePushed(
        source=processor,
        destination=processor,
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


def test_metrics_adapter_emits_tts_request_synthesis_and_realtime_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_metrics(monkeypatch)
    processor = FakeTTS()
    adapter = MetricsFrameAdapter()

    async def scenario() -> None:
        await adapter.on_push_frame(
            _pushed(
                processor,
                MetricsFrame(
                    [
                        TTSUsageMetricsData(
                            processor="KokoroTTSService", model=None, value=20
                        ),
                        ProcessingMetricsData(
                            processor="KokoroTTSService", model=None, value=0.25
                        ),
                    ]
                ),
            )
        )
        await adapter.on_push_frame(
            _pushed(
                processor,
                TTSAudioRawFrame(
                    audio=b"\0" * 32_000, sample_rate=16_000, num_channels=1
                ),
            )
        )
        await adapter.on_push_frame(_pushed(processor, BotStoppedSpeakingFrame()))

    asyncio.run(scenario())

    assert (
        "therapy_tts_requests_total",
        1,
        {"language_group": "unknown", "outcome": "success"},
    ) in calls
    assert (
        "therapy_tts_synthesis_seconds",
        0.25,
        {"language_group": "unknown", "outcome": "success"},
    ) in calls
    assert (
        "therapy_tts_realtime_factor",
        0.25,
        {"language_group": "unknown", "outcome": "success"},
    ) in calls


def test_metrics_adapter_emits_bounded_tts_failure_without_error_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_metrics(monkeypatch)
    processor = FakeTTS()
    adapter = MetricsFrameAdapter()

    async def scenario() -> None:
        await adapter.on_push_frame(_pushed(processor, TTSStartedFrame()))
        await adapter.on_push_frame(
            _pushed(processor, ErrorFrame("private-tts-error-canary"))
        )

    asyncio.run(scenario())

    assert (
        "therapy_tts_requests_total",
        1,
        {"language_group": "unknown", "outcome": "error"},
    ) in calls
    assert any(
        name == "therapy_tts_synthesis_seconds"
        and attributes["outcome"] == "error"
        for name, _, attributes in calls
    )
    assert "private-tts-error-canary" not in repr(calls)


def test_turn_and_barge_in_metrics_are_content_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_metrics(monkeypatch)
    pushed: list[Frame] = []

    async def capture_frame(
        frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        del direction
        pushed.append(frame)

    turn = BotTextRelay(lambda: "en", lambda: "text")
    presence = PresenceRelay(ReplyModality())
    monkeypatch.setattr(turn, "push_frame", capture_frame)
    monkeypatch.setattr(presence, "push_frame", capture_frame)

    async def process(
        processor: BotTextRelay | PresenceRelay, frame: Frame
    ) -> None:
        await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    async def scenario() -> None:
        await process(turn, LLMTextFrame("private-turn-canary"))
        await process(turn, LLMFullResponseEndFrame())
        await process(turn, LLMFullResponseEndFrame())
        await process(presence, BotStartedSpeakingFrame())
        await process(presence, UserStartedSpeakingFrame())
        await process(presence, BotStoppedSpeakingFrame())
        await process(presence, BotStartedSpeakingFrame())
        await process(presence, UserStartedSpeakingFrame())
        await process(presence, CancelFrame())

    asyncio.run(scenario())

    turns = [
        attributes
        for name, _, attributes in calls
        if name == "therapy_conversation_turns_total"
    ]
    assert turns == [
        {"modality": "text", "language_group": "en", "outcome": "success"},
        {"modality": "text", "language_group": "en", "outcome": "incomplete"},
    ]
    assert [
        attributes["outcome"]
        for name, _, attributes in calls
        if name == "therapy_barge_ins_total"
    ] == ["success", "error"]
    assert [
        attributes["outcome"]
        for name, _, attributes in calls
        if name == "therapy_barge_in_stop_seconds"
    ] == ["success", "error"]
    assert "private-turn-canary" not in repr(calls)
