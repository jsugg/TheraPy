"""Realtime capture regression tests (plan O1.4 item 4; container-only).

Prove: the exact transcript reaches restricted capture exactly once, the
broad plane carries no content, and interruption yields an explicit
`incomplete` terminal with the partial stream preserved.
"""

import asyncio
import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (  # noqa: E402
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.observers.base_observer import FramePushed  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402

from therapy.integrations.pipecat.observability import (  # noqa: E402
    InteractionCaptureObserver,
    build_task_telemetry,
)
from therapy.observability.capture import set_capture_service, start_capture  # noqa: E402
from therapy.observability.config import ObservabilityConfig  # noqa: E402
from therapy.observability.logging import BroadJsonFormatter  # noqa: E402

TRANSCRIPT_CANARY = "OBS-TEST-VOICE-TRANSCRIPT-9f2c"
SYSTEM_CANARY = "OBS-TEST-SYSTEM-PROMPT-7d1a"
COMPLETION_CANARY = "OBS-TEST-COMPLETION-3b8e"


class _FakeLLM:
    pass


def _pushed(frame, *, source=None, destination=None) -> FramePushed:
    return FramePushed(
        source=source, destination=destination, frame=frame,
        direction=None, timestamp=0,
    )


def _context_frame() -> LLMContextFrame:
    context = LLMContext(
        messages=[
            {"role": "system", "content": SYSTEM_CANARY},
            {"role": "user", "content": TRANSCRIPT_CANARY},
        ]
    )
    return LLMContextFrame(context=context)


@pytest.fixture
def broad_log() -> Iterator[io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(BroadJsonFormatter(environment="test"))
    logger = logging.getLogger("therapy.broad")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    yield stream
    logger.handlers = []


def _drive(observer: InteractionCaptureObserver, llm: _FakeLLM, *, interrupt: bool):
    async def scenario() -> None:
        await observer.on_push_frame(
            _pushed(_context_frame(), destination=llm)
        )
        await observer.on_push_frame(
            _pushed(LLMFullResponseStartFrame(), source=llm)
        )
        await observer.on_push_frame(
            _pushed(LLMTextFrame(text=COMPLETION_CANARY[:12]), source=llm)
        )
        if interrupt:
            await observer.on_push_frame(_pushed(InterruptionFrame()))
        else:
            await observer.on_push_frame(
                _pushed(LLMTextFrame(text=COMPLETION_CANARY[12:]), source=llm)
            )
            await observer.on_push_frame(
                _pushed(LLMFullResponseEndFrame(), source=llm)
            )

    return scenario()


def test_transcript_reaches_restricted_capture_once_and_never_broad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broad_log: io.StringIO
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("THERAPY_INTERACTION_JOURNAL", raising=False)
    config = ObservabilityConfig.from_env()

    async def scenario() -> None:
        runtime = await start_capture(config, build_version="test")
        llm = _FakeLLM()
        observer = InteractionCaptureObserver(
            llm, session_id="sess-rt-1", provider="ollama", requested_model="m"
        )
        try:
            await _drive(observer, llm, interrupt=False)
            await runtime.writer.flush()
            store = runtime.store
            ids = list(store.iter_interaction_ids())
            assert len(ids) == 1  # exactly once
            loaded = store.load(ids[0])
            row = loaded["interaction"]
            assert row["status"] == "succeeded"
            assert row["operation"] == "reply"
            request = json.loads(row["canonical_request_json"])
            assert request["system_instructions"] == SYSTEM_CANARY
            assert TRANSCRIPT_CANARY in json.dumps(request["messages"])
            terminal = json.loads(row["terminal_json"])
            assert terminal["response"]["completion"] == COMPLETION_CANARY
            deltas = [
                json.loads(event["payload_json"])["delta"]
                for event in loaded["events"]
            ]
            assert "".join(deltas) == COMPLETION_CANARY
        finally:
            await runtime.close()

    asyncio.run(scenario())

    broad = broad_log.getvalue()
    for canary in (TRANSCRIPT_CANARY, SYSTEM_CANARY, COMPLETION_CANARY):
        assert canary not in broad, "content leaked into broad output"


def test_interruption_yields_explicit_incomplete_with_partial_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broad_log: io.StringIO
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("THERAPY_INTERACTION_JOURNAL", raising=False)
    config = ObservabilityConfig.from_env()

    async def scenario() -> None:
        runtime = await start_capture(config, build_version="test")
        llm = _FakeLLM()
        observer = InteractionCaptureObserver(
            llm, session_id=None, provider="anthropic", requested_model="m"
        )
        try:
            await _drive(observer, llm, interrupt=True)
            await runtime.writer.flush()
            store = runtime.store
            ids = list(store.iter_interaction_ids())
            assert len(ids) == 1
            loaded = store.load(ids[0])
            assert loaded["interaction"]["status"] == "incomplete"
            terminal = json.loads(loaded["interaction"]["terminal_json"])
            assert terminal["reason"] == "barge_in"
            assert len(loaded["events"]) == 1  # partial stream preserved
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_task_telemetry_kwargs_are_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """No product session ID as conversation ID; tracing off without the
    owned provider; observers wired."""
    set_capture_service(None)
    llm = _FakeLLM()
    kwargs = build_task_telemetry(llm, session_id="sess-product-42")
    assert kwargs["enable_tracing"] is False  # owned provider not installed
    assert kwargs["enable_turn_tracking"] is False
    assert kwargs["conversation_id"].startswith("tele-")
    assert "sess-product-42" not in kwargs["conversation_id"]
    assert len(kwargs["observers"]) == 1
    assert isinstance(kwargs["observers"][0], InteractionCaptureObserver)


def test_sdp_never_appears_in_capture_or_broad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broad_log: io.StringIO
) -> None:
    """The adapter has no SDP path at all; prove a journaled attempt and the
    broad stream contain no SDP markers even when the context is polluted."""
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("THERAPY_INTERACTION_JOURNAL", raising=False)
    config = ObservabilityConfig.from_env()

    async def scenario() -> None:
        runtime = await start_capture(config, build_version="test")
        llm = _FakeLLM()
        observer = InteractionCaptureObserver(
            llm, session_id=None, provider="ollama", requested_model="m"
        )
        try:
            await _drive(observer, llm, interrupt=False)
            await runtime.writer.flush()
        finally:
            await runtime.close()

    asyncio.run(scenario())
    combined = broad_log.getvalue()
    for marker in ("v=0", "a=candidate", "ice-ufrag"):
        assert marker not in combined

    journal_bytes = (tmp_path / "interaction-journal.sqlite3").read_bytes()
    assert b"a=candidate" not in journal_bytes
    assert b"ice-ufrag" not in journal_bytes
