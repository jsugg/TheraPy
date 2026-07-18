"""Realtime capture regression tests (plan O1.4 item 4; container-only).

Prove: the exact transcript reaches restricted capture exactly once, the
broad plane carries no content, and interruption yields an explicit
`incomplete` terminal with the partial stream preserved.
"""

import asyncio
import io
import json
import logging
from collections.abc import Coroutine, Iterator
from pathlib import Path

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (  # noqa: E402
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.observers.base_observer import FramePushed  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.frame_processor import (  # noqa: E402
    FrameDirection,
    FrameProcessor,
)

from therapy.integrations.pipecat.observability import (  # noqa: E402
    InteractionCaptureObserver,
    build_task_telemetry,
)
from therapy.observability.capture import (  # noqa: E402
    set_capture_service,
    start_capture,
)
from therapy.observability.config import ObservabilityConfig  # noqa: E402
from therapy.observability.journal import (  # noqa: E402
    JournalStore,
    LoadedInteraction,
)
from therapy.observability.logging import BroadJsonFormatter  # noqa: E402

TRANSCRIPT_CANARY = "OBS-TEST-VOICE-TRANSCRIPT-9f2c"
SYSTEM_CANARY = "OBS-TEST-SYSTEM-PROMPT-7d1a"
COMPLETION_CANARY = "OBS-TEST-COMPLETION-3b8e"


class _FakeLLM(FrameProcessor):
    pass


def _pushed(
    frame: Frame,
    *,
    source: FrameProcessor | None = None,
    destination: FrameProcessor | None = None,
) -> FramePushed:
    fallback = _FakeLLM()
    return FramePushed(
        source=source or fallback,
        destination=destination or fallback,
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
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


def _drive(
    observer: InteractionCaptureObserver, llm: _FakeLLM, *, interrupt: bool
) -> Coroutine[object, object, None]:
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


def _loaded(store: JournalStore, interaction_id: str) -> LoadedInteraction:
    loaded = store.load(interaction_id)
    assert loaded is not None
    return loaded


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
            assert runtime.writer is not None
            await runtime.writer.flush()
            store = runtime.store
            assert store is not None
            ids = list(store.iter_interaction_ids())
            assert len(ids) == 1  # exactly once
            loaded = _loaded(store, ids[0])
            row = loaded["interaction"]
            assert row["status"] == "succeeded"
            assert row["operation"] == "reply"
            canonical_request_json = row["canonical_request_json"]
            assert canonical_request_json is not None
            request = json.loads(canonical_request_json)
            assert request["system_instructions"] == SYSTEM_CANARY
            assert TRANSCRIPT_CANARY in json.dumps(request["messages"])
            terminal_json = row["terminal_json"]
            assert terminal_json is not None
            terminal = json.loads(terminal_json)
            assert terminal["response"]["completion"] == COMPLETION_CANARY
            deltas: list[str] = []
            for event in loaded["events"]:
                payload_json = event["payload_json"]
                assert payload_json is not None
                payload = json.loads(payload_json)
                delta = payload["delta"]
                assert isinstance(delta, str)
                deltas.append(delta)
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
            assert runtime.writer is not None
            await runtime.writer.flush()
            store = runtime.store
            assert store is not None
            ids = list(store.iter_interaction_ids())
            assert len(ids) == 1
            loaded = _loaded(store, ids[0])
            assert loaded["interaction"]["status"] == "incomplete"
            terminal_json = loaded["interaction"]["terminal_json"]
            assert terminal_json is not None
            terminal = json.loads(terminal_json)
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
    conversation_id = kwargs["conversation_id"]
    observers = kwargs["observers"]
    assert isinstance(conversation_id, str)
    assert isinstance(observers, list)
    assert conversation_id.startswith("tele-")
    assert "sess-product-42" not in conversation_id
    assert len(observers) == 2  # capture + MetricsFrame adapter
    assert isinstance(observers[0], InteractionCaptureObserver)


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
            assert runtime.writer is not None
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
