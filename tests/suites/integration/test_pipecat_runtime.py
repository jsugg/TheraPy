"""Contract tests for the Pipecat adapter using controlled vendor doubles."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import cast

import pytest
from fastapi import HTTPException

from therapy.integrations.pipecat.runtime import (
    PipecatVoiceGateway,
    PipelineRunner,
    _Connection,
    _load_pipecat,
    _VendorRequest,
)
from therapy.voice.contracts import (
    ConnectionConflict,
    SessionTarget,
    VoiceUnavailable,
    WebRTCOffer,
)


@dataclass
class FakeConnection:
    """Minimal peer connection owned by the fake request handler."""

    pc_id: str
    disconnected: bool = False

    async def disconnect(self) -> None:
        self.disconnected = True


@dataclass
class FakeRequestHandler:
    """Mimic Pipecat's new-connection and renegotiation callback behavior."""

    requests: list[WebRTCOffer] = field(default_factory=list)
    connections: dict[str, FakeConnection] = field(default_factory=dict)
    callback_count: int = 0
    closed: bool = False
    failure: HTTPException | None = None
    answer: dict[str, object] | None = None

    async def handle_web_request(
        self,
        request: _VendorRequest,
        webrtc_connection_callback: Callable[[object], Awaitable[None]],
    ) -> dict[str, object] | None:
        offer = request
        assert isinstance(offer, WebRTCOffer)
        self.requests.append(offer)
        if self.failure is not None:
            raise self.failure
        if offer.pc_id and offer.pc_id in self.connections:
            return {
                "sdp": "renegotiated-answer",
                "type": "answer",
                "pc_id": offer.pc_id,
            }
        connection = FakeConnection(pc_id=f"peer-{len(self.connections) + 1}")
        self.connections[connection.pc_id] = connection
        self.callback_count += 1
        # Pipecat 1.5 catches callback exceptions and still returns an answer.
        try:
            await webrtc_connection_callback(connection)
        except Exception:
            pass
        return self.answer or {
            "sdp": "answer-sdp",
            "type": "answer",
            "pc_id": connection.pc_id,
        }

    async def close(self) -> None:
        self.closed = True
        for connection in self.connections.values():
            await connection.disconnect()


def make_gateway(
    handler: FakeRequestHandler,
    pipeline_runner: PipelineRunner,
    *,
    cancel_timeout: float = 0.2,
) -> PipecatVoiceGateway:
    """Build an adapter without importing the optional Pipecat distribution."""
    return PipecatVoiceGateway(
        handler=handler,
        request_factory=lambda offer: offer,
        pipeline_runner=pipeline_runner,
        cancel_timeout=cancel_timeout,
    )


def test_real_pipecat_factory_maps_fields_redacts_and_suppresses_logs() -> None:
    try:
        handler, request_factory = _load_pipecat()
    except ModuleNotFoundError:
        pytest.skip("Pipecat realtime stack is not installed")
        return
    from loguru import logger as vendor_logger
    from pipecat.transports.smallwebrtc import request_handler

    offer = WebRTCOffer(
        sdp="private-sdp",
        type="offer",
        pc_id="peer-1",
        restart_pc=True,
        request_data={"private": "metadata"},
    )

    request = request_factory(offer)

    assert isinstance(request, request_handler.SmallWebRTCRequest)
    assert request.sdp == offer.sdp
    assert request.type == offer.type
    assert request.pc_id == offer.pc_id
    assert request.restart_pc is True
    assert request.request_data == offer.request_data
    assert "private-sdp" not in repr(request)
    assert "metadata" not in repr(request)

    captured: list[str] = []
    sink_id = vendor_logger.add(captured.append, format="{message}")
    try:
        vendor_module = ModuleType("pipecat.private")
        vendor_module.__dict__["logger"] = vendor_logger
        exec(
            "def emit(): logger.warning('private-conversation-content')",
            vendor_module.__dict__,
        )
        cast(Callable[[], None], vendor_module.__dict__["emit"])()
    finally:
        vendor_logger.remove(sink_id)
    assert captured == []
    asyncio.run(handler.close())


def test_adapter_maps_owned_offer_and_answer_without_leaking_vendor_objects() -> None:
    async def scenario() -> None:
        handler = FakeRequestHandler()
        stop = asyncio.Event()
        seen: list[tuple[_Connection, SessionTarget]] = []

        async def run_pipeline(connection: _Connection, target: SessionTarget) -> None:
            seen.append((connection, target))
            await stop.wait()

        gateway = make_gateway(handler, run_pipeline)
        offer = WebRTCOffer(
            sdp="private-offer",
            type="offer",
            restart_pc=True,
            request_data={"client": "pwa"},
        )
        target = SessionTarget(session_id=None, new_session=True)

        answer = await gateway.negotiate(offer, target)

        assert answer.as_payload() == {
            "sdp": "answer-sdp",
            "type": "answer",
            "pc_id": "peer-1",
        }
        assert handler.requests == [offer]
        assert seen == [(handler.connections["peer-1"], target)]
        stop.set()
        await gateway.close()

    asyncio.run(scenario())


def test_disconnect_only_stops_matching_peer_and_gateway_remains_reusable() -> None:
    async def scenario() -> None:
        handler = FakeRequestHandler()
        stopped = asyncio.Event()

        async def run_pipeline(connection: _Connection, target: SessionTarget) -> None:
            del connection, target
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        gateway = make_gateway(handler, run_pipeline)
        target = SessionTarget(session_id=None, new_session=True)
        first = await gateway.negotiate(WebRTCOffer(sdp="first", type="offer"), target)

        assert await gateway.disconnect("another-peer") is False
        assert await gateway.disconnect(first.pc_id) is True
        assert stopped.is_set()
        assert handler.connections[first.pc_id].disconnected is True

        second = await gateway.negotiate(WebRTCOffer(sdp="second", type="offer"), target)
        assert second.pc_id != first.pc_id
        await gateway.close()

    asyncio.run(scenario())


def test_renegotiation_does_not_launch_duplicate_pipeline() -> None:
    async def scenario() -> None:
        handler = FakeRequestHandler()
        stop = asyncio.Event()
        starts = 0

        async def run_pipeline(connection: _Connection, target: SessionTarget) -> None:
            nonlocal starts
            starts += 1
            await stop.wait()

        gateway = make_gateway(handler, run_pipeline)
        target = SessionTarget(session_id=None, new_session=True)
        first = await gateway.negotiate(WebRTCOffer("one", "offer"), target)
        second = await gateway.negotiate(
            WebRTCOffer("two", "offer", pc_id=first.pc_id), target
        )

        assert second.pc_id == first.pc_id
        assert handler.callback_count == 1
        assert starts == 1
        stop.set()
        await gateway.close()

    asyncio.run(scenario())


def test_preemption_drains_old_pipeline_before_starting_replacement() -> None:
    async def scenario() -> None:
        handler = FakeRequestHandler()
        order: list[str] = []
        hold = asyncio.Event()

        async def run_pipeline(connection: _Connection, target: SessionTarget) -> None:
            order.append(f"start:{connection.pc_id}")
            try:
                await hold.wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0)
                order.append(f"stop:{connection.pc_id}")
                raise

        gateway = make_gateway(handler, run_pipeline)
        target = SessionTarget(session_id=None, new_session=True)
        await gateway.negotiate(WebRTCOffer("one", "offer"), target)
        await gateway.negotiate(WebRTCOffer("two", "offer"), target)

        assert order == ["start:peer-1", "stop:peer-1", "start:peer-2"]
        assert handler.connections["peer-1"].disconnected is True
        hold.set()
        await gateway.close()

    asyncio.run(scenario())


def test_preemption_timeout_refuses_overlap_and_disconnects_new_peer() -> None:
    async def scenario() -> None:
        handler = FakeRequestHandler()
        release = asyncio.Event()
        starts = 0

        async def stubborn_pipeline(
            connection: _Connection, target: SessionTarget
        ) -> None:
            nonlocal starts
            starts += 1
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()

        gateway = make_gateway(handler, stubborn_pipeline, cancel_timeout=0.01)
        target = SessionTarget(session_id=None, new_session=True)
        await gateway.negotiate(WebRTCOffer("one", "offer"), target)

        with pytest.raises(VoiceUnavailable, match="failed to start"):
            await gateway.negotiate(WebRTCOffer("two", "offer"), target)

        assert starts == 1
        assert handler.connections["peer-2"].disconnected is True
        release.set()
        await asyncio.sleep(0)
        await gateway.close()

    asyncio.run(scenario())


def test_startup_failure_is_detected_despite_handler_swallowing_callback() -> None:
    async def scenario() -> None:
        handler = FakeRequestHandler()

        async def broken_pipeline(
            connection: _Connection, target: SessionTarget
        ) -> None:
            raise RuntimeError("sensitive provider detail")

        gateway = make_gateway(handler, broken_pipeline)

        with pytest.raises(VoiceUnavailable, match="failed to start"):
            await gateway.negotiate(
                WebRTCOffer("private-offer", "offer"),
                SessionTarget(session_id=None, new_session=True),
            )

        assert handler.connections["peer-1"].disconnected is True
        assert handler.closed is True
        await gateway.close()

    asyncio.run(scenario())


def test_background_failure_is_observed_without_logging_failure_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def scenario() -> None:
        handler = FakeRequestHandler()
        fail = asyncio.Event()

        async def broken_later(connection: _Connection, target: SessionTarget) -> None:
            await fail.wait()
            raise RuntimeError("private transcript text")

        gateway = make_gateway(handler, broken_later)
        await gateway.negotiate(
            WebRTCOffer("private-sdp", "offer"),
            SessionTarget(session_id=None, new_session=True),
        )
        fail.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert gateway._pipeline_task is None
        await gateway.close()

    with caplog.at_level(logging.ERROR):
        asyncio.run(scenario())

    assert "failure_type=RuntimeError" in caplog.text
    assert "private transcript text" not in caplog.text
    assert "private-sdp" not in caplog.text


def test_handler_conflict_and_invalid_answer_are_translated() -> None:
    async def idle_pipeline(connection: _Connection, target: SessionTarget) -> None:
        await asyncio.Event().wait()

    async def scenario() -> None:
        conflict_handler = FakeRequestHandler(failure=HTTPException(status_code=400))
        conflict_gateway = make_gateway(conflict_handler, idle_pipeline)
        with pytest.raises(ConnectionConflict):
            await conflict_gateway.negotiate(
                WebRTCOffer("offer", "offer"),
                SessionTarget(session_id=None, new_session=True),
            )
        await conflict_gateway.close()

        invalid_handler = FakeRequestHandler(
            answer={"sdp": "answer", "type": "wrong", "pc_id": "peer-1"}
        )
        invalid_gateway = make_gateway(invalid_handler, idle_pipeline)
        with pytest.raises(VoiceUnavailable, match="invalid WebRTC answer"):
            await invalid_gateway.negotiate(
                WebRTCOffer("offer", "offer"),
                SessionTarget(session_id=None, new_session=True),
            )
        await invalid_gateway.close()

    asyncio.run(scenario())


def test_close_is_idempotent_drains_pipeline_and_rejects_new_offers() -> None:
    async def scenario() -> None:
        handler = FakeRequestHandler()
        cancelled = asyncio.Event()

        async def run_pipeline(connection: _Connection, target: SessionTarget) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        gateway = make_gateway(handler, run_pipeline)
        await gateway.negotiate(
            WebRTCOffer("offer", "offer"),
            SessionTarget(session_id=None, new_session=True),
        )

        await gateway.close()
        await gateway.close()

        assert cancelled.is_set()
        assert handler.closed is True
        assert gateway._pipeline_task is None
        assert all(
            connection.disconnected for connection in handler.connections.values()
        )
        with pytest.raises(VoiceUnavailable, match="shutting down"):
            await gateway.negotiate(
                WebRTCOffer("later", "offer"),
                SessionTarget(session_id=None, new_session=True),
            )

    asyncio.run(scenario())
