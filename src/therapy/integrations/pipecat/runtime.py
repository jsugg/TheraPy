"""Pipecat WebRTC signaling and single-user pipeline lifecycle adapter."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from typing import TYPE_CHECKING, Protocol, cast

from fastapi import HTTPException

from therapy.voice.contracts import (
    ConnectionConflict,
    InvalidOffer,
    SessionTarget,
    VoiceUnavailable,
    WebRTCAnswer,
    WebRTCOffer,
)

if TYPE_CHECKING:
    from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection

logger = logging.getLogger(__name__)


class _Connection(Protocol):
    @property
    def pc_id(self) -> str: ...

    async def disconnect(self) -> None: ...


class _VendorRequest(Protocol):
    @property
    def sdp(self) -> str: ...

    @property
    def type(self) -> str: ...

    @property
    def pc_id(self) -> str | None: ...

    @property
    def restart_pc(self) -> bool | None: ...

    @property
    def request_data(self) -> object | None: ...


class _RequestHandler(Protocol):
    async def handle_web_request(
        self,
        request: _VendorRequest,
        webrtc_connection_callback: Callable[[object], Awaitable[None]],
    ) -> dict[str, object] | None: ...

    async def close(self) -> None: ...


type RequestFactory = Callable[[WebRTCOffer], _VendorRequest]
type PipelineRunner = Callable[
    [_Connection, SessionTarget], Coroutine[object, object, None]
]
type FinalizerDrainer = Callable[[float], Awaitable[None]]


def _load_pipecat() -> tuple[_RequestHandler, RequestFactory]:
    """Load the optional realtime stack only when the first offer needs it."""
    from loguru import logger as vendor_logger

    # Pipecat logs full LLM contexts at DEBUG and rejected app messages (which
    # include typed turns) at WARNING. Namespace suppression is the only robust
    # guarantee that conversation content never reaches logs; TheraPy's adapter
    # emits its own content-free lifecycle diagnostics instead (SPEC §8).
    vendor_logger.disable("pipecat")
    from pipecat.transports.smallwebrtc.request_handler import (
        SmallWebRTCRequest,
        SmallWebRTCRequestHandler,
    )

    class RedactedSmallWebRTCRequest(SmallWebRTCRequest):
        """Prevent Pipecat failure logs from rendering SDP or request data."""

        def __repr__(self) -> str:
            return (
                "SmallWebRTCRequest(sdp=<redacted>, "
                f"type={self.type!r}, pc_id={self.pc_id!r}, "
                f"restart_pc={self.restart_pc!r}, request_data=<redacted>)"
            )

        __str__ = __repr__

    def request_factory(offer: WebRTCOffer) -> _VendorRequest:
        return RedactedSmallWebRTCRequest(
            sdp=offer.sdp,
            type=offer.type,
            pc_id=offer.pc_id,
            restart_pc=offer.restart_pc,
            request_data=offer.request_data,
        )

    return cast(_RequestHandler, SmallWebRTCRequestHandler()), request_factory


async def _default_pipeline_runner(
    connection: _Connection, target: SessionTarget
) -> None:
    """Load and run the Pipecat pipeline only when a voice offer arrives."""
    from therapy.integrations.pipecat.pipeline import run_bot

    await run_bot(
        cast("SmallWebRTCConnection", connection),
        new_session=target.new_session,
        resume_session_id=target.session_id,
    )


async def _default_finalizer_drainer(timeout: float) -> None:
    """Load the Pipecat finalizer registry only for the production runner."""
    from therapy.integrations.pipecat.pipeline import drain_session_finalizers

    await drain_session_finalizers(timeout)


async def _noop_finalizer_drainer(timeout: float) -> None:
    del timeout


class PipecatVoiceGateway:
    """Own Pipecat signaling, peer connections, and pipeline task lifecycle."""

    def __init__(
        self,
        *,
        handler: _RequestHandler | None = None,
        request_factory: RequestFactory | None = None,
        pipeline_runner: PipelineRunner | None = None,
        finalizer_drainer: FinalizerDrainer | None = None,
        cancel_timeout: float = 10.0,
    ) -> None:
        if cancel_timeout <= 0:
            raise ValueError("cancel_timeout must be positive")
        if (handler is None) != (request_factory is None):
            raise ValueError("handler and request_factory must be provided together")
        if handler is None or request_factory is None:
            handler, request_factory = _load_pipecat()
        self._handler = handler
        self._request_factory = request_factory
        using_default_pipeline = pipeline_runner is None
        self._pipeline_runner = pipeline_runner or _default_pipeline_runner
        self._finalizer_drainer = finalizer_drainer or (
            _default_finalizer_drainer
            if using_default_pipeline
            else _noop_finalizer_drainer
        )
        self._cancel_timeout = cancel_timeout
        self._lock = asyncio.Lock()
        self._pipeline_task: asyncio.Task[None] | None = None
        self._pipeline_connection: _Connection | None = None
        self._closed = False

    async def negotiate(
        self, offer: WebRTCOffer, target: SessionTarget
    ) -> WebRTCAnswer:
        """Negotiate an offer and atomically replace the single live pipeline."""
        async with self._lock:
            if self._closed:
                raise VoiceUnavailable("Voice runtime is shutting down")

            callback_failure: BaseException | None = None
            callback_connection: _Connection | None = None

            async def start_pipeline(connection: object) -> None:
                nonlocal callback_failure, callback_connection
                callback_connection = cast(_Connection, connection)
                try:
                    await self._replace_pipeline(callback_connection, target)
                except asyncio.CancelledError:
                    await self._disconnect_safely(callback_connection)
                    raise
                except Exception as exc:
                    callback_failure = exc
                    await self._disconnect_safely(callback_connection)

            request = self._request_factory(offer)
            try:
                raw_answer = await self._handler.handle_web_request(
                    request=request,
                    webrtc_connection_callback=start_pipeline,
                )
            except HTTPException as exc:
                await self._abort_started_connection(callback_connection)
                if exc.status_code in {400, 409}:
                    raise ConnectionConflict(
                        "WebRTC offer conflicts with the active connection"
                    ) from exc
                if exc.status_code == 422:
                    raise InvalidOffer("Pipecat rejected the WebRTC offer") from exc
                raise VoiceUnavailable("Voice signaling request failed") from exc
            except Exception as exc:
                await self._abort_started_connection(callback_connection)
                logger.error(
                    "Pipecat signaling failed failure_type=%s", type(exc).__name__
                )
                raise VoiceUnavailable("Voice signaling request failed") from exc

            if callback_failure is not None:
                # Pipecat stores the connection in its map *after* the callback
                # returns. A callback-time disconnect therefore fires too early
                # to remove it; close the handler after it returns to clear the
                # unusable peer deterministically. The handler remains reusable.
                await self._clear_handler_connections_safely()
                logger.error(
                    "Pipecat pipeline startup failed failure_type=%s",
                    type(callback_failure).__name__,
                )
                raise VoiceUnavailable(
                    "Voice pipeline failed to start"
                ) from callback_failure
            if raw_answer is None:
                await self._abort_started_connection(callback_connection)
                raise VoiceUnavailable("Voice signaling produced no answer")
            try:
                return self._map_answer(raw_answer)
            except VoiceUnavailable:
                await self._abort_started_connection(callback_connection)
                raise

    async def close(self) -> None:
        """Idempotently reject new work and drain pipelines and peer connections."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            stop_failure: VoiceUnavailable | None = None
            try:
                await self._stop_pipeline()
            except VoiceUnavailable as exc:
                stop_failure = exc
            try:
                await asyncio.wait_for(
                    self._handler.close(), timeout=self._cancel_timeout
                )
            except TimeoutError:
                logger.error("Pipecat request handler close timed out")
                stop_failure = VoiceUnavailable("Voice peer shutdown timed out")
            except Exception as exc:
                logger.error(
                    "Pipecat request handler close failed failure_type=%s",
                    type(exc).__name__,
                )
                stop_failure = VoiceUnavailable("Voice peer shutdown failed")

            task = self._pipeline_task
            if task is not None and not task.done():
                done, _ = await asyncio.wait({task}, timeout=self._cancel_timeout)
                if not done:
                    stop_failure = VoiceUnavailable("Voice pipeline shutdown timed out")
            if task is not None and task.done():
                self._consume_task(task)
            self._pipeline_task = None
            self._pipeline_connection = None
            if stop_failure is not None:
                raise stop_failure

    async def disconnect(self, peer_id: str) -> bool:
        """Stop the matching peer without shutting down the reusable gateway."""
        if not peer_id:
            raise ValueError("peer_id must not be empty")
        async with self._lock:
            connection = self._pipeline_connection
            if self._closed or connection is None or connection.pc_id != peer_id:
                return False
            await self._stop_pipeline()
            try:
                await self._finalizer_drainer(self._cancel_timeout)
            except TimeoutError as exc:
                raise VoiceUnavailable(
                    "Voice session finalization did not stop in time"
                ) from exc
            return True

    async def _replace_pipeline(
        self, connection: _Connection, target: SessionTarget
    ) -> None:
        """Drain the prior pipeline before allowing a replacement to start."""
        await self._stop_pipeline()
        task = asyncio.create_task(
            self._pipeline_runner(connection, target), name="therapy-pipecat-pipeline"
        )
        self._pipeline_task = task
        self._pipeline_connection = connection
        task.add_done_callback(self._observe_pipeline_completion)

        # Give synchronous setup and its first await one loop turn. This catches
        # missing provider configuration and model initialization failures before
        # returning a successful SDP answer.
        await asyncio.sleep(0)
        if task.done():
            failure = self._consume_task(task)
            if failure is None:
                raise VoiceUnavailable("Voice pipeline stopped during startup")
            raise VoiceUnavailable("Voice pipeline failed during startup") from failure

    async def _stop_pipeline(self) -> None:
        """Cancel, await, and disconnect the currently tracked pipeline."""
        task = self._pipeline_task
        connection = self._pipeline_connection
        if task is not None and not task.done():
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=self._cancel_timeout)
            if not done:
                raise VoiceUnavailable("Previous voice pipeline did not stop in time")
        if task is not None and task.done():
            self._consume_task(task)
        if connection is not None:
            try:
                await asyncio.wait_for(
                    connection.disconnect(), timeout=self._cancel_timeout
                )
            except TimeoutError as exc:
                raise VoiceUnavailable(
                    "Previous voice peer did not close in time"
                ) from exc
            except Exception as exc:
                raise VoiceUnavailable("Previous voice peer failed to close") from exc
        if self._pipeline_task is task:
            self._pipeline_task = None
            self._pipeline_connection = None

    async def _disconnect_safely(self, connection: _Connection) -> None:
        """Best-effort cleanup for a connection whose pipeline could not start."""
        try:
            await asyncio.wait_for(
                connection.disconnect(), timeout=self._cancel_timeout
            )
        except Exception as exc:
            logger.error(
                "Failed to disconnect unusable Pipecat peer failure_type=%s",
                type(exc).__name__,
            )

    async def _clear_handler_connections_safely(self) -> None:
        """Best-effort clear of Pipecat's connection map after callback failure."""
        try:
            await asyncio.wait_for(self._handler.close(), timeout=self._cancel_timeout)
        except Exception as exc:
            logger.error(
                "Failed to clear rejected Pipecat peers failure_type=%s",
                type(exc).__name__,
            )

    async def _abort_started_connection(self, connection: _Connection | None) -> None:
        """Drain any pipeline started before signaling ultimately failed."""
        if connection is None:
            return
        try:
            await self._stop_pipeline()
            return
        except VoiceUnavailable as exc:
            logger.error(
                "Failed to drain rejected Pipecat pipeline failure_type=%s",
                type(exc).__name__,
            )
        await self._disconnect_safely(connection)

    def _observe_pipeline_completion(self, task: asyncio.Task[None]) -> None:
        """Consume every background result and surface unexpected failures."""
        failure = self._consume_task(task)
        if failure is not None:
            logger.error(
                "Pipecat pipeline stopped unexpectedly failure_type=%s",
                type(failure).__name__,
            )
        if self._pipeline_task is task:
            self._pipeline_task = None
            self._pipeline_connection = None

    @staticmethod
    def _consume_task(task: asyncio.Task[None]) -> BaseException | None:
        """Retrieve a completed task's exception without leaking cancellation."""
        if not task.done() or task.cancelled():
            return None
        with suppress(asyncio.CancelledError):
            return task.exception()
        return None

    @staticmethod
    def _map_answer(raw_answer: dict[str, object]) -> WebRTCAnswer:
        """Validate Pipecat's untrusted response before crossing the owned port."""
        sdp = raw_answer.get("sdp")
        answer_type = raw_answer.get("type")
        pc_id = raw_answer.get("pc_id")
        if (
            not isinstance(sdp, str)
            or answer_type != "answer"
            or not isinstance(pc_id, str)
        ):
            raise VoiceUnavailable("Voice runtime returned an invalid WebRTC answer")
        return WebRTCAnswer(sdp=sdp, type="answer", pc_id=pc_id)
