"""FastAPI backend: WebRTC signaling + PWA static serving (SPEC §5)."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from therapy import __version__
from therapy.memory import MemoryStore

STATIC_DIR = Path(__file__).parent / "static"


class _WebRTCHandler(Protocol):
    """Subset of SmallWebRTCRequestHandler used by this module."""

    async def close(self) -> None: ...

    async def handle_web_request(
        self,
        request: object,
        on_connection: Callable[[object], Awaitable[None]],
    ) -> dict[str, object] | None: ...


_webrtc_handler: _WebRTCHandler | None = None
_bot_tasks: set[asyncio.Task[None]] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if _webrtc_handler is not None:
        await _webrtc_handler.close()
    for task in _bot_tasks:
        task.cancel()


app = FastAPI(title="TheraPy", version=__version__, lifespan=lifespan)


@lru_cache(maxsize=1)
def _store() -> MemoryStore:
    """Return the lazily initialized local memory store."""
    return MemoryStore()


def _handler() -> _WebRTCHandler:
    """Return the lazily initialized SmallWebRTC request handler."""
    global _webrtc_handler
    if _webrtc_handler is None:
        from pipecat.transports.smallwebrtc.request_handler import (
            SmallWebRTCRequestHandler,
        )

        _webrtc_handler = SmallWebRTCRequestHandler()
    return _webrtc_handler


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/api/offer")
async def offer(request: Request) -> dict[str, object]:
    """SDP offer → answer; each new connection gets its own pipeline."""
    from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequest
    from therapy.agent import run_bot

    body = await request.json()

    async def on_connection(connection: object) -> None:
        task = asyncio.create_task(run_bot(connection))
        _bot_tasks.add(task)
        task.add_done_callback(_bot_tasks.discard)

    answer = await _handler().handle_web_request(
        SmallWebRTCRequest.from_dict(body), on_connection
    )
    return answer or {}


@app.get("/api/sessions")
def sessions() -> dict[str, list[dict[str, object]]]:
    """Return session summaries with turn counts for the review UI."""
    store = _store()
    session_rows: list[dict[str, object]] = []
    for session in store.sessions():
        session_row: dict[str, object] = dict(session)
        session_row["turn_count"] = len(store.session_turns(str(session["id"])))
        session_rows.append(session_row)
    return {"sessions": session_rows}


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str) -> dict[str, object]:
    """Return one session and its ordered transcript turns."""
    store = _store()
    session = next(
        (session for session in store.sessions() if session["id"] == session_id),
        None,
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session, "turns": store.session_turns(session_id)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
