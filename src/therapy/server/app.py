"""FastAPI backend: WebRTC signaling + PWA static serving (SPEC §5)."""

import asyncio
import os
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from therapy import __version__
from therapy.memory import MemoryStore
from therapy.server import live

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


def launch_bot(
    connection: object,
    bot: Callable[[object], Coroutine[object, object, None]],
) -> asyncio.Task[None]:
    """Start a pipeline for a connection, preempting any existing one.

    v1 is single-user (SPEC §2): a second live pipeline is never a second
    user, only a reconnect — and every pipeline loads its own STT/TTS
    models, so letting them stack is how the container runs out of memory.
    The newest connection always wins.
    """
    for task in list(_bot_tasks):
        task.cancel()
    task = asyncio.create_task(bot(connection))
    _bot_tasks.add(task)
    task.add_done_callback(_bot_tasks.discard)
    return task


@app.post("/api/offer")
async def offer(request: Request) -> dict[str, object]:
    """SDP offer → answer; the new connection preempts the previous pipeline.

    `?new_session=1` skips reconnect-resume so the pipeline always opens a
    fresh session — test scripts need isolated sessions; real clients
    resume an interrupted one (SPEC §8). `?session=<id>` continues that
    specific session (the history browser's explicit choice).
    """
    from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequest
    from therapy.agent import run_bot

    body = await request.json()
    new_session = request.query_params.get("new_session") == "1"
    resume_session_id = request.query_params.get("session")

    async def on_connection(connection: object) -> None:
        launch_bot(
            connection,
            lambda conn: run_bot(
                conn, new_session=new_session, resume_session_id=resume_session_id
            ),
        )

    answer = await _handler().handle_web_request(
        SmallWebRTCRequest.from_dict(body), on_connection
    )
    return answer or {}


@app.get("/api/ice-config")
def ice_config() -> dict[str, object]:
    """TURN credentials for the compose relay (SPEC §5 clients).

    The client builds the URL itself from the page's hostname —
    `turn:<host>:<port>` — so the same config works on localhost, LAN,
    and the tailnet. Credentials are static (tailnet-only exposure; the
    VPN identity is the auth boundary for the MVP, SPEC §8).
    """
    return {
        "username": os.environ.get("THERAPY_TURN_USER", "therapy"),
        "credential": os.environ.get("THERAPY_TURN_PASSWORD", "therapy-local"),
        "port": int(os.environ.get("THERAPY_TURN_PORT", "3478")),
    }


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


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    """Delete one session (turns + archived audio) from the review UI.

    Wholesale wipe stays a deliberate CLI act (`python -m therapy.memory
    delete --yes`); this is the per-conversation eraser.
    """
    store = _store()
    if not store.has_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    if live.is_active(session_id):
        raise HTTPException(
            status_code=409, detail="Session is live — disconnect first"
        )
    store.delete_session(session_id)
    return {"deleted": session_id}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
