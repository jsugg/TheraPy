"""FastAPI backend: WebRTC signaling + PWA static serving (SPEC §5)."""

import asyncio
import os
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from therapy import __version__
from therapy.memory import MemoryStore
from therapy.memory.store import RowDict, resume_window_secs
from therapy.server import live
from therapy.server.protocol import session_state_message

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


def _resolve_session(
    store: MemoryStore, *, new_session: bool, explicit: str | None
) -> tuple[str | None, bool]:
    """Resolve which stored session a new connection will land in.

    Mirrors run_bot's resume logic so the offer response can tell the client
    up front which session it is joining. The client then loads that
    transcript over HTTP instead of waiting for the data-channel `session`
    replay — pipecat clears its outbound queue and disables it if the data
    channel is slow to open (10 s), which is exactly what happens on the
    phone over the TURN relay, so the replay was silently dropped and the
    resumed transcript never rendered (field test 2026-07-10).

    Returns the resolved session id (None for a fresh session) and whether
    connecting resumes it.
    """
    if explicit:
        return (explicit, True) if store.has_session(explicit) else (None, False)
    if new_session:
        return None, False
    resumed = store.resume_candidate(resume_window_secs())
    return resumed, resumed is not None


@app.post("/api/offer")
async def offer(request: Request) -> dict[str, object]:
    """SDP offer → answer; the new connection preempts the previous pipeline.

    `?new_session=1` skips reconnect-resume so the pipeline always opens a
    fresh session — test scripts need isolated sessions; real clients
    resume an interrupted one (SPEC §8). `?session=<id>` continues that
    specific session (the history browser's explicit choice).

    The answer carries `session_id`/`resumed` so the client can load the
    transcript over HTTP; the resolved id is also what run_bot joins, so the
    two never disagree.
    """
    from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequest
    from therapy.agent import run_bot

    body = await request.json()
    store = _store()
    new_session = request.query_params.get("new_session") == "1"
    resolved, resumed = _resolve_session(
        store,
        new_session=new_session,
        explicit=request.query_params.get("session"),
    )
    # Carry the resumed transcript in the answer so the client renders it
    # synchronously on connect — an async fetch after connect raced a reconnect
    # (rendering the wrong session) and live turns (duplicating/mis-ordering).
    state = session_state_message(
        resolved or "", resumed, store.session_turns(resolved) if resolved else []
    )

    async def on_connection(connection: object) -> None:
        launch_bot(
            connection,
            # run_bot joins exactly `resolved`: a real id resumes it, None opens
            # a fresh session. new_session=(resolved is None) stops run_bot from
            # picking its own resume candidate and disagreeing with this answer.
            lambda conn: run_bot(
                conn, new_session=resolved is None, resume_session_id=resolved
            ),
        )

    answer = await _handler().handle_web_request(
        SmallWebRTCRequest.from_dict(body), on_connection
    )
    return {
        **(answer or {}),
        "session_id": resolved,
        "resumed": state["resumed"],
        "turns": state["turns"],
    }


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


@app.get("/api/resumable")
def resumable() -> dict[str, object]:
    """Whether connecting now would resume a session (SPEC §8).

    The client labels its connect button accordingly — "Resume" must not
    look like "Start", or a user expecting a fresh conversation lands in
    an old one unawares.
    """
    return {"session_id": _store().resume_candidate(resume_window_secs())}


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


def _client_turns(turns: Sequence[RowDict]) -> list[dict[str, object]]:
    """Shape stored turns for the review UI.

    Exposes whether a turn has archived audio (so the client can offer a
    playback control) without leaking the host filesystem path the store
    keeps internally.
    """
    fields = ("id", "session_id", "ts", "role", "modality", "language", "text")
    shaped: list[dict[str, object]] = []
    for turn in turns:
        row: dict[str, object] = {key: turn[key] for key in fields}
        row["has_audio"] = bool(turn.get("audio_path"))
        shaped.append(row)
    return shaped


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
    return {"session": session, "turns": _client_turns(store.session_turns(session_id))}


@app.get("/api/sessions/{session_id}/turns/{turn_id}/audio")
def turn_audio(session_id: str, turn_id: int) -> FileResponse:
    """Serve a turn's archived voice WAV for in-transcript playback (SPEC §8).

    The client asks by session + turn id; the store resolves the path from
    its own record, so no client-supplied path reaches the filesystem. Raw
    utterance audio stays on the host, within the tailnet the whole app is
    already scoped to.
    """
    path = _store().turn_audio_path(session_id, turn_id)
    if path is None:
        raise HTTPException(status_code=404, detail="No audio for this turn")
    return FileResponse(path, media_type="audio/wav")


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, request: Request) -> dict[str, str]:
    """Rename a session (user edit of the auto-generated title)."""
    body = await request.json()
    title = str(body.get("title", "")).strip()[:80]
    if not title:
        raise HTTPException(status_code=400, detail="Title must not be empty")
    if not _store().set_title(session_id, title):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"id": session_id, "title": title}


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
