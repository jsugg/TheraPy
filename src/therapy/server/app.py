"""FastAPI backend: WebRTC signaling + PWA static serving (SPEC §5)."""

import json
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from therapy.knowledge.research import ResearchKB
    from therapy.knowledge.user_model import UserModel

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from therapy import __version__
from therapy.memory import MemoryStore
from therapy.memory.store import RowDict, resume_window_secs
from therapy.server import live
from therapy.server.protocol import session_state_message
from therapy.voice.contracts import (
    ConnectionConflict,
    InvalidOffer,
    SessionTarget,
    VoiceUnavailable,
    WebRTCOffer,
)
from therapy.voice.ports import VoiceGateway

STATIC_DIR = Path(__file__).parent / "static"
MAX_OFFER_BODY_BYTES = 256 * 1024

_voice_gateway: VoiceGateway | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _voice_gateway
    yield
    gateway, _voice_gateway = _voice_gateway, None
    if gateway is not None:
        await gateway.close()


app = FastAPI(title="TheraPy", version=__version__, lifespan=lifespan)


@lru_cache(maxsize=1)
def _store() -> MemoryStore:
    """Return the lazily initialized local memory store."""
    return MemoryStore()


@lru_cache(maxsize=1)
def _model() -> "UserModel":
    """Return the lazily initialized property-graph user model.

    Shares the store's data directory (both open `therapy.db`), so the review
    UI reads and edits the same graph the conversation writes.
    """
    from therapy.knowledge.user_model import UserModel

    return UserModel()


@lru_cache(maxsize=1)
def _research() -> "ResearchKB":
    """Return the lazily initialized curated research knowledge base."""
    from therapy.knowledge.research import ResearchKB

    return ResearchKB()


def get_voice_gateway() -> VoiceGateway:
    """Return the lazy process-wide voice runtime implementation."""
    global _voice_gateway
    if _voice_gateway is None:
        from therapy.integrations.pipecat.runtime import PipecatVoiceGateway

        _voice_gateway = PipecatVoiceGateway()
    return _voice_gateway


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


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
async def offer(
    request: Request,
    gateway: Annotated[VoiceGateway, Depends(get_voice_gateway)],
) -> dict[str, object]:
    """SDP offer → answer; the new connection preempts the previous pipeline.

    `?new_session=1` skips reconnect-resume so the pipeline always opens a
    fresh session — test scripts need isolated sessions; real clients
    resume an interrupted one (SPEC §8). `?session=<id>` continues that
    specific session (the history browser's explicit choice).

    The answer carries `session_id`/`resumed` so the client can load the
    transcript over HTTP; the resolved id is also what run_bot joins, so the
    two never disagree.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_OFFER_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Offer body is too large")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from None
    body = await request.body()
    if len(body) > MAX_OFFER_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Offer body is too large")
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, RecursionError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON body") from None
    try:
        webrtc_offer = WebRTCOffer.from_payload(payload)
    except InvalidOffer as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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

    try:
        answer = await gateway.negotiate(
            webrtc_offer,
            SessionTarget(session_id=resolved, new_session=resolved is None),
        )
    except ConnectionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidOffer as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except VoiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        **answer.as_payload(),
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


# --------------------------------------------------------------------- #
# Review-UI sovereignty (W7): the user's model of themselves, browsable,
# editable, deletable. "The assistant is its curator, not its owner."
# --------------------------------------------------------------------- #


@app.get("/api/graph")
def graph() -> dict[str, object]:
    """Return the whole user-model graph plus boundaries and pending inbox."""
    model = _model()
    return {
        "nodes": model.nodes(),
        "edges": model.edges(),
        "boundaries": model.boundaries(),
        "pending_insights": model.pending_insights(),
    }


@app.get("/api/graph/pending")
def pending_insights() -> dict[str, object]:
    """Proposed patterns awaiting the user's confirm/reject (W4 inbox)."""
    return {"pending_insights": _model().pending_insights()}


@app.patch("/api/graph/nodes/{node_id}")
async def edit_node(node_id: int, request: Request) -> dict[str, object]:
    """Edit a node's statement or its `never_initiate` flag (user sovereignty)."""
    body = await request.json()
    statement = body.get("statement")
    never_initiate = body.get("never_initiate")
    if statement is None and never_initiate is None:
        raise HTTPException(status_code=400, detail="Nothing to update")
    ok = _model().edit_node(
        node_id,
        statement=str(statement).strip() if statement is not None else None,
        never_initiate=bool(never_initiate) if never_initiate is not None else None,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Node not found")
    return {"node": _model().get_node(node_id)}


@app.post("/api/graph/nodes/{node_id}/confirm")
def confirm_node(node_id: int) -> dict[str, object]:
    """Explicitly validate a claim — the only path to `confirmed` (SPEC §3)."""
    if not _model().confirm_node(node_id):
        raise HTTPException(status_code=404, detail="Node not found")
    return {"node": _model().get_node(node_id)}


@app.post("/api/graph/nodes/{node_id}/reject")
def reject_node(node_id: int) -> dict[str, object]:
    """Reject a proposed pattern, demoting it back to an observation."""
    if not _model().reject_node(node_id):
        raise HTTPException(status_code=404, detail="Node not found")
    return {"node": _model().get_node(node_id)}


@app.delete("/api/graph/nodes/{node_id}")
def delete_node(node_id: int) -> dict[str, object]:
    """Delete a node (tombstoned so distillation cannot re-learn it)."""
    if not _model().delete_node(node_id):
        raise HTTPException(status_code=404, detail="Node not found")
    return {"deleted": node_id}


@app.delete("/api/graph/edges/{edge_id}")
def delete_edge(edge_id: int) -> dict[str, object]:
    """Delete an edge (tombstoned against re-learning)."""
    if not _model().delete_edge(edge_id):
        raise HTTPException(status_code=404, detail="Edge not found")
    return {"deleted": edge_id}


@app.get("/api/graph/boundaries")
def boundaries() -> dict[str, object]:
    """Return the editable `never_store` / `never_initiate` boundaries."""
    return {"boundaries": _model().boundaries()}


@app.post("/api/graph/boundaries")
async def add_boundary(request: Request) -> dict[str, object]:
    """Add a `never_store` pattern or `never_initiate` topic."""
    body = await request.json()
    kind = str(body.get("kind", ""))
    value = str(body.get("value", "")).strip()
    if kind not in {"never_store", "never_initiate"} or not value:
        raise HTTPException(status_code=400, detail="kind and value required")
    _model().add_boundary(kind, value)
    return {"boundaries": _model().boundaries()}


@app.delete("/api/graph/boundaries")
async def remove_boundary(request: Request) -> dict[str, object]:
    """Remove a boundary by kind + value."""
    body = await request.json()
    kind = str(body.get("kind", ""))
    value = str(body.get("value", ""))
    if not _model().remove_boundary(kind, value):
        raise HTTPException(status_code=404, detail="Boundary not found")
    return {"boundaries": _model().boundaries()}


@app.get("/api/research")
def research_documents() -> dict[str, object]:
    """List the curated research documents the user has ingested."""
    return {"documents": _research().documents()}


@app.get("/api/research/query")
def research_query(q: str, k: int = 3) -> dict[str, object]:
    """Answer a psychoeducation query from the literature, with citations."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    return _research().psychoeducation(q, k=k)


@app.get("/api/crisis-resources")
def crisis_resources_config() -> dict[str, object]:
    """The configurable crisis hotlines/contacts surfaced in the UI (W8)."""
    from therapy.dialogue.policy import crisis_contacts, crisis_resources

    return {"contacts": crisis_contacts(), "resources": crisis_resources()}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
