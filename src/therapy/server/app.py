"""FastAPI backend: WebRTC signaling + PWA static serving (SPEC §5)."""

import asyncio
import json
import os
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

if TYPE_CHECKING:
    from therapy.acceptance import AcceptanceAgent
    from therapy.data import DataSovereignty
    from therapy.dialogue.outreach import ProactivityService
    from therapy.knowledge.insight import InsightService
    from therapy.knowledge.research import ResearchKB
    from therapy.knowledge.user_model import UserModel

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from therapy import __version__
from therapy.memory import MemoryStore
from therapy.memory.store import RowDict, resume_window_secs
from therapy.server import live
from therapy.server.protocol import session_state_message
from therapy.server.schemas import (
    AcceptanceAgentTurn,
    AcceptanceOutreachRun,
    BoundaryRequest,
    ClientTelemetryBatch,
    DeleteAllRequest,
    GraphEdgePatch,
    GraphNodePatch,
    InsightSnoozeRequest,
    ProactivityChannelPatch,
    PushSubscriptionRequest,
    RenameSessionRequest,
    ResearchBlockCorrection,
)
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
    from therapy.dialogue.outreach import ProactivityScheduler
    from therapy.observability import telemetry
    from therapy.observability.capture import start_capture
    from therapy.observability.config import ObservabilityConfig
    from therapy.observability.logging import emit_event

    emit_event(
        "app.starting", component="server", operation="lifecycle",
        outcome="success",
    )
    if os.getenv("THERAPY_TEST_MODE") == "1" and os.getenv(
        "THERAPY_ENVIRONMENT", ""
    ) not in ("test", "acceptance"):
        # Test-mode routes outside an explicit test deployment are a
        # misconfiguration alarm (plan O3.1), never silent.
        import logging as logging_module

        emit_event(
            "test_mode_outside_test_deployment",
            severity=logging_module.ERROR,
            component="server",
            operation="lifecycle",
            outcome="error",
        )

    # Interaction capture (plan O1.1/O1.2): journal opens and recovers before
    # any LLM boundary can run; failure degrades visibly, never blocks start.
    capture_runtime = await start_capture(
        ObservabilityConfig.from_env(), build_version=__version__
    )

    scheduler = ProactivityScheduler(_proactivity())
    scheduler_task = asyncio.create_task(scheduler.run(), name="therapy-proactivity")

    emit_event(
        "app.ready", component="server", operation="lifecycle", outcome="success"
    )
    try:
        yield
    finally:
        # Shutdown order (plan O1.1): scheduler -> finalizers/gateway ->
        # journal/exporter flush (bounded) -> OTel. Never waits indefinitely.
        emit_event(
            "app.stopping", component="server", operation="lifecycle",
            outcome="success",
        )
        scheduler.stop()
        try:
            await asyncio.wait_for(scheduler_task, 10.0)
        except TimeoutError:
            # A hung tick must not stall shutdown, but its cancellation is
            # awaited so the drain is bounded AND complete (O3 audit).
            scheduler_task.cancel()
            try:
                await asyncio.wait_for(scheduler_task, 5.0)
            except (TimeoutError, asyncio.CancelledError):
                emit_event(
                    "component.scheduler", component="server",
                    operation="shutdown", outcome="timeout",
                )
        except asyncio.CancelledError:
            pass
        gateway, _voice_gateway = _voice_gateway, None
        if gateway is not None:
            try:
                await asyncio.wait_for(gateway.close(), 15.0)
            except Exception as exc:
                # Swallowed for shutdown progress, but never silently (O3
                # audit): one bounded component event, class only.
                emit_event(
                    "component.gateway", component="voice",
                    operation="shutdown", outcome="error",
                    error_type=type(exc).__name__,
                )
        await capture_runtime.close()
        telemetry.shutdown()
        emit_event(
            "app.stopped", component="server", operation="lifecycle",
            outcome="success",
        )


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
    if os.getenv("THERAPY_TEST_MODE") == "1":
        return _acceptance_agent().research
    from therapy.knowledge.research import ResearchKB

    return ResearchKB()


@lru_cache(maxsize=1)
def _insights() -> "InsightService":
    """Return the durable insight queue service."""
    from therapy.knowledge.insight import InsightService

    return InsightService(_model())


@lru_cache(maxsize=1)
def _proactivity() -> "ProactivityService":
    """Return the persistent outreach service used by API and scheduler."""
    from therapy.dialogue.outreach import ProactivityService

    return ProactivityService(model=_model())


@lru_cache(maxsize=1)
def _data() -> "DataSovereignty":
    """Return the complete owner-data coordinator over shared services."""
    from therapy.data import DataSovereignty

    return DataSovereignty(
        store=_store(),
        model=_model(),
        research=_research(),
        proactivity=_proactivity(),
    )


@lru_cache(maxsize=1)
def _acceptance_agent() -> "AcceptanceAgent":
    """Return the deterministic agent only when explicit test mode is active."""
    from therapy.acceptance import AcceptanceAgent

    return AcceptanceAgent(_model().data_dir)


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


@app.get("/ready")
def ready() -> dict[str, object]:
    """Readiness model (obs plan O3.1): bounded checks, enums only.

    External LLM/push state may degrade readiness detail but never fails
    liveness or restarts the process. No paths, errors, or IDs in the body.
    """
    import shutil
    import socket
    import time

    checks: dict[str, str] = {}
    data_dir = Path(os.environ.get("THERAPY_DATA_DIR", "data"))
    db_path = data_dir / "therapy.db"

    try:
        connection = sqlite3.connect(db_path, timeout=2.0)
        try:
            connection.execute("SELECT 1").fetchone()
            row = connection.execute("PRAGMA user_version").fetchone()
            checks["db"] = "ready"
            version = int(row[0]) if row else -1
            # Frozen enum: known versions only, never an arbitrary number.
            checks["schema"] = f"v{version}" if 0 <= version <= 16 else "unknown"
        finally:
            connection.close()
    except sqlite3.Error:
        checks["db"] = "degraded"
        checks["schema"] = "unknown"

    try:
        db_bytes = db_path.stat().st_size if db_path.exists() else 0
        wal_path = Path(str(db_path) + "-wal")
        wal_bytes = wal_path.stat().st_size if wal_path.exists() else 0
        checks["db_size"] = "ok" if db_bytes < 1_073_741_824 else "large"
        checks["wal_size"] = "ok" if wal_bytes < 67_108_864 else "large"
    except OSError:
        checks["db_size"] = "unknown"
        checks["wal_size"] = "unknown"

    try:
        probe = data_dir / ".ready-probe"
        probe.write_text("ok")
        probe.unlink()
        free = shutil.disk_usage(data_dir).free
        checks["data_dir"] = "ready" if free > 512 * 1024 * 1024 else "degraded"
    except OSError:
        checks["data_dir"] = "degraded"

    from therapy.dialogue import outreach
    from therapy.observability.capture import capture_service
    from therapy.observability.health import registry
    from therapy.observability.telemetry import state as telemetry_state

    last_tick = outreach.last_scheduler_tick()
    if last_tick is None:
        checks["scheduler"] = "starting"
    else:
        checks["scheduler"] = (
            "ready" if time.time() - last_tick < 150 else "degraded"
        )

    # TURN synthetic reachability: external relay state degrades detail
    # only (enum ready|unreachable), never overall readiness (plan O3.1).
    try:
        with socket.create_connection(
            (
                os.environ.get("THERAPY_TURN_PROBE_HOST", "turn"),
                int(os.environ.get("THERAPY_TURN_PORT", "3478")),
            ),
            timeout=0.5,
        ):
            checks["turn"] = "ready"
    except OSError:
        checks["turn"] = "unreachable"

    capture = capture_service()
    checks["capture"] = (
        "ready" if capture is not None and capture.writer is not None else "degraded"
    )
    checks["telemetry"] = "ready" if telemetry_state().enabled else "disabled"
    checks["voice"] = "ready" if _voice_gateway is not None else "starting"
    for name, snapshot in registry().snapshot().items():
        checks[f"component.{name}"] = str(snapshot["state"])

    degraded = [name for name, state in checks.items() if state == "degraded"]
    return {
        "status": "degraded" if degraded else "ready",
        "checks": checks,
    }


@contextmanager
def _audit(operation: str, component: str = "data") -> Generator[None, None, None]:
    """Minimal content-free audit event for destructive/research/data
    operations (obs plan O3.1); never IDs, names, paths, or payloads.

    Exactly one terminal event fires AFTER the wrapped operation resolves,
    with a bounded outcome — a validation reject or crash must never be
    recorded as success (O3 audit finding)."""
    from therapy.observability.logging import emit_event

    outcome = "success"
    try:
        yield
    except HTTPException as exc:
        outcome = "rejected" if exc.status_code < 500 else "error"
        raise
    except Exception:
        outcome = "error"
        raise
    finally:
        emit_event(
            "owner.audit",
            component=component,
            operation=operation,
            outcome=outcome,
        )


def _read_rows(route_class: str, count: int) -> None:
    """Bounded result-count evidence for read routes (plan O3.1).

    Only the frozen route class and count bucket become labels — no
    filters, SQL, content, or IDs."""
    from therapy.observability.model import count_bucket
    from therapy.observability.telemetry import record_metric

    record_metric(
        "therapy_http_read_rows_total",
        1,
        {"route_class": route_class, "bucket": count_bucket(count)},
    )


def resolve_session(
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
    from therapy.observability.telemetry import broad_span, record_metric

    # Stage children (plan O3.1): validate / resolve / transcript-state /
    # negotiate, so the waterfall isolates each cost. No body/SDP/IDs broadly.
    try:
        with broad_span("offer.validate", component="voice", operation="validate"):
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > MAX_OFFER_BODY_BYTES:
                        raise HTTPException(
                            status_code=413, detail="Offer body is too large"
                        )
                except ValueError:
                    raise HTTPException(
                        status_code=400, detail="Invalid Content-Length"
                    ) from None
            body = await request.body()
            if len(body) > MAX_OFFER_BODY_BYTES:
                raise HTTPException(
                    status_code=413, detail="Offer body is too large"
                )
            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, RecursionError, UnicodeDecodeError):
                raise HTTPException(
                    status_code=400, detail="Invalid JSON body"
                ) from None
            try:
                webrtc_offer = WebRTCOffer.from_payload(payload)
            except InvalidOffer as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        record_metric("therapy_offers_total", 1, {"outcome": "rejected"})
        raise

    store = _store()
    new_session = request.query_params.get("new_session") == "1"
    with broad_span("offer.resolve_session", component="voice", operation="resolve"):
        resolved, resumed = resolve_session(
            store,
            new_session=new_session,
            explicit=request.query_params.get("session"),
        )
    # Carry the resumed transcript in the answer so the client renders it
    # synchronously on connect — an async fetch after connect raced a reconnect
    # (rendering the wrong session) and live turns (duplicating/mis-ordering).
    with broad_span(
        "offer.transcript_state", component="voice", operation="read"
    ):
        state = session_state_message(
            resolved or "", resumed, store.session_turns(resolved) if resolved else []
        )

    try:
        with broad_span("offer.negotiate", component="voice", operation="negotiate"):
            answer = await gateway.negotiate(
                webrtc_offer,
                SessionTarget(session_id=resolved, new_session=resolved is None),
            )
    except ConnectionConflict as exc:
        record_metric("therapy_offers_total", 1, {"outcome": "conflict"})
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidOffer as exc:
        record_metric("therapy_offers_total", 1, {"outcome": "rejected"})
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except VoiceUnavailable as exc:
        record_metric("therapy_offers_total", 1, {"outcome": "unavailable"})
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    record_metric(
        "therapy_offers_total", 1, {"outcome": "resumed" if resumed else "fresh"}
    )
    return {
        **answer.as_payload(),
        "session_id": resolved,
        "resumed": state["resumed"],
        "turns": state["turns"],
    }


@app.post("/api/voice/disconnect")
async def disconnect_voice(
    pc_id: Annotated[str, Query(min_length=1, max_length=256)],
    gateway: Annotated[VoiceGateway, Depends(get_voice_gateway)],
) -> dict[str, bool]:
    """End only the caller's current peer, leaving the voice runtime reusable."""
    from therapy.observability.telemetry import record_metric

    try:
        disconnected = await gateway.disconnect(pc_id)
    except VoiceUnavailable as exc:
        record_metric(
            "therapy_voice_signal_total",
            1,
            {"operation": "disconnect", "outcome": "unavailable"},
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    record_metric(
        "therapy_voice_signal_total",
        1,
        {
            "operation": "disconnect",
            "outcome": "disconnected" if disconnected else "stale",
        },
    )
    return {"disconnected": disconnected}


@app.get("/api/ice-config")
def ice_config() -> dict[str, object]:
    """TURN credentials for the compose relay (SPEC §5 clients).

    The client builds the URL itself from the page's hostname —
    `turn:<host>:<port>` — so the same config works on localhost, LAN,
    and the tailnet. Credentials are static (tailnet-only exposure; the
    VPN identity is the auth boundary for the MVP, SPEC §8).
    """
    from therapy.observability.telemetry import record_metric

    record_metric(
        "therapy_voice_signal_total",
        1,
        {"operation": "ice_config", "outcome": "success"},
    )
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
    from therapy.observability.telemetry import record_metric

    candidate = _store().resume_candidate(resume_window_secs())
    record_metric(
        "therapy_voice_signal_total",
        1,
        {
            "operation": "resumable",
            "outcome": "resumable" if candidate else "none",
        },
    )
    return {"session_id": candidate}


@app.get("/api/sessions")
def sessions() -> dict[str, list[dict[str, object]]]:
    """Return session summaries with turn counts for the review UI."""
    store = _store()
    session_rows: list[dict[str, object]] = []
    for session in store.sessions():
        session_row: dict[str, object] = dict(session)
        session_row["turn_count"] = len(store.session_turns(str(session["id"])))
        session_rows.append(session_row)
    _read_rows("sessions", len(session_rows))
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
    turns = _client_turns(store.session_turns(session_id))
    _read_rows("session_detail", len(turns))
    return {"session": session, "turns": turns}


@app.get("/api/sessions/{session_id}/turns/{turn_id}/audio")
def turn_audio(session_id: str, turn_id: int, request: Request) -> FileResponse:
    """Serve a turn's archived voice WAV for in-transcript playback (SPEC §8).

    The client asks by session + turn id; the store resolves the path from
    its own record, so no client-supplied path reaches the filesystem. Raw
    utterance audio stays on the host, within the tailnet the whole app is
    already scoped to.
    """
    from therapy.observability.telemetry import record_metric

    path = _store().turn_audio_path(session_id, turn_id)
    if path is None:
        record_metric("therapy_audio_serve_total", 1, {"outcome": "missing"})
        raise HTTPException(status_code=404, detail="No audio for this turn")
    # range|full outcome + response bytes (plan O3.1); never the archive path.
    outcome = "range" if request.headers.get("range") else "full"
    record_metric("therapy_audio_serve_total", 1, {"outcome": outcome})
    try:
        record_metric("therapy_audio_serve_bytes", Path(path).stat().st_size)
    except OSError:
        pass
    return FileResponse(path, media_type="audio/wav")


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, body: RenameSessionRequest) -> dict[str, str]:
    """Rename a session (user edit of the auto-generated title)."""
    if not _store().set_title(session_id, body.title):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"id": session_id, "title": body.title}


@app.delete("/api/sessions/{session_id}")
def delete_session(
    session_id: str,
    mode: Literal["keep_knowledge", "remove_derived"] = "keep_knowledge",
) -> dict[str, object]:
    """Delete conversation artifacts under the owner's selected knowledge policy."""
    from therapy.observability.telemetry import broad_span

    # Distinct staged children (plan O3.1): guard / evidence-policy / delete
    # (the store adds db-delete and audio-delete children of its own).
    with _audit("delete_session", "memory"):
        store = _store()
        with broad_span(
            "session_delete.guard", component="memory", operation="guard"
        ):
            if not store.has_session(session_id):
                raise HTTPException(status_code=404, detail="Session not found")
            if live.is_active(session_id):
                raise HTTPException(
                    status_code=409, detail="Session is live — disconnect first"
                )
        with broad_span(
            "session_delete.evidence_policy",
            component="knowledge",
            operation="delete",
        ):
            provenance = _model().delete_session_evidence(
                session_id, remove_derived=mode == "remove_derived"
            )
        store.delete_session(session_id)
        return {
            "deleted": session_id,
            "mode": mode,
            "learned_knowledge_survives": mode == "keep_knowledge",
            "provenance": provenance,
        }


# --------------------------------------------------------------------- #
# Review-UI sovereignty (W7): the user's model of themselves, browsable,
# editable, deletable. "The assistant is its curator, not its owner."
# --------------------------------------------------------------------- #


@app.get("/api/graph")
def graph(
    node_type: str | None = None,
    edge_type: str | None = None,
    status: str | None = None,
    source: str | None = None,
) -> dict[str, object]:
    """Return filterable graph data and the durable review queue."""
    model = _model()
    try:
        nodes = model.nodes(type_=node_type, status=status)
        edges = model.edges(type_=edge_type, status=status)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if source is not None:
        if source not in {"conversation", "distilled", "user-stated", "inferred"}:
            raise HTTPException(status_code=422, detail="Unknown claim source")
        nodes = [node for node in nodes if node["source"] == source]
        edges = [edge for edge in edges if edge["source"] == source]
    insights = _insights()
    insights.sync_proposals()
    _read_rows("graph", len(nodes) + len(edges))
    return {
        "nodes": nodes,
        "edges": edges,
        "boundaries": model.boundaries(),
        "pending_insights": insights.list(),
    }


@app.get("/api/graph/pending")
def pending_insights(
    state: Literal[
        "queued", "delivered", "snoozed", "confirmed", "rejected", "dismissed"
    ]
    | None = None,
) -> dict[str, object]:
    """Return durable pending insight queue records."""
    service = _insights()
    service.sync_proposals()
    pending = service.list(state=state)
    _read_rows("graph_pending", len(pending))
    return {"pending_insights": pending}


@app.get("/api/graph/nodes/{node_id}")
def node_detail(node_id: int) -> dict[str, object]:
    """Return a node with normalized evidence and lifecycle audit."""
    model = _model()
    node = model.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return {
        "node": node,
        "evidence": model.evidence("node", node_id),
        "lifecycle": model.lifecycle_events("node", node_id),
    }


@app.get("/api/graph/edges/{edge_id}")
def edge_detail(edge_id: int) -> dict[str, object]:
    """Return an edge with normalized evidence and lifecycle audit."""
    model = _model()
    edge = model.get_edge(edge_id)
    if edge is None:
        raise HTTPException(status_code=404, detail="Edge not found")
    return {
        "edge": edge,
        "evidence": model.evidence("edge", edge_id),
        "lifecycle": model.lifecycle_events("edge", edge_id),
    }


@app.patch("/api/graph/nodes/{node_id}")
def edit_node(node_id: int, body: GraphNodePatch) -> dict[str, object]:
    """Edit a node's statement or its `never_initiate` flag (user sovereignty)."""
    insights = _insights()
    insights.sync_proposals()
    ok = _model().edit_node(
        node_id,
        statement=body.statement,
        never_initiate=body.never_initiate,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Node not found")
    insights.dismiss_claim("node", node_id)
    return {"node": _model().get_node(node_id)}


@app.post("/api/graph/nodes/{node_id}/confirm")
def confirm_node(node_id: int) -> dict[str, object]:
    """Explicitly validate a claim — the only path to `confirmed` (SPEC §3)."""
    model = _model()
    node = model.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    if node["status"] != "proposed":
        raise HTTPException(
            status_code=409, detail="Only proposed nodes can be confirmed"
        )
    if not _insights().resolve_claim("node", node_id, "confirmed"):
        raise HTTPException(status_code=409, detail="Node state changed")
    return {"node": model.get_node(node_id)}


@app.post("/api/graph/nodes/{node_id}/reject")
def reject_node(node_id: int) -> dict[str, object]:
    """Durably reject one proposed node at its evidence snapshot."""
    model = _model()
    node = model.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    if node["status"] != "proposed":
        raise HTTPException(
            status_code=409, detail="Only proposed nodes can be rejected"
        )
    if not _insights().resolve_claim("node", node_id, "rejected"):
        raise HTTPException(status_code=409, detail="Node state changed")
    return {"node": model.get_node(node_id)}


@app.delete("/api/graph/nodes/{node_id}")
def delete_node(node_id: int) -> dict[str, object]:
    """Delete a node (tombstoned so distillation cannot re-learn it)."""
    with _audit("delete_node", "knowledge"):
        if not _model().delete_node(node_id):
            raise HTTPException(status_code=404, detail="Node not found")
        return {"deleted": node_id}


@app.patch("/api/graph/edges/{edge_id}")
def edit_edge(edge_id: int, body: GraphEdgePatch) -> dict[str, object]:
    """Apply an authoritative owner edit to an edge statement."""
    insights = _insights()
    insights.sync_proposals()
    if not _model().edit_edge(edge_id, statement=body.statement):
        raise HTTPException(status_code=404, detail="Edge not found")
    insights.dismiss_claim("edge", edge_id)
    return {"edge": _model().get_edge(edge_id)}


@app.post("/api/graph/edges/{edge_id}/confirm")
def confirm_edge(edge_id: int) -> dict[str, object]:
    """Explicitly validate one proposed edge."""
    model = _model()
    edge = model.get_edge(edge_id)
    if edge is None:
        raise HTTPException(status_code=404, detail="Edge not found")
    if edge["status"] != "proposed":
        raise HTTPException(
            status_code=409, detail="Only proposed edges can be confirmed"
        )
    if not _insights().resolve_claim("edge", edge_id, "confirmed"):
        raise HTTPException(status_code=409, detail="Edge state changed")
    return {"edge": model.get_edge(edge_id)}


@app.post("/api/graph/edges/{edge_id}/reject")
def reject_edge(edge_id: int) -> dict[str, object]:
    """Durably reject one proposed edge at its evidence snapshot."""
    model = _model()
    edge = model.get_edge(edge_id)
    if edge is None:
        raise HTTPException(status_code=404, detail="Edge not found")
    if edge["status"] != "proposed":
        raise HTTPException(
            status_code=409, detail="Only proposed edges can be rejected"
        )
    if not _insights().resolve_claim("edge", edge_id, "rejected"):
        raise HTTPException(status_code=409, detail="Edge state changed")
    return {"edge": model.get_edge(edge_id)}


@app.delete("/api/graph/edges/{edge_id}")
def delete_edge(edge_id: int) -> dict[str, object]:
    """Delete an edge (tombstoned against re-learning)."""
    with _audit("delete_edge", "knowledge"):
        if not _model().delete_edge(edge_id):
            raise HTTPException(status_code=404, detail="Edge not found")
        return {"deleted": edge_id}


@app.get("/api/graph/boundaries")
def boundaries() -> dict[str, object]:
    """Return the editable `never_store` / `never_initiate` boundaries."""
    return {"boundaries": _model().boundaries()}


@app.post("/api/graph/boundaries")
def add_boundary(body: BoundaryRequest) -> dict[str, object]:
    """Add a `never_store` pattern or `never_initiate` topic."""
    _model().add_boundary(body.kind, body.value)
    return {"boundaries": _model().boundaries()}


@app.delete("/api/graph/boundaries")
def remove_boundary(body: BoundaryRequest) -> dict[str, object]:
    """Remove a boundary by kind + value."""
    with _audit("remove_boundary", "knowledge"):
        if not _model().remove_boundary(body.kind, body.value):
            raise HTTPException(status_code=404, detail="Boundary not found")
        return {"boundaries": _model().boundaries()}


@app.post("/api/insights/{insight_id}/confirm")
def confirm_insight(insight_id: str) -> dict[str, object]:
    """Confirm the exact durable queue record selected by the owner."""
    insight = _insights().resolve(insight_id, "confirmed")
    if insight is None:
        raise HTTPException(
            status_code=409, detail="Insight is missing or not resolvable"
        )
    return {"insight": insight}


@app.post("/api/insights/{insight_id}/reject")
def reject_insight(insight_id: str) -> dict[str, object]:
    """Reject the exact durable queue record selected by the owner."""
    insight = _insights().resolve(insight_id, "rejected")
    if insight is None:
        raise HTTPException(
            status_code=409, detail="Insight is missing or not resolvable"
        )
    return {"insight": insight}


@app.post("/api/insights/{insight_id}/snooze")
def snooze_insight(insight_id: str, body: InsightSnoozeRequest) -> dict[str, object]:
    """Snooze an unresolved insight for an owner-selected duration."""
    if not _insights().snooze(insight_id, days=body.days):
        raise HTTPException(
            status_code=409, detail="Insight is missing or not snoozable"
        )
    return {"insight_id": insight_id, "state": "snoozed", "days": body.days}


@app.post("/api/insights/{insight_id}/dismiss")
def dismiss_insight(insight_id: str) -> dict[str, object]:
    """Dismiss a delivery without mutating the graph claim."""
    if not _insights().dismiss(insight_id):
        raise HTTPException(status_code=409, detail="Insight is missing or resolved")
    return {"insight_id": insight_id, "state": "dismissed"}


@app.get("/api/insights/{insight_id}/history")
def insight_history(insight_id: str) -> dict[str, object]:
    """Return delivery and owner-resolution audit events."""
    return {"history": _insights().history(insight_id)}


@app.get("/api/research")
def research_documents() -> dict[str, object]:
    """List the curated research documents the user has ingested."""
    documents = _research().documents()
    _read_rows("research_documents", len(documents))
    return {"documents": documents}


@app.post("/api/research/ingest")
async def research_ingest(
    file: Annotated[UploadFile, File(description="Owner-local research source")],
    source_title: Annotated[str | None, Form(max_length=300)] = None,
    source_ref: Annotated[str | None, Form(max_length=1_000)] = None,
    force: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    """Validate, extract/OCR, preserve, and index one local source."""
    from therapy.knowledge.research_ingest import MAX_SOURCE_BYTES

    with _audit("research_ingest", "research"):
        filename = file.filename or ""
        payload = await file.read(MAX_SOURCE_BYTES + 1)
        await file.close()
        try:
            result = _research().ingest_bytes(
                payload,
                filename,
                file.content_type,
                source_title=source_title,
                source_ref=source_ref,
                force=force,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "ingest": result,
            "document": _research().document(result["document_id"]),
        }


@app.get("/api/research/query")
def research_query(
    q: Annotated[str, Query(min_length=1, max_length=2_000)],
    k: Annotated[int, Query(ge=1, le=20)] = 3,
    threshold: Annotated[float, Query(ge=0, le=1)] = 0.28,
) -> dict[str, object]:
    """Answer a psychoeducation query from the literature, with citations."""
    results = _research().query(q, k=k, threshold=threshold)
    _read_rows("research_query", len(results))
    return {
        "answer": "\n\n".join(
            f"{result['text']} {result['citation']}" for result in results
        ),
        "sources": [
            {
                "document_id": result["document_id"],
                "title": result["source_title"],
                "ref": result["source_ref"],
                "page": result["page"],
                "section": result["heading"],
                "anchor": result["anchor"],
                "citation": result["citation"],
            }
            for result in results
        ],
    }


@app.get("/api/research/{document_id}")
def research_document(document_id: int) -> dict[str, object]:
    """Return source metadata and OCR/digital block preview."""
    document = _research().document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Research document not found")
    return {"document": document}


@app.patch("/api/research/{document_id}/blocks/{anchor}")
def correct_research_block(
    document_id: int, anchor: str, body: ResearchBlockCorrection
) -> dict[str, object]:
    """Apply one owner OCR correction and rebuild the semantic index."""
    with _audit("correct_research_block", "research"):
        try:
            changed = _research().correct_block(document_id, anchor, body.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not changed:
            raise HTTPException(status_code=404, detail="Research block not found")
        return {"document": _research().document(document_id)}


@app.post("/api/research/{document_id}/reindex")
def reindex_research(document_id: int) -> dict[str, int]:
    """Rebuild one document using the configured model/policy version."""
    with _audit("reindex_research", "research"):
        if _research().document(document_id) is None:
            raise HTTPException(
                status_code=404, detail="Research document not found"
            )
        return {"chunks_indexed": _research().reindex(document_id)}


@app.delete("/api/research/{document_id}")
def delete_research(document_id: int) -> dict[str, int]:
    """Delete one source artifact, extraction, and semantic index."""
    with _audit("delete_research", "research"):
        if not _research().delete_document(document_id):
            raise HTTPException(
                status_code=404, detail="Research document not found"
            )
        return {"deleted": document_id}


@app.get("/api/proactivity")
def proactivity_settings() -> dict[str, object]:
    """Return all four opt-in channel settings."""
    channels = _proactivity().settings()
    _read_rows("proactivity_settings", len(channels))
    return {"channels": channels}


@app.put("/api/proactivity/{channel}")
def update_proactivity(
    channel: Literal["push", "greeting", "check_in", "digest"],
    body: ProactivityChannelPatch,
) -> dict[str, object]:
    """Validate and persist one channel's owner controls."""
    try:
        settings = _proactivity().update_settings(
            channel,
            enabled=body.enabled,
            timezone=body.timezone,
            quiet_start=body.quiet_start,
            quiet_end=body.quiet_end,
            schedule_time=body.schedule_time,
            schedule_day=body.schedule_day,
            frequency=body.frequency,
            topic=body.topic,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"channel": settings}


@app.get("/api/proactivity/jobs")
def proactivity_jobs() -> dict[str, object]:
    """Return the observable persistent outreach ledger."""
    return {"jobs": _proactivity().jobs()}


@app.get("/api/push/public-key")
def push_public_key() -> dict[str, str]:
    """Return the local VAPID application-server public key."""
    return {"public_key": _proactivity().vapid.public_key()}


@app.post("/api/push/subscriptions")
def add_push_subscription(body: PushSubscriptionRequest) -> dict[str, str]:
    """Persist an encrypted Web Push subscription endpoint and keys."""
    subscription_id = _proactivity().subscribe(body.endpoint, body.p256dh, body.auth)
    return {"subscription_id": subscription_id}


@app.delete("/api/push/subscriptions/{subscription_id}")
def remove_push_subscription(subscription_id: str) -> dict[str, str]:
    """Deactivate a browser subscription."""
    if not _proactivity().unsubscribe(subscription_id):
        raise HTTPException(status_code=404, detail="Push subscription not found")
    return {"deleted": subscription_id}


@app.get("/api/proactivity/in-app")
def in_app_outreach(consume: bool = True) -> dict[str, object]:
    """Queue today's opted-in greeting and return local unseen outreach."""
    service = _proactivity()
    job_id = service.queue_greeting()
    if job_id is not None:
        service.deliver(job_id)
    return {"messages": service.in_app_messages(consume=consume)}


@app.get("/api/proactivity/digests")
def proactivity_digests() -> dict[str, object]:
    """Return owner-local written daily/weekly reflection digests."""
    digests = _proactivity().digests()
    _read_rows("proactivity_digests", len(digests))
    return {"digests": digests}


def _assert_no_live_sessions() -> None:
    if any(live.is_active(str(session["id"])) for session in _store().sessions()):
        raise HTTPException(
            status_code=409,
            detail="Disconnect live conversations before data restore/delete",
        )


@app.get("/api/data/export")
def export_owner_data() -> Response:
    """Download one complete inspectable owner-data JSON snapshot."""
    with _audit("export_owner_data", "data"):
        payload = _data().export_json()
        return Response(
            payload,
            media_type="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="therapy-export.json"'
            },
        )


@app.post("/api/data/restore")
async def restore_owner_data(
    file: Annotated[UploadFile, File(description="TheraPy owner-data JSON export")],
) -> dict[str, object]:
    """Validate completely and restore a prior owner snapshot with rollback."""
    with _audit("restore_owner_data", "data"):
        _assert_no_live_sessions()
        maximum_upload = 720 * 1024 * 1024
        payload = await file.read(maximum_upload + 1)
        await file.close()
        if len(payload) > maximum_upload:
            raise HTTPException(
                status_code=413, detail="Restore snapshot is too large"
            )
        try:
            decoded = json.loads(payload)
            if not isinstance(decoded, dict):
                raise ValueError("restore snapshot root must be an object")
            raw_snapshot = cast(dict[object, object], decoded)
            if not all(isinstance(key, str) for key in raw_snapshot):
                raise ValueError("restore snapshot field names must be strings")
            result = _data().restore_snapshot(
                cast(dict[str, object], raw_snapshot)
            )
        except (json.JSONDecodeError, ValueError, sqlite3.IntegrityError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"restored": result}


@app.delete("/api/data")
def delete_owner_data(body: DeleteAllRequest) -> dict[str, bool]:
    """Erase every Phase 4 personal/corpus store after exact confirmation."""
    del body
    with _audit("delete_owner_data", "data"):
        _assert_no_live_sessions()
        _data().delete_all()
        return {"deleted": True}


def _require_acceptance_capture() -> None:
    """Acceptance routes forbid disabled capture (obs plan §4, O3.1)."""
    from therapy.observability.capture import capture_service
    from therapy.observability.model import CaptureMode

    service = capture_service()
    if service is not None and service.mode is CaptureMode.DISABLED:
        raise HTTPException(
            status_code=409,
            detail="acceptance routes require capture (mode is disabled)",
        )


@app.post("/api/testing/agent/turn")
async def acceptance_agent_turn(body: AcceptanceAgentTurn) -> dict[str, object]:
    """Run a deterministic production-shaped agent turn in explicit test mode."""
    if os.getenv("THERAPY_TEST_MODE") != "1":
        raise HTTPException(status_code=404, detail="Not found")
    _require_acceptance_capture()
    try:
        result = await _acceptance_agent().turn(
            body.text,
            body.language,
            session_id=body.session_id,
            finalize=body.finalize,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    return dict(result)


@app.post("/api/testing/proactivity/run")
def acceptance_proactivity_run(body: AcceptanceOutreachRun) -> dict[str, object]:
    """Exercise persistent enqueue/delivery with an explicit acceptance clock."""
    if os.getenv("THERAPY_TEST_MODE") != "1":
        raise HTTPException(status_code=404, detail="Not found")
    _require_acceptance_capture()
    service = _proactivity()
    try:
        job_id = service.enqueue(
            body.channel,
            body.due_at,
            idempotency_key=body.idempotency_key,
            topic=body.topic,
        )
        job = service.deliver(job_id, now=body.now)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"job": job}




# Process-wide token bucket for the client telemetry endpoint (O4.1):
# never keyed by client IP, refills 1 token/s up to 60.
_client_bucket = {"tokens": 60.0, "updated": 0.0}
_CLIENT_BUCKET_CAPACITY = 60.0
MAX_CLIENT_TELEMETRY_BYTES = 16 * 1024


def _client_bucket_take() -> bool:
    import time as _time

    now = _time.monotonic()
    elapsed = now - _client_bucket["updated"]
    _client_bucket["updated"] = now
    _client_bucket["tokens"] = min(
        _CLIENT_BUCKET_CAPACITY, _client_bucket["tokens"] + elapsed
    )
    if _client_bucket["tokens"] < 1.0:
        return False
    _client_bucket["tokens"] -= 1.0
    return True


def _client_telemetry_rejected(reason: str) -> None:
    """Fixed-schema broad evidence for a rejected batch (O4.1) — a bounded
    reason only, never payload values, headers, or origins."""
    from therapy.observability.logging import emit_event

    emit_event(
        "client_telemetry_rejected",
        component="server",
        operation="client_telemetry",
        outcome=reason,
        rate_limited=True,
    )


def _client_origin_allowed(origin: str | None, request: Request) -> bool:
    """Same-origin only: an Origin header is required, opaque/malformed
    origins are rejected, and scheme + netloc must both match the request
    (the proxy scheme wins when forwarded)."""
    if not origin or origin == "null":
        return False
    from urllib.parse import urlsplit

    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    if parsed.path or parsed.query or parsed.fragment:
        return False
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    scheme = (
        forwarded_proto.split(",")[0].strip().lower()
        if forwarded_proto
        else request.url.scheme
    )
    return (
        parsed.netloc == request.headers.get("host", "")
        and parsed.scheme == scheme
    )


@contextmanager
def _client_trace_parent(traceparent: str | None) -> Generator[None, None, None]:
    """Adopt only a valid remote W3C traceparent; ignore all other context."""
    if not traceparent:
        yield
        return

    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    extracted = TraceContextTextMapPropagator().extract(
        carrier={"traceparent": traceparent},
        context=otel_context.Context(),
    )
    span_context = trace.get_current_span(extracted).get_span_context()
    if not span_context.is_valid or not span_context.is_remote:
        yield
        return

    token = otel_context.attach(extracted)
    try:
        yield
    finally:
        otel_context.detach(token)


async def _ingest_client_telemetry(request: Request) -> dict[str, str]:
    """Validate and aggregate one strict, same-origin client batch."""
    length = request.headers.get("content-length")
    if length is not None:
        try:
            declared_length = int(length)
        except ValueError:
            _client_telemetry_rejected("length_malformed")
            raise HTTPException(
                status_code=400, detail="malformed content-length"
            ) from None
        if declared_length > MAX_CLIENT_TELEMETRY_BYTES:
            _client_telemetry_rejected("too_large")
            raise HTTPException(status_code=413, detail="telemetry batch too large")
    if not _client_origin_allowed(request.headers.get("origin"), request):
        _client_telemetry_rejected("origin_rejected")
        raise HTTPException(status_code=403, detail="same-origin only")
    if not _client_bucket_take():
        _client_telemetry_rejected("rate_limited")
        raise HTTPException(status_code=429, detail="telemetry rate limited")

    raw = await request.body()
    if len(raw) > MAX_CLIENT_TELEMETRY_BYTES:
        _client_telemetry_rejected("too_large")
        raise HTTPException(status_code=413, detail="telemetry batch too large")
    from pydantic import ValidationError

    try:
        body = ClientTelemetryBatch.model_validate_json(raw)
    except ValidationError:
        # Never reflect payload-derived validation details back or into logs.
        _client_telemetry_rejected("schema_error")
        raise HTTPException(
            status_code=422, detail="invalid telemetry batch"
        ) from None

    from therapy.observability.telemetry import record_metric

    for event in body.events:
        candidate = event.candidate_type or "host"
        if event.name == "webrtc_sample":
            if event.rtt_ms is not None:
                record_metric(
                    "therapy_webrtc_rtt_seconds",
                    event.rtt_ms / 1000,
                    {"candidate_type": candidate},
                )
            if event.jitter_ms is not None:
                record_metric(
                    "therapy_webrtc_jitter_seconds",
                    event.jitter_ms / 1000,
                    {"candidate_type": candidate},
                )
            if event.packet_loss_ratio is not None:
                record_metric(
                    "therapy_webrtc_packet_loss_ratio",
                    event.packet_loss_ratio,
                    {"candidate_type": candidate},
                )
            if event.bitrate_kbps is not None:
                record_metric(
                    "therapy_webrtc_bitrate_kbps",
                    event.bitrate_kbps,
                    {"candidate_type": candidate},
                )
            if event.bytes_delta is not None:
                record_metric(
                    "therapy_webrtc_bytes_total",
                    event.bytes_delta,
                    {"candidate_type": candidate},
                )
            if event.concealed_samples is not None:
                record_metric(
                    "therapy_webrtc_concealed_samples_total",
                    event.concealed_samples,
                    {"candidate_type": candidate},
                )
        elif event.name in ("peer_state", "ice_state"):
            record_metric(
                "therapy_webrtc_connection_total",
                1,
                {"candidate_type": candidate, "outcome": event.outcome},
            )
        elif event.name == "data_channel_state" and event.duration_ms is not None:
            record_metric(
                "therapy_webrtc_data_channel_open_seconds",
                event.duration_ms / 1000,
                {"outcome": event.outcome},
            )
        if event.dropped_events:
            record_metric(
                "therapy_client_dropped_events_total",
                event.dropped_events,
                {"name": event.name},
            )
        record_metric(
            "therapy_client_events_total",
            1,
            {"name": event.name, "outcome": event.outcome},
        )
    return {"status": "accepted"}


@app.post("/api/telemetry/client")
async def client_telemetry(request: Request) -> dict[str, str]:
    """Strict first-party browser telemetry (obs plan O4.1).

    Same-origin only; bounded size + token bucket BEFORE parsing; events
    aggregate to metrics and are never persisted; only fixed-schema
    rejection evidence is logged. A controlled W3C parent correlates this
    content-free span with the page's offer without entering the event body.
    """
    if os.environ.get("THERAPY_CLIENT_TELEMETRY", "0") != "1":
        raise HTTPException(status_code=404, detail="Not found")

    from therapy.observability.telemetry import broad_span

    with _client_trace_parent(request.headers.get("traceparent")):
        with broad_span(
            "client_telemetry", component="server", operation="client_telemetry"
        ):
            return await _ingest_client_telemetry(request)


@app.get("/api/crisis-resources")
def crisis_resources_config() -> dict[str, object]:
    """The configurable crisis hotlines/contacts surfaced in the UI (W8)."""
    from therapy.dialogue.policy import (
        CrisisConfigurationError,
        crisis_contacts,
        crisis_resources,
    )

    try:
        contacts = crisis_contacts()
    except CrisisConfigurationError as exc:
        # Config validity only — never crisis activity inference (plan O3.1).
        import logging as logging_module

        from therapy.observability.logging import emit_event

        emit_event(
            "crisis_config_invalid",
            severity=logging_module.ERROR,
            component="server",
            operation="crisis_config",
            outcome="error",
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "contacts": contacts,
        "resources": crisis_resources(),
        "editing": "environment-only",
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
