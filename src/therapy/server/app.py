"""FastAPI backend: WebRTC signaling + PWA static serving (SPEC §5)."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from therapy import __version__
from therapy.agent import run_bot

STATIC_DIR = Path(__file__).parent / "static"

_webrtc_handler = SmallWebRTCRequestHandler()
_bot_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await _webrtc_handler.close()
    for task in _bot_tasks:
        task.cancel()


app = FastAPI(title="TheraPy", version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/api/offer")
async def offer(request: Request) -> dict:
    """SDP offer → answer; each new connection gets its own pipeline."""
    body = await request.json()

    async def on_connection(connection) -> None:
        task = asyncio.create_task(run_bot(connection))
        _bot_tasks.add(task)
        task.add_done_callback(_bot_tasks.discard)

    answer = await _webrtc_handler.handle_web_request(
        SmallWebRTCRequest.from_dict(body), on_connection
    )
    return answer or {}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
