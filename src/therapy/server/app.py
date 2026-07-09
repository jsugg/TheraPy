"""Application entrypoint. Phase 0: health check + placeholder root."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from therapy import __version__

app = FastAPI(title="TheraPy", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return "<!doctype html><title>TheraPy</title><h1>TheraPy</h1><p>Phase 0 — scaffold. See docs/SPEC.md.</p>"
