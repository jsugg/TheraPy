"""Shared test configuration and helpers for every suite.

Conveniences that live here so individual tests stay lean:

1. **Auto-marking by folder.** A test under ``tests/suites/<suite>/`` is tagged
   with the matching marker (``unit`` / ``integration`` / ``e2e``) automatically,
   so no file needs a ``pytestmark`` just to declare its suite. Selection then
   works everywhere: ``pytest -m unit``, ``pytest -m e2e``, etc. (the default
   ``addopts = -m "not e2e"`` keeps the browser suite opt-in).

2. **Repo root on ``sys.path``.** So ``from scripts.watchdog import ...`` resolves
   no matter how pytest was launched (plain ``pytest`` does not add the CWD the
   way ``python -m pytest`` does) — no per-file root-walk needed.

3. **Isolated storage + a ready TestClient.** The ``data_dir`` fixture redirects
   ``THERAPY_DATA_DIR`` at a throwaway temp dir — enforcing the "never touch the
   real /data" invariant centrally (docs/SPEC.md Hardening 7-9) — and ``client``
   hands back a FastAPI TestClient with the app's cached singletons reset around
   each test. Any test hitting the app must go through ``client`` so the reset
   is centralized (no module-level TestClient that leaks state or storage).

4. **Common building blocks** — ``repo_root``, ``free_port``, ``wait_until``, and
   ``ok_server`` — replace helpers that were copy-pasted across the reliability
   and watcher suites.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

_SUITES = ("unit", "integration", "e2e")

# Make the repo importable (scripts/ is not a package) regardless of launcher.
REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tag each test with its suite marker based on the folder it lives in."""
    for item in items:
        parts = item.path.parts
        for suite in _SUITES:
            if suite in parts:
                item.add_marker(getattr(pytest.mark, suite))
                break


# --- Isolated storage + TestClient ------------------------------------------


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point THERAPY_DATA_DIR at an isolated temp dir for the duration of a test."""
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    return tmp_path


def _reset_app_singletons() -> None:
    """Clear the app's lru_cache singletons so each test rebuilds them fresh."""
    from therapy.server import app as app_mod

    for cached in (
        "_store",
        "_model",
        "_research",
        "_insights",
        "_proactivity",
        "_data",
        "_acceptance_agent",
    ):
        singleton = getattr(app_mod, cached, None)
        if singleton is not None and hasattr(singleton, "cache_clear"):
            singleton.cache_clear()


@pytest.fixture
def client(data_dir: Path) -> Iterator[TestClient]:
    """A FastAPI TestClient whose lru_cache singletons are reset for isolation.

    Depends on ``data_dir`` so the app reads/writes only the throwaway dir. The
    singletons are cleared before *and* after the test: before so a prior test's
    cached store (pinned to a prior temp dir) can't bleed in, after so the next
    test starts clean even if it does not take this fixture.
    """
    from fastapi.testclient import TestClient

    from therapy.server import app as app_mod

    _reset_app_singletons()
    with TestClient(app_mod.app) as test_client:
        yield test_client
    _reset_app_singletons()


# --- Common building blocks -------------------------------------------------


@pytest.fixture
def repo_root() -> Path:
    """The repository root (the dir holding pyproject.toml)."""
    return REPO_ROOT


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def free_port() -> Callable[[], int]:
    """Return a callable that hands out an unused loopback TCP port."""
    return _free_port


def _wait_until(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition was not met before timeout")


@pytest.fixture
def wait_until() -> Callable[..., None]:
    """Return a poller that blocks until ``predicate()`` is true or it times out."""
    return _wait_until


class _OkHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return  # silence the default request logging


@pytest.fixture
def ok_server() -> Iterator[str]:
    """A local HTTP server answering 200 to any GET; yields its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OkHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{int(server.server_address[1])}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)
