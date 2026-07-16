"""Fixtures for the PWA browser E2E (Playwright — run with `pytest -m e2e`).

Everything runs against an *isolated* server: a second uvicorn on a test port
with its own temp data dir. The E2E therefore never touches the real /data —
a scripted client sharing the container's store polluted the owner's sessions
during field testing (docs/SPEC.md Hardening 7-9), and this keeps the browser
tests deterministic besides.
"""

import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator

import pytest


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session")
def e2e_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Start an isolated uvicorn with a throwaway data dir; yield its base URL."""
    data_dir = tmp_path_factory.mktemp("e2e-data")
    port = _free_port()
    env = {
        # Inherit the container env (THERAPY_LLM=ollama, OLLAMA_BASE_URL, …)
        # so typed turns get real replies, but redirect storage to the temp dir.
        **_os_environ(),
        "THERAPY_DATA_DIR": str(data_dir),
        "THERAPY_RESUME_WINDOW_SECS": "900",
        "THERAPY_TEST_MODE": "1",
        "THERAPY_CLIENT_TELEMETRY": "1",
    }
    proc = subprocess.Popen(
        [
            # the owned launcher: logging/OTel bootstrap BEFORE the app
            # imports (obs plan O1.1); uvicorn host/port come from env
            sys.executable, "-m", "therapy.server",
        ],
        env={**env, "THERAPY_HOST": "127.0.0.1", "THERAPY_PORT": str(port)},
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _await_health(base_url, proc)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)


def _await_health(base_url: str, proc: subprocess.Popen, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"e2e server exited early (code {proc.returncode})")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("e2e server did not become healthy in time")


@pytest.fixture(scope="session")
def browser_type_launch_args(browser_type_launch_args: dict) -> dict:
    # Fake mic so getUserMedia resolves headlessly without a real device and
    # without a permission dialog.
    return {
        **browser_type_launch_args,
        "args": [
            *browser_type_launch_args.get("args", []),
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
            "--autoplay-policy=no-user-gesture-required",
        ],
    }


@pytest.fixture
def browser_context_args(browser_context_args: dict) -> dict:
    return {**browser_context_args, "permissions": ["microphone", "notifications"]}
