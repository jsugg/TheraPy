import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.hostwatch import StackWatch, probe, run  # noqa: E402

import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402


class _OkHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _StopAfterRecoverWatch(StackWatch):
    recoveries: list[str]

    def recover(self) -> str:
        self.recoveries.append("container")
        self.request_stop()
        return "container"


def _wait_until(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition was not met before timeout")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_probe_returns_true_for_local_http_200() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OkHandler)
    port = int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        assert probe(f"http://127.0.0.1:{port}/health", timeout=1.0) is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_probe_returns_false_for_closed_port() -> None:
    port = _free_port()

    assert probe(f"http://127.0.0.1:{port}/health", timeout=0.2) is False


def test_run_returns_true_for_successful_command() -> None:
    assert run([sys.executable, "-c", "pass"], timeout=1.0) is True


def test_run_returns_false_for_nonzero_exit() -> None:
    assert run([sys.executable, "-c", "raise SystemExit(1)"], timeout=1.0) is False


def test_run_returns_false_for_timeout_promptly() -> None:
    start = time.monotonic()

    assert (
        run([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1)
        is False
    )

    assert time.monotonic() - start < 3.0


def test_run_returns_false_for_nonexistent_binary() -> None:
    cmd = ["definitely-not-a-real-hostwatch-binary"]

    assert run(cmd, timeout=1.0) is False


def test_recover_restarts_container_when_daemon_answers() -> None:
    calls: list[tuple[list[str], float]] = []
    daemon_check_cmd = ["docker", "version"]
    container_cmd = ["docker", "restart", "therapy"]
    provider_restart_cmds = [["osascript", "-e", "quit"], ["open", "-a", "OrbStack"]]
    post_restart_cmd = ["docker", "compose", "up", "-d"]

    def fake_runner(cmd: list[str], timeout: float) -> bool:
        calls.append((cmd, timeout))
        return True

    watcher = StackWatch(
        health_url="http://127.0.0.1:8000/health",
        probe_timeout=0.1,
        interval=0.1,
        max_failures=1,
        container_cmd=container_cmd,
        daemon_check_cmd=daemon_check_cmd,
        provider_restart_cmds=provider_restart_cmds,
        post_restart_cmd=post_restart_cmd,
        settle=0.0,
        runner=fake_runner,
    )

    assert watcher.recover() == "container"
    assert calls == [(daemon_check_cmd, 15), (container_cmd, 120)]
    assert all(cmd not in provider_restart_cmds for cmd, _timeout in calls)


def test_recover_restarts_provider_when_daemon_is_wedged() -> None:
    calls: list[tuple[list[str], float]] = []
    daemon_check_cmd = ["docker", "version"]
    container_cmd = ["docker", "restart", "therapy"]
    provider_restart_cmds = [["osascript", "-e", "quit"], ["open", "-a", "OrbStack"]]
    post_restart_cmd = ["docker", "compose", "up", "-d"]

    def fake_runner(cmd: list[str], timeout: float) -> bool:
        calls.append((cmd, timeout))
        return cmd != daemon_check_cmd

    watcher = StackWatch(
        health_url="http://127.0.0.1:8000/health",
        probe_timeout=0.1,
        interval=0.1,
        max_failures=1,
        container_cmd=container_cmd,
        daemon_check_cmd=daemon_check_cmd,
        provider_restart_cmds=provider_restart_cmds,
        post_restart_cmd=post_restart_cmd,
        settle=0.0,
        runner=fake_runner,
    )

    assert watcher.recover() == "provider"
    assert calls == [
        (daemon_check_cmd, 15),
        (provider_restart_cmds[0], 60),
        (provider_restart_cmds[1], 60),
        (post_restart_cmd, 300),
    ]


def test_watch_forever_recovers_after_max_failures() -> None:
    probe_count = 0

    def fake_prober(_url: str, _timeout: float) -> bool:
        nonlocal probe_count
        probe_count += 1
        return False

    def fake_runner(_cmd: list[str], _timeout: float) -> bool:
        return True

    def fake_sleeper(_seconds: float) -> None:
        return

    watcher = _StopAfterRecoverWatch(
        health_url="http://127.0.0.1:8000/health",
        probe_timeout=0.1,
        interval=0.1,
        max_failures=2,
        container_cmd=["docker", "restart", "therapy"],
        daemon_check_cmd=["docker", "version"],
        provider_restart_cmds=[["open", "-a", "OrbStack"]],
        post_restart_cmd=["docker", "compose", "up", "-d"],
        settle=0.1,
        prober=fake_prober,
        runner=fake_runner,
        sleeper=fake_sleeper,
    )
    watcher.recoveries = []
    thread = threading.Thread(target=watcher.watch_forever)
    thread.start()
    try:
        _wait_until(lambda: bool(watcher.recoveries), timeout=1.0)
    finally:
        watcher.request_stop()
        thread.join(timeout=1.0)

    assert watcher.recoveries == ["container"]
    assert probe_count == 2
    assert not thread.is_alive()
