from __future__ import annotations

import os
import socket
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# scripts/ is not a package; make it importable regardless of how pytest
# was launched (`python -m pytest` adds the CWD, plain `pytest` does not).
sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.watchdog import Watchdog, probe  # noqa: E402


class _OkHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, _format: str, *_args: object) -> None:
        return


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


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


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


def test_probe_returns_false_on_timeout() -> None:
    stop = threading.Event()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen()
        server.settimeout(0.1)
        port = int(server.getsockname()[1])

        def accept_without_response() -> None:
            connections: list[socket.socket] = []
            try:
                while not stop.is_set():
                    try:
                        connection, _address = server.accept()
                    except TimeoutError:
                        continue
                    except OSError:
                        return
                    connections.append(connection)
            finally:
                for connection in connections:
                    connection.close()

        thread = threading.Thread(target=accept_without_response)
        thread.start()
        try:
            assert probe(f"http://127.0.0.1:{port}/health", timeout=0.1) is False
        finally:
            stop.set()
            server.close()
            thread.join(timeout=1.0)


def test_watchdog_restarts_crashed_child_until_requested_stop() -> None:
    watchdog = Watchdog(
        cmd=[sys.executable, "-c", "import time; time.sleep(0.2)"],
        url="http://127.0.0.1:9/health",
        interval=0.05,
        timeout=0.05,
        max_failures=100,
        grace=60.0,
    )
    thread = threading.Thread(target=watchdog.run_forever)
    thread.start()
    try:
        _wait_until(lambda: watchdog.restart_count >= 2)
    finally:
        watchdog.request_stop()
        thread.join(timeout=2.0)

    assert watchdog.restart_count >= 2
    assert not thread.is_alive()


def test_watchdog_restarts_hung_child_and_kills_original_pid() -> None:
    port = _free_port()
    child_code = (
        "import socket\n"
        f"port = {port}\n"
        "server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "server.bind(('127.0.0.1', port))\n"
        "server.listen()\n"
        "connections = []\n"
        "while True:\n"
        "    connection, _address = server.accept()\n"
        "    connections.append(connection)\n"
    )
    watchdog = Watchdog(
        cmd=[sys.executable, "-c", child_code],
        url=f"http://127.0.0.1:{port}/health",
        interval=0.1,
        timeout=0.2,
        max_failures=2,
        grace=0.0,
    )
    thread = threading.Thread(target=watchdog.run_forever)
    thread.start()
    try:
        _wait_until(lambda: watchdog.child is not None)
        original_child = watchdog.child
        assert original_child is not None
        original_pid = original_child.pid

        _wait_until(
            lambda: watchdog.restart_count >= 1
            and watchdog.child is not None
            and watchdog.child.pid != original_pid,
            timeout=4.0,
        )
    finally:
        watchdog.request_stop()
        thread.join(timeout=2.0)

    assert watchdog.restart_count >= 1
    assert not _pid_is_alive(original_pid)
    assert not thread.is_alive()
