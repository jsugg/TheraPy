from __future__ import annotations

import json
import os
import socket
import sys
import threading

import scripts.watchdog as watchdog_module
from scripts.watchdog import Watchdog, probe

ALLOWED_LOG_FIELDS = frozenset(
    {
        "timestamp",
        "severity",
        "event.name",
        "service.name",
        "service.version",
        "service.instance.id",
        "deployment.environment",
        "trace_id",
        "span_id",
        "component",
        "operation",
        "outcome",
        "duration_ms",
        "error.type",
        "retry_count",
        "count",
    }
)
REQUIRED_LOG_FIELDS = frozenset(
    {
        "timestamp",
        "severity",
        "event.name",
        "service.name",
        "service.version",
        "service.instance.id",
        "deployment.environment",
        "component",
        "operation",
        "outcome",
    }
)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_probe_returns_true_for_local_http_200(ok_server) -> None:
    assert probe(f"{ok_server}/health", timeout=1.0) is True


def test_probe_returns_false_for_closed_port(free_port) -> None:
    port = free_port()

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


def test_watchdog_restarts_crashed_child_until_requested_stop(wait_until) -> None:
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
        wait_until(lambda: watchdog.restart_count >= 2)
    finally:
        watchdog.request_stop()
        thread.join(timeout=2.0)

    assert watchdog.restart_count >= 2
    assert not thread.is_alive()


def test_watchdog_restarts_hung_child_and_kills_original_pid(
    free_port, wait_until
) -> None:
    port = free_port()
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
        wait_until(lambda: watchdog.child is not None)
        original_child = watchdog.child
        assert original_child is not None
        original_pid = original_child.pid

        wait_until(
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


def test_watchdog_stdout_uses_only_broad_schema_and_hides_child_command(
    capsys,
) -> None:
    command_marker = "full-child-command-must-not-be-logged"
    watchdog = Watchdog(
        cmd=[
            sys.executable,
            "-c",
            f"import time; time.sleep(5)  # {command_marker}",
        ],
        url="http://127.0.0.1:9/health",
        interval=1.0,
        timeout=0.05,
        max_failures=3,
        grace=0.0,
    )
    try:
        watchdog.start_child()
        assert watchdog._probe_failed() is False
    finally:
        watchdog.stop_child()

    output = capsys.readouterr().out
    records = [json.loads(line) for line in output.splitlines()]
    assert records
    assert command_marker not in output
    assert all(REQUIRED_LOG_FIELDS <= record.keys() for record in records)
    assert all(record.keys() <= ALLOWED_LOG_FIELDS for record in records)
    assert all(record["component"] == "watchdog" for record in records)
    probe_failure = next(
        record
        for record in records
        if record["event.name"] == "watchdog.probe.failed"
    )
    assert probe_failure["count"] == 1


def test_watchdog_plain_fallback_survives_broken_json(monkeypatch, capsys) -> None:
    def broken_dumps(*_args, **_kwargs) -> str:
        raise RuntimeError("broken encoder")

    monkeypatch.setattr(watchdog_module.json, "dumps", broken_dumps)

    watchdog_module._log(
        "watchdog.emitter.failure",
        operation="probe",
        outcome="error",
    )

    assert capsys.readouterr().out == "[watchdog] watchdog.emitter.failure\n"
