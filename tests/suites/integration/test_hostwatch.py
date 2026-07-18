import importlib
import json
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

import scripts.hostwatch as hostwatch_module
from scripts.hostwatch import StackWatch, notify_macos, probe, run, write_state
from tests.type_contracts import FreePort, WaitUntil


class _HostwatchArguments(Protocol):
    notify: str


class _ArgumentParser(Protocol):
    def __call__(self, args: list[str]) -> _HostwatchArguments: ...


class _HostwatchLogger(Protocol):
    def __call__(self, event_name: str, **fields: object) -> None: ...


class _RecordBuilder(Protocol):
    def __call__(
        self, *, scenario: str, duration_s: float, result: str
    ) -> dict[str, object]: ...

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


class _StopAfterRecoverWatch(StackWatch):
    recoveries: list[str]

    def recover(self) -> str:
        self.recoveries.append("container")
        self.request_stop()
        return "container"


class _VerifierDependencyStub:
    pass


def _stub_verifier_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    aiortc = ModuleType("aiortc")
    aiortc.__dict__.update(
        {
            "RTCConfiguration": _VerifierDependencyStub,
            "RTCIceServer": _VerifierDependencyStub,
            "RTCPeerConnection": _VerifierDependencyStub,
            "RTCSessionDescription": _VerifierDependencyStub,
        }
    )
    aiortc_mediastreams = ModuleType("aiortc.mediastreams")
    aiortc_mediastreams.__dict__["MediaStreamTrack"] = _VerifierDependencyStub
    av = ModuleType("av")
    av.__dict__["AudioFrame"] = _VerifierDependencyStub
    kokoro = ModuleType("kokoro_onnx")
    kokoro.__dict__["Kokoro"] = _VerifierDependencyStub
    pipecat = ModuleType("pipecat")
    pipecat_services = ModuleType("pipecat.services")
    pipecat_kokoro = ModuleType("pipecat.services.kokoro")
    pipecat_tts = ModuleType("pipecat.services.kokoro.tts")
    pipecat_tts.__dict__["KOKORO_CACHE_DIR"] = Path("/tmp")
    for name, module in {
        "aiortc": aiortc,
        "aiortc.mediastreams": aiortc_mediastreams,
        "av": av,
        "kokoro_onnx": kokoro,
        "pipecat": pipecat,
        "pipecat.services": pipecat_services,
        "pipecat.services.kokoro": pipecat_kokoro,
        "pipecat.services.kokoro.tts": pipecat_tts,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_probe_returns_true_for_local_http_200(ok_server: str) -> None:
    assert probe(f"{ok_server}/health", timeout=1.0) is True


def test_probe_returns_false_for_closed_port(free_port: FreePort) -> None:
    port = free_port()

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


def test_watch_forever_recovers_after_max_failures(wait_until: WaitUntil) -> None:
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
        wait_until(lambda: bool(watcher.recoveries), timeout=1.0)
    finally:
        watcher.request_stop()
        thread.join(timeout=1.0)

    assert watcher.recoveries == ["container"]
    assert probe_count == 2
    assert not thread.is_alive()


def test_hostwatch_stdout_uses_only_broad_schema_and_hides_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    command_marker = "full-host-command-must-not-be-logged"

    def fake_runner(_cmd: list[str], _timeout: float) -> bool:
        return True

    watcher = StackWatch(
        health_url="http://127.0.0.1:8000/health",
        probe_timeout=0.1,
        interval=0.1,
        max_failures=1,
        container_cmd=["docker", "restart", command_marker],
        daemon_check_cmd=["docker", "version", command_marker],
        provider_restart_cmds=[["open", "-a", command_marker]],
        post_restart_cmd=["docker", "compose", "up", command_marker],
        settle=0.0,
        runner=fake_runner,
    )

    assert watcher.recover() == "container"

    output = capsys.readouterr().out
    from therapy.observability.interactions import require_json_object

    records = [
        require_json_object(json.loads(line), "test.hostwatch.log")
        for line in output.splitlines()
    ]
    assert records
    assert command_marker not in output
    assert all(REQUIRED_LOG_FIELDS <= record.keys() for record in records)
    assert all(record.keys() <= ALLOWED_LOG_FIELDS for record in records)
    assert all(record["component"] == "hostwatch" for record in records)


def test_hostwatch_plain_fallback_survives_broken_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken_dumps(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("broken encoder")

    monkeypatch.setattr(hostwatch_module.json, "dumps", broken_dumps)

    logger_value: object = getattr(hostwatch_module, "_log", None)
    if not callable(logger_value):
        raise TypeError("hostwatch logger is unavailable")
    logger = cast(_HostwatchLogger, logger_value)
    logger(
        "hostwatch.emitter.failure",
        operation="probe",
        outcome="error",
    )

    assert capsys.readouterr().out == "[hostwatch] hostwatch.emitter.failure\n"


def test_hostwatch_state_file_is_owner_only(tmp_path: Path) -> None:
    state_file = tmp_path / "hostwatch.json"

    write_state(state_file, {"consecutive_failures": 1})

    assert stat.S_IMODE(state_file.stat().st_mode) == 0o600
    assert json.loads(state_file.read_text()) == {"consecutive_failures": 1}


def test_hostwatch_state_replace_is_atomic_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_file = tmp_path / "hostwatch.json"
    initial = {"consecutive_failures": 1}
    write_state(state_file, initial)

    def crash_before_replace(_source: object, _destination: object) -> None:
        raise OSError("simulated crash")

    monkeypatch.setattr(hostwatch_module.os, "replace", crash_before_replace)

    with pytest.raises(OSError, match="simulated crash"):
        write_state(state_file, {"consecutive_failures": 2})

    assert json.loads(state_file.read_text()) == initial
    assert list(tmp_path.glob(f".{state_file.name}.*.tmp")) == []


def test_macos_notification_is_bounded_and_docker_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(
        cmd: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(hostwatch_module.subprocess, "run", fake_run)

    assert notify_macos("provider-restart") is True
    assert calls[0][0][0:2] == ["osascript", "-e"]
    assert "provider-restart" in calls[0][0][2]
    assert "docker" not in " ".join(calls[0][0]).lower()
    assert "shell" not in calls[0][1]


def test_hostwatch_notify_cli_accepts_only_none_or_macos() -> None:
    parser_value: object = getattr(hostwatch_module, "_parse_args", None)
    if not callable(parser_value):
        raise TypeError("hostwatch argument parser is unavailable")
    parse_args = cast(_ArgumentParser, parser_value)
    assert parse_args([]).notify == "none"
    assert parse_args(["--notify", "macos"]).notify == "macos"
    with pytest.raises(SystemExit):
        parse_args(["--notify", "arbitrary-command"])


@pytest.mark.parametrize(
    ("module_name", "scenario"),
    [
        ("scripts.verify_relay_connectivity", "relay-only"),
        ("scripts.verify_voice_text_loop", "voice-text-loop"),
        ("scripts.verify_memory_continuity", "memory-continuity"),
        ("scripts.verify_longitudinal_loop", "longitudinal-loop"),
    ],
)
def test_verification_script_record_builder(
    module_name: str, scenario: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_verifier_dependencies(monkeypatch)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)

    builder_value: object = getattr(module, "build_verification_record", None)
    if not callable(builder_value):
        raise TypeError("verification record builder is unavailable")
    builder = cast(_RecordBuilder, builder_value)
    record = builder(
        scenario=scenario,
        duration_s=1.25,
        result="pass",
    )

    assert record == {
        "record": "verification",
        "script": module_name.rsplit(".", maxsplit=1)[-1],
        "build": record["build"],
        "scenario": scenario,
        "duration_s": 1.25,
        "result": "pass",
        "environment": "test",
    }
    assert isinstance(record["build"], str)
