"""Host-side Docker supervisor for two-layer recovery (SPEC §5 deployment).

The in-container watchdog handles hung app processes while Docker still
answers. This host watcher escalates further: restart the container when the
Docker daemon answers, or restart the Docker provider when the VM itself is
wedged and every Docker CLI call hangs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

DEFAULT_HEALTH_URL = "http://localhost:8000/health"
DEFAULT_INTERVAL = 30.0
DEFAULT_TIMEOUT = 10.0
DEFAULT_FAILURES = 4
DEFAULT_SETTLE = 180.0
DEFAULT_CONTAINER_CMD = ["docker", "restart", "therapy-therapy-1"]
DEFAULT_DAEMON_CHECK_CMD = ["docker", "version", "--format", "{{.Server.Version}}"]
DEFAULT_PROVIDER_RESTART_CMDS = [
    ["osascript", "-e", 'quit app "OrbStack"'],
    ["sleep", "10"],
    ["open", "-a", "OrbStack"],
    ["sleep", "25"],
]
DEFAULT_POST_RESTART_CMD = ["docker", "compose", "up", "-d"]
DEFAULT_STATE_FILE = Path.home() / ".therapy" / "hostwatch-state.json"
NOTIFICATION_EVENTS = frozenset({"container-restart", "provider-restart"})

type Prober = Callable[[str, float], bool]
type Runner = Callable[[list[str], float], bool]
type Sleeper = Callable[[float], object]
type NotificationEvent = Literal["container-restart", "provider-restart"]
type Notifier = Callable[[NotificationEvent], bool]
type Operation = Literal["start", "stop", "restart", "probe"]
type Outcome = Literal["success", "error", "timeout"]
type Severity = Literal["INFO", "WARNING", "ERROR"]
type StateValue = str | float | int | None


def _service_version() -> str:
    """Return the installed build version without importing the application."""
    try:
        return version("therapy")
    except PackageNotFoundError:
        return "unknown"


SERVICE_VERSION = _service_version()


def _log(
    event_name: str,
    *,
    operation: Operation,
    outcome: Outcome,
    severity: Severity = "INFO",
    duration_ms: float | None = None,
    error_type: str | None = None,
    retry_count: int | None = None,
    count: int | None = None,
) -> None:
    """Write one fixed-schema hostwatch record to host stdout."""
    try:
        created = time.time()
        payload: dict[str, str | float | int] = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(created)
            )
            + f".{int(created % 1 * 1000):03d}Z",
            "severity": severity,
            "event.name": event_name,
            "service.name": "therapy",
            "service.version": SERVICE_VERSION,
            "service.instance.id": f"hostwatch-{os.getpid()}",
            "deployment.environment": _environment(),
            "component": "hostwatch",
            "operation": operation,
            "outcome": outcome,
        }
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if error_type is not None:
            payload["error.type"] = error_type
        if retry_count is not None:
            payload["retry_count"] = retry_count
        if count is not None:
            payload["count"] = count
        print(json.dumps(payload, sort_keys=True), flush=True)
    except Exception:
        print(f"[hostwatch] {event_name}", flush=True)


def _environment() -> str:
    """Return the bounded deployment environment used by broad telemetry."""
    environment = os.environ.get("THERAPY_ENVIRONMENT", "development")
    if environment in {"development", "test", "dogfood", "vps-test"}:
        return environment
    return "development"


def probe(url: str, timeout: float) -> bool:
    """Return True iff one HTTP GET receives status 200."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def run(cmd: list[str], timeout: float) -> bool:
    """Run a command with a timeout, returning True iff it exits with code 0."""
    try:
        completed = subprocess.run(
            cmd,
            timeout=timeout,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def write_state(path: Path, state: Mapping[str, StateValue]) -> None:
    """Atomically replace one owner-readable hostwatch state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(json.dumps(state, sort_keys=True))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def notify_macos(event_name: NotificationEvent) -> bool:
    """Send one bounded generic macOS hostwatch notification."""
    if event_name not in NOTIFICATION_EVENTS:
        raise ValueError("unsupported notification event")
    script = (
        f'display notification "Host supervisor event: {event_name}" '
        'with title "TheraPy"'
    )
    try:
        completed = subprocess.run(
            ["osascript", "-e", script],
            timeout=10,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


class StackWatch:
    """Supervise host-visible Docker health with container/provider recovery."""

    def __init__(
        self,
        health_url: str,
        probe_timeout: float,
        interval: float,
        max_failures: int,
        container_cmd: list[str],
        daemon_check_cmd: list[str],
        provider_restart_cmds: list[list[str]],
        post_restart_cmd: list[str],
        settle: float,
        prober: Prober = probe,
        runner: Runner = run,
        sleeper: Sleeper = time.sleep,
        state_file: Path | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        if probe_timeout <= 0:
            raise ValueError("probe_timeout must be greater than 0")
        if interval <= 0:
            raise ValueError("interval must be greater than 0")
        if max_failures < 1:
            raise ValueError("max_failures must be at least 1")
        if settle < 0:
            raise ValueError("settle must be greater than or equal to 0")
        if not container_cmd:
            raise ValueError("container_cmd must contain at least one argument")
        if not daemon_check_cmd:
            raise ValueError("daemon_check_cmd must contain at least one argument")
        if not post_restart_cmd:
            raise ValueError("post_restart_cmd must contain at least one argument")

        self.health_url = health_url
        self.probe_timeout = probe_timeout
        self.interval = interval
        self.max_failures = max_failures
        self.container_cmd = container_cmd
        self.daemon_check_cmd = daemon_check_cmd
        self.provider_restart_cmds = provider_restart_cmds
        self.post_restart_cmd = post_restart_cmd
        self.settle = settle
        self.prober = prober
        self.runner = runner
        self.sleeper = sleeper
        self.state_file = state_file
        self.notifier = notifier
        self._started = time.monotonic()
        self._last_success = 0.0
        self._last_success_at: float | None = None
        self._failures = 0
        self._recovery_count = 0
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        """Ask the watch loop to stop."""
        self._stop_requested.set()

    def recover(self) -> str:
        """Recover the stack, returning the escalation layer used."""
        if self.runner(self.daemon_check_cmd, 15):
            succeeded = self.runner(self.container_cmd, 120)
            _log(
                "hostwatch.container.restart",
                operation="restart",
                outcome="success" if succeeded else "error",
                severity="INFO" if succeeded else "ERROR",
                retry_count=self._recovery_count + 1,
            )
            self._notify("container-restart")
            return "container"

        succeeded = True
        for cmd in self.provider_restart_cmds:
            if not self.runner(cmd, 60):
                succeeded = False
        if not self.runner(self.post_restart_cmd, 300):
            succeeded = False
        _log(
            "hostwatch.provider.restart",
            operation="restart",
            outcome="success" if succeeded else "error",
            severity="INFO" if succeeded else "ERROR",
            retry_count=self._recovery_count + 1,
        )
        self._notify("provider-restart")
        return "provider"

    def watch_forever(self) -> None:
        """Probe forever, escalating after consecutive failures."""
        self._persist_state("start")
        while not self._stop_requested.is_set():
            if self.prober(self.health_url, self.probe_timeout):
                if self._failures:
                    _log(
                        "hostwatch.probe.recovered",
                        operation="probe",
                        outcome="success",
                        duration_ms=self._since_last_success_ms(),
                        count=self._failures,
                    )
                self._failures = 0
                self._last_success = time.monotonic()
                self._last_success_at = time.time()
            else:
                self._failures += 1
                _log(
                    "hostwatch.probe.failed",
                    operation="probe",
                    outcome="error",
                    severity="WARNING",
                    duration_ms=self._since_last_success_ms(),
                    count=self._failures,
                )
                if self._failures >= self.max_failures:
                    _log(
                        "hostwatch.recovery.started",
                        operation="restart",
                        outcome="error",
                        severity="WARNING",
                        count=self._failures,
                    )
                    self.recover()
                    self._recovery_count += 1
                    self._sleep(self.settle)
                    self._failures = 0

            self._persist_state("probe")
            self._sleep(self.interval)

    def _sleep(self, seconds: float) -> None:
        """Sleep with immediate stop support in production and injectable tests."""
        if seconds <= 0 or self._stop_requested.is_set():
            return
        if self.sleeper is time.sleep:
            self._stop_requested.wait(seconds)
            return
        self.sleeper(seconds)

    def _notify(self, event_name: NotificationEvent) -> None:
        """Send and record a configured bounded recovery notification."""
        if self.notifier is None:
            return
        try:
            succeeded = self.notifier(event_name)
        except Exception as exc:
            _log(
                "hostwatch.notification.sent",
                operation="restart",
                outcome="error",
                severity="ERROR",
                error_type=type(exc).__name__,
            )
            return
        _log(
            "hostwatch.notification.sent",
            operation="restart",
            outcome="success" if succeeded else "error",
            severity="INFO" if succeeded else "ERROR",
        )

    def _persist_state(self, operation: Operation) -> None:
        """Persist non-sensitive recovery state when a state path is configured."""
        if self.state_file is None:
            return
        state: dict[str, StateValue] = {
            "consecutive_failures": self._failures,
            "last_success_at": self._last_success_at,
            "recovery_count": self._recovery_count,
            "status": "degraded" if self._failures else "healthy",
        }
        try:
            write_state(self.state_file, state)
        except Exception as exc:
            _log(
                "hostwatch.state.write",
                operation=operation,
                outcome="error",
                severity="ERROR",
                error_type=type(exc).__name__,
            )

    def _since_last_success_ms(self) -> float:
        """Return uptime or time since the last successful probe."""
        reference = self._last_success or self._started
        return round((time.monotonic() - reference) * 1000, 3)


def _float_env(name: str, default: float) -> float:
    """Parse a float environment variable."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float") from exc


def _int_env(name: str, default: int) -> int:
    """Parse an integer environment variable."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bounded hostwatch notification adapter selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notify", choices=("none", "macos"), default="none")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build a host watcher from environment variables and run it."""
    args = _parse_args(argv)
    try:
        watcher = StackWatch(
            health_url=os.environ.get("HOSTWATCH_URL", DEFAULT_HEALTH_URL),
            probe_timeout=_float_env("HOSTWATCH_TIMEOUT", DEFAULT_TIMEOUT),
            interval=_float_env("HOSTWATCH_INTERVAL", DEFAULT_INTERVAL),
            max_failures=_int_env("HOSTWATCH_FAILURES", DEFAULT_FAILURES),
            container_cmd=DEFAULT_CONTAINER_CMD,
            daemon_check_cmd=DEFAULT_DAEMON_CHECK_CMD,
            provider_restart_cmds=DEFAULT_PROVIDER_RESTART_CMDS,
            post_restart_cmd=DEFAULT_POST_RESTART_CMD,
            settle=_float_env("HOSTWATCH_SETTLE", DEFAULT_SETTLE),
            state_file=Path(
                os.environ.get("HOSTWATCH_STATE_FILE", str(DEFAULT_STATE_FILE))
            ).expanduser(),
            notifier=notify_macos if args.notify == "macos" else None,
        )
    except ValueError as exc:
        _log(
            "hostwatch.configuration.invalid",
            operation="start",
            outcome="error",
            severity="ERROR",
            error_type=type(exc).__name__,
        )
        return 2

    watcher.watch_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
