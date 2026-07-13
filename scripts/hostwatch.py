"""Host-side Docker supervisor for two-layer recovery (SPEC §5 deployment).

The in-container watchdog handles hung app processes while Docker still
answers. This host watcher escalates further: restart the container when the
Docker daemon answers, or restart the Docker provider when the VM itself is
wedged and every Docker CLI call hangs.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable

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

type Prober = Callable[[str, float], bool]
type Runner = Callable[[list[str], float], bool]
type Sleeper = Callable[[float], object]


def _log(message: str) -> None:
    """Write one hostwatch log line to host stdout."""
    print(f"[hostwatch] {message}", flush=True)


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
        completed = subprocess.run(cmd, timeout=timeout, check=False)
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
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        """Ask the watch loop to stop."""
        self._stop_requested.set()

    def recover(self) -> str:
        """Recover the stack, returning the escalation layer used."""
        if self.runner(self.daemon_check_cmd, 15):
            self.runner(self.container_cmd, 120)
            return "container"

        for cmd in self.provider_restart_cmds:
            self.runner(cmd, 60)
        self.runner(self.post_restart_cmd, 300)
        return "provider"

    def watch_forever(self) -> None:
        """Probe forever, escalating after consecutive failures."""
        failures = 0
        while not self._stop_requested.is_set():
            if self.prober(self.health_url, self.probe_timeout):
                failures = 0
            else:
                failures += 1
                _log(f"probe failed ({failures}/{self.max_failures})")
                if failures >= self.max_failures:
                    _log("failure threshold reached; starting recovery")
                    action = self.recover()
                    _log(f"recovery action: {action}")
                    self._sleep(self.settle)
                    failures = 0

            self._sleep(self.interval)

    def _sleep(self, seconds: float) -> None:
        """Sleep with immediate stop support in production and injectable tests."""
        if seconds <= 0 or self._stop_requested.is_set():
            return
        if self.sleeper is time.sleep:
            self._stop_requested.wait(seconds)
            return
        self.sleeper(seconds)


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


def main() -> int:
    """Build a host watcher from environment variables and run it."""
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
        )
    except ValueError as exc:
        _log(f"configuration error: {exc}")
        return 2

    watcher.watch_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
