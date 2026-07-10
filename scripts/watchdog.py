"""PID-1 HTTP watchdog for deployment reliability (SPEC §5).

A wedged event loop can keep the server process alive while HTTP handling is
dead, so process-liveness checks alone are insufficient. Docker restart
policies only cover process exits; this supervisor restarts the app when the
health endpoint stops answering.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from types import FrameType

DEFAULT_CMD = (
    "uv run --no-dev uvicorn therapy.server.app:app --host 0.0.0.0 --port 8000"
)
DEFAULT_URL = "http://127.0.0.1:8000/health"
DEFAULT_INTERVAL = 15.0
DEFAULT_TIMEOUT = 10.0
DEFAULT_FAILURES = 3
DEFAULT_GRACE = 120.0
STOP_TIMEOUT = 10.0
LOOP_SLEEP = 0.05

_SignalHandler = Callable[[int, FrameType | None], object] | int | signal.Handlers | None


def _log(message: str) -> None:
    """Write one watchdog log line to container stdout."""
    print(f"[watchdog] {message}", flush=True)


def _exit_code(returncode: int | None) -> int:
    """Convert a subprocess returncode into a shell-style exit code."""
    if returncode is None:
        return 0
    if returncode < 0:
        return 128 + abs(returncode)
    return returncode


def probe(url: str, timeout: float) -> bool:
    """Return True iff one HTTP GET receives status 200."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


class Watchdog:
    """Supervise a child process with process and HTTP health checks."""

    def __init__(
        self,
        cmd: list[str],
        url: str,
        interval: float,
        timeout: float,
        max_failures: int,
        grace: float,
    ) -> None:
        if not cmd:
            raise ValueError("cmd must contain at least one argument")
        if interval <= 0:
            raise ValueError("interval must be greater than 0")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        if max_failures < 1:
            raise ValueError("max_failures must be at least 1")
        if grace < 0:
            raise ValueError("grace must be greater than or equal to 0")

        self.cmd = cmd
        self.url = url
        self.interval = interval
        self.timeout = timeout
        self.max_failures = max_failures
        self.grace = grace
        self.child: subprocess.Popen[bytes] | None = None
        self.restart_count = 0

        self._consecutive_failures = 0
        self._last_exit_code = 0
        self._last_start = 0.0
        self._next_probe = 0.0
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        """Ask the run loop to stop without using process signals."""
        self._stop_requested.set()

    def start_child(self) -> None:
        """Start the supervised command in its own process session."""
        _log(f"starting child: {shlex.join(self.cmd)}")
        self.child = subprocess.Popen(self.cmd, start_new_session=True)
        self._last_start = time.monotonic()
        self._next_probe = self._last_start + self.grace
        self._consecutive_failures = 0

    def stop_child(self) -> None:
        """Terminate the child process group, escalating to SIGKILL if needed."""
        child = self.child
        if child is None:
            self._last_exit_code = 0
            return

        if child.poll() is not None:
            self._last_exit_code = _exit_code(child.returncode)
            self.child = None
            return

        _log(f"stopping child pid={child.pid}")
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            child.wait(timeout=STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            _log(f"child pid={child.pid} ignored SIGTERM; sending SIGKILL")
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()

        self._last_exit_code = _exit_code(child.returncode)
        self.child = None

    def run_forever(self) -> int:
        """Run until a stop request or signal, returning the child exit code."""
        previous_handlers = self._install_signal_handlers()
        try:
            self.start_child()
            while not self._stop_requested.is_set():
                if self._restart_exited_child():
                    continue

                now = time.monotonic()
                if now >= self._next_probe:
                    if self._probe_failed():
                        continue
                    self._next_probe = time.monotonic() + self.interval

                sleep_for = max(0.0, min(LOOP_SLEEP, self._next_probe - now))
                self._stop_requested.wait(sleep_for)
        finally:
            self.stop_child()
            self._restore_signal_handlers(previous_handlers)

        return self._last_exit_code

    def _install_signal_handlers(self) -> list[tuple[int, _SignalHandler]]:
        """Install stop handlers when running in the main thread."""
        if threading.current_thread() is not threading.main_thread():
            return []

        previous_handlers: list[tuple[int, _SignalHandler]] = []

        def handle_stop(signum: int, _frame: FrameType | None) -> None:
            _log(f"received signal {signum}; shutting down")
            self.request_stop()

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers.append((signum, signal.getsignal(signum)))
            signal.signal(signum, handle_stop)
        return previous_handlers

    def _restore_signal_handlers(
        self, previous_handlers: list[tuple[int, _SignalHandler]]
    ) -> None:
        """Restore signal handlers replaced by run_forever."""
        for signum, handler in previous_handlers:
            signal.signal(signum, handler)

    def _restart_exited_child(self) -> bool:
        """Restart after a process crash; return True when restarted."""
        child = self.child
        if child is None or child.poll() is None:
            return False

        self._last_exit_code = _exit_code(child.returncode)
        _log(f"child pid={child.pid} exited with code {self._last_exit_code}; restarting")
        self.child = None
        self.restart_count += 1
        self.start_child()
        return True

    def _probe_failed(self) -> bool:
        """Probe health and restart the child after too many failures."""
        if probe(self.url, self.timeout):
            if self._consecutive_failures:
                _log("health probe recovered")
            self._consecutive_failures = 0
            return False

        self._consecutive_failures += 1
        _log(
            "health probe failed "
            f"({self._consecutive_failures}/{self.max_failures})"
        )
        if self._consecutive_failures < self.max_failures:
            return False

        _log("health failure threshold reached; restarting child")
        self.stop_child()
        self.restart_count += 1
        self.start_child()
        return True


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
    """Build a watchdog from environment variables and run it."""
    try:
        cmd = shlex.split(os.environ.get("WATCHDOG_CMD", DEFAULT_CMD))
        watchdog = Watchdog(
            cmd=cmd,
            url=os.environ.get("WATCHDOG_URL", DEFAULT_URL),
            interval=_float_env("WATCHDOG_INTERVAL", DEFAULT_INTERVAL),
            timeout=_float_env("WATCHDOG_TIMEOUT", DEFAULT_TIMEOUT),
            max_failures=_int_env("WATCHDOG_FAILURES", DEFAULT_FAILURES),
            grace=_float_env("WATCHDOG_GRACE", DEFAULT_GRACE),
        )
    except ValueError as exc:
        _log(f"configuration error: {exc}")
        return 2

    return watchdog.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
