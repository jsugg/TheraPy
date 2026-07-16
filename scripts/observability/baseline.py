"""Telemetry-off baseline measurement (plan O0.4).

Drives scripted workloads against a RUNNING TheraPy instance and records
distributions for HTTP reads, research ingest/query/delete, owner-data
export, scheduler delivery, and agent turns, alongside container CPU/RSS
samples and DB/log byte sizes. Voice TTFA comes from the existing
in-container `verify_voice_text_loop.py` run (invoked separately; see the
evidence report for the exact command).

Test-only routes (agent turn, scheduler run) exist only when the target
instance sets THERAPY_TEST_MODE=1. The live dogfood container correctly
does not; launch a disposable instance for those workloads and point
`--server` at it. Workloads that hit missing test routes are recorded as
skipped, never failed.

Deliberately NOT measured here (recorded as remaining measurements):
- data restore: destructive against the live owner dogfood volume; measured
  in the isolated disposable environment during O3 drills;
- event-loop lag and TTFA stage decomposition: require O1/O2 owned
  instrumentation; the O2 off/on comparison harness reuses this script so
  both sides share identical workloads.

Writes raw JSON to `.local/obs-baseline/baseline-<label>.json` (gitignored)
and prints a content-free summary. Raw logs never enter the repo.

Usage:
    .venv/bin/python scripts/observability/baseline.py --label off
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
FIXTURES = REPO_ROOT / "tests/fixtures/observability/research"


def _dist(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"count": 0}
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 4),
        "p50": round(statistics.median(ordered), 4),
        "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 4),
        "max": round(ordered[-1], 4),
        "mean": round(statistics.fmean(ordered), 4),
    }


class StatsSampler(threading.Thread):
    """Samples `docker stats` for one container (name substring) until stopped."""

    def __init__(self, interval: float = 2.0, container: str = "therapy-therapy") -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.container = container
        self.samples: list[dict[str, float]] = []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    [
                        "docker", "stats", "--no-stream", "--format",
                        "{{.Name}} {{.CPUPerc}} {{.MemUsage}}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                ).stdout
                for line in out.splitlines():
                    parts = line.split()
                    if len(parts) < 3 or self.container not in parts[0]:
                        continue
                    cpu = float(parts[1].rstrip("%"))
                    raw_mem = parts[2]
                    factor = 1e9 if "GiB" in raw_mem else 1e6
                    rss = float(raw_mem.rstrip("MGiB")) * factor
                    self.samples.append({"cpu_pct": cpu, "rss_bytes": rss})
            except (subprocess.SubprocessError, ValueError):
                pass
            self._halt.wait(self.interval)

    def stop(self) -> list[dict[str, float]]:
        self._halt.set()
        self.join(timeout=20)
        return self.samples


class Workloads:
    def __init__(self, client: httpx.Client, server: str) -> None:
        self.client = client
        self.server = server

    def _timed(self, method: str, path: str, **kw) -> tuple[float, httpx.Response]:
        start = time.perf_counter()
        response = self.client.request(method, f"{self.server}{path}", **kw)
        return time.perf_counter() - start, response

    def http_reads(self) -> dict[str, object]:
        results: dict[str, list[float]] = {
            "health": [], "sessions": [], "graph": [], "research_list": [],
        }
        for _ in range(50):
            duration, response = self._timed("GET", "/health")
            response.raise_for_status()
            results["health"].append(duration)
        for _ in range(20):
            for name, path in (
                ("sessions", "/api/sessions"),
                ("graph", "/api/graph"),
                ("research_list", "/api/research"),
            ):
                duration, response = self._timed("GET", path)
                response.raise_for_status()
                results[name].append(duration)
        return {name: _dist(values) for name, values in results.items()}

    def research(self) -> dict[str, object]:
        ingest_times: list[float] = []
        query_times: list[float] = []
        delete_times: list[float] = []
        doc_ids: list[str] = []
        for source in (
            "sleep_hygiene_synthetic.txt",
            "boundaries_worksheet_synthetic.txt",
        ):
            payload = (FIXTURES / source).read_bytes()
            duration, response = self._timed(
                "POST",
                "/api/research/ingest",
                files={"file": (f"baseline-{source}", payload, "text/plain")},
                data={"source_title": f"baseline {source}", "force": "true"},
            )
            ingest_times.append(duration)
            if response.status_code < 300:
                body = response.json()
                doc_id = body.get("ingest", {}).get("document_id")
                if doc_id:
                    doc_ids.append(str(doc_id))
        for _ in range(20):
            duration, response = self._timed(
                "GET",
                "/api/research/query",
                params={"q": "what helps me fall asleep faster"},
            )
            response.raise_for_status()
            query_times.append(duration)
        for doc_id in doc_ids:
            duration, _ = self._timed("DELETE", f"/api/research/{doc_id}")
            delete_times.append(duration)
        return {
            "ingest": _dist(ingest_times),
            "query": _dist(query_times),
            "delete": _dist(delete_times),
            "ingested_documents": len(doc_ids),
        }

    def data_export(self) -> dict[str, object]:
        times: list[float] = []
        sizes: list[int] = []
        for _ in range(5):
            duration, response = self._timed("GET", "/api/data/export")
            response.raise_for_status()
            times.append(duration)
            sizes.append(len(response.content))
        return {"export": _dist(times), "export_bytes": max(sizes) if sizes else 0}

    def agent_turns(self, turns: int) -> dict[str, object]:
        times: list[float] = []
        failures = 0
        phrases = [
            "I slept a little better this week and wanted to note it.",
            "Work felt calmer after I set that boundary we discussed.",
            "Quiero repasar lo que dije sobre mi rutina de sueño.",
        ]
        languages = ["en", "en", "es"]
        for index in range(turns):
            duration, response = self._timed(
                "POST",
                "/api/testing/agent/turn",
                json={
                    "text": phrases[index % 3],
                    "language": languages[index % 3],
                },
                timeout=300.0,
            )
            if response.status_code == 404:
                return {"skipped": "test mode disabled on target instance"}
            if response.status_code >= 300:
                failures += 1
            else:
                times.append(duration)
        return {"agent_turn": _dist(times), "failures": failures}

    def scheduler(self) -> dict[str, object]:
        times: list[float] = []
        failures = 0
        now = datetime.now(UTC).isoformat()
        for index in range(3):
            duration, response = self._timed(
                "POST",
                "/api/testing/proactivity/run",
                json={
                    "channel": "greeting",
                    "due_at": now,
                    "now": now,
                    "idempotency_key": f"baseline-{int(time.time())}-{index}",
                },
                timeout=120.0,
            )
            if response.status_code == 404:
                return {"skipped": "test mode disabled on target instance"}
            if response.status_code >= 300:
                failures += 1
            else:
                times.append(duration)
        return {"scheduler_run": _dist(times), "failures": failures}


def storage_sizes() -> dict[str, object]:
    sizes: dict[str, object] = {}
    try:
        out = subprocess.run(
            ["docker", "compose", "exec", "-T", "therapy", "sh", "-c",
             "du -sb /data 2>/dev/null | cut -f1"],
            capture_output=True, text=True, timeout=60, cwd=REPO_ROOT,
        ).stdout.strip().splitlines()
        if out:
            sizes["data_dir_bytes"] = int(out[0])
    except (subprocess.SubprocessError, ValueError) as exc:
        sizes["error"] = type(exc).__name__
    try:
        logs = subprocess.run(
            ["docker", "compose", "logs", "--no-color", "therapy"],
            capture_output=True, cwd=REPO_ROOT, timeout=120,
        ).stdout
        sizes["log_bytes_since_start"] = len(logs)
        sizes["log_lines_since_start"] = logs.count(b"\n")
    except subprocess.SubprocessError as exc:
        sizes["log_error"] = type(exc).__name__
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="off", help="baseline label (off/on)")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("--agent-turns", type=int, default=5)
    parser.add_argument("--skip-llm", action="store_true",
                        help="skip agent-turn/scheduler workloads entirely")
    parser.add_argument("--container", default="therapy-therapy",
                        help="docker stats name substring to sample")
    args = parser.parse_args()

    report: dict[str, object] = {
        "label": args.label,
        "started_at": datetime.now(UTC).isoformat(),
        "server": args.server,
    }

    idle = StatsSampler(container=args.container)
    idle.start()
    time.sleep(args.idle_seconds)
    idle_samples = idle.stop()
    report["idle"] = {
        "cpu_pct": _dist([s["cpu_pct"] for s in idle_samples]),
        "rss_bytes": _dist([s["rss_bytes"] for s in idle_samples]),
    }

    load = StatsSampler(container=args.container)
    load.start()
    with httpx.Client(timeout=60.0) as client:
        workloads = Workloads(client, args.server)
        report["http_reads"] = workloads.http_reads()
        report["research"] = workloads.research()
        report["data_export"] = workloads.data_export()
        if not args.skip_llm:
            report["agent_turns"] = workloads.agent_turns(args.agent_turns)
            report["scheduler"] = workloads.scheduler()
    load_samples = load.stop()
    report["under_load"] = {
        "cpu_pct": _dist([s["cpu_pct"] for s in load_samples]),
        "rss_bytes": _dist([s["rss_bytes"] for s in load_samples]),
    }

    report["storage"] = storage_sizes()
    report["finished_at"] = datetime.now(UTC).isoformat()

    out_dir = REPO_ROOT / ".local/obs-baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baseline-{args.label}-{int(time.time())}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
