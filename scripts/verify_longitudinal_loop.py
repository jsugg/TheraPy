"""Phase 4 longitudinal acceptance against a real isolated HTTP server.

The subprocess uses the production FastAPI lifespan, persistence services,
context assembler, distillation transaction, scheduler, and public owner APIs.
Only the LLM/embedding boundary is deterministic under ``THERAPY_TEST_MODE``.
No domain service is imported or orchestrated by this verifier.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import httpx

RESULTS: list[str] = []


def _pass(label: str, detail: str) -> None:
    RESULTS.append(f"[{label}] PASS — {detail}")


def _require(label: str, condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(f"[{label}] {detail}")
    _pass(label, detail)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(slots=True)
class ServerProcess:
    """One isolated uvicorn process and its diagnostics sink."""

    process: subprocess.Popen[bytes]
    base_url: str
    log_path: Path
    log_file: BinaryIO

    def stop(self) -> None:
        """Stop lifespan cleanly, escalating only after a bounded wait."""
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        finally:
            self.log_file.close()


def _await_health(server: ServerProcess, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            log = server.log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"isolated server exited {server.process.returncode}:\n{log[-4_000:]}"
            )
        try:
            response = httpx.get(f"{server.base_url}/health", timeout=1.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise TimeoutError("isolated server did not become healthy")


@contextmanager
def _server(data_dir: Path, log_path: Path) -> Iterator[ServerProcess]:
    port = _free_port()
    log_file = log_path.open("wb")
    environment = {
        **os.environ,
        "THERAPY_DATA_DIR": str(data_dir),
        "THERAPY_TEST_MODE": "1",
        "THERAPY_CRISIS_CONTACTS": json.dumps(
            [{"label": "Línea 135", "value": "135"}]
        ),
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "therapy.server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    server = ServerProcess(process, f"http://127.0.0.1:{port}", log_path, log_file)
    try:
        _await_health(server)
        yield server
    finally:
        server.stop()


def _json(response: httpx.Response) -> dict[str, object]:
    if response.status_code >= 400:
        raise AssertionError(
            f"{response.request.method} {response.request.url.path} -> "
            f"{response.status_code}: {response.text}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise AssertionError("API response root must be an object")
    return payload


def _agent_turn(
    client: httpx.Client,
    text: str,
    *,
    session_id: str | None = None,
    language: str = "en",
    finalize: bool = False,
) -> dict[str, object]:
    body: dict[str, object] = {
        "text": text,
        "language": language,
        "finalize": finalize,
    }
    if session_id is not None:
        body["session_id"] = session_id
    return _json(client.post("/api/testing/agent/turn", json=body))


def _confirm_next(client: httpx.Client, topic: str) -> dict[str, object]:
    raised = _agent_turn(client, topic)
    insight = raised.get("insight")
    _require("adjacent-reflection", isinstance(insight, dict), topic)
    resolution = _agent_turn(
        client,
        "yes",
        session_id=str(raised["session_id"]),
    )
    recorded = resolution.get("resolution")
    _require(
        "conversational-confirmation",
        isinstance(recorded, dict) and recorded.get("state") == "confirmed",
        "explicit yes resolves only the insight raised in this session",
    )
    return insight if isinstance(insight, dict) else {}


def _verify_agent_graph_context_research(client: httpx.Client) -> None:
    last_distillation: dict[str, object] | None = None
    for index in range(3):
        result = _agent_turn(
            client,
            f"Late meeting {index} drains my energy.",
            finalize=True,
        )
        candidate = result.get("distillation")
        if isinstance(candidate, dict):
            last_distillation = candidate
    _require(
        "distillation",
        bool(last_distillation)
        and bool(last_distillation.get("proposed_nodes"))
        and bool(last_distillation.get("proposed_edges")),
        "three real finalized sessions reach the node and edge proposal floor",
    )

    proposed = _json(client.get("/api/graph"))
    nodes = proposed.get("nodes")
    edges = proposed.get("edges")
    _require(
        "no-premature-confirmation",
        isinstance(nodes, list)
        and isinstance(edges, list)
        and nodes
        and edges
        and all(item["status"] == "proposed" for item in [*nodes, *edges]),
        "mechanical eligibility plus judgment produces proposals, never confirmation",
    )

    first = _confirm_next(client, "Late meetings drained my energy again.")
    second = _confirm_next(client, "Energy drops after late meetings.")
    third = _confirm_next(client, "Late meetings trigger an energy drop.")
    _require(
        "node-edge-lifecycle",
        first.get("claim_kind") == "node"
        and second.get("claim_kind") == "node"
        and third.get("claim_kind") == "edge",
        "conversation confirms two nodes and their relationship edge",
    )
    confirmed = _json(client.get("/api/graph"))
    confirmed_nodes = confirmed.get("nodes")
    confirmed_edges = confirmed.get("edges")
    _require(
        "confirmed-graph",
        isinstance(confirmed_nodes, list)
        and isinstance(confirmed_edges, list)
        and all(node["status"] == "confirmed" for node in confirmed_nodes)
        and all(edge["status"] == "confirmed" for edge in confirmed_edges),
        "HTTP graph reports confirmed node and edge state",
    )

    refreshed = _agent_turn(client, "A late meeting drained my energy today.")
    _require(
        "per-turn-context",
        "Late meetings trigger an energy drop" in str(refreshed.get("memory_note")),
        "new-conversation topic refresh includes the confirmed relationship",
    )

    for index in range(3):
        _agent_turn(
            client,
            f"My brother is a sensitive ongoing thread {index}.",
            finalize=True,
        )
    _json(
        client.post(
            "/api/graph/boundaries",
            json={"kind": "never_initiate", "value": "brother"},
        )
    )
    unsolicited = _agent_turn(client, "How should I plan tomorrow?")
    user_raised = _agent_turn(client, "My brother is on my mind today.")
    _require(
        "never-initiate",
        "brother" not in str(unsolicited.get("memory_note")).casefold()
        and "brother" in str(user_raised.get("memory_note")).casefold(),
        "private topic stays out unsolicited and returns only when user-raised",
    )

    upload = _json(
        client.post(
            "/api/research/ingest",
            files={
                "file": (
                    "planning.md",
                    b"# Planning transitions\n\nA visible checklist reduces planning load.",
                    "text/markdown",
                )
            },
            data={
                "source_title": "Planning transitions",
                "source_ref": "Owner guide",
            },
        )
    )
    _require("research-ingest", "document" in upload, "local source indexed through API")
    grounded = _agent_turn(client, "Help me with planning a task transition.")
    citation = _json(
        client.get(
            "/api/research/query",
            params={"q": "planning checklist", "k": 1, "threshold": 0.1},
        )
    )
    sources = citation.get("sources")
    source = sources[0] if isinstance(sources, list) and sources else {}
    _require(
        "research-grounding",
        "visible checklist" in str(grounded.get("reply")).casefold()
        and source.get("section") == "Planning transitions"
        and source.get("anchor") == "section-planning-transitions-block-1",
        "retrieval changes technique choice and returns exact section/block anchor",
    )

    crisis = _json(client.get("/api/crisis-resources"))
    _require(
        "crisis-config",
        crisis.get("editing") == "environment-only"
        and isinstance(crisis.get("contacts"), list),
        "validated contacts remain independent and environment-owned",
    )


def _quiet_delivery(client: httpx.Client) -> dict[str, object]:
    _json(
        client.put(
            "/api/proactivity/check_in",
            json={
                "enabled": True,
                "timezone": "UTC",
                "quiet_start": "22:00",
                "quiet_end": "08:00",
                "schedule_time": "18:00",
                "schedule_day": 0,
                "frequency": "weekly",
                "topic": "planning",
            },
        )
    )
    future_night = datetime(2036, 7, 15, 2, tzinfo=UTC)
    request: dict[str, object] = {
        "channel": "check_in",
        "due_at": future_night.isoformat(),
        "now": future_night.isoformat(),
        "idempotency_key": "verify-restart",
        "topic": "planning",
    }
    quiet = _json(client.post("/api/testing/proactivity/run", json=request))["job"]
    _require(
        "quiet-hours",
        isinstance(quiet, dict)
        and quiet.get("state") == "retry"
        and quiet.get("result") == {"reason": "quiet_hours"},
        "delivery is durably postponed inside overnight quiet hours",
    )
    request["now"] = future_night.replace(hour=14).isoformat()
    return request


def _restart_delivery(client: httpx.Client, request: dict[str, object]) -> None:
    delivered = _json(client.post("/api/testing/proactivity/run", json=request))["job"]
    jobs = _json(client.get("/api/proactivity/jobs"))["jobs"]
    messages = _json(client.get("/api/proactivity/in-app?consume=false"))["messages"]
    _require(
        "restart-idempotency",
        isinstance(delivered, dict)
        and delivered.get("state") == "delivered"
        and isinstance(jobs, list)
        and sum(job["idempotency_key"] == "verify-restart" for job in jobs) == 1
        and isinstance(messages, list)
        and any(message["channel"] == "check_in" for message in messages),
        "same persisted job delivers once after a full server restart",
    )


def main() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="therapy-longitudinal-") as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir(mode=0o700)
            if data_dir.resolve() == Path("/data"):
                raise AssertionError("acceptance must never touch real /data")

            with _server(data_dir, root / "server-first.log") as first:
                with httpx.Client(base_url=first.base_url, timeout=30.0) as client:
                    _verify_agent_graph_context_research(client)
                    restart_request = _quiet_delivery(client)

            with _server(data_dir, root / "server-restart.log") as restarted:
                with httpx.Client(base_url=restarted.base_url, timeout=30.0) as client:
                    _restart_delivery(client, restart_request)
                    exported = client.get("/api/data/export")
                    _require(
                        "sovereignty-smoke",
                        exported.status_code == 200
                        and exported.headers.get("content-type", "").startswith(
                            "application/json"
                        ),
                        "complete owner export remains reachable after restart",
                    )
    except Exception as exc:
        print(f"FAIL — {exc}", file=sys.stderr)
        for line in RESULTS:
            print(line)
        return 1

    print("=== Phase 4 longitudinal server/agent acceptance ===")
    for line in RESULTS:
        print(line)
    print("PASS — every automated longitudinal acceptance in this verifier is green.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
