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
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import BinaryIO, Literal, cast

import httpx

from therapy.observability.interactions import JsonValue, require_json_object

RESULTS: list[str] = []
SCRIPT_NAME = "verify_longitudinal_loop"
SCENARIOS = frozenset({"longitudinal-loop"})

type VerificationResult = Literal["pass", "fail"]


def _object_list(value: object, label: str) -> list[dict[str, JsonValue]]:
    """Validate an API field as a list of JSON objects."""
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return [
        require_json_object(item, f"{label}[{index}]")
        for index, item in enumerate(cast(list[object], value))
    ]


def build_verification_record(
    *, scenario: str, duration_s: float, result: VerificationResult
) -> dict[str, str | float]:
    """Build the final bounded verification record."""
    if scenario not in SCENARIOS:
        raise ValueError("unsupported verification scenario")
    try:
        build = version("therapy")
    except PackageNotFoundError:
        build = "unknown"
    return {
        "record": "verification",
        "script": SCRIPT_NAME,
        "build": build,
        "scenario": scenario,
        "duration_s": duration_s,
        "result": result,
        "environment": "test",
    }


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
def _server(data_dir: Path, log_path: Path) -> Generator[ServerProcess, None, None]:
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
            # owned launcher (obs plan O1.1): bootstrap before app import
            sys.executable,
            "-m",
            "therapy.server",
        ],
        env={
            **environment,
            "THERAPY_HOST": "127.0.0.1",
            "THERAPY_PORT": str(port),
            "THERAPY_LOG_LEVEL": "WARNING",
        },
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    server = ServerProcess(process, f"http://127.0.0.1:{port}", log_path, log_file)
    try:
        _await_health(server)
        yield server
    finally:
        server.stop()


def _json(response: httpx.Response) -> dict[str, JsonValue]:
    if response.status_code >= 400:
        raise AssertionError(
            f"{response.request.method} {response.request.url.path} -> "
            f"{response.status_code}: {response.text}"
        )
    payload: object = response.json()
    return require_json_object(payload, "API response")


def _agent_turn(
    client: httpx.Client,
    text: str,
    *,
    session_id: str | None = None,
    language: str = "en",
    finalize: bool = False,
) -> dict[str, JsonValue]:
    body: dict[str, object] = {
        "text": text,
        "language": language,
        "finalize": finalize,
    }
    if session_id is not None:
        body["session_id"] = session_id
    return _json(client.post("/api/testing/agent/turn", json=body))


def _confirm_next(client: httpx.Client, topic: str) -> dict[str, JsonValue]:
    raised = _agent_turn(client, topic)
    insight = raised.get("insight")
    _require("adjacent-reflection", isinstance(insight, dict), topic)
    insight_object = require_json_object(insight, "agent insight")
    resolution = _agent_turn(
        client,
        "yes",
        session_id=str(raised["session_id"]),
    )
    recorded = resolution.get("resolution")
    recorded_object = require_json_object(recorded, "agent insight resolution")
    _require(
        "conversational-confirmation",
        recorded_object.get("state") == "confirmed",
        "explicit yes resolves only the insight raised in this session",
    )
    return insight_object


def _verify_agent_graph_context_research(client: httpx.Client) -> None:
    last_distillation: dict[str, JsonValue] | None = None
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
    proposed_nodes = _object_list(nodes, "proposed graph nodes")
    proposed_edges = _object_list(edges, "proposed graph edges")
    _require(
        "no-premature-confirmation",
        bool(proposed_nodes)
        and bool(proposed_edges)
        and all(
            item.get("status") == "proposed"
            for item in [*proposed_nodes, *proposed_edges]
        ),
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
    typed_confirmed_nodes = _object_list(confirmed_nodes, "confirmed graph nodes")
    typed_confirmed_edges = _object_list(confirmed_edges, "confirmed graph edges")
    _require(
        "confirmed-graph",
        all(node.get("status") == "confirmed" for node in typed_confirmed_nodes)
        and all(edge.get("status") == "confirmed" for edge in typed_confirmed_edges),
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
    typed_sources = _object_list(sources, "research query sources")
    source = typed_sources[0] if typed_sources else {}
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
    quiet_response = _json(client.post("/api/testing/proactivity/run", json=request))
    quiet = require_json_object(quiet_response.get("job"), "quiet-hours job")
    _require(
        "quiet-hours",
        quiet.get("state") == "retry"
        and quiet.get("result") == {"reason": "quiet_hours"},
        "delivery is durably postponed inside overnight quiet hours",
    )
    request["now"] = future_night.replace(hour=14).isoformat()
    return request


def _restart_delivery(client: httpx.Client, request: dict[str, object]) -> None:
    delivery_response = _json(client.post("/api/testing/proactivity/run", json=request))
    delivered = require_json_object(delivery_response.get("job"), "delivered job")
    jobs_response = _json(client.get("/api/proactivity/jobs"))
    jobs = _object_list(jobs_response.get("jobs"), "proactivity jobs")
    messages_response = _json(client.get("/api/proactivity/in-app?consume=false"))
    messages = _object_list(messages_response.get("messages"), "in-app messages")
    _require(
        "restart-idempotency",
        delivered.get("state") == "delivered"
        and sum(job["idempotency_key"] == "verify-restart" for job in jobs) == 1
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


def _run_with_record() -> int:
    """Run the verifier and always leave its machine record last on stdout."""
    started = time.monotonic()
    exit_code = 1
    try:
        exit_code = main()
        return exit_code
    finally:
        result: VerificationResult = "pass" if exit_code == 0 else "fail"
        record = build_verification_record(
            scenario="longitudinal-loop",
            duration_s=time.monotonic() - started,
            result=result,
        )
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    raise SystemExit(_run_with_record())
