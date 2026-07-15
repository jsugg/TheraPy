"""Phase 4 vertical acceptance through HTTP production entry points."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


@pytest.fixture
def test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable deterministic external-boundary doubles before app startup."""
    monkeypatch.setenv("THERAPY_TEST_MODE", "1")


@pytest.fixture
def acceptance_client(test_mode: None, client: TestClient) -> TestClient:
    """Return the centrally isolated client after test mode is active."""
    del test_mode
    return client


def _agent_turn(
    client: TestClient,
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
    response = client.post("/api/testing/agent/turn", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _seed_late_meeting_evidence(client: TestClient) -> None:
    for index in range(3):
        result = _agent_turn(
            client,
            f"Late meeting {index} drains my energy.",
            finalize=True,
        )
        distillation = result["distillation"]
        assert isinstance(distillation, dict)
    assert distillation["proposed_nodes"]
    assert distillation["proposed_edges"]


def _confirm_next(client: TestClient, topic: str) -> dict[str, object]:
    reflected = _agent_turn(client, topic)
    insight = reflected["insight"]
    assert isinstance(insight, dict), reflected
    confirmed = _agent_turn(client, "yes", session_id=str(reflected["session_id"]))
    resolution = confirmed["resolution"]
    assert isinstance(resolution, dict)
    assert resolution["state"] == "confirmed"
    assert "confirmed" in str(confirmed["reply"])
    return insight


def test_real_agent_loop_context_privacy_research_and_proactivity(
    acceptance_client: TestClient,
) -> None:
    client = acceptance_client
    _seed_late_meeting_evidence(client)

    proposed = client.get("/api/graph").json()
    assert {node["status"] for node in proposed["nodes"]} == {"proposed"}
    assert {edge["status"] for edge in proposed["edges"]} == {"proposed"}

    first = _confirm_next(client, "Late meetings drained my energy again.")
    second = _confirm_next(client, "Energy drops after late meetings.")
    third = _confirm_next(client, "Late meetings trigger an energy drop.")
    assert first["claim_kind"] == "node"
    assert second["claim_kind"] == "node"
    assert third["claim_kind"] == "edge"

    confirmed = client.get("/api/graph").json()
    assert all(node["status"] == "confirmed" for node in confirmed["nodes"])
    assert confirmed["edges"][0]["status"] == "confirmed"
    refreshed = _agent_turn(client, "A late meeting drained my energy today.")
    assert "Late meetings trigger an energy drop" in str(refreshed["memory_note"])

    for index in range(3):
        _agent_turn(
            client,
            f"My brother is a sensitive ongoing thread {index}.",
            finalize=True,
        )
    boundary = client.post(
        "/api/graph/boundaries",
        json={"kind": "never_initiate", "value": "brother"},
    )
    assert boundary.status_code == 200
    unsolicited = _agent_turn(client, "How should I plan tomorrow?")
    assert "brother" not in str(unsolicited["memory_note"]).casefold()
    user_raised = _agent_turn(client, "My brother is on my mind today.")
    assert "brother" in str(user_raised["memory_note"]).casefold()

    upload = client.post(
        "/api/research/ingest",
        files={
            "file": (
                "planning.md",
                b"# Planning transitions\n\nA visible checklist reduces planning load.",
                "text/markdown",
            )
        },
        data={"source_title": "Planning transitions", "source_ref": "Owner guide"},
    )
    assert upload.status_code == 200, upload.text
    grounded = _agent_turn(client, "Help me with planning a task transition.")
    assert "visible checklist" in str(grounded["reply"]).casefold()
    cited = client.get(
        "/api/research/query",
        params={"q": "planning checklist", "k": 1, "threshold": 0.1},
    )
    assert cited.status_code == 200, cited.text
    source = cited.json()["sources"][0]
    assert source["page"] is None
    assert source["section"] == "Planning transitions"
    assert source["anchor"] == "section-planning-transitions-block-1"

    settings = {
        "enabled": True,
        "timezone": "UTC",
        "quiet_start": "22:00",
        "quiet_end": "08:00",
        "schedule_time": "18:00",
        "schedule_day": 0,
        "frequency": "weekly",
        "topic": "planning",
    }
    assert client.put("/api/proactivity/check_in", json=settings).status_code == 200
    future_night = datetime(2036, 7, 15, 2, tzinfo=UTC)
    request = {
        "channel": "check_in",
        "due_at": future_night.isoformat(),
        "now": future_night.isoformat(),
        "idempotency_key": "acceptance-restart",
        "topic": "planning",
    }
    quiet = client.post("/api/testing/proactivity/run", json=request)
    assert quiet.status_code == 200, quiet.text
    assert quiet.json()["job"]["state"] == "retry"
    assert quiet.json()["job"]["result"] == {"reason": "quiet_hours"}

    from therapy.server import app as app_module

    app_module._proactivity.cache_clear()
    daytime = future_night.replace(hour=14)
    request["now"] = daytime.isoformat()
    delivered = client.post("/api/testing/proactivity/run", json=request)
    assert delivered.status_code == 200, delivered.text
    assert delivered.json()["job"]["state"] == "delivered"
    jobs = client.get("/api/proactivity/jobs").json()["jobs"]
    assert sum(job["idempotency_key"] == "acceptance-restart" for job in jobs) == 1
    messages = client.get("/api/proactivity/in-app?consume=false").json()["messages"]
    assert any(message["channel"] == "check_in" for message in messages)


def test_acceptance_routes_are_hidden_without_test_mode(client: TestClient) -> None:
    assert client.post(
        "/api/testing/agent/turn", json={"text": "hello", "language": "en"}
    ).status_code == 404
    assert client.post(
        "/api/testing/proactivity/run",
        json={
            "channel": "check_in",
            "due_at": "2036-07-15T02:00:00+00:00",
            "now": "2036-07-15T02:00:00+00:00",
            "idempotency_key": "hidden",
        },
    ).status_code == 404
