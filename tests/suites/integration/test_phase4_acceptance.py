"""Phase 4 vertical acceptance through HTTP production entry points."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest

from tests.type_contracts import HttpTestClient
from therapy.observability.interactions import require_json_object


@pytest.fixture
def test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable deterministic external-boundary doubles before app startup."""
    monkeypatch.setenv("THERAPY_TEST_MODE", "1")


@pytest.fixture
def acceptance_client(
    test_mode: None, client: HttpTestClient
) -> HttpTestClient:
    """Return the centrally isolated client after test mode is active."""
    del test_mode
    return client


def _agent_turn(
    client: HttpTestClient,
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
    return _json_object(response.json())


def _json_object(value: object) -> dict[str, object]:
    """Validate one JSON object returned by the test HTTP boundary."""
    return cast(dict[str, object], require_json_object(value, "test.response"))


def _json_objects(value: object) -> list[dict[str, object]]:
    """Validate one JSON array of objects returned by the test boundary."""
    if not isinstance(value, list):
        raise TypeError("test response field must be an array")
    return [_json_object(item) for item in cast(list[object], value)]


def _seed_late_meeting_evidence(client: HttpTestClient) -> None:
    distillation: dict[str, object] = {}
    for index in range(3):
        result = _agent_turn(
            client,
            f"Late meeting {index} drains my energy.",
            finalize=True,
        )
        candidate = result["distillation"]
        distillation = _json_object(candidate)
    assert distillation["proposed_nodes"]
    assert distillation["proposed_edges"]


def _confirm_next(client: HttpTestClient, topic: str) -> dict[str, object]:
    reflected = _agent_turn(client, topic)
    insight = reflected["insight"]
    insight_object = _json_object(insight)
    confirmed = _agent_turn(client, "yes", session_id=str(reflected["session_id"]))
    resolution = confirmed["resolution"]
    resolution_object = _json_object(resolution)
    assert resolution_object["state"] == "confirmed"
    assert "confirmed" in str(confirmed["reply"])
    return insight_object


def test_real_agent_loop_context_privacy_research_and_proactivity(
    acceptance_client: HttpTestClient,
) -> None:
    client = acceptance_client
    _seed_late_meeting_evidence(client)

    proposed = _json_object(client.get("/api/graph").json())
    proposed_nodes = _json_objects(proposed["nodes"])
    proposed_edges = _json_objects(proposed["edges"])
    assert {node["status"] for node in proposed_nodes} == {"proposed"}
    assert {edge["status"] for edge in proposed_edges} == {"proposed"}

    first = _confirm_next(client, "Late meetings drained my energy again.")
    second = _confirm_next(client, "Energy drops after late meetings.")
    third = _confirm_next(client, "Late meetings trigger an energy drop.")
    claim_kinds = [first["claim_kind"], second["claim_kind"], third["claim_kind"]]
    assert claim_kinds.count("node") == 2
    assert claim_kinds.count("edge") == 1

    confirmed = _json_object(client.get("/api/graph").json())
    confirmed_nodes = _json_objects(confirmed["nodes"])
    confirmed_edges = _json_objects(confirmed["edges"])
    assert all(node["status"] == "confirmed" for node in confirmed_nodes)
    assert confirmed_edges[0]["status"] == "confirmed"
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
    cited_payload = _json_object(cited.json())
    source = _json_objects(cited_payload["sources"])[0]
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
    quiet_job = _json_object(_json_object(quiet.json())["job"])
    assert quiet_job["state"] == "retry"
    assert quiet_job["result"] == {"reason": "quiet_hours"}

    from therapy.server import app as app_module

    proactivity_factory = getattr(app_module, "_proactivity", None)
    cache_clear = getattr(proactivity_factory, "cache_clear", None)
    if not callable(cache_clear):
        raise TypeError("proactivity cache is unavailable")
    cache_clear()
    daytime = future_night.replace(hour=14)
    request["now"] = daytime.isoformat()
    delivered = client.post("/api/testing/proactivity/run", json=request)
    assert delivered.status_code == 200, delivered.text
    delivered_job = _json_object(_json_object(delivered.json())["job"])
    assert delivered_job["state"] == "delivered"
    jobs_payload = _json_object(client.get("/api/proactivity/jobs").json())
    jobs = _json_objects(jobs_payload["jobs"])
    assert sum(job["idempotency_key"] == "acceptance-restart" for job in jobs) == 1
    messages_payload = _json_object(
        client.get("/api/proactivity/in-app?consume=false").json()
    )
    messages = _json_objects(messages_payload["messages"])
    assert any(message["channel"] == "check_in" for message in messages)


def test_acceptance_routes_are_hidden_without_test_mode(
    client: HttpTestClient,
) -> None:
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
