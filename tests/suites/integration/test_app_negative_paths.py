"""Negative-path contracts for owner-facing HTTP endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import httpx
import pytest

from tests.type_contracts import HttpTestClient
from therapy.knowledge.user_model import UserModel
from therapy.memory.store import MemoryStore


def _assert_validation(response: httpx.Response, field: str) -> None:
    """Assert a structured FastAPI validation error names the rejected field."""
    assert response.status_code == 422
    payload: object = response.json()
    assert isinstance(payload, dict)
    payload_dict = cast("dict[str, object]", payload)
    detail = payload_dict.get("detail")
    assert isinstance(detail, list)
    for item in cast("list[object]", detail):
        if not isinstance(item, dict):
            continue
        item_dict = cast("dict[str, object]", item)
        location = item_dict.get("loc")
        if (
            isinstance(location, list)
            and location
            and cast("list[object]", location)[-1] == field
        ):
            return
    pytest.fail(f"validation response did not identify {field!r}")


def _proposed_node(model: UserModel, statement: str) -> int:
    node_id: int | None = None
    for index in range(3):
        node_id = model.upsert_node(
            "pattern",
            statement,
            session_id=f"{statement}-{index}",
            evidence_key=f"{statement}-evidence-{index}",
        )
    assert node_id is not None
    assert model.propose(node_id)
    return node_id


def _proposed_edge(model: UserModel, suffix: str) -> int:
    source = model.add_user_statement("pattern", f"Source {suffix}.")
    target = model.add_user_statement("pattern", f"Target {suffix}.")
    assert source is not None
    assert target is not None
    edge_id: int | None = None
    for index in range(3):
        edge_id = model.upsert_edge(
            source,
            target,
            "triggers",
            statement=f"Source triggers target {suffix}.",
            session_id=f"edge-{suffix}-{index}",
            evidence_key=f"edge-{suffix}-evidence-{index}",
        )
    assert edge_id is not None
    assert model.propose_edge(edge_id)
    return edge_id


class _StaleInsights:
    """Resolve no claim, emulating a concurrent state transition."""

    def resolve_claim(self, claim_kind: str, claim_id: int, resolution: str) -> bool:
        del claim_kind, claim_id, resolution
        return False


class _IngestFailure:
    def __init__(self, error: ValueError | RuntimeError) -> None:
        self.error = error

    def ingest_bytes(
        self,
        data: bytes,
        filename: str,
        declared_type: str | None,
        *,
        source_title: str | None = None,
        source_ref: str | None = None,
        force: bool = False,
    ) -> dict[str, object]:
        del data, filename, declared_type, source_title, source_ref, force
        raise self.error


class _MissingResearch:
    def document(self, document_id: int) -> None:
        del document_id
        return None

    def correct_block(self, document_id: int, anchor: str, text: str) -> bool:
        del document_id, anchor, text
        return False

    def delete_document(self, document_id: int) -> bool:
        del document_id
        return False


class _CorrectionFailure(_MissingResearch):
    def correct_block(self, document_id: int, anchor: str, text: str) -> bool:
        del document_id, anchor, text
        raise ValueError("corrected block must contain visible text")


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/api/graph/nodes/999999", {"detail": "Node not found"}),
        ("PATCH", "/api/graph/nodes/999999", {"detail": "Node not found"}),
        ("POST", "/api/graph/nodes/999999/confirm", {"detail": "Node not found"}),
        ("POST", "/api/graph/nodes/999999/reject", {"detail": "Node not found"}),
        ("DELETE", "/api/graph/nodes/999999", {"detail": "Node not found"}),
        ("GET", "/api/graph/edges/999999", {"detail": "Edge not found"}),
        ("PATCH", "/api/graph/edges/999999", {"detail": "Edge not found"}),
        ("POST", "/api/graph/edges/999999/confirm", {"detail": "Edge not found"}),
        ("POST", "/api/graph/edges/999999/reject", {"detail": "Edge not found"}),
        ("DELETE", "/api/graph/edges/999999", {"detail": "Edge not found"}),
    ],
)
def test_graph_missing_entities_return_exact_errors(
    client: HttpTestClient, method: str, path: str, body: dict[str, str]
) -> None:
    kwargs: dict[str, object] = {}
    if method == "PATCH":
        kwargs["json"] = {"statement": "Valid edit."}
    response = client.request(method, path, **kwargs)
    assert response.status_code == 404
    assert response.json() == body


@pytest.mark.parametrize(
    ("path", "payload", "field"),
    [
        ("/api/graph/nodes/1", {}, "body"),
        ("/api/graph/nodes/1", {"statement": ""}, "statement"),
        ("/api/graph/edges/1", {}, "statement"),
        ("/api/graph/edges/1", {"statement": ""}, "statement"),
    ],
)
def test_graph_patch_payloads_are_strictly_validated(
    client: HttpTestClient, path: str, payload: dict[str, str], field: str
) -> None:
    _assert_validation(client.patch(path, json=payload), field)


def test_node_state_guards_reject_non_proposals_and_stale_resolutions(
    data_dir: Path, client: HttpTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from therapy.server import app as app_module

    model = UserModel(data_dir)
    observation = model.upsert_node("pattern", "Observed only once.")
    assert observation is not None
    confirm = client.post(f"/api/graph/nodes/{observation}/confirm")
    reject = client.post(f"/api/graph/nodes/{observation}/reject")
    assert confirm.status_code == 409
    assert confirm.json() == {"detail": "Only proposed nodes can be confirmed"}
    assert reject.status_code == 409
    assert reject.json() == {"detail": "Only proposed nodes can be rejected"}

    rejected_id = _proposed_node(model, "Reject this node.")
    rejected = client.post(f"/api/graph/nodes/{rejected_id}/reject")
    assert rejected.status_code == 200
    assert rejected.json()["node"]["status"] == "rejected"

    stale_confirm = _proposed_node(model, "Stale node confirmation.")
    stale_reject = _proposed_node(model, "Stale node rejection.")
    monkeypatch.setattr(app_module, "_insights", lambda: _StaleInsights())
    confirm = client.post(f"/api/graph/nodes/{stale_confirm}/confirm")
    reject = client.post(f"/api/graph/nodes/{stale_reject}/reject")
    assert confirm.status_code == 409
    assert confirm.json() == {"detail": "Node state changed"}
    assert reject.status_code == 409
    assert reject.json() == {"detail": "Node state changed"}


def test_edge_state_guards_reject_non_proposals_and_stale_resolutions(
    data_dir: Path, client: HttpTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from therapy.server import app as app_module

    model = UserModel(data_dir)
    source = model.add_user_statement("pattern", "Observation source.")
    target = model.add_user_statement("pattern", "Observation target.")
    assert source is not None
    assert target is not None
    observation = model.upsert_edge(
        source, target, "triggers", statement="Observed edge once."
    )
    assert observation is not None
    confirm = client.post(f"/api/graph/edges/{observation}/confirm")
    reject = client.post(f"/api/graph/edges/{observation}/reject")
    assert confirm.status_code == 409
    assert confirm.json() == {"detail": "Only proposed edges can be confirmed"}
    assert reject.status_code == 409
    assert reject.json() == {"detail": "Only proposed edges can be rejected"}

    rejected_id = _proposed_edge(model, "rejected")
    rejected = client.post(f"/api/graph/edges/{rejected_id}/reject")
    assert rejected.status_code == 200
    assert rejected.json()["edge"]["status"] == "rejected"

    stale_confirm = _proposed_edge(model, "stale-confirm")
    stale_reject = _proposed_edge(model, "stale-reject")
    monkeypatch.setattr(app_module, "_insights", lambda: _StaleInsights())
    confirm = client.post(f"/api/graph/edges/{stale_confirm}/confirm")
    reject = client.post(f"/api/graph/edges/{stale_reject}/reject")
    assert confirm.status_code == 409
    assert confirm.json() == {"detail": "Edge state changed"}
    assert reject.status_code == 409
    assert reject.json() == {"detail": "Edge state changed"}


def test_graph_and_insight_missing_state_guards_are_exact(
    client: HttpTestClient,
) -> None:
    boundaries = client.get("/api/graph/boundaries")
    assert boundaries.status_code == 200
    assert boundaries.json() == {"boundaries": []}

    missing_boundary = client.request(
        "DELETE",
        "/api/graph/boundaries",
        json={"kind": "never_store", "value": "unknown"},
    )
    assert missing_boundary.status_code == 404
    assert missing_boundary.json() == {"detail": "Boundary not found"}

    expected = {
        "confirm": "Insight is missing or not resolvable",
        "reject": "Insight is missing or not resolvable",
        "snooze": "Insight is missing or not snoozable",
        "dismiss": "Insight is missing or resolved",
    }
    for operation, detail in expected.items():
        kwargs: dict[str, object] = {}
        if operation == "snooze":
            kwargs["json"] = {"days": 1}
        response = client.post(f"/api/insights/missing/{operation}", **kwargs)
        assert response.status_code == 409
        assert response.json() == {"detail": detail}


@pytest.mark.parametrize(
    ("error", "status_code"),
    [(ValueError("unsupported source"), 422), (RuntimeError("extractor unavailable"), 503)],
)
def test_research_ingest_maps_domain_failures(
    client: HttpTestClient,
    monkeypatch: pytest.MonkeyPatch,
    error: ValueError | RuntimeError,
    status_code: int,
) -> None:
    from therapy.server import app as app_module

    monkeypatch.setattr(app_module, "_research", lambda: _IngestFailure(error))
    response = client.post(
        "/api/research/ingest",
        files={"file": ("note.txt", b"content", "text/plain")},
    )
    assert response.status_code == status_code
    assert response.json() == {"detail": str(error)}


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"q": ""}, "q"),
        ({"q": "focus", "k": 0}, "k"),
        ({"q": "focus", "k": 21}, "k"),
        ({"q": "focus", "threshold": -0.1}, "threshold"),
        ({"q": "focus", "threshold": 1.1}, "threshold"),
    ],
)
def test_research_query_guards_are_structured(
    client: HttpTestClient, params: dict[str, str | int | float], field: str
) -> None:
    _assert_validation(client.get("/api/research/query", params=params), field)


def test_research_missing_and_correction_errors_are_exact(
    client: HttpTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from therapy.server import app as app_module

    missing = _MissingResearch()
    monkeypatch.setattr(app_module, "_research", lambda: missing)
    expected = {"detail": "Research document not found"}
    for method, path in (
        ("GET", "/api/research/999999"),
        ("POST", "/api/research/999999/reindex"),
        ("DELETE", "/api/research/999999"),
    ):
        response = client.request(method, path)
        assert response.status_code == 404
        assert response.json() == expected

    block = client.patch(
        "/api/research/999999/blocks/missing", json={"text": "replacement"}
    )
    assert block.status_code == 404
    assert block.json() == {"detail": "Research block not found"}
    _assert_validation(
        client.patch("/api/research/1/blocks/anchor", json={"text": ""}), "text"
    )

    monkeypatch.setattr(app_module, "_research", lambda: _CorrectionFailure())
    failure = client.patch(
        "/api/research/1/blocks/anchor", json={"text": "visible replacement"}
    )
    assert failure.status_code == 422
    assert failure.json() == {"detail": "corrected block must contain visible text"}


@pytest.mark.parametrize(
    ("channel", "override", "field"),
    [
        ("email", {}, "channel"),
        ("check_in", {"enabled": 1}, "enabled"),
        ("check_in", {"quiet_start": "24:00"}, "quiet_start"),
        ("check_in", {"schedule_day": 7}, "schedule_day"),
        ("check_in", {"frequency": "monthly"}, "frequency"),
        ("check_in", {"topic": "x" * 501}, "topic"),
    ],
)
def test_proactivity_rejects_invalid_channel_settings(
    client: HttpTestClient, channel: str, override: dict[str, object], field: str
) -> None:
    body: dict[str, object] = {
        "enabled": True,
        "timezone": "UTC",
        "quiet_start": "22:00",
        "quiet_end": "08:00",
        "schedule_time": "18:30",
        "schedule_day": 6,
        "frequency": "weekly",
        "topic": "planning",
    }
    body.update(override)
    _assert_validation(client.put(f"/api/proactivity/{channel}", json=body), field)


def test_proactivity_rejects_missing_fields_and_unknown_timezone(
    client: HttpTestClient,
) -> None:
    _assert_validation(client.put("/api/proactivity/check_in", json={}), "enabled")
    body = {
        "enabled": True,
        "timezone": "No/Such_Zone",
        "quiet_start": "22:00",
        "quiet_end": "08:00",
        "schedule_time": "18:30",
    }
    response = client.put("/api/proactivity/check_in", json=body)
    assert response.status_code == 422
    assert response.json() == {"detail": "Unknown IANA timezone: 'No/Such_Zone'"}


def test_audio_guards_hide_foreign_and_traversal_shaped_sessions(
    data_dir: Path, client: HttpTestClient
) -> None:
    store = MemoryStore(data_dir)
    owner_session = store.create_session()
    foreign_session = store.create_session()
    turn_id = store.add_turn(
        owner_session, "user", "voice", "en", "Hello.", audio=b"\x01\x00" * 80
    )
    expected = {"detail": "No audio for this turn"}
    paths = [
        f"/api/sessions/{foreign_session}/turns/{turn_id}/audio",
        f"/api/sessions/{owner_session}/turns/999999/audio",
        "/api/sessions/%2E%2E/turns/1/audio",
        "/api/sessions/..%5Cetc%5Cpasswd/turns/1/audio",
    ]
    for path in paths:
        response = client.get(path)
        assert response.status_code == 404
        assert response.json() == expected
