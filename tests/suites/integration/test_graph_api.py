"""Tests for the review-UI sovereignty API: graph + research + crisis (W7/W8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from therapy.knowledge.user_model import UserModel
from therapy.server.app import _research


def test_graph_endpoint_returns_nodes_edges_and_boundaries(
    data_dir: Path, client
) -> None:
    model = UserModel(data_dir)
    model.upsert_node("identity", "Is a teacher.", source="user-stated")
    model.add_boundary("never_initiate", "my ex")

    payload = client.get("/api/graph").json()

    assert any(n["statement"] == "Is a teacher." for n in payload["nodes"])
    assert {"kind": "never_initiate", "value": "my ex"} == {
        "kind": payload["boundaries"][0]["kind"],
        "value": payload["boundaries"][0]["value"],
    }


def test_confirm_edit_and_delete_node(data_dir: Path, client) -> None:
    model = UserModel(data_dir)
    nid = model.upsert_node("routine", "Runs on Tuesdays.")

    confirmed = client.post(f"/api/graph/nodes/{nid}/confirm").json()
    assert confirmed["node"]["status"] == "confirmed"

    edited = client.patch(
        f"/api/graph/nodes/{nid}", json={"statement": "Runs on Wednesdays."}
    ).json()
    assert edited["node"]["statement"] == "Runs on Wednesdays."
    assert edited["node"]["user_edited"] is True

    assert client.delete(f"/api/graph/nodes/{nid}").json() == {"deleted": nid}
    assert client.get("/api/graph").json()["nodes"] == []


def test_boundaries_add_and_remove(client) -> None:
    added = client.post(
        "/api/graph/boundaries",
        json={"kind": "never_store", "value": "salary"},
    )
    assert added.status_code == 200
    assert any(b["value"] == "salary" for b in added.json()["boundaries"])

    removed = client.request(
        "DELETE",
        "/api/graph/boundaries",
        json={"kind": "never_store", "value": "salary"},
    )
    assert removed.status_code == 200
    assert removed.json()["boundaries"] == []


def test_research_query_returns_citation(client) -> None:
    _research().ingest(
        "Body doubling for ADHD",
        "Brown 2021, ADHD Practice Review",
        "Body doubling supports task initiation. Working beside another person "
        "makes it easier to start a task when you cannot get going.",
    )
    payload = client.get(
        "/api/research/query", params={"q": "how do I start a task"}
    ).json()
    assert payload["answer"]
    assert payload["sources"][0]["ref"] == "Brown 2021, ADHD Practice Review"


def test_crisis_resources_config_endpoint(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "THERAPY_CRISIS_CONTACTS",
        '[{"label": "Línea 135", "value": "135"}]',
    )
    payload = client.get("/api/crisis-resources").json()
    assert payload["contacts"][0]["value"] == "135"
    assert "135" in payload["resources"]
