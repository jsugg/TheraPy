"""Owner sovereignty API: graph lifecycle, corpus, proactivity, and crisis."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.type_contracts import HttpTestClient
from therapy.knowledge.embeddings import EmbeddingMetadata
from therapy.knowledge.research import IngestResult, ResearchKB
from therapy.knowledge.research_ingest import OCRBackend, OCRBlock
from therapy.knowledge.user_model import UserModel

if TYPE_CHECKING:
    from PIL.Image import Image


class APIEmbedder:
    @property
    def metadata(self) -> EmbeddingMetadata:
        return EmbeddingMetadata("api-test", "v1", 2)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        folded = text.casefold()
        vector = np.asarray(
            [float(any(word in folded for word in ("task", "start", "initiation"))), 0.1],
            dtype=np.float32,
        )
        return vector / np.linalg.norm(vector)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class APIOCR:
    @property
    def metadata(self) -> dict[str, object]:
        return {"engine": "api-local", "version": "1", "languages": ["eng"]}

    def recognize(self, image: Image) -> list[OCRBlock]:
        del image
        return [
            OCRBlock("A visible first step supports task initiation.", 0.96),
            OCRBlock("uncertain wrd", 0.42),
        ]


class APIResearchKB(ResearchKB):
    def ingest_bytes(
        self,
        data: bytes,
        filename: str,
        declared_type: str | None,
        *,
        source_title: str | None = None,
        source_ref: str | None = None,
        force: bool = False,
        ocr_backend: OCRBackend | None = None,
    ) -> IngestResult:
        return super().ingest_bytes(
            data,
            filename,
            declared_type,
            source_title=source_title,
            source_ref=source_ref,
            force=force,
            ocr_backend=ocr_backend or APIOCR(),
        )


def _proposed_node(model: UserModel, statement: str = "Runs after lunch.") -> int:
    node_id: int | None = None
    for index in range(3):
        node_id = model.upsert_node(
            "pattern",
            statement,
            session_id=f"node-session-{index}",
            evidence_key=f"node-evidence-{index}",
        )
    assert node_id is not None
    assert model.propose(node_id)
    return node_id


def _proposed_edge(model: UserModel) -> int:
    source = model.add_user_statement("pattern", "Late meetings happen.")
    target = model.add_user_statement("pattern", "Energy drops afterward.")
    assert source is not None
    assert target is not None
    edge_id: int | None = None
    for index in range(3):
        edge_id = model.upsert_edge(
            source,
            target,
            "triggers",
            statement="Late meetings trigger an energy drop.",
            session_id=f"edge-session-{index}",
            evidence_key=f"edge-evidence-{index}",
        )
    assert edge_id is not None
    assert model.propose_edge(edge_id)
    return edge_id


def test_graph_endpoint_filters_and_returns_exact_registries(
    data_dir: Path, client: HttpTestClient
) -> None:
    model = UserModel(data_dir)
    model.add_user_statement("identity_fact", "Is a teacher.")
    model.add_user_statement("goal", "Finish the portfolio.")
    model.add_boundary("never_initiate", "my ex")

    payload = client.get(
        "/api/graph", params={"node_type": "identity_fact", "source": "user-stated"}
    ).json()

    assert [node["statement"] for node in payload["nodes"]] == ["Is a teacher."]
    assert payload["boundaries"][0]["value"] == "my ex"
    assert client.get("/api/graph", params={"status": "invented"}).status_code == 422


def test_observation_confirmation_conflicts_but_proposal_can_confirm_edit_and_delete(
    data_dir: Path, client: HttpTestClient
) -> None:
    model = UserModel(data_dir)
    observation = model.upsert_node("pattern", "Runs on Tuesdays.")
    assert client.post(f"/api/graph/nodes/{observation}/confirm").status_code == 409
    node_id = _proposed_node(model)

    confirmed = client.post(f"/api/graph/nodes/{node_id}/confirm")
    assert confirmed.status_code == 200
    assert confirmed.json()["node"]["status"] == "confirmed"
    queue = client.get("/api/graph/pending").json()["pending_insights"]
    assert next(item for item in queue if item["claim_id"] == node_id)["state"] == "confirmed"
    edited = client.patch(
        f"/api/graph/nodes/{node_id}", json={"statement": "Runs on Wednesdays."}
    ).json()
    assert edited["node"]["statement"] == "Runs on Wednesdays."
    assert edited["node"]["user_edited"] is True
    assert client.patch(f"/api/graph/nodes/{node_id}", json={"extra": 1}).status_code == 422

    detail = client.get(f"/api/graph/nodes/{node_id}").json()
    assert len(detail["evidence"]) == 5
    assert any(event["event_type"] == "owner_edit" for event in detail["lifecycle"])
    assert client.delete(f"/api/graph/nodes/{node_id}").json() == {"deleted": node_id}


def test_edge_has_equal_edit_confirm_reject_delete_and_audit_api(
    data_dir: Path, client: HttpTestClient
) -> None:
    model = UserModel(data_dir)
    edge_id = _proposed_edge(model)

    confirmed = client.post(f"/api/graph/edges/{edge_id}/confirm").json()
    assert confirmed["edge"]["status"] == "confirmed"
    queue = client.get("/api/graph/pending").json()["pending_insights"]
    assert next(item for item in queue if item["claim_id"] == edge_id)["state"] == "confirmed"
    edited = client.patch(
        f"/api/graph/edges/{edge_id}",
        json={"statement": "Long late meetings reliably drain energy."},
    ).json()
    assert edited["edge"]["user_edited"] is True
    detail = client.get(f"/api/graph/edges/{edge_id}").json()
    assert detail["evidence"]
    assert detail["lifecycle"]
    assert client.delete(f"/api/graph/edges/{edge_id}").status_code == 200


def test_pending_insight_queue_resolves_exact_selected_claim(
    data_dir: Path, client: HttpTestClient
) -> None:
    model = UserModel(data_dir)
    first = _proposed_node(model, "Skips lunch under deadline pressure.")
    second = _proposed_node(model, "Planning aloud reduces friction.")
    queued = client.get("/api/graph/pending").json()["pending_insights"]
    selected = next(item for item in queued if item["claim_id"] == second)

    response = client.post(f"/api/insights/{selected['id']}/confirm")

    assert response.status_code == 200
    second_node = model.get_node(second)
    first_node = model.get_node(first)
    assert second_node is not None
    assert first_node is not None
    assert second_node["status"] == "confirmed"
    assert first_node["status"] == "proposed"
    history = client.get(f"/api/insights/{selected['id']}/history").json()["history"]
    assert history[-1]["event_type"] == "confirmed"

    edited_id = _proposed_node(model, "Starts planning with a visible first step.")
    assert client.patch(
        f"/api/graph/nodes/{edited_id}",
        json={"statement": "Starts planning from one visible step."},
    ).status_code == 200
    edited_queue = client.get("/api/graph/pending").json()["pending_insights"]
    assert next(item for item in edited_queue if item["claim_id"] == edited_id)[
        "state"
    ] == "dismissed"


def test_boundaries_are_strictly_validated_and_removable(
    client: HttpTestClient,
) -> None:
    added = client.post(
        "/api/graph/boundaries", json={"kind": "never_store", "value": "salary"}
    )
    assert added.status_code == 200
    assert client.post(
        "/api/graph/boundaries", json={"kind": "made_up", "value": "x"}
    ).status_code == 422
    removed = client.request(
        "DELETE",
        "/api/graph/boundaries",
        json={"kind": "never_store", "value": "salary"},
    )
    assert removed.status_code == 200
    assert removed.json()["boundaries"] == []


def test_research_upload_query_preview_reindex_and_delete(
    data_dir: Path, client: HttpTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from therapy.server import app as app_module

    kb = ResearchKB(data_dir, embedder=APIEmbedder())
    monkeypatch.setattr(app_module, "_research", lambda: kb)
    uploaded = client.post(
        "/api/research/ingest",
        files={"file": ("guide.md", b"# Starting\n\nA visible step supports task initiation.", "text/markdown")},
        data={"source_title": "Starting guide", "source_ref": "Brown 2026"},
    )
    assert uploaded.status_code == 200
    document_id = uploaded.json()["ingest"]["document_id"]

    query = client.get("/api/research/query", params={"q": "start task"}).json()
    assert query["sources"][0]["anchor"] == "section-starting-block-1"
    assert "Brown 2026" == query["sources"][0]["ref"]
    detail = client.get(f"/api/research/{document_id}").json()["document"]
    assert detail["blocks"][0]["heading"] == "Starting"
    assert client.post(f"/api/research/{document_id}/reindex").status_code == 200
    assert client.delete(f"/api/research/{document_id}").json() == {"deleted": document_id}


def test_research_api_accepts_full_source_matrix_with_ocr_review_and_citations(
    data_dir: Path, client: HttpTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from therapy.server import app as app_module

    image_module = pytest.importorskip("PIL.Image")
    pymupdf = pytest.importorskip("pymupdf")
    image = image_module.new("RGB", (160, 80), "white")
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "A visible first step supports task initiation.")
    pdf_payload = pdf.tobytes()
    pdf.close()
    sources = (
        ("notes.txt", b"A visible first step supports task initiation.", "text/plain", "text"),
        (
            "guide.md",
            b"# Starting tasks\n\nA visible first step supports task initiation.",
            "text/markdown",
            "markdown",
        ),
        (
            "guide.html",
            b"<h2>Starting tasks</h2><p>A visible first step supports task initiation.</p>",
            "text/html",
            "html",
        ),
        ("paper.pdf", pdf_payload, "application/pdf", "pdf"),
        ("scan.png", image_buffer.getvalue(), "image/png", "image"),
    )
    kb = APIResearchKB(data_dir, embedder=APIEmbedder())
    monkeypatch.setattr(app_module, "_research", lambda: kb)
    document_ids: list[int] = []

    for filename, payload, media_type, expected_format in sources:
        uploaded = client.post(
            "/api/research/ingest",
            files={"file": (filename, payload, media_type)},
            data={"source_title": filename, "source_ref": f"Owner {filename}"},
        )
        assert uploaded.status_code == 200, uploaded.text
        document = uploaded.json()["document"]
        document_id = int(document["id"])
        document_ids.append(document_id)
        assert document["format"] == expected_format
        sources_payload = client.get(
            "/api/research/query",
            params={"q": "start task", "k": 20, "threshold": 0},
        ).json()["sources"]
        citation = next(
            item for item in sources_payload if item["document_id"] == document_id
        )
        assert citation["anchor"]
        assert f"#{citation['anchor']}" in citation["citation"]

    image_document = client.get(f"/api/research/{document_ids[-1]}").json()["document"]
    assert image_document["status"] == "review_required"
    assert image_document["ocr_metadata"]["engine"] == "api-local"
    review_block = next(block for block in image_document["blocks"] if block["needs_review"])
    corrected = client.patch(
        f"/api/research/{document_ids[-1]}/blocks/{review_block['anchor']}",
        json={"text": "A planning checklist supports task initiation."},
    )
    assert corrected.status_code == 200
    assert corrected.json()["document"]["status"] == "indexed"


def test_proactivity_api_defaults_off_and_validates_timezone(
    client: HttpTestClient,
) -> None:
    channels = client.get("/api/proactivity").json()["channels"]
    assert [channel["enabled"] for channel in channels] == [False] * 4
    body = {
        "enabled": True,
        "timezone": "America/Sao_Paulo",
        "quiet_start": "22:00",
        "quiet_end": "08:00",
        "schedule_time": "18:30",
        "schedule_day": 6,
        "frequency": "weekly",
        "topic": "weekly planning",
    }
    assert client.put("/api/proactivity/check_in", json=body).status_code == 200
    body["timezone"] = "No/Such_Zone"
    assert client.put("/api/proactivity/check_in", json=body).status_code == 422


def test_crisis_resources_config_endpoint(
    client: HttpTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "THERAPY_CRISIS_CONTACTS", '[{"label": "Línea 135", "value": "135"}]'
    )
    payload = client.get("/api/crisis-resources").json()
    assert payload["contacts"][0]["value"] == "135"
    assert "135" in payload["resources"]

    monkeypatch.setenv("THERAPY_CRISIS_CONTACTS", "not-json")
    malformed = client.get("/api/crisis-resources")
    assert malformed.status_code == 503
    assert "invalid JSON" in malformed.json()["detail"]


def test_owner_audit_emits_terminal_bounded_outcomes(
    client: HttpTestClient,
) -> None:
    """O3 audit: rejected destructive calls must never be audited as success."""
    import io
    import json
    import logging

    from therapy.observability.logging import BroadJsonFormatter

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        BroadJsonFormatter(service_version="0.1.0", environment="test")
    )
    logger = logging.getLogger("therapy.broad")
    previous_handlers = logger.handlers
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        assert client.delete("/api/graph/nodes/999999").status_code == 404
        assert client.post(
            "/api/graph/boundaries", json={"kind": "never_store", "value": "probe"}
        ).status_code == 200
        assert client.request(
            "DELETE",
            "/api/graph/boundaries",
            json={"kind": "never_store", "value": "probe"},
        ).status_code == 200
    finally:
        logger.handlers = previous_handlers

    audits = [
        event
        for event in map(json.loads, stream.getvalue().splitlines())
        if event.get("event.name") == "owner.audit"
    ]
    by_operation = {event["operation"]: event["outcome"] for event in audits}
    assert by_operation["delete_node"] == "rejected"
    assert by_operation["remove_boundary"] == "success"
