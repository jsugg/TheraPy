"""Tests for the local research knowledge base."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from therapy.knowledge.research import ResearchKB, embed

_BODY_TITLE = "Body doubling for ADHD"
_BODY_REF = "Brown 2021, ADHD Practice Review"
_BODY_TEXT = (
    "Body doubling supports task initiation in ADHD. Working beside another "
    "person can make it easier to start a task when you cannot get going."
)
_SENSORY_TITLE = "Deep pressure for sensory regulation"
_SENSORY_REF = "Lee 2020, Occupational Therapy Review"
_SENSORY_TEXT = (
    "Sensory regulation can use deep pressure to reduce overload and support "
    "a calmer level of alertness."
)


def _ingest_examples(kb: ResearchKB) -> tuple[int, int]:
    body_id = kb.ingest(_BODY_TITLE, _BODY_REF, _BODY_TEXT)
    sensory_id = kb.ingest(
        _SENSORY_TITLE,
        _SENSORY_REF,
        _SENSORY_TEXT,
    )
    return body_id, sensory_id


def test_embed_is_deterministic_normalized_and_zero_for_empty_text() -> None:
    first = embed("Body doubling helps task initiation.")
    second = embed("Body doubling helps task initiation.")
    empty = embed("")

    assert first == second
    assert abs(float(np.linalg.norm(first)) - 1.0) < 1e-12
    assert empty == [0.0] * 256


def test_query_ranks_the_relevant_body_doubling_chunk_first(
    tmp_path: Path,
) -> None:
    kb = ResearchKB(tmp_path)
    _ingest_examples(kb)

    results = kb.query("how do I start a task when I can't get going?")

    assert results
    assert results[0]["source_title"] == _BODY_TITLE
    assert "Body doubling" in str(results[0]["text"])
    assert float(results[0]["score"]) > 0.0


def test_psychoeducation_returns_an_answer_with_its_source(
    tmp_path: Path,
) -> None:
    kb = ResearchKB(tmp_path)
    _ingest_examples(kb)

    result = kb.psychoeducation(
        "What can body doubling do for task initiation in ADHD?"
    )

    assert result["answer"]
    assert {"title": _BODY_TITLE, "ref": _BODY_REF} in result["sources"]


def test_ground_returns_bare_text_and_none_for_an_empty_kb(
    tmp_path: Path,
) -> None:
    kb = ResearchKB(tmp_path)
    _ingest_examples(kb)

    grounded = kb.ground("deep pressure for sensory regulation")

    assert isinstance(grounded, str)
    assert "deep pressure" in grounded.lower()
    assert _SENSORY_REF not in grounded
    assert ResearchKB(tmp_path / "empty").ground("sensory regulation") is None


def test_delete_document_removes_its_chunks_from_retrieval(
    tmp_path: Path,
) -> None:
    kb = ResearchKB(tmp_path)
    body_id, _ = _ingest_examples(kb)

    assert kb.delete_document(body_id) is True
    assert kb.delete_document(body_id) is False

    results = kb.query("how do I start a task?", k=10)
    assert all(result["source_title"] != _BODY_TITLE for result in results)
    assert any(result["source_title"] == _SENSORY_TITLE for result in results)
    assert all(
        document["source_title"] != _BODY_TITLE
        for document in kb.documents()
    )


def test_export_all_contains_documents_without_vectors(tmp_path: Path) -> None:
    kb = ResearchKB(tmp_path)
    _ingest_examples(kb)

    snapshot = kb.export_all()
    documents = snapshot["documents"]

    assert isinstance(documents, list)
    assert {document["source_title"] for document in documents} == {
        _BODY_TITLE,
        _SENSORY_TITLE,
    }
    assert all("vector" not in document for document in documents)
    assert all(
        isinstance(chunk, str)
        for document in documents
        for chunk in document["chunks"]
    )
    assert json.loads(json.dumps(snapshot)) == snapshot
