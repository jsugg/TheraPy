"""Local semantic research ingest, OCR review, citations, and lifecycle."""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from therapy.knowledge.embeddings import EmbeddingMetadata
from therapy.knowledge.research import CHUNK_POLICY_VERSION, ResearchKB
from therapy.knowledge.research_ingest import OCRBlock, extract_source

_BODY_TITLE = "Body doubling for ADHD"
_BODY_REF = "Brown 2021, ADHD Practice Review"
_BODY_TEXT = (
    "Body doubling supports task initiation. Working beside another person "
    "can make it easier to start a task when you cannot get going."
)
_SENSORY_TITLE = "Deep pressure for sensory regulation"
_SENSORY_REF = "Lee 2020, Occupational Therapy Review"
_SENSORY_TEXT = (
    "Sensory regulation can use deep pressure to reduce overload and support "
    "a calmer level of alertness."
)


class SemanticEmbedder:
    """Tiny multilingual concept backend; production never uses this test double."""

    def __init__(self, revision: str = "test-v1") -> None:
        self._metadata = EmbeddingMetadata("semantic-test", revision, 4)

    @property
    def metadata(self) -> EmbeddingMetadata:
        return self._metadata

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        folded = text.casefold()
        concepts = (
            ("task", "start", "initiation", "tarea", "começar", "iniciar", "doubling"),
            ("sensory", "pressure", "overload", "sensorial", "pressão", "sobrecarga"),
            ("sleep", "rest", "sono", "dormir", "sueño"),
            ("planning", "checklist", "plan", "lista"),
        )
        vector = np.asarray(
            [sum(token in folded for token in aliases) for aliases in concepts],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class FakeOCR:
    @property
    def metadata(self) -> dict[str, object]:
        return {"engine": "fake-local", "version": "1", "languages": ["eng", "spa", "por"]}

    def recognize(self, image) -> list[OCRBlock]:
        del image
        return [
            OCRBlock("Body doubling helps start a task.", 0.96, (1, 2, 100, 20)),
            OCRBlock("uncertain wrd", 0.42, (1, 30, 100, 48)),
        ]


def _kb(path: Path, revision: str = "test-v1") -> ResearchKB:
    return ResearchKB(path, embedder=SemanticEmbedder(revision))


def _ingest_examples(kb: ResearchKB) -> tuple[int, int]:
    return (
        kb.ingest(_BODY_TITLE, _BODY_REF, _BODY_TEXT),
        kb.ingest(_SENSORY_TITLE, _SENSORY_REF, _SENSORY_TEXT),
    )


def test_multilingual_semantic_query_and_exact_citation_anchor(tmp_path: Path) -> None:
    kb = _kb(tmp_path)
    _ingest_examples(kb)

    spanish = kb.query("¿cómo puedo iniciar una tarea?", threshold=0.1)
    portuguese = kb.query("pressão para sobrecarga sensorial", threshold=0.1)

    assert spanish[0]["source_title"] == _BODY_TITLE
    assert portuguese[0]["source_title"] == _SENSORY_TITLE
    assert spanish[0]["anchor"] == "section-document-block-1"
    assert "#section-document-block-1" in spanish[0]["citation"]


def test_psychoeducation_is_only_retrieved_text_with_source_attribution(
    tmp_path: Path,
) -> None:
    kb = _kb(tmp_path)
    _ingest_examples(kb)

    result = kb.psychoeducation("task initiation", k=1)

    assert _BODY_TEXT in result["answer"]
    assert result["sources"][0]["ref"] == _BODY_REF
    assert result["sources"][0]["anchor"] == "section-document-block-1"
    assert result["sources"][0]["citation"] in result["answer"]


def test_silent_grounding_labels_untrusted_research_and_not_user_memory(
    tmp_path: Path,
) -> None:
    kb = _kb(tmp_path)
    _ingest_examples(kb)

    context = kb.ground("deep pressure for sensory regulation")

    assert context is not None
    assert "UNTRUSTED" in context
    assert "never a fact about the user" in context
    assert _SENSORY_REF not in context
    assert _kb(tmp_path / "empty").ground("sensory regulation") is None


def test_content_hash_dedup_and_explicit_reingest(tmp_path: Path) -> None:
    kb = _kb(tmp_path)
    payload = b"# Planning\n\nA checklist supports task initiation."

    first = kb.ingest_bytes(payload, "guide.md", "text/markdown")
    duplicate = kb.ingest_bytes(payload, "guide.md", "text/markdown")
    forced = kb.ingest_bytes(payload, "guide.md", "text/markdown", force=True)

    assert duplicate["document_id"] == first["document_id"]
    assert duplicate["deduplicated"] is True
    assert forced["document_id"] == first["document_id"]
    assert forced["deduplicated"] is False
    assert len(kb.documents()) == 1


@pytest.mark.parametrize(
    ("payload", "filename", "media_type", "error"),
    [
        (b"", "empty.txt", "text/plain", "empty"),
        (b"text", "../escape.txt", "text/plain", "path"),
        (b"not a pdf", "paper.pdf", "application/pdf", "signature"),
        (b"RIFFxxxxNOPE", "image.webp", "image/webp", "signature"),
        (b"\xff", "invalid.txt", "text/plain", "UTF-8"),
        (b"contains\x00nul", "nul.txt", "text/plain", "NUL"),
        (b"plain text", "note.txt", "application/pdf", "MIME"),
        (b"\x89PNG\r\n\x1a\n", "scan.png", "image/jpeg", "MIME"),
    ],
)
def test_ingest_boundary_rejects_malformed_paths_signatures_encoding_and_mime(
    payload: bytes, filename: str, media_type: str, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        extract_source(payload, filename, media_type)


def test_plain_text_and_markdown_anchors_are_stable() -> None:
    plain = extract_source(
        b"First bounded paragraph.\n\nSecond bounded paragraph.",
        "notes.txt",
        "text/plain",
    )
    markdown = extract_source(
        b"# First heading\n\nOne.\n\n## Second heading\n\nTwo.",
        "notes.md",
        "text/markdown",
    )

    assert [block.anchor for block in plain.blocks] == [
        "section-document-block-1",
        "section-document-block-2",
    ]
    assert [block.anchor for block in markdown.blocks] == [
        "section-first-heading-block-1",
        "section-second-heading-block-1",
    ]


def test_html_active_content_is_stripped_and_heading_anchor_is_stable(
    tmp_path: Path,
) -> None:
    html = b"""
    <html><nav>Ignore navigation</nav><script>alert('never')</script>
    <h2>Task initiation</h2><p>Use a visible first step.</p>
    <style>.secret{}</style></html>
    """
    first = extract_source(html, "guide.html", "text/html")
    second = extract_source(html, "guide.html", "text/html")

    assert first.blocks == second.blocks
    assert first.blocks[0].anchor == "section-task-initiation-block-1"
    assert "alert" not in first.markdown
    assert "navigation" not in first.markdown


def test_image_ocr_preview_excludes_low_confidence_until_owner_correction(
    tmp_path: Path,
) -> None:
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", (120, 60), "white")
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    kb = _kb(tmp_path)

    result = kb.ingest_bytes(
        payload.getvalue(),
        "scan.png",
        "image/png",
        ocr_backend=FakeOCR(),
    )
    detail = kb.document(result["document_id"])

    assert result["status"] == "review_required"
    assert result["review_blocks"] == 1
    assert detail is not None
    assert detail["ocr_metadata"]["engine"] == "fake-local"
    initial = kb.query("uncertain", threshold=0.0)
    assert all("uncertain wrd" not in item["text"] for item in initial)

    assert kb.correct_block(
        result["document_id"], "image-1-block-2", "Planning checklist"
    )
    corrected = kb.document(result["document_id"])
    assert corrected is not None
    assert corrected["status"] == "indexed"
    assert kb.query("planning checklist", threshold=0.1)[0]["text"] == "Planning checklist"


def test_digital_pdf_preserves_page_anchor_without_ocr(tmp_path: Path) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Body doubling supports task initiation in daily routines.")
    payload = document.tobytes()
    document.close()
    kb = _kb(tmp_path)

    result = kb.ingest_bytes(payload, "paper.pdf", "application/pdf")
    detail = kb.document(result["document_id"])

    assert detail is not None
    assert detail["blocks"][0]["anchor"] == "page-1-block-1"
    assert detail["blocks"][0]["extraction_method"] == "digital"
    hit = kb.query("task initiation", threshold=0.1)[0]
    assert hit["page"] == 1
    assert "p. 1" in hit["citation"]


def test_scanned_pdf_pages_use_local_ocr_with_reviewable_page_anchors(
    tmp_path: Path,
) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    document.new_page()
    payload = document.tobytes()
    document.close()
    kb = _kb(tmp_path)

    result = kb.ingest_bytes(
        payload,
        "scan.pdf",
        "application/pdf",
        ocr_backend=FakeOCR(),
    )
    detail = kb.document(result["document_id"])

    assert detail is not None
    assert detail["status"] == "review_required"
    assert detail["ocr_metadata"]["engine"] == "fake-local"
    assert [block["anchor"] for block in detail["blocks"]] == [
        "page-1-block-1",
        "page-1-block-2",
    ]
    assert all(block["extraction_method"] == "ocr" for block in detail["blocks"])


def test_model_revision_change_triggers_deterministic_reindex(tmp_path: Path) -> None:
    first = _kb(tmp_path, "revision-one")
    document_id = first.ingest(_BODY_TITLE, _BODY_REF, _BODY_TEXT)
    assert first.document(document_id)["index_model_revision"] == "revision-one"

    second = _kb(tmp_path, "revision-two")
    assert second.query("task initiation", threshold=0.1)
    detail = second.document(document_id)
    assert detail is not None
    assert detail["index_model_revision"] == "revision-two"
    assert detail["chunk_policy_version"] == CHUNK_POLICY_VERSION


def test_export_contains_artifact_and_delete_removes_all_corpus_bytes(
    tmp_path: Path,
) -> None:
    kb = _kb(tmp_path)
    document_id = kb.ingest(_BODY_TITLE, _BODY_REF, _BODY_TEXT)

    snapshot = kb.export_all()
    document = snapshot["documents"][0]
    assert document["artifact_base64"]
    assert json.loads(json.dumps(snapshot)) == snapshot

    artifact = tmp_path / str(document["artifact_path"])
    assert artifact.exists()
    assert kb.delete_document(document_id)
    assert not artifact.exists()
    assert kb.documents() == []


def test_legacy_lexical_tables_migrate_without_source_loss(tmp_path: Path) -> None:
    database = tmp_path / "therapy.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE research_docs (
                id INTEGER PRIMARY KEY, source_title TEXT, source_ref TEXT, ingested_at TEXT
            );
            CREATE TABLE research_chunks (
                id INTEGER PRIMARY KEY, doc_id INTEGER, ord INTEGER, text TEXT, vector TEXT
            );
            INSERT INTO research_docs VALUES (1, 'Legacy paper', 'Doe 2020', '2026-01-01');
            INSERT INTO research_chunks VALUES (1, 1, 0, 'Task initiation support.', '[]');
            """
        )

    kb = _kb(tmp_path)

    assert kb.documents()[0]["source_title"] == "Legacy paper"
    assert kb.query("task initiation", threshold=0.1)[0]["source_ref"] == "Doe 2020"
