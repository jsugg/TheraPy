"""Local semantic research ingest, OCR review, citations, and lifecycle."""

from __future__ import annotations

import io
import json
import logging
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from therapy.knowledge.embeddings import (
    MODEL_DIMENSION,
    EmbeddingMetadata,
    FastEmbedBackend,
)
from therapy.knowledge.research import CHUNK_POLICY_VERSION, ResearchKB
from therapy.knowledge.research_ingest import (
    OCR_TIMEOUT_SECONDS,
    OCRBlock,
    TesseractOCR,
    extract_source,
)

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

type MetricCall = tuple[str, float, dict[str, str]]

_METRIC_ENUMS = {
    "therapy_research_ingests_total": {
        "format": {"pdf", "image", "html", "markdown", "text", "unknown"},
        "outcome": {"success", "error", "timeout", "rejected"},
        "deduplicated": {"true", "false"},
    },
    "therapy_research_stage_seconds": {
        "stage": {"validate", "extract", "ocr", "artifact", "db", "embed", "index"},
        "outcome": {"success", "error", "timeout", "rejected"},
    },
    "therapy_research_queries_total": {"outcome": {"hit", "no_hit", "error"}},
    "therapy_research_reindex_total": {
        "outcome": {"success", "error", "timeout", "rejected"}
    },
    "therapy_research_divergence_total": {
        "kind": {"db_without_file", "file_without_db"}
    },
    "therapy_ocr_runs_total": {"outcome": {"success", "timeout", "error", "skipped"}},
    "therapy_ocr_seconds": {"outcome": {"success", "timeout", "error"}},
    "therapy_embedding_batches_total": {
        "cache": {"cold", "warm"},
        "outcome": {"success", "error"},
    },
    "therapy_embedding_seconds": {"cache": {"cold", "warm"}},
}


@pytest.fixture
def metric_calls(monkeypatch: pytest.MonkeyPatch) -> list[MetricCall]:
    from therapy.observability import telemetry

    calls: list[MetricCall] = []

    def capture(name: str, value: float, attrs: dict[str, str] | None = None) -> None:
        calls.append((name, value, attrs or {}))

    monkeypatch.setattr(telemetry, "record_metric", capture)
    return calls


def _assert_metric_cardinality(calls: list[MetricCall]) -> None:
    """Assert target callers emit only their declared finite dimensions."""
    from therapy.observability.metrics import INSTRUMENT_INDEX

    for name, _, attributes in calls:
        expected = _METRIC_ENUMS.get(name)
        if expected is None:
            continue
        declared = INSTRUMENT_INDEX[name].attributes
        assert expected.keys() == declared.keys()
        assert attributes.keys() == expected.keys()
        for key, value in attributes.items():
            if declared[key] is not None:
                assert expected[key] == set(declared[key])
            assert value in expected[key]


class StubEmbeddingModel:
    """Minimal FastEmbed-compatible model for local telemetry tests."""

    def embed(self, texts: list[str]) -> Iterator[np.ndarray]:
        for _ in texts:
            yield np.ones(MODEL_DIMENSION, dtype=np.float32)


class StubSpan:
    """Capture bounded integer span attributes."""

    def __init__(self) -> None:
        self.attributes: dict[str, int] = {}

    def set_attribute(self, key: str, value: int) -> None:
        self.attributes[key] = value


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
        return {
            "engine": "fake-local",
            "version": "1",
            "languages": ["eng", "spa", "por"],
        }

    def recognize(self, image) -> list[OCRBlock]:
        del image
        return [
            OCRBlock("Body doubling helps start a task.", 0.96, (1, 2, 100, 20)),
            OCRBlock("uncertain wrd", 0.42, (1, 30, 100, 48)),
        ]


class TimeoutOCR:
    """OCR double that proves timeout classification without subprocesses."""

    @property
    def metadata(self) -> dict[str, object]:
        return {}

    def recognize(self, image: object) -> list[OCRBlock]:
        del image
        raise TimeoutError("private-ocr-timeout-payload-canary")


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


def test_content_hash_dedup_and_explicit_reingest(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
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
    assert (
        "therapy_research_ingests_total",
        1,
        {"format": "markdown", "outcome": "success", "deduplicated": "true"},
    ) in metric_calls
    _assert_metric_cardinality(metric_calls)


def test_research_observability_success_spans_and_cardinality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
) -> None:
    from therapy.observability import telemetry

    span_calls: list[tuple[str, str, str, StubSpan]] = []

    @contextmanager
    def capture_span(
        name: str, *, component: str, operation: str
    ) -> Iterator[StubSpan]:
        span = StubSpan()
        span_calls.append((name, component, operation, span))
        yield span

    monkeypatch.setattr(telemetry, "broad_span", capture_span)
    kb = _kb(tmp_path)
    result = kb.ingest_bytes(
        b"Private-text-canary supports task initiation.",
        "private-name-canary.txt",
        "text/plain",
        source_title="Private-title-canary",
        source_ref="Private-ref-canary",
    )

    assert result["deduplicated"] is False
    assert kb.query("task initiation", threshold=0.1)
    assert kb.query("private-query-canary", threshold=1.0) == []

    assert (
        "therapy_research_ingests_total",
        1,
        {"format": "text", "outcome": "success", "deduplicated": "false"},
    ) in metric_calls
    stages = {
        attrs["stage"]
        for name, _, attrs in metric_calls
        if name == "therapy_research_stage_seconds" and attrs["outcome"] == "success"
    }
    assert {"validate", "extract", "artifact", "db", "embed", "index"} <= stages
    assert (
        "therapy_research_reindex_total",
        1,
        {"outcome": "success"},
    ) in metric_calls
    query_outcomes = [
        attrs["outcome"]
        for name, _, attrs in metric_calls
        if name == "therapy_research_queries_total"
    ]
    assert query_outcomes == ["hit", "no_hit"]
    assert {name for name, _, _, _ in span_calls} >= {
        "research.ingest_bytes",
        "research.reindex",
        "research.query",
        "research.reindex.document",
    }
    document_spans = [
        span for name, _, _, span in span_calls if name == "research.reindex.document"
    ]
    assert document_spans
    assert document_spans[0].attributes == {
        "research.document_count": 1,
        "research.block_count": 1,
        "research.chunk_count": 1,
        "research.excluded_review_count": 0,
    }
    _assert_metric_cardinality(metric_calls)
    broad_payload = repr(metric_calls) + repr(span_calls)
    assert all(
        canary not in broad_payload.casefold()
        for canary in (
            "private-text-canary",
            "private-name-canary",
            "private-title-canary",
            "private-ref-canary",
            "private-query-canary",
        )
    )


def test_invalid_ingest_records_rejected_bounded_outcome(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    kb = _kb(tmp_path)

    with pytest.raises(ValueError, match="unsupported source format"):
        kb.ingest_bytes(
            b"private-invalid-content-canary",
            "private-invalid-name-canary.exe",
            "application/octet-stream",
        )

    assert (
        "therapy_research_ingests_total",
        1,
        {"format": "unknown", "outcome": "rejected", "deduplicated": "false"},
    ) in metric_calls
    assert any(
        name == "therapy_research_stage_seconds"
        and attrs == {"stage": "validate", "outcome": "rejected"}
        for name, _, attrs in metric_calls
    )
    _assert_metric_cardinality(metric_calls)
    assert "private-invalid" not in repr(metric_calls).casefold()


def test_unexpected_db_failure_records_error_and_file_without_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
) -> None:
    from therapy.observability import logging as observability_logging

    event_calls: list[tuple[str, dict[str, object]]] = []

    def capture_event(event_name: str, **kwargs: object) -> None:
        event_calls.append((event_name, kwargs))

    def fail_insert(*args: object) -> None:
        del args
        raise RuntimeError("private-exception-payload-canary")

    monkeypatch.setattr(observability_logging, "emit_event", capture_event)
    kb = _kb(tmp_path)
    monkeypatch.setattr(kb, "_insert_blocks", fail_insert)

    with pytest.raises(RuntimeError, match="private-exception"):
        kb.ingest_bytes(
            b"private-db-content-canary",
            "private-db-name-canary.txt",
            "text/plain",
        )

    assert (
        "therapy_research_ingests_total",
        1,
        {"format": "text", "outcome": "error", "deduplicated": "false"},
    ) in metric_calls
    assert any(
        name == "therapy_research_stage_seconds"
        and attrs == {"stage": "db", "outcome": "error"}
        for name, _, attrs in metric_calls
    )
    assert (
        "therapy_research_divergence_total",
        1.0,
        {"kind": "file_without_db"},
    ) in metric_calls
    assert event_calls == [
        (
            "research_artifact_divergence",
            {
                "component": "research",
                "operation": "artifact_write",
                "outcome": "file_without_db",
                "severity": logging.ERROR,
                "error_type": "artifact_consistency",
                "count": 1,
            },
        )
    ]
    _assert_metric_cardinality(metric_calls)
    broad_payload = (repr(metric_calls) + repr(event_calls)).casefold()
    assert "private-db" not in broad_payload
    assert "private-exception" not in broad_payload


def test_delete_detects_db_without_file_without_leaking_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
) -> None:
    from therapy.observability import logging as observability_logging

    event_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        observability_logging,
        "emit_event",
        lambda event_name, **kwargs: event_calls.append((event_name, kwargs)),
    )
    kb = _kb(tmp_path)
    result = kb.ingest_bytes(
        b"private-delete-content-canary",
        "private-delete-name-canary.txt",
        "text/plain",
    )
    detail = kb.document(result["document_id"])
    assert detail is not None
    artifact = tmp_path / str(detail["artifact_path"])
    artifact.unlink()
    metric_calls.clear()

    assert kb.delete_document(result["document_id"])
    assert (
        "therapy_research_divergence_total",
        1.0,
        {"kind": "db_without_file"},
    ) in metric_calls
    assert event_calls[-1][1]["outcome"] == "db_without_file"
    assert any(
        name == "therapy_research_stage_seconds"
        and attrs == {"stage": "artifact", "outcome": "success"}
        for name, _, attrs in metric_calls
    )
    _assert_metric_cardinality(metric_calls)
    broad_payload = (repr(metric_calls) + repr(event_calls)).casefold()
    assert "private-delete" not in broad_payload
    assert str(tmp_path).casefold() not in broad_payload


def test_reindex_and_query_exceptions_record_error_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
) -> None:
    kb = _kb(tmp_path)
    document_id = kb.ingest(_BODY_TITLE, _BODY_REF, _BODY_TEXT)
    metric_calls.clear()

    def fail_embedding(*args: object) -> list[np.ndarray]:
        del args
        raise RuntimeError("private-embedding-error-canary")

    monkeypatch.setattr(kb.embedder, "embed_documents", fail_embedding)
    with pytest.raises(RuntimeError, match="private-embedding"):
        kb.reindex(document_id)
    monkeypatch.setattr(kb.embedder, "embed_query", fail_embedding)
    with pytest.raises(RuntimeError, match="private-embedding"):
        kb.query("private-query-error-canary")

    assert (
        "therapy_research_reindex_total",
        1,
        {"outcome": "error"},
    ) in metric_calls
    assert (
        "therapy_research_queries_total",
        1,
        {"outcome": "error"},
    ) in metric_calls
    embed_errors = [
        attrs
        for name, _, attrs in metric_calls
        if name == "therapy_research_stage_seconds"
        and attrs == {"stage": "embed", "outcome": "error"}
    ]
    assert len(embed_errors) == 2
    _assert_metric_cardinality(metric_calls)
    assert "private-embedding" not in repr(metric_calls).casefold()


def test_duplicate_retry_repairs_index_after_initial_embedding_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
) -> None:
    kb = _kb(tmp_path)
    embed_documents = kb.embedder.embed_documents

    def fail_embedding(texts: list[str]) -> list[np.ndarray]:
        del texts
        raise RuntimeError("private-initial-index-failure-canary")

    monkeypatch.setattr(kb.embedder, "embed_documents", fail_embedding)
    payload = b"A checklist supports task initiation."
    with pytest.raises(RuntimeError, match="private-initial-index"):
        kb.ingest_bytes(payload, "private-retry-name-canary.txt", "text/plain")
    monkeypatch.setattr(kb.embedder, "embed_documents", embed_documents)

    retried = kb.ingest_bytes(payload, "private-retry-name-canary.txt", "text/plain")

    assert retried["deduplicated"] is True
    detail = kb.document(retried["document_id"])
    assert detail is not None
    assert detail["index_model_revision"] == kb.embedder.metadata.revision
    assert kb.query("task initiation", threshold=0.1)
    assert (
        "therapy_research_reindex_total",
        1,
        {"outcome": "error"},
    ) in metric_calls
    assert (
        "therapy_research_reindex_total",
        1,
        {"outcome": "success"},
    ) in metric_calls
    _assert_metric_cardinality(metric_calls)
    assert "private-" not in repr(metric_calls).casefold()


def test_fastembed_records_cold_warm_success_and_load_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
) -> None:
    backend = FastEmbedBackend(tmp_path)
    model = StubEmbeddingModel()

    def load_model() -> StubEmbeddingModel:
        backend._model = model
        return model

    monkeypatch.setattr(backend, "_load", load_model)
    backend.embed_documents(["private-passage-canary"])
    backend.embed_query("private-query-canary")
    backend._model = None

    def fail_load() -> StubEmbeddingModel:
        raise RuntimeError("private-model-load-canary")

    monkeypatch.setattr(backend, "_load", fail_load)
    with pytest.raises(RuntimeError, match="private-model-load"):
        backend.embed_query("private-failing-query-canary")

    batches = [
        attrs
        for name, _, attrs in metric_calls
        if name == "therapy_embedding_batches_total"
    ]
    assert batches == [
        {"cache": "cold", "outcome": "success"},
        {"cache": "warm", "outcome": "success"},
        {"cache": "cold", "outcome": "error"},
    ]
    assert [
        attrs["cache"]
        for name, _, attrs in metric_calls
        if name == "therapy_embedding_seconds"
    ] == ["cold", "warm", "cold"]
    _assert_metric_cardinality(metric_calls)
    broad_payload = repr(metric_calls).casefold()
    assert "private-" not in broad_payload
    assert str(tmp_path).casefold() not in broad_payload


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
    tmp_path: Path, metric_calls: list[MetricCall]
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
    assert (
        kb.query("planning checklist", threshold=0.1)[0]["text"] == "Planning checklist"
    )
    assert (
        "therapy_ocr_runs_total",
        1.0,
        {"outcome": "success"},
    ) in metric_calls
    assert any(
        name == "therapy_ocr_seconds" and attrs == {"outcome": "success"}
        for name, _, attrs in metric_calls
    )
    assert any(
        name == "therapy_research_stage_seconds"
        and attrs == {"stage": "ocr", "outcome": "success"}
        for name, _, attrs in metric_calls
    )
    _assert_metric_cardinality(metric_calls)


def test_ocr_timeout_records_timeout_without_payload(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", (16, 16), "white")
    payload = io.BytesIO()
    image.save(payload, format="PNG")

    with pytest.raises(TimeoutError, match="private-ocr-timeout"):
        _kb(tmp_path).ingest_bytes(
            payload.getvalue(),
            "private-ocr-name-canary.png",
            "image/png",
            ocr_backend=TimeoutOCR(),
        )

    assert (
        "therapy_ocr_runs_total",
        1.0,
        {"outcome": "timeout"},
    ) in metric_calls
    assert any(
        name == "therapy_ocr_seconds" and attrs == {"outcome": "timeout"}
        for name, _, attrs in metric_calls
    )
    assert any(
        name == "therapy_research_stage_seconds"
        and attrs == {"stage": "ocr", "outcome": "timeout"}
        for name, _, attrs in metric_calls
    )
    assert (
        "therapy_research_ingests_total",
        1,
        {"format": "image", "outcome": "timeout", "deduplicated": "false"},
    ) in metric_calls
    _assert_metric_cardinality(metric_calls)
    assert "private-ocr" not in repr(metric_calls).casefold()


def test_tesseract_cold_start_emits_bounded_version_validity_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from therapy.observability import logging as observability_logging

    event_calls: list[tuple[str, dict[str, object]]] = []
    timeouts: list[float] = []

    def image_to_osd(image: object, *, timeout: float) -> str:
        del image
        timeouts.append(timeout)
        return ""

    def image_to_data(image: object, **kwargs: object) -> dict[str, list[object]]:
        del image
        timeouts.append(float(kwargs["timeout"]))
        return {"text": []}

    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        SimpleNamespace(
            get_languages=lambda config: ["eng"],
            get_tesseract_version=lambda: "private-version-canary",
            image_to_osd=image_to_osd,
            image_to_data=image_to_data,
            TesseractError=RuntimeError,
            Output=SimpleNamespace(DICT="dict"),
        ),
    )
    monkeypatch.setattr(
        observability_logging,
        "emit_event",
        lambda event_name, **kwargs: event_calls.append((event_name, kwargs)),
    )
    backend = TesseractOCR("eng")

    assert backend.metadata["engine"] == "tesseract"
    assert backend.metadata["engine"] == "tesseract"
    image_module = pytest.importorskip("PIL.Image")
    assert backend.recognize(image_module.new("RGB", (16, 16), "white")) == []

    assert event_calls == [
        (
            "research_ocr_cold_start",
            {
                "component": "research_ocr",
                "operation": "validate_backend",
                "outcome": "success",
                "severity": logging.INFO,
                "error_type": None,
            },
        )
    ]
    assert timeouts == [OCR_TIMEOUT_SECONDS, OCR_TIMEOUT_SECONDS]
    assert "private-version" not in repr(event_calls).casefold()


def test_digital_pdf_preserves_page_anchor_without_ocr(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72), "Body doubling supports task initiation in daily routines."
    )
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
    assert (
        "therapy_ocr_runs_total",
        1.0,
        {"outcome": "skipped"},
    ) in metric_calls
    assert not any(
        name == "therapy_ocr_seconds" and attrs == {"outcome": "skipped"}
        for name, _, attrs in metric_calls
    )
    _assert_metric_cardinality(metric_calls)


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
