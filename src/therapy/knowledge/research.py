"""Research knowledge base: curated local RAG over vetted literature.

Separate from the user model — this store knows *the territory*
(neurodiversity research, CBT/OT technique), not the user. Manual,
user-owned curation; no auto-scraping. Two retrieval modes (SPEC §5):
silent grounding (default) and source-attributed psychoeducation on
demand. Phase 4.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from heapq import nsmallest
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_EMBEDDING_DIM: int = 256
_TOKEN_RE: re.Pattern[str] = re.compile(r"[a-z0-9]+")
_PARAGRAPH_RE: re.Pattern[str] = re.compile(r"\n\s*\n+")
_SENTENCE_RE: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+")


def embed(text: str, dim: int = 256) -> list[float]:
    """Create a deterministic, L2-normalized hashed term-frequency vector.

    Args:
        text: Text to embed.
        dim: Number of vector dimensions.

    Returns:
        A list of normalized floating-point term frequencies.

    Raises:
        ValueError: If `dim` is less than one.
    """
    if dim < 1:
        raise ValueError("dim must be >= 1")

    vector = np.zeros(dim, dtype=np.float64)
    for token in _TOKEN_RE.findall(text.lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, byteorder="big") % dim
        vector[index] += 1.0

    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector.tolist()


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _wrap_words(text: str, limit: int) -> list[str]:
    """Split oversized text into non-empty pieces bounded by `limit`."""
    pieces: list[str] = []
    current = ""
    for word in text.split():
        if len(word) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(
                word[offset : offset + limit]
                for offset in range(0, len(word), limit)
            )
            continue

        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= limit:
            current = candidate
        else:
            pieces.append(current)
            current = word

    if current:
        pieces.append(current)
    return pieces


def _split_chunks(text: str, limit: int) -> list[str]:
    """Split text near paragraph and sentence boundaries."""
    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in _PARAGRAPH_RE.split(text.strip())
        if paragraph.strip()
    ]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) <= limit:
            units = [paragraph]
        else:
            sentences = [
                sentence.strip()
                for sentence in _SENTENCE_RE.split(paragraph)
                if sentence.strip()
            ]
            units = [
                piece
                for sentence in sentences
                for piece in _wrap_words(sentence, limit)
            ]

        for unit_index, unit in enumerate(units):
            separator = "\n\n" if unit_index == 0 else " "
            candidate = unit if not current else f"{current}{separator}{unit}"
            if len(candidate) <= limit:
                current = candidate
            else:
                chunks.append(current)
                current = unit

    if current:
        chunks.append(current)
    return chunks


def _decode_vector(raw: object) -> NDArray[np.float64]:
    """Decode and validate a stored embedding vector."""
    if not isinstance(raw, str):
        raise ValueError("stored research vector must be JSON text")

    try:
        values: object = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("stored research vector is invalid JSON") from error

    if (
        not isinstance(values, list)
        or len(values) != _EMBEDDING_DIM
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        )
    ):
        raise ValueError(
            f"stored research vector must contain {_EMBEDDING_DIM} numbers"
        )

    vector = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("stored research vector must contain finite numbers")
    return vector


class ResearchKB:
    """Store and retrieve curated research in a local SQLite database."""

    def __init__(self, data_dir: Path | None = None) -> None:
        """Create the data directory and initialize the research schema.

        Args:
            data_dir: Base directory for the shared `therapy.db` file. When
                omitted, use `THERAPY_DATA_DIR`, then `./data`.
        """
        self.data_dir = data_dir or Path(
            os.environ.get("THERAPY_DATA_DIR", "./data")
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.data_dir / "therapy.db"
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a configured SQLite connection for one method call."""
        connection = sqlite3.connect(self._db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
        finally:
            connection.close()

    def _init_schema(self) -> None:
        """Create research tables if this is a new database."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_docs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_title TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    ingested_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL
                        REFERENCES research_docs(id) ON DELETE CASCADE,
                    ord INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    vector TEXT NOT NULL
                );
                """
            )

    def _search(self, text: str, k: int) -> list[dict[str, object]]:
        """Return ranked chunks for an embedded query."""
        if k < 0:
            raise ValueError("k must be >= 0")
        if k == 0:
            return []

        query_vector = np.asarray(
            embed(text, dim=_EMBEDDING_DIM), dtype=np.float64
        )
        with self._connect() as connection:
            rows: list[sqlite3.Row] = connection.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    c.text,
                    c.vector,
                    d.source_title,
                    d.source_ref
                FROM research_chunks AS c
                JOIN research_docs AS d ON d.id = c.doc_id
                """
            ).fetchall()

        if not rows:
            return []

        vectors = [_decode_vector(row["vector"]) for row in rows]
        matrix = np.vstack(vectors)
        query_norm = float(np.linalg.norm(query_vector))
        denominators = np.linalg.norm(matrix, axis=1) * query_norm
        dot_products = matrix @ query_vector
        scores = np.divide(
            dot_products,
            denominators,
            out=np.zeros_like(dot_products),
            where=denominators > 0.0,
        )
        scores = np.clip(scores, 0.0, 1.0)

        indices = nsmallest(
            min(k, len(rows)),
            range(len(rows)),
            key=lambda index: (
                -float(scores[index]),
                int(rows[index]["chunk_id"]),
            ),
        )
        return [
            {
                "text": str(rows[index]["text"]),
                "source_title": str(rows[index]["source_title"]),
                "source_ref": str(rows[index]["source_ref"]),
                "score": float(scores[index]),
            }
            for index in indices
        ]

    def ingest(
        self,
        source_title: str,
        source_ref: str,
        text: str,
        *,
        chunk_chars: int = 400,
    ) -> int:
        """Chunk, embed, and store one curated research document.

        Args:
            source_title: Human-readable title for the source.
            source_ref: Citation string for the source.
            text: Vetted source text to store.
            chunk_chars: Approximate maximum number of characters per chunk.

        Returns:
            The new research document ID.

        Raises:
            ValueError: If `chunk_chars` is less than one.
            RuntimeError: If SQLite does not return a document ID.
        """
        if chunk_chars < 1:
            raise ValueError("chunk_chars must be >= 1")

        chunks = _split_chunks(text, chunk_chars)
        encoded_chunks = [
            json.dumps(embed(chunk, dim=_EMBEDDING_DIM), separators=(",", ":"))
            for chunk in chunks
        ]

        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO research_docs (
                        source_title, source_ref, ingested_at
                    )
                    VALUES (?, ?, ?)
                    """,
                    (source_title, source_ref, _utc_now()),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("SQLite did not return a document ID")
                doc_id = int(cursor.lastrowid)
                connection.executemany(
                    """
                    INSERT INTO research_chunks (
                        doc_id, ord, text, vector
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        (doc_id, order, chunk, encoded_chunks[order])
                        for order, chunk in enumerate(chunks)
                    ),
                )
        return doc_id

    def query(self, text: str, k: int = 3) -> list[dict[str, object]]:
        """Return the highest-scoring research chunks.

        Args:
            text: Query text to embed and compare.
            k: Maximum number of chunks to return.

        Returns:
            Ranked chunk dictionaries with source metadata and cosine scores.

        Raises:
            ValueError: If `k` is negative.
        """
        return self._search(text, k)

    def ground(self, topic: str) -> str | None:
        """Return only the best relevant chunk for silent grounding.

        Args:
            topic: Topic to ground.

        Returns:
            The best chunk text, or `None` when no positive match exists.
        """
        results = self._search(topic, 1)
        if not results or float(results[0]["score"]) <= 0.0:
            return None
        return str(results[0]["text"])

    def psychoeducation(
        self, query: str, k: int = 3
    ) -> dict[str, object]:
        """Return source-attributed research text for psychoeducation.

        Args:
            query: Question or topic to retrieve.
            k: Maximum number of chunks to include.

        Returns:
            An answer assembled from chunks and unique sources in rank order.

        Raises:
            ValueError: If `k` is negative.
        """
        results = self._search(query, k)
        sources: list[dict[str, str]] = []
        seen_sources: set[tuple[str, str]] = set()
        for result in results:
            title = str(result["source_title"])
            source_ref = str(result["source_ref"])
            source_key = (title, source_ref)
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({"title": title, "ref": source_ref})

        return {
            "answer": "\n\n".join(str(result["text"]) for result in results),
            "sources": sources,
        }

    def documents(self) -> list[dict[str, object]]:
        """Return ingested documents newest first for review."""
        with self._connect() as connection:
            rows: list[sqlite3.Row] = connection.execute(
                """
                SELECT id, source_title, source_ref, ingested_at
                FROM research_docs
                ORDER BY ingested_at DESC, id DESC
                """
            ).fetchall()

        return [
            {
                "id": int(row["id"]),
                "source_title": str(row["source_title"]),
                "source_ref": str(row["source_ref"]),
                "ingested_at": str(row["ingested_at"]),
            }
            for row in rows
        ]

    def delete_document(self, doc_id: int) -> bool:
        """Delete one research document and its chunks.

        Args:
            doc_id: Research document ID to delete.

        Returns:
            `True` if a document was deleted, otherwise `False`.
        """
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM research_docs WHERE id = ?",
                    (doc_id,),
                )
        return cursor.rowcount > 0

    def export_all(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot without embedding vectors."""
        with self._connect() as connection:
            rows: list[sqlite3.Row] = connection.execute(
                """
                SELECT
                    d.id,
                    d.source_title,
                    d.source_ref,
                    d.ingested_at,
                    c.id AS chunk_id,
                    c.text AS chunk_text
                FROM research_docs AS d
                LEFT JOIN research_chunks AS c ON c.doc_id = d.id
                ORDER BY d.ingested_at DESC, d.id DESC, c.ord ASC, c.id ASC
                """
            ).fetchall()

        documents: list[dict[str, object]] = []
        documents_by_id: dict[int, dict[str, object]] = {}
        chunks_by_id: dict[int, list[str]] = {}
        for row in rows:
            doc_id = int(row["id"])
            if doc_id not in documents_by_id:
                chunks_by_id[doc_id] = []
                document: dict[str, object] = {
                    "id": doc_id,
                    "source_title": str(row["source_title"]),
                    "source_ref": str(row["source_ref"]),
                    "ingested_at": str(row["ingested_at"]),
                    "chunks": chunks_by_id[doc_id],
                }
                documents_by_id[doc_id] = document
                documents.append(document)

            if row["chunk_id"] is not None:
                chunks_by_id[doc_id].append(str(row["chunk_text"]))

        return {"documents": documents}
