"""Versioned local multilingual embedding boundary for graph/research retrieval.

Production uses a quantized ONNX multilingual-e5-small model through FastEmbed.
Text is processed locally; only model artifacts may be downloaded into the
owner-controlled cache on first use. Tests inject a deterministic backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "fd1525a9fd15316a2d503bf26ab031a61d056e98"
MODEL_DIMENSION = 384
MODEL_FILE = "onnx/model_O4.onnx"


@dataclass(frozen=True, slots=True)
class EmbeddingMetadata:
    """Stable identity stored with every index generation."""

    name: str
    revision: str
    dimension: int


class EmbeddingBackend(Protocol):
    """Local dense embedding port."""

    @property
    def metadata(self) -> EmbeddingMetadata:
        """Return model identity and vector dimension."""
        ...

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        """Embed passage text locally."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one retrieval query locally."""
        ...


class FastEmbedBackend:
    """Lazy CPU ONNX backend with owner-local versioned cache."""

    def __init__(self, data_dir: Path | None = None) -> None:
        root = data_dir or Path(os.environ.get("THERAPY_DATA_DIR", "./data"))
        self.cache_dir = root / "models" / "embeddings" / MODEL_REVISION
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: object | None = None

    @property
    def metadata(self) -> EmbeddingMetadata:
        return EmbeddingMetadata(MODEL_NAME, MODEL_REVISION, MODEL_DIMENSION)

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType
        except ImportError as error:
            raise RuntimeError(
                "fastembed is required for local semantic retrieval; install project dependencies"
            ) from error
        supported = {str(item["model"]) for item in TextEmbedding.list_supported_models()}
        if MODEL_NAME not in supported:
            TextEmbedding.add_custom_model(
                model=MODEL_NAME,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=MODEL_NAME),
                dim=MODEL_DIMENSION,
                model_file=MODEL_FILE,
            )
        offline = os.environ.get("THERAPY_EMBEDDINGS_OFFLINE", "0").casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._model = TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(self.cache_dir),
            local_files_only=offline,
            revision=MODEL_REVISION,
            threads=max(1, min(os.cpu_count() or 1, 4)),
        )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        """Embed passages with E5-required task prefix."""
        if not texts:
            return []
        model = self._load()
        vectors = list(model.embed([f"passage: {text}" for text in texts]))
        return [self._validated(vector) for vector in vectors]

    def embed_query(self, text: str) -> np.ndarray:
        """Embed query with E5-required task prefix."""
        if not text.strip():
            raise ValueError("query text must not be empty")
        model = self._load()
        vector = next(iter(model.embed([f"query: {text}"])))
        return self._validated(vector)

    def _validated(self, value: object) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (MODEL_DIMENSION,) or not np.isfinite(vector).all():
            raise RuntimeError(
                f"embedding model returned invalid shape/data: {vector.shape}"
            )
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise RuntimeError("embedding model returned zero vector")
        return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity for normalized or unnormalized finite vectors."""
    if left.shape != right.shape:
        raise ValueError("embedding dimensions differ")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


__all__ = [
    "EmbeddingBackend",
    "EmbeddingMetadata",
    "FastEmbedBackend",
    "MODEL_DIMENSION",
    "MODEL_NAME",
    "MODEL_REVISION",
    "cosine_similarity",
]
