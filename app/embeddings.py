"""Embedding providers.

The public interface is the :class:`Embedder` protocol so the pipeline can be
tested with lightweight fakes. The default implementation uses fastembed
(ONNX runtime) which needs neither GPU nor API keys and runs fully offline.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Minimal interface required by the RAG pipeline."""

    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _needs_e5_prefix(model_name: str) -> bool:
    """Multilingual E5 models expect 'query:' / 'passage:' prefixes."""
    return "e5" in model_name.lower()


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.atleast_2d(vectors).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class FastEmbedEmbedder:
    """Local ONNX embeddings via fastembed."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # lazy import: heavy dependency

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self._e5_prefix = _needs_e5_prefix(model_name)
        probe = self._model.embed([self._prepare_document("dimension probe")])
        self.dim = len(next(iter(probe)))

    def _prepare_document(self, text: str) -> str:
        return f"passage: {text}" if self._e5_prefix else text

    def _prepare_query(self, text: str) -> str:
        return f"query: {text}" if self._e5_prefix else text

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prepared = [self._prepare_document(text) for text in texts]
        vectors = np.array(list(self._model.embed(prepared)), dtype=np.float32)
        return _normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.array(list(self._model.embed([self._prepare_query(text)])), dtype=np.float32)
        return _normalize(vector)[0]
