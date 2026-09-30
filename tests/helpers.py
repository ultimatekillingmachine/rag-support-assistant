"""Shared test helpers (kept out of conftest so tests can import them directly)."""

from __future__ import annotations

import zlib

import numpy as np

from app.vectorstore import StoredChunk


class FakeEmbedder:
    """Deterministic bag-of-words embedder: no model downloads in tests.

    Not semantically strong, but enough to assert that *retrieval wiring*
    works: queries sharing tokens with a document rank it first.
    """

    dim = 64

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed_query(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in str(text).lower().split():
            vector[zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector)) or 1.0
        return vector / norm


def stored_chunk(
    slug: str,
    title: str,
    text: str,
    audience: str = "customer",
    index: int = 0,
) -> StoredChunk:
    return StoredChunk(
        slug=slug,
        title=title,
        category="test",
        audience=audience,
        updated_at="2026-01-01",
        chunk_index=index,
        text=text,
    )


class FakeReranker:
    """Deterministic stand-in for a cross-encoder: scores by keyword presence.

    Tests need *some* model that reorders candidates predictably, without
    downloading ~1 GB of weights. ``positive``/``negative`` control the score
    range so gate behaviour can be exercised too.
    """

    name = "fake"

    def __init__(self, keyword: str, positive: float = 0.9, negative: float = 0.1) -> None:
        self.keyword = keyword
        self.positive = positive
        self.negative = negative

    def score(self, query: str, documents: list[str]) -> list[float]:
        needle = self.keyword.lower()
        return [
            self.positive if needle in document.lower() else self.negative
            for document in documents
        ]
