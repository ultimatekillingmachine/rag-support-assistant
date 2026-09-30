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
