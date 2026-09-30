"""Optional second-stage ranking (cross-encoder reranking).

Why a second stage at all: the first stage compares the question and each
document *separately* (so it can search thousands of pieces quickly). A
cross-encoder reads the question and one document **together** and judges how
well they actually match. It is far slower, so it is only applied to the small
list of candidates the first stage produced — a classic "recall then precision"
pipeline: the cheap search finds candidates, the expensive model orders them.

Interface mirrors :class:`app.embeddings.Embedder` so tests can inject a fake
and CI never downloads model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.vectorstore import SearchHit


class Reranker(Protocol):
    """Minimal interface required by the RAG pipeline."""

    name: str

    def score(self, query: str, documents: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class RerankResult:
    hits: list[SearchHit]
    top_score: float | None


def rerank_hits(
    hits: list[SearchHit],
    scores: list[float],
    top_k: int,
    max_chunks_per_slug: int = 0,
) -> RerankResult:
    """Reorder hits by cross-encoder score (highest first) and cut to ``top_k``.

    The original retrieval score is preserved in ``SearchHit.score`` only when
    no reranker is used; after reranking the score field carries the model
    judgement, which is what the relevance gate should look at.
    """
    if len(scores) != len(hits):
        raise ValueError(f"expected {len(hits)} scores, got {len(scores)}")
    ordered = sorted(
        zip(hits, scores, strict=True),
        key=lambda pair: -pair[1],
    )
    reordered = [SearchHit(chunk=hit.chunk, score=float(score)) for hit, score in ordered]

    if max_chunks_per_slug > 0:
        counts: dict[str, int] = {}
        kept: list[SearchHit] = []
        for hit in reordered:
            slug = hit.chunk.slug
            if counts.get(slug, 0) >= max_chunks_per_slug:
                continue
            counts[slug] = counts.get(slug, 0) + 1
            kept.append(hit)
        reordered = kept

    top = reordered[:top_k]
    return RerankResult(hits=top, top_score=top[0].score if top else None)


class CrossEncoderReranker:
    """Local ONNX cross-encoder via fastembed (multilingual by default)."""

    name = "cross_encoder"

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazy: heavy import

        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)
        self._batch_size = batch_size

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        return [
            float(value)
            for value in self._model.rerank(query, documents, batch_size=self._batch_size)
        ]
