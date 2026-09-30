"""Tests for rerank helpers and reranker integration (with a fake reranker)."""

from __future__ import annotations

import pytest

from app.rerank import rerank_hits
from app.retrieval import VectorRetriever
from app.vectorstore import SearchHit
from tests.helpers import FakeReranker, stored_chunk


def _hits() -> list[SearchHit]:
    return [
        SearchHit(chunk=stored_chunk("a", "A", "alpha text", index=0), score=0.9),
        SearchHit(chunk=stored_chunk("b", "B", "beta text", index=0), score=0.8),
        SearchHit(chunk=stored_chunk("c", "C", "gamma text", index=0), score=0.7),
    ]


def test_rerank_hits_reorders_by_model_score() -> None:
    result = rerank_hits(_hits(), scores=[0.1, 0.9, 0.5], top_k=3)
    assert [hit.chunk.slug for hit in result.hits] == ["b", "c", "a"]
    assert result.top_score == 0.9


def test_rerank_hits_respects_top_k() -> None:
    result = rerank_hits(_hits(), scores=[0.1, 0.9, 0.5], top_k=2)
    assert [hit.chunk.slug for hit in result.hits] == ["b", "c"]


def test_rerank_hits_applies_diversity_cap() -> None:
    hits = [
        SearchHit(chunk=stored_chunk("a", "A", "x", index=0), score=0.9),
        SearchHit(chunk=stored_chunk("a", "A", "x", index=1), score=0.85),
        SearchHit(chunk=stored_chunk("b", "B", "y", index=0), score=0.8),
    ]
    result = rerank_hits(hits, scores=[0.9, 0.8, 0.7], top_k=3, max_chunks_per_slug=1)
    assert [hit.chunk.slug for hit in result.hits] == ["a", "b"]


def test_rerank_hits_rejects_mismatched_scores() -> None:
    with pytest.raises(ValueError, match="expected 3 scores"):
        rerank_hits(_hits(), scores=[0.5], top_k=3)


async def test_vector_retriever_uses_reranker_order(populated_store) -> None:
    store, embedder = populated_store
    # The fixture's article body says "Вернуть товар ...", so the keyword must
    # appear there for the return article to win.
    reranker = FakeReranker(keyword="вернуть")
    retriever = VectorRetriever(store, embedder, candidate_pool=5, reranker=reranker)

    hits = await retriever.retrieve("как вернуть покупку", top_k=3, audience="customer")

    assert hits[0].chunk.slug == "return-policy"
    assert hits[0].score == 0.9  # score comes from the reranker, not the vector


async def test_vector_retriever_with_reranker_gate_refuses(populated_store) -> None:
    store, embedder = populated_store
    retriever = VectorRetriever(
        store,
        embedder,
        candidate_pool=3,
        min_relevance_score=0.99,
        reranker=FakeReranker(keyword="доставка", positive=0.1),
    )
    assert await retriever.retrieve("доставка в москву", top_k=3) == []
