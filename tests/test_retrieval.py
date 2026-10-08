"""Tests for retrieval strategies: RRF fusion, relevance gate, hybrid behavior."""

from __future__ import annotations

import pytest

from app.retrieval import (
    HybridRetriever,
    VectorRetriever,
    limit_per_slug,
    relevance_gate_passed,
    rrf_fuse,
)
from app.vectorstore import SearchHit
from tests.helpers import stored_chunk


def test_limit_per_slug_dedups_articles() -> None:
    hits = [
        SearchHit(chunk=stored_chunk("a", "A", "x", index=0), score=0.9),
        SearchHit(chunk=stored_chunk("a", "A", "x", index=1), score=0.8),
        SearchHit(chunk=stored_chunk("b", "B", "y", index=0), score=0.7),
    ]
    kept = limit_per_slug(hits, max_per_slug=1)
    assert [hit.chunk.slug for hit in kept] == ["a", "b"]
    assert limit_per_slug(hits, max_per_slug=0) == hits


def test_rrf_fuse_merges_rankings_by_rank_not_scale() -> None:
    fused = rrf_fuse([[("a", 0), ("b", 0)], [("b", 0), ("c", 0)]], k=60)
    scores = dict(fused)
    order = [key for key, _ in fused]
    assert order[0] == ("b", 0)  # high in both rankings
    assert scores[("b", 0)] > scores[("a", 0)] > scores[("c", 0)]


def test_rrf_weights_shift_fusion() -> None:
    # Heavy vector weight -> the vector ranking wins.
    fused = rrf_fuse([[("a", 0)], [("b", 0)]], k=60, weights=[1.0, 0.1])
    assert fused[0][0] == ("a", 0)
    # Overwhelming lexical weight -> the BM25 ranking wins.
    fused = rrf_fuse([[("a", 0)], [("b", 0)]], k=60, weights=[0.1, 1.0])
    assert fused[0][0] == ("b", 0)


def test_relevance_gate_passed_edges() -> None:
    assert relevance_gate_passed(None, 0.8) is False
    assert relevance_gate_passed(0.79, 0.8) is False
    assert relevance_gate_passed(0.85, 0.8) is True
    # threshold <= 0 disables the gate
    assert relevance_gate_passed(None, 0.0) is True
    assert relevance_gate_passed(0.01, 0.0) is True


async def test_vector_retriever_finds_delivery_chunk(populated_store) -> None:
    store, embedder = populated_store
    hits = await VectorRetriever(store, embedder).retrieve("доставка в Москву", top_k=3)
    assert hits
    assert hits[0].chunk.slug == "delivery-time"


async def test_hybrid_retriever_respects_audience(populated_store) -> None:
    store, embedder = populated_store
    retriever = HybridRetriever(store, embedder, min_relevance_score=0.0)

    as_customer = await retriever.retrieve(
        "регламент эскалации операторов", top_k=3, audience="customer"
    )
    assert all(hit.chunk.audience == "customer" for hit in as_customer)

    as_operator = await retriever.retrieve(
        "регламент эскалации операторов", top_k=3, audience="operator"
    )
    assert any(hit.chunk.slug == "internal-escalation" for hit in as_operator)


async def test_hybrid_gate_refuses_unrelated_query(populated_store) -> None:
    store, embedder = populated_store
    retriever = HybridRetriever(store, embedder, min_relevance_score=0.99)
    hits = await retriever.retrieve("рецепт борща со сметаной", top_k=3)
    assert hits == []


async def test_hybrid_gate_allows_matching_query(populated_store) -> None:
    store, embedder = populated_store
    retriever = HybridRetriever(store, embedder, min_relevance_score=0.2)
    hits = await retriever.retrieve("доставка в Москву", top_k=3)
    assert hits
    assert hits[0].chunk.slug == "delivery-time"


async def test_hybrid_reports_cosine_scores_not_fusion_scores(populated_store) -> None:
    """The API field must mean the same thing in both retrieval modes."""
    store, embedder = populated_store
    vector_hits = await VectorRetriever(store, embedder).retrieve("доставка в Москву", top_k=3)
    hybrid_hits = await HybridRetriever(store, embedder).retrieve("доставка в Москву", top_k=3)

    assert hybrid_hits
    # Cosine similarity is in [0, 1] and clearly larger than an RRF score
    # (which is on the order of 1/60).
    assert hybrid_hits[0].score > 0.1
    assert hybrid_hits[0].score == pytest.approx(vector_hits[0].score, abs=1e-4)


async def test_hybrid_orders_by_fusion_but_keeps_known_scores(populated_store) -> None:
    store, embedder = populated_store
    hits = await HybridRetriever(store, embedder).retrieve("доставка в москву", top_k=5)
    assert all(0.0 <= hit.score <= 1.0 for hit in hits)


async def test_vector_retriever_gate_refuses_unrelated_query(populated_store) -> None:
    store, embedder = populated_store
    retriever = VectorRetriever(store, embedder, min_relevance_score=0.99)
    assert await retriever.retrieve("рецепт борща со специями", top_k=3) == []
