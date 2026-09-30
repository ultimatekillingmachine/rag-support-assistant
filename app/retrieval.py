"""Retrieval strategies.

* :class:`VectorRetriever` — the Phase 1 baseline (pure embedding similarity);
* :class:`HybridRetriever` — BM25 + vector candidates fused with Reciprocal
  Rank Fusion (Cormack et al., 2009), plus an optional relevance gate that
  refuses to answer when even the best cosine score is too low.

RRF is used instead of weighted score sums because cosine similarities and
BM25 scores live on different, uncalibrated scales: RRF only needs *ranks*,
which makes fusion robust and hyperparameter-light (single constant ``k``).
"""

from __future__ import annotations

from typing import Protocol

from app.embeddings import Embedder
from app.lexical import BM25, tokenize
from app.vectorstore import SearchHit, VectorStore

ChunkKey = tuple[str, int]  # (slug, chunk_index)


class Retriever(Protocol):
    """What the RAG pipeline needs from a retrieval stage."""

    name: str

    async def retrieve(
        self, query: str, top_k: int, audience: str | None = None
    ) -> list[SearchHit]: ...


def relevance_gate_passed(best_vector_score: float | None, threshold: float) -> bool:
    """Whether retrieval results are confident enough to be shown to a user.

    Absolute cosine thresholds are model-specific and must be tuned on the
    labelled eval set (``scripts/run_eval`` prints false-refusal statistics).
    ``threshold <= 0`` disables the gate.
    """
    if threshold <= 0:
        return True
    return best_vector_score is not None and best_vector_score >= threshold


def rrf_fuse(
    rankings: list[list[ChunkKey]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[ChunkKey, float]]:
    """Fuse ranked lists of chunk keys with (optionally weighted) RRF.

    Weights let a stronger retriever contribute more than a weaker one while
    keeping fusion purely rank-based (no score calibration needed).
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[ChunkKey, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])


def limit_per_slug(hits: list[SearchHit], max_per_slug: int) -> list[SearchHit]:
    """Keep retrieval diverse: at most ``max_per_slug`` chunks per article.

    Without this, long articles flood the context window with near-duplicate
    chunks and crowd out the actually relevant document (observed in eval:
    4/5 slots taken by one article). ``max_per_slug <= 0`` disables the cap.
    """
    if max_per_slug <= 0:
        return hits
    counts: dict[str, int] = {}
    kept: list[SearchHit] = []
    for hit in hits:
        slug = hit.chunk.slug
        if counts.get(slug, 0) >= max_per_slug:
            continue
        counts[slug] = counts.get(slug, 0) + 1
        kept.append(hit)
    return kept


class VectorRetriever:
    """Pure vector similarity search (baseline)."""

    name = "vector"

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        candidate_pool: int = 20,
        min_relevance_score: float = 0.0,
        max_chunks_per_slug: int = 1,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._candidate_pool = candidate_pool
        self._min_relevance_score = min_relevance_score
        self._max_chunks_per_slug = max_chunks_per_slug

    async def retrieve(
        self, query: str, top_k: int, audience: str | None = None
    ) -> list[SearchHit]:
        query_vector = self._embedder.embed_query(query)
        pool = max(top_k, self._candidate_pool)
        hits = await self._store.search(query_vector, pool, audience=audience)
        best_score = hits[0].score if hits else None
        if not relevance_gate_passed(best_score, self._min_relevance_score):
            return []
        return limit_per_slug(hits, self._max_chunks_per_slug)[:top_k]


class HybridRetriever:
    """BM25 + vector search fused with RRF, with an optional relevance gate."""

    name = "hybrid"

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        rrf_k: int = 60,
        candidate_pool: int = 20,
        min_relevance_score: float = 0.0,
        vector_weight: float = 1.0,
        lexical_weight: float = 0.5,
        max_chunks_per_slug: int = 1,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._rrf_k = rrf_k
        self._candidate_pool = candidate_pool
        self._min_relevance_score = min_relevance_score
        self._vector_weight = vector_weight
        self._lexical_weight = lexical_weight
        self._max_chunks_per_slug = max_chunks_per_slug

    async def retrieve(
        self, query: str, top_k: int, audience: str | None = None
    ) -> list[SearchHit]:
        query_vector = self._embedder.embed_query(query)
        vector_hits = await self._store.search(
            query_vector, self._candidate_pool, audience=audience
        )

        best_vector_score = vector_hits[0].score if vector_hits else None
        if not relevance_gate_passed(best_vector_score, self._min_relevance_score):
            return []

        chunks = await self._store.list_chunks(audience=audience)
        if not chunks:
            return []

        # Lexical candidates over the same audience-filtered corpus.
        corpus = [tokenize(f"{chunk.title} {chunk.text}") for chunk in chunks]
        lexical_scores = BM25(corpus).scores(tokenize(query))
        lexical_order = sorted(
            range(len(chunks)), key=lambda index: -lexical_scores[index]
        )[: self._candidate_pool]
        lexical_keys: list[ChunkKey] = [
            (chunks[index].slug, chunks[index].chunk_index)
            for index in lexical_order
            if lexical_scores[index] > 0
        ]

        vector_keys: list[ChunkKey] = [
            (hit.chunk.slug, hit.chunk.chunk_index) for hit in vector_hits
        ]
        fused = rrf_fuse(
            [vector_keys, lexical_keys],
            k=self._rrf_k,
            weights=[self._vector_weight, self._lexical_weight],
        )

        by_key = {(chunk.slug, chunk.chunk_index): chunk for chunk in chunks}
        results: list[SearchHit] = []
        for key, fused_score in fused[: self._candidate_pool]:
            chunk = by_key.get(key)
            if chunk is not None:
                results.append(SearchHit(chunk=chunk, score=round(fused_score, 6)))
        return limit_per_slug(results, self._max_chunks_per_slug)[:top_k]


def create_retriever(store: VectorStore, embedder: Embedder, config) -> Retriever:
    """Build the configured retriever (see Settings for the knobs)."""
    if config.hybrid_enabled:
        return HybridRetriever(
            store,
            embedder,
            rrf_k=config.rrf_k,
            candidate_pool=config.candidate_pool,
            min_relevance_score=config.min_relevance_score,
            vector_weight=config.vector_weight,
            lexical_weight=config.lexical_weight,
            max_chunks_per_slug=config.max_chunks_per_slug,
        )
    return VectorRetriever(
        store,
        embedder,
        candidate_pool=config.candidate_pool,
        min_relevance_score=config.min_relevance_score,
        max_chunks_per_slug=config.max_chunks_per_slug,
    )
