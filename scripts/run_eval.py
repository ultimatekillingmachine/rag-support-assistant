"""CLI: ``python -m scripts.run_eval`` — retrieval evaluation harness.

Compares retrieval modes on the labelled eval set:

* ``vector`` — Phase 1 baseline (pure embedding similarity);
* ``hybrid`` — BM25 + vector fused with RRF, gate disabled;
* ``hybrid+gate`` — same, with the relevance gate active.

Reported per mode: hit@1/3/5 and MRR over **answerable** questions, plus
relevance-gate statistics over **out-of-scope** questions (``expected_slugs``
empty): how many were correctly refused and how many answerable questions were
over-refused. Deliberately LLM-free: cheap, local, deterministic — CI-safe.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.embeddings import FastEmbedEmbedder
from app.rerank import CrossEncoderReranker, Reranker
from app.retrieval import HybridRetriever, Retriever, VectorRetriever
from app.vectorstore import create_store

EVAL_PATH = Path("data/eval/questions.jsonl")


def load_questions(path: Path = EVAL_PATH) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


def first_hit_rank(expected: list[str], retrieved: list[str]) -> int | None:
    for rank, slug in enumerate(retrieved, start=1):
        if slug in expected:
            return rank
    return None


@dataclass
class ModeStats:
    name: str
    answerable: int = 0
    hit1: int = 0
    hit3: int = 0
    hit5: int = 0
    rr_sum: float = 0.0
    refused_answerable: int = 0
    out_of_scope: int = 0
    refused_out_of_scope: int = 0
    latency_ms_sum: float = 0.0
    total: int = 0
    misses: list[str] = field(default_factory=list)

    @property
    def hit1_rate(self) -> float:
        return self.hit1 / self.answerable

    @property
    def hit3_rate(self) -> float:
        return self.hit3 / self.answerable

    @property
    def hit5_rate(self) -> float:
        return self.hit5 / self.answerable

    @property
    def mrr(self) -> float:
        return self.rr_sum / self.answerable

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_sum / self.total


async def evaluate_mode(
    retriever: Retriever, questions: list[dict], top_k: int
) -> ModeStats:
    stats = ModeStats(name=retriever.name)
    for item in questions:
        expected: list[str] = item["expected_slugs"]
        started = time.perf_counter()
        hits = await retriever.retrieve(
            item["question"],
            top_k,
            audience=None,
        )
        stats.latency_ms_sum += (time.perf_counter() - started) * 1000
        stats.total += 1

        if expected:
            stats.answerable += 1
            if not hits:
                stats.refused_answerable += 1
                stats.misses.append(f"{item['id']} REFUSED: {item['question'][:60]}")
                continue
            retrieved = [hit.chunk.slug for hit in hits]
            rank = first_hit_rank(expected, retrieved)
            if rank == 1:
                stats.hit1 += 1
            if rank is not None and rank <= 3:
                stats.hit3 += 1
            if rank is not None and rank <= 5:
                stats.hit5 += 1
            if rank is not None:
                stats.rr_sum += 1.0 / rank
            else:
                stats.misses.append(
                    f"{item['id']} rank>5: {item['question'][:60]} -> {retrieved}"
                )
        else:
            stats.out_of_scope += 1
            if not hits:
                stats.refused_out_of_scope += 1
    return stats


async def run() -> None:
    store = create_store(settings)
    await store.init()
    embedder = FastEmbedEmbedder(settings.embedding_model)
    questions = load_questions()
    answerable = sum(1 for item in questions if item["expected_slugs"])

    print(f"embedding model: {settings.embedding_model}")
    print(
        f"top_k: {settings.top_k} | questions: {len(questions)} "
        f"({answerable} answerable, {len(questions) - answerable} out-of-scope)"
    )

    reranker: Reranker | None = None
    if settings.rerank_enabled:
        reranker = CrossEncoderReranker(
            model_name=settings.rerank_model,
            batch_size=settings.rerank_batch_size,
        )

    # The evaluation measures retrieval quality per labelled question; the
    # audience filter is a security concern covered by tests/test_auth.py, so
    # the harness retrieves over the whole corpus (audience=None = no filter).
    modes: list[tuple[str, Retriever]] = [
        (
            "vector",
            VectorRetriever(
                store,
                embedder,
                candidate_pool=settings.candidate_pool,
                max_chunks_per_slug=settings.max_chunks_per_slug,
            ),
        ),
        (
            f"vector+gate({settings.min_relevance_score:g})",
            VectorRetriever(
                store,
                embedder,
                candidate_pool=settings.candidate_pool,
                min_relevance_score=settings.min_relevance_score,
                max_chunks_per_slug=settings.max_chunks_per_slug,
            ),
        ),
        (
            "hybrid",
            HybridRetriever(
                store,
                embedder,
                rrf_k=settings.rrf_k,
                candidate_pool=settings.candidate_pool,
                min_relevance_score=0.0,
                vector_weight=settings.vector_weight,
                lexical_weight=settings.lexical_weight,
                max_chunks_per_slug=settings.max_chunks_per_slug,
            ),
        ),
        (
            f"hybrid+gate({settings.min_relevance_score:g})",
            HybridRetriever(
                store,
                embedder,
                rrf_k=settings.rrf_k,
                candidate_pool=settings.candidate_pool,
                min_relevance_score=settings.min_relevance_score,
                vector_weight=settings.vector_weight,
                lexical_weight=settings.lexical_weight,
                max_chunks_per_slug=settings.max_chunks_per_slug,
            ),
        ),
    ]

    if reranker is not None:
        modes.append(
            (
                "vector+rerank",
                VectorRetriever(
                    store,
                    embedder,
                    candidate_pool=settings.candidate_pool,
                    max_chunks_per_slug=settings.max_chunks_per_slug,
                    reranker=reranker,
                ),
            )
        )
        modes.append(
            (
                "vector+rerank+gate",
                VectorRetriever(
                    store,
                    embedder,
                    candidate_pool=settings.candidate_pool,
                    min_relevance_score=settings.min_relevance_score,
                    max_chunks_per_slug=settings.max_chunks_per_slug,
                    reranker=reranker,
                ),
            )
        )

    header = (
        f"{'mode':<16} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6} "
        f"{'refused(out)':>13} {'over-refused(in)':>17} {'avg ms':>7}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for label, retriever in modes:
        stats = await evaluate_mode(retriever, questions, settings.top_k)
        print(
            f"{label:<16} {stats.hit1_rate:>6.3f} {stats.hit3_rate:>6.3f} "
            f"{stats.hit5_rate:>6.3f} {stats.mrr:>6.3f} "
            f"{stats.refused_out_of_scope}/{stats.out_of_scope:>11} "
            f"{stats.refused_answerable}/{stats.answerable:>15} "
            f"{stats.avg_latency_ms:>7.1f}"
        )
        for miss in stats.misses[:5]:
            print(f"    ! {miss}")

    await store.dispose()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()

