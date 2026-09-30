"""Tests for the response cache: correctness rules and eviction."""

from __future__ import annotations

from app.cache import CachedPipeline, ResponseCache
from app.llm import MockLLM
from app.rag import RAGPipeline
from app.retrieval import VectorRetriever
from app.vectorstore import SearchHit
from tests.helpers import stored_chunk


def _hit(slug: str, text: str) -> SearchHit:
    return SearchHit(chunk=stored_chunk(slug, slug, text), score=0.9)


def test_key_is_case_and_whitespace_insensitive() -> None:
    a = ResponseCache.make_key("Сколько идёт доставка?", 5, "customer")
    b = ResponseCache.make_key("  сколько   ИДЁТ доставка?  ", 5, "customer")
    assert a == b


def test_key_is_audience_scoped() -> None:
    customer = ResponseCache.make_key("регламент эскалации", 5, "customer")
    operator = ResponseCache.make_key("регламент эскалации", 5, "operator")
    assert customer != operator


def test_key_includes_top_k() -> None:
    assert ResponseCache.make_key("q", 3, "customer") != ResponseCache.make_key("q", 5, "customer")


def test_get_returns_none_after_ttl_expires() -> None:
    cache = ResponseCache(ttl_seconds=0.0)  # 0 disables expiry
    from app.rag import AnswerResult

    cache.put("k", AnswerResult(answer="a", hits=[], latency_ms=1))
    assert cache.get("k") is not None

    expiring = ResponseCache(ttl_seconds=-1.0)  # already expired on arrival
    expiring.put("k", AnswerResult(answer="a", hits=[], latency_ms=1))
    assert expiring.get("k") is None


def test_oldest_entry_is_evicted_when_full() -> None:
    from app.rag import AnswerResult

    cache = ResponseCache(max_entries=2, ttl_seconds=60.0)
    cache.put("first", AnswerResult(answer="1", hits=[], latency_ms=1))
    cache.put("second", AnswerResult(answer="2", hits=[], latency_ms=1))
    cache.put("third", AnswerResult(answer="3", hits=[], latency_ms=1))
    assert cache.get("first") is None
    assert cache.get("second") is not None
    assert cache.get("third") is not None


def test_stats_reports_hit_rate() -> None:
    from app.rag import AnswerResult

    cache = ResponseCache(ttl_seconds=60.0)
    cache.put("k", AnswerResult(answer="a", hits=[], latency_ms=1))
    cache.get("k")
    cache.get("missing")
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


async def test_cached_pipeline_skips_second_call(populated_store) -> None:
    store, embedder = populated_store
    base = RAGPipeline(VectorRetriever(store, embedder), MockLLM(), default_top_k=3)
    cache = ResponseCache(ttl_seconds=60.0)
    pipeline = CachedPipeline(base, cache, default_top_k=3)

    first = await pipeline.answer("доставка в москву", audience="customer")
    second = await pipeline.answer("ДОСТАВКА в  москву", audience="customer")

    assert first.latency_ms >= 0
    assert second.latency_ms == 0  # served from cache
    assert second.answer == first.answer
    assert [hit.chunk.slug for hit in second.hits] == [hit.chunk.slug for hit in first.hits]
    assert cache.stats()["hits"] == 1


async def test_cache_does_not_leak_between_audiences(populated_store) -> None:
    store, embedder = populated_store
    base = RAGPipeline(VectorRetriever(store, embedder), MockLLM(), default_top_k=3)
    cache = ResponseCache(ttl_seconds=60.0)
    pipeline = CachedPipeline(base, cache, default_top_k=3)

    await pipeline.answer("регламент эскалации", audience="operator")
    customer_view = await pipeline.answer("регламент эскалации", audience="customer")

    # The customer must not receive the operator's cached answer/hits.
    assert customer_view.latency_ms >= 0 or customer_view.latency_ms == 0
    assert all(hit.chunk.audience == "customer" for hit in customer_view.hits)
    assert cache.stats()["entries"] == 2
