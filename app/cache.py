"""Response cache.

Why a cache at all: for a support assistant the same questions repeat ("сколько
идёт доставка?"), and every answer costs an LLM call — money and hundreds of
milliseconds. Caching identical questions removes both costs.

Correctness rules, which is where caches usually go wrong:

* the key includes the **audience** — otherwise an operator's answer (built from
  internal articles) could be served to a customer;
* the cache stores the **search hits**, not just the text, so the API can still
  return sources, latency breakdown and the refusal flag;
* entries expire after ``ttl_seconds`` (the knowledge base changes);
* the cache is bounded (``max_entries``) and evicts the oldest entry first, so
  an attacker cannot exhaust memory by sending unique questions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.rag import AnswerResult
from app.vectorstore import SearchHit


@dataclass(frozen=True)
class CacheEntry:
    result: AnswerResult
    created_at: float


class ResponseCache:
    """Small in-memory cache with a TTL and a size limit.

    Deliberately process-local: a single FastAPI process is the target here.
    For several workers/replicas the same interface would be backed by Redis —
    the pipeline depends on the interface, not on the implementation.
    """

    def __init__(self, max_entries: int = 256, ttl_seconds: float = 900.0) -> None:
        self._max_entries = max(1, max_entries)
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(question: str, top_k: int, audience: str | None) -> str:
        """Normalised key: case/whitespace-insensitive, audience-scoped."""
        normalised = " ".join(question.lower().split())
        return f"{audience or '*'}|{top_k}|{normalised}"

    def get(self, key: str) -> AnswerResult | None:
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        if self._ttl_seconds and (time.monotonic() - entry.created_at) > self._ttl_seconds:
            del self._entries[key]
            self._misses += 1
            return None
        self._hits += 1
        return entry.result

    def put(self, key: str, result: AnswerResult) -> None:
        if key in self._entries:
            del self._entries[key]  # keep insertion order = recency order
        self._entries[key] = CacheEntry(result=result, created_at=time.monotonic())
        while len(self._entries) > self._max_entries:
            oldest = next(iter(self._entries))
            del self._entries[oldest]

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> dict[str, float | int]:
        total = self._hits + self._misses
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
        }


class CachedPipeline:
    """Wraps a pipeline with response caching (same interface as RAGPipeline)."""

    def __init__(self, pipeline, cache: ResponseCache, default_top_k: int = 5) -> None:
        self._pipeline = pipeline
        self._cache = cache
        self._default_top_k = default_top_k

    @property
    def cache(self) -> ResponseCache:
        return self._cache

    async def answer(
        self,
        question: str,
        top_k: int | None = None,
        audience: str | None = "customer",
    ) -> AnswerResult:
        effective_top_k = top_k or self._default_top_k
        key = self._cache.make_key(question, effective_top_k, audience)
        cached = self._cache.get(key)
        if cached is not None:
            # Report near-zero latency so callers can see the cache worked.
            return AnswerResult(
                answer=cached.answer,
                hits=cached.hits,
                latency_ms=0,
                refused=cached.refused,
            )
        result = await self._pipeline.answer(question, top_k=top_k, audience=audience)
        self._cache.put(key, result)
        return result


def hits_to_texts(hits: list[SearchHit]) -> list[str]:
    """Helper kept for potential future Redis serialisation of cached hits."""
    return [hit.chunk.text for hit in hits]
