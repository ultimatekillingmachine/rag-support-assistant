"""CLI: ``python -m scripts.ask_cli "Сколько идёт доставка в Москву?" [customer|operator]``."""

from __future__ import annotations

import asyncio
import sys

from app.auth import ROLE_AUDIENCE_FILTER, ROLE_CUSTOMER, ROLE_OPERATOR
from app.config import settings
from app.embeddings import FastEmbedEmbedder
from app.llm import get_llm
from app.rag import RAGPipeline
from app.retrieval import create_retriever
from app.vectorstore import create_store


async def main(question: str, role: str) -> None:
    store = create_store(settings)
    await store.init()
    embedder = FastEmbedEmbedder(settings.embedding_model)
    retriever = create_retriever(store, embedder, settings)
    pipeline = RAGPipeline(retriever, get_llm(settings), default_top_k=settings.top_k)

    # Local CLI: the role is simulated by resolving the same audience filter the
    # API would derive from a bearer token.
    audience = ROLE_AUDIENCE_FILTER[role]
    result = await pipeline.answer(question, audience=audience)

    print("=== ANSWER ===")
    print(result.answer)
    if result.refused:
        print("\n(отказ: уверенность ниже порога)")
    print("\n=== SOURCES ===")
    for ref, hit in enumerate(result.hits, start=1):
        print(f"[{ref}] {hit.chunk.title} ({hit.chunk.slug})  score={hit.score:.3f}")
    print(f"\nretriever: {retriever.name} | role: {role} | latency: {result.latency_ms} ms")
    await store.dispose()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: python -m scripts.ask_cli "ваш вопрос" [customer|operator]')
        raise SystemExit(2)
    role_arg = sys.argv[2] if len(sys.argv) > 2 else ROLE_CUSTOMER
    if role_arg not in ROLE_AUDIENCE_FILTER:
        print(f"unknown role: {role_arg} (expected {ROLE_CUSTOMER} or {ROLE_OPERATOR})")
        raise SystemExit(2)
    asyncio.run(main(sys.argv[1], role_arg))

