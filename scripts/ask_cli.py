"""CLI: ``python -m scripts.ask_cli "Сколько идёт доставка в Москву?" [customer|operator]``."""

from __future__ import annotations

import asyncio
import sys

from app.config import settings
from app.embeddings import FastEmbedEmbedder
from app.llm import get_llm
from app.rag import RAGPipeline
from app.retrieval import create_retriever
from app.vectorstore import create_store


async def main(question: str, audience: str) -> None:
    store = create_store(settings)
    await store.init()
    embedder = FastEmbedEmbedder(settings.embedding_model)
    retriever = create_retriever(store, embedder, settings)
    pipeline = RAGPipeline(retriever, get_llm(settings), default_top_k=settings.top_k)

    result = await pipeline.answer(question, audience=audience)

    print("=== ANSWER ===")
    print(result.answer)
    if result.refused:
        print("\n(отказ: relevance gate не нашёл достаточно уверенного контекста)")
    print("\n=== SOURCES ===")
    for ref, hit in enumerate(result.hits, start=1):
        print(f"[{ref}] {hit.chunk.title} ({hit.chunk.slug})  score={hit.score:.3f}")
    print(f"\nretriever: {retriever.name} | latency: {result.latency_ms} ms")
    await store.dispose()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('usage: python -m scripts.ask_cli "ваш вопрос" [customer|operator]')
        raise SystemExit(2)
    audience_arg = sys.argv[2] if len(sys.argv) > 2 else "customer"
    asyncio.run(main(sys.argv[1], audience_arg))

