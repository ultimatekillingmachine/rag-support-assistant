"""CLI: ``python -m scripts.ingest_kb`` — load KB, chunk, embed, store.

Idempotent: performs a full re-index (reset + insert). Incremental re-index
arrives in Phase 3.
"""

from __future__ import annotations

import asyncio
import time

from app.chunking import Chunk, chunk_article
from app.config import settings
from app.embeddings import FastEmbedEmbedder
from app.kb import load_kb
from app.vectorstore import StoredChunk, create_store


def build_embed_pairs(articles) -> list[tuple[str, Chunk]]:
    """Compose (embedding_text, chunk) pairs.

    The embedding input is prefixed with the article title: chunk-level
    retrieval then matches against both the local section and the global topic.
    """
    pairs: list[tuple[str, Chunk]] = []
    for article in articles:
        chunks = chunk_article(
            article,
            max_chars=settings.chunk_max_chars,
            overlap=settings.chunk_overlap_chars,
        )
        for chunk in chunks:
            pairs.append((f"{article.title}. {chunk.text}", chunk))
    return pairs


def to_stored(chunk: Chunk) -> StoredChunk:
    return StoredChunk(
        slug=chunk.slug,
        title=chunk.title,
        category=chunk.category,
        audience=chunk.audience,
        updated_at=chunk.updated_at,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
    )


async def main() -> None:
    started = time.perf_counter()

    articles = load_kb(settings.kb_dir)
    pairs = build_embed_pairs(articles)
    print(f"loaded {len(articles)} articles -> {len(pairs)} chunks")

    embedder = FastEmbedEmbedder(settings.embedding_model)
    print(f"embedding model: {settings.embedding_model} (dim={embedder.dim})")
    embeddings = embedder.embed_documents([text for text, _ in pairs])

    store = create_store(settings)
    await store.init()
    await store.reset()
    inserted = await store.upsert([to_stored(chunk) for _, chunk in pairs], embeddings)
    total = await store.count()
    await store.dispose()

    elapsed = time.perf_counter() - started
    print(f"inserted {inserted} chunks (store total: {total}) in {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
