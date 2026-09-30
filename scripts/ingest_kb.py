"""CLI: ``python -m scripts.ingest_kb [--full]`` — build/refresh the search index.

Incremental by default (Phase 3): every article is hashed and compared with what
is already stored, so only new or changed articles are re-embedded and only
deleted articles are removed. Re-embedding the whole corpus on every deploy
wastes minutes of CPU for nothing.

Two hashes cooperate:

* **revision** — content hash of the article's chunks: changes when text changes;
* **fingerprint** — revision plus chunking and embedding settings: changes when
  the text, the chunk size or the embedding model changes.

The fingerprint is what makes re-indexing *correct*: switching the embedding
model must invalidate stored vectors even though the text is unchanged. Use
``--full`` to force a complete rebuild (also the right choice the first time).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import time
from dataclasses import dataclass

from app.chunking import Chunk, chunk_article
from app.config import Settings
from app.config import settings as default_settings
from app.embeddings import Embedder, FastEmbedEmbedder
from app.kb import KBArticle, load_kb
from app.vectorstore import StoredChunk, VectorStore, create_store


@dataclass
class IngestPlan:
    """Which articles need work and which are already up to date."""

    added: list[str]
    changed: list[str]
    unchanged: list[str]
    removed: list[str]

    @property
    def to_index(self) -> list[str]:
        return self.added + self.changed


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_hash(chunks: list[Chunk]) -> str:
    """Content hash of an article's chunks (order matters)."""
    return _hash("\n".join(chunk.text for chunk in chunks))


def fingerprint(revision: str, config: Settings) -> str:
    """Revision combined with the settings that produced the stored vectors."""
    return _hash(
        "|".join(
            [
                revision,
                config.embedding_model,
                str(config.chunk_max_chars),
                str(config.chunk_overlap_chars),
            ]
        )
    )


def build_embed_pairs(article: KBArticle, config: Settings) -> list[tuple[str, Chunk]]:
    """Compose (embedding_text, chunk) pairs for one article.

    The embedding input is prefixed with the article title: chunk-level
    retrieval then matches against both the local section and the global topic.
    """
    chunks = chunk_article(
        article,
        max_chars=config.chunk_max_chars,
        overlap=config.chunk_overlap_chars,
    )
    return [(f"{article.title}. {chunk.text}", chunk) for chunk in chunks]


def plan_ingest(
    articles: list[KBArticle],
    stored: dict[str, tuple[str, str]],
    config: Settings,
) -> tuple[IngestPlan, dict[str, str]]:
    """Compare the knowledge base on disk with what is already indexed."""
    added: list[str] = []
    changed: list[str] = []
    unchanged: list[str] = []
    revisions: dict[str, str] = {}

    for article in articles:
        pairs = build_embed_pairs(article, config)
        revision = chunk_hash([chunk for _, chunk in pairs])
        revisions[article.slug] = revision

        current = stored.get(article.slug)
        if current is None:
            added.append(article.slug)
        elif current[1] != fingerprint(revision, config):
            # Different fingerprint -> the text changed, or the chunking /
            # embedding settings changed; both require fresh vectors.
            changed.append(article.slug)
        else:
            unchanged.append(article.slug)

    removed = [slug for slug in stored if slug not in revisions]
    return IngestPlan(added, changed, unchanged, removed), revisions


def to_stored(chunk: Chunk, revision: str, fingerprint_value: str) -> StoredChunk:
    return StoredChunk(
        slug=chunk.slug,
        title=chunk.title,
        category=chunk.category,
        audience=chunk.audience,
        updated_at=chunk.updated_at,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        revision=revision,
        fingerprint=fingerprint_value,
    )


async def run_ingest(
    config: Settings | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    full: bool = False,
) -> dict[str, int]:
    """Refresh the index and return counters (also used by the tests)."""
    config = config or default_settings
    articles = load_kb(config.kb_dir)
    articles_by_slug = {article.slug: article for article in articles}

    pairs_by_slug = {
        slug: build_embed_pairs(article, config) for slug, article in articles_by_slug.items()
    }
    revisions = {
        slug: chunk_hash([chunk for _, chunk in pairs]) for slug, pairs in pairs_by_slug.items()
    }

    owns_store = store is None
    store = store or create_store(config)
    await store.init()
    try:
        stored = {} if full else await store.article_versions()
        plan, _ = plan_ingest(articles, stored, config)

        to_index = list(articles_by_slug) if full else plan.to_index
        if to_index:
            embedder = embedder or FastEmbedEmbedder(config.embedding_model)
            texts = [text for slug in to_index for text, _ in pairs_by_slug[slug]]
            chunks = [chunk for slug in to_index for _, chunk in pairs_by_slug[slug]]
            embeddings = embedder.embed_documents(texts)

            # Replace whole articles: a partial update could leave stale chunks.
            await store.delete_slugs(to_index)
            await store.upsert(
                [
                    to_stored(
                        chunk,
                        revisions[chunk.slug],
                        fingerprint(revisions[chunk.slug], config),
                    )
                    for chunk in chunks
                ],
                embeddings,
            )

        removed_count = 0 if full else await store.delete_slugs(plan.removed)

        return {
            "articles": len(articles),
            "added": len(plan.added) if not full else len(articles),
            "changed": len(plan.changed) if not full else 0,
            "unchanged": len(plan.unchanged) if not full else 0,
            "removed": removed_count,
            "reembedded_articles": len(to_index),
            "chunks": await store.count(),
        }
    finally:
        if owns_store:
            await store.dispose()


async def main(full: bool) -> None:
    started = time.perf_counter()
    counters = await run_ingest(full=full)
    elapsed = time.perf_counter() - started

    print(f"mode: {'full rebuild' if full else 'incremental'}")
    print(
        f"articles: {counters['articles']} (+{counters['added']} / "
        f"~{counters['changed']} / ={counters['unchanged']}, -{counters['removed']})"
    )
    print(
        f"re-embedded articles: {counters['reembedded_articles']} | "
        f"chunks in store: {counters['chunks']} | took {elapsed:.1f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or refresh the RAG search index.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="rebuild every article from scratch (default: only new/changed ones)",
    )
    arguments = parser.parse_args()
    asyncio.run(main(full=arguments.full))
