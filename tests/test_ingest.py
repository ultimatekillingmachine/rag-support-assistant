"""Tests for incremental ingestion: change detection and re-embedding scope."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.kb import load_kb
from app.vectorstore import SqliteVectorStore
from scripts.ingest_kb import (
    build_embed_pairs,
    chunk_hash,
    fingerprint,
    plan_ingest,
    run_ingest,
)
from tests.helpers import FakeEmbedder

ARTICLE = """---
slug: demo-{n}
title: Статья {n}
category: demo
audience: customer
updated_at: 2026-01-01
---

## Раздел

Текст статьи номер {n} про доставку и возврат.
"""


def _write_article(root: Path, number: int, body: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    text = body if body is not None else ARTICLE.format(n=number)
    (root / f"article-{number}.md").write_text(text, encoding="utf-8")


def _config(kb_dir: Path, **overrides) -> Settings:
    values = {"kb_dir": kb_dir, "embedding_model": "fake-model"}
    values.update(overrides)
    return Settings(**values)


def test_fingerprint_changes_with_embedding_model() -> None:
    base = _config(Path("kb"))
    other = _config(Path("kb"), embedding_model="another-model")
    assert fingerprint("rev", base) != fingerprint("rev", other)


def test_fingerprint_changes_with_chunk_settings() -> None:
    base = _config(Path("kb"))
    other = _config(Path("kb"), chunk_max_chars=999)
    assert fingerprint("rev", base) != fingerprint("rev", other)


def test_fingerprint_is_stable_for_same_inputs() -> None:
    config = _config(Path("kb"))
    assert fingerprint("rev", config) == fingerprint("rev", config)


def test_chunk_hash_detects_text_changes(tmp_path: Path) -> None:
    _write_article(tmp_path, 1)
    config = _config(tmp_path)
    article = load_kb(tmp_path)[0]

    original = chunk_hash([chunk for _, chunk in build_embed_pairs(article, config)])
    updated = chunk_hash(
        [chunk for _, chunk in build_embed_pairs(article, config)]
    )
    assert original == updated

    _write_article(tmp_path, 1, ARTICLE.format(n=1) + "\nДополнительный абзац.")
    changed_article = load_kb(tmp_path)[0]
    changed = chunk_hash([chunk for _, chunk in build_embed_pairs(changed_article, config)])
    assert changed != original


def test_plan_classifies_added_changed_unchanged_removed(tmp_path: Path) -> None:
    _write_article(tmp_path, 1)
    _write_article(tmp_path, 2)
    config = _config(tmp_path)
    articles = load_kb(tmp_path)

    # Nothing stored yet -> everything is new.
    plan, revisions = plan_ingest(articles, {}, config)
    assert sorted(plan.added) == ["demo-1", "demo-2"]
    assert plan.unchanged == []

    # Stored and identical -> unchanged; a stale slug -> removed.
    stored = {slug: ("rev", fingerprint(rev, config)) for slug, rev in revisions.items()}
    plan, _ = plan_ingest(articles, stored, config)
    assert plan.unchanged == ["demo-1", "demo-2"]
    assert plan.to_index == []

    stored_with_ghost = dict(stored)
    stored_with_ghost["demo-99"] = ("rev", "fp")
    plan, _ = plan_ingest(articles, stored_with_ghost, config)
    assert plan.removed == ["demo-99"]

    # Text edited -> changed, not new.
    _write_article(tmp_path, 1, ARTICLE.format(n=1) + "\nНовый абзац.")
    plan, _ = plan_ingest(load_kb(tmp_path), stored, config)
    assert plan.changed == ["demo-1"]
    assert plan.unchanged == ["demo-2"]


async def test_run_ingest_only_reembeds_changed_articles(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    _write_article(kb_dir, 1)
    _write_article(kb_dir, 2)
    config = _config(kb_dir, sqlite_path=tmp_path / "index.db")
    embedder = FakeEmbedder()
    store = SqliteVectorStore(tmp_path / "index.db")

    first = await run_ingest(config=config, embedder=embedder, store=store)
    assert first["added"] == 2
    assert first["reembedded_articles"] == 2

    # Second run with no changes: nothing is re-embedded.
    second = await run_ingest(config=config, embedder=embedder, store=store)
    assert second["reembedded_articles"] == 0
    assert second["unchanged"] == 2
    assert second["chunks"] == first["chunks"]

    # Change one article: only that one is re-embedded.
    _write_article(kb_dir, 2, ARTICLE.format(n=2) + "\nОбновление.")
    third = await run_ingest(config=config, embedder=embedder, store=store)
    assert third["reembedded_articles"] == 1
    assert third["changed"] == 1
    assert third["unchanged"] == 1

    # Remove one article: its chunks disappear from the index.
    (kb_dir / "article-1.md").unlink()
    fourth = await run_ingest(config=config, embedder=embedder, store=store)
    assert fourth["removed"] >= 1
    assert fourth["articles"] == 1

    await store.dispose()
