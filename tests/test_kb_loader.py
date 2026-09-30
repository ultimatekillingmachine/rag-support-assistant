"""Tests for the knowledge base loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.kb import KBError, load_article, load_kb, parse_frontmatter

GOOD_ARTICLE = """---
slug: demo-article
title: Демонстрационная статья
category: demo
audience: customer
updated_at: 2026-01-01
---

# Заголовок

Текст статьи.
"""


def test_parse_frontmatter_ok() -> None:
    meta, body = parse_frontmatter(GOOD_ARTICLE)
    assert meta["slug"] == "demo-article"
    assert meta["title"] == "Демонстрационная статья"
    assert body.startswith("# Заголовок")


def test_parse_frontmatter_without_opening_delimiter_raises() -> None:
    with pytest.raises(KBError, match="first line"):
        parse_frontmatter("slug: x\n---\nbody")


def test_parse_frontmatter_unclosed_raises() -> None:
    with pytest.raises(KBError, match="not closed"):
        parse_frontmatter("---\nslug: x\n")


def test_load_article_defaults_audience_to_customer(tmp_path: Path) -> None:
    path = tmp_path / "article.md"
    path.write_text(
        "---\nslug: a\ntitle: A\ncategory: c\n---\n\nText.",
        encoding="utf-8",
    )
    article = load_article(path)
    assert article.audience == "customer"
    assert article.updated_at == ""


def test_load_article_requires_core_keys(tmp_path: Path) -> None:
    path = tmp_path / "broken.md"
    path.write_text("---\nslug: a\n---\n\nText.", encoding="utf-8")
    with pytest.raises(KBError, match="missing frontmatter keys"):
        load_article(path)


def test_load_kb_skips_readme_and_detects_duplicate_slugs(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("not an article", encoding="utf-8")
    (tmp_path / "one.md").write_text(GOOD_ARTICLE, encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "two.md").write_text(
        GOOD_ARTICLE.replace("demo-article", "demo-article"),
        encoding="utf-8",
    )
    with pytest.raises(KBError, match="duplicate article slugs"):
        load_kb(tmp_path)

    (sub / "two.md").write_text(
        GOOD_ARTICLE.replace("demo-article", "other-article"),
        encoding="utf-8",
    )
    articles = load_kb(tmp_path)
    assert sorted(article.slug for article in articles) == ["demo-article", "other-article"]
