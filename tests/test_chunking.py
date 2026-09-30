"""Tests for the markdown-aware chunker."""

from __future__ import annotations

from pathlib import Path

from app.chunking import _split_sections, _windows, chunk_article
from app.kb import KBArticle


def make_article(body: str) -> KBArticle:
    return KBArticle(
        slug="test-article",
        title="Тестовая статья",
        category="test",
        audience="customer",
        updated_at="2026-01-01",
        path=Path("test-article.md"),
        body=body,
    )


def test_split_sections_keeps_headings() -> None:
    body = (
        "Вводный текст.\n\n## Раздел один\nТекст раздела один."
        "\n\n## Раздел два\nТекст раздела два."
    )
    sections = _split_sections(body)
    assert sections[0] == ("", "Вводный текст.")
    assert sections[1][0] == "Раздел один"
    assert "Текст раздела один." in sections[1][1]
    assert sections[2][0] == "Раздел два"


def test_windows_cover_long_text_with_overlap() -> None:
    text = " ".join(f"слово{i}" for i in range(500))
    windows = _windows(text, max_chars=300, overlap=50)
    assert len(windows) > 1
    assert all(len(window) <= 300 for window in windows)
    # Consecutive windows share some content (overlap).
    first_tail = windows[0][-40:]
    assert first_tail.split()[-1] in windows[1]


def test_windows_short_text_is_single_window() -> None:
    assert _windows("короткий текст", max_chars=300, overlap=50) == ["короткий текст"]


def test_chunk_article_prefixes_heading_and_keeps_metadata() -> None:
    article = make_article(
        "## Сроки\nДоставка занимает 2 дня.\n\n## Оплата\nОплата картой или СБП."
    )
    chunks = chunk_article(article, max_chars=500, overlap=50)
    assert [chunk.heading for chunk in chunks] == ["Сроки", "Оплата"]
    assert chunks[0].text.startswith("Сроки")
    assert chunks[0].slug == "test-article"
    assert chunks[0].category == "test"
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
