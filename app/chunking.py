"""Markdown-aware chunking with whitespace-safe windows and character overlap."""

from __future__ import annotations

from dataclasses import dataclass

from app.kb import KBArticle


@dataclass(frozen=True)
class Chunk:
    slug: str
    title: str
    category: str
    audience: str
    updated_at: str
    chunk_index: int
    heading: str
    text: str


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split a markdown body into ``(heading, section_text)`` pairs.

    Text before the first heading gets an empty heading.
    """
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_heading, text))

    for line in body.splitlines():
        if line.lstrip().startswith("#"):
            flush()
            current_heading = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return sections


def _windows(text: str, max_chars: int, overlap: int) -> list[str]:
    """Hard-split a long text into overlapping windows, preferring line/space cuts."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    windows: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            cut = text.rfind("\n", start, end)
            if cut == -1 or cut <= start + max_chars // 2:
                cut = text.rfind(" ", start, end)
            if cut != -1 and cut > start + max_chars // 2:
                end = cut
        windows.append(text[start:end].strip())
        if end >= length:
            break
        new_start = end - overlap
        start = new_start if new_start > start else end
    return [window for window in windows if window]


def chunk_article(article: KBArticle, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Split an article into retrieval chunks.

    Each chunk keeps its section heading as a prefix so that the embedded text
    carries local context even when the article title is not visible.
    """
    chunks: list[Chunk] = []
    index = 0
    for heading, section_text in _split_sections(article.body):
        for window in _windows(section_text, max_chars, overlap):
            text = f"{heading}\n\n{window}" if heading else window
            chunks.append(
                Chunk(
                    slug=article.slug,
                    title=article.title,
                    category=article.category,
                    audience=article.audience,
                    updated_at=article.updated_at,
                    chunk_index=index,
                    heading=heading,
                    text=text,
                )
            )
            index += 1
    return chunks
