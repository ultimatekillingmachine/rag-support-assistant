"""Loading knowledge base articles from markdown files with minimal frontmatter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FRONTMATTER_DELIMITER = "---"


@dataclass(frozen=True)
class KBArticle:
    slug: str
    title: str
    category: str
    audience: str
    updated_at: str
    path: Path
    body: str


class KBError(ValueError):
    """Raised when an article is malformed."""


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse a minimal `key: value` frontmatter block.

    Returns ``(metadata, body)``. Raises :class:`KBError` when the block is
    missing, unclosed, or contains a line without ``key: value``.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
        raise KBError("frontmatter must start with '---' on the first line")
    meta: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == FRONTMATTER_DELIMITER:
            body = "\n".join(lines[index + 1 :]).strip()
            return meta, body
        if ":" not in line:
            raise KBError(f"invalid frontmatter line: {line!r}")
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    raise KBError("frontmatter block is not closed with '---'")


def load_article(path: Path) -> KBArticle:
    """Load and validate a single KB article."""
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    missing = {"slug", "title", "category"} - meta.keys()
    if missing:
        raise KBError(f"{path}: missing frontmatter keys: {sorted(missing)}")
    if not body:
        raise KBError(f"{path}: empty article body")
    return KBArticle(
        slug=meta["slug"],
        title=meta["title"],
        category=meta["category"],
        audience=meta.get("audience", "customer"),
        updated_at=meta.get("updated_at", ""),
        path=path,
        body=body,
    )


def load_kb(kb_dir: Path) -> list[KBArticle]:
    """Load all markdown articles under ``kb_dir`` (README.md is skipped)."""
    if not kb_dir.is_dir():
        raise KBError(f"knowledge base directory not found: {kb_dir}")
    articles: list[KBArticle] = []
    for path in sorted(kb_dir.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        articles.append(load_article(path))
    slugs = [article.slug for article in articles]
    duplicates = {slug for slug in slugs if slugs.count(slug) > 1}
    if duplicates:
        raise KBError(f"duplicate article slugs: {sorted(duplicates)}")
    return articles
