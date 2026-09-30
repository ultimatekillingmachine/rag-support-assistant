"""Lexical retrieval: a small, dependency-free BM25 (Okapi) implementation.

Why hand-rolled BM25 instead of FTS5/Elasticsearch at this stage: the corpus is
small, the dev setup stays zero-dependency, and — the real reason — the ranking
math is explicit, readable, and unit-testable. The Postgres path (Phase 3) will
use ``tsvector`` + GIN index for the same purpose.
"""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-zа-яё0-9-]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization that keeps Cyrillic, digits and hyphens."""
    return TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 over a list of pre-tokenized documents."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._term_freqs: list[Counter[str]] = [Counter(doc) for doc in documents]
        self._doc_lengths = [len(doc) for doc in documents]
        self._avg_len = (sum(self._doc_lengths) / len(documents)) if documents else 0.0

        doc_count = len(documents)
        doc_freq: Counter[str] = Counter()
        for term_freq in self._term_freqs:
            doc_freq.update(term_freq.keys())
        self._idf = {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        """Score every document against the query tokens."""
        results: list[float] = []
        for term_freq, doc_len in zip(self._term_freqs, self._doc_lengths, strict=True):
            score = 0.0
            for term in query_tokens:
                freq = term_freq.get(term)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * doc_len / (self._avg_len or 1.0)
                )
                score += idf * freq * (self.k1 + 1) / denominator
            results.append(score)
        return results
