"""Tests for the BM25 implementation."""

from __future__ import annotations

from app.lexical import BM25, tokenize


def test_tokenize_handles_cyrillic_hyphens_digits_case() -> None:
    tokens = tokenize("Доставка СБП-платежа за 3 дня")
    assert tokens == ["доставка", "сбп-платежа", "за", "3", "дня"]


def test_bm25_ranks_matching_document_first() -> None:
    docs = [
        tokenize("доставка в москву занимает два дня"),
        tokenize("возврат товара в течение семи дней"),
    ]
    scores = BM25(docs).scores(tokenize("доставка москву"))
    assert scores[0] > scores[1]
    assert scores[1] == 0.0


def test_bm25_rare_terms_weigh_more_than_common_ones() -> None:
    docs = [
        tokenize("доставка курьером"),
        tokenize("доставка самовывоз"),
        tokenize("гарантия на холодильник"),
    ]
    bm25 = BM25(docs)
    common_score = bm25.scores(tokenize("доставка"))[0]
    rare_score = bm25.scores(tokenize("холодильник"))[2]
    assert rare_score > common_score


def test_bm25_empty_corpus_returns_empty_scores() -> None:
    assert BM25([]).scores(tokenize("доставка")) == []
