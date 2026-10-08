"""Tests for the compact Russian stemmer and its use in lexical search."""

from __future__ import annotations

from app.lexical import BM25, tokenize
from app.stemming import stem_word


def test_word_forms_collapse_to_one_stem() -> None:
    forms = ["доставка", "доставки", "доставке", "доставку", "доставкой"]
    stems = {stem_word(form) for form in forms}
    assert len(stems) == 1


def test_verb_and_noun_of_a_guarantee_collapse() -> None:
    assert stem_word("гарантия") == stem_word("гарантии")


def test_short_words_and_digits_are_untouched() -> None:
    for token in ["сбп", "нa", "3", "tm", "20"]:
        assert stem_word(token) == token


def test_hyphenated_technology_terms_keep_their_prefix() -> None:
    # "сбп-платежа" -> the ending is stripped, the meaningful prefix stays.
    assert stem_word("сбп-платежа").startswith("сбп-")


def test_superlative_and_soft_sign_cleanup() -> None:
    assert stem_word("быстрейший") == "быстр"
    assert not stem_word("доставь").endswith("ь")


def test_tokenize_with_stemming_collapses_inflections() -> None:
    plain = tokenize("доставки товара")
    stemmed = tokenize("доставки товара", stemming=True)
    assert plain != stemmed
    assert stemmed == [stem_word("доставки"), stem_word("товара")]


def test_bm25_with_stemming_matches_a_different_word_form() -> None:
    # Document uses "доставка", the query uses "доставку" — without stemming
    # there is no shared token, with stemming there is.
    documents = [tokenize("доставка по москве занимает два дня", stemming=True)]
    scores = BM25(documents).scores(tokenize("доставку в москву", stemming=True))
    assert scores[0] > 0

    plain_documents = [tokenize("доставка по москве занимает два дня")]
    plain_scores = BM25(plain_documents).scores(tokenize("доставку в москву"))
    assert plain_scores[0] == 0
