"""A compact Russian stemmer (Snowball-style), written without dependencies.

Why this exists: the lexical half of hybrid search matches **exact** tokens, so
for BM25 the words "доставка" and "доставки" are unrelated even though a human
reads them as one word. Russian is heavily inflected, and that is exactly why
BM25 alone under-performed the embedding model on this corpus (measured in
``scripts/run_eval``).

Scope, stated honestly: this implements the main steps of the Snowball Russian
algorithm (RV-based suffix stripping plus cleanup), not a byte-for-byte port. It
is short, readable and covered by tests. For production-grade morphology across
several languages the next step would be a real library (pymorphy, spaCy) or a
stemming tokenizer inside Postgres (``russian`` dictionary configuration).
"""

from __future__ import annotations

VOWELS = frozenset("аеиоуыэюя")

# Suffix groups, longest match first (the tuples below are sorted at import).
# Lists follow Snowball's Russian ending tables, trimmed to endings that occur
# in service/technical Russian.
# Note: the bare gerund "в" is handled separately — Snowball allows it only
# after "а" or "я", otherwise "бонусов" would turn into "бонусо".
_PERFECTIVE_GERUND = (
    "ившись", "ывшись", "ивши", "ывши", "вшись", "ив", "ыв",
)
_REFLEXIVE = ("ся", "сь")
_ADJECTIVE = (
    "ими", "ыми", "его", "ого", "ему", "ому", "ее", "ие", "ые", "ое", "ей",
    "ий", "ый", "ой", "ем", "им", "ым", "ом", "их", "ых", "ую", "юю", "ая",
    "яя", "ою", "ею",
)
_PARTICIPLE = (
    "ивш", "ывш", "ующ", "ем", "нн", "вш", "ющ", "щ", "енн", "онн", "ащ", "ящ",
)
_VERB = (
    "ейте", "уйте", "ила", "ыла", "ена", "ите", "или", "ыли", "ило", "ыло",
    "ено", "ены", "ить", "ыть", "ишь", "ует", "уют", "ила", "ыла", "ила",
    "ит", "ыт", "ен", "ил", "ыл", "им", "ым", "ей", "уй", "ла", "ло", "ли",
    "ем", "ет", "ют", "ую", "ю",
)
_NOUN = (
    "иями", "ями", "ами", "иях", "иям", "ям", "ием", "ьей", "ьям", "ьях",
    "иев", "ьев", "ией", "ия", "ья", "ий", "ые", "ии", "ие", "ье", "ев", "ов",
    "ах", "ях", "ам", "ом", "ем", "ий", "ый", "ой", "ей", "ию", "ью", "ю",
    "а", "я", "о", "е", "у", "ы", "ь", "и",
)
_DERIVATIONAL = ("ость", "ост")
_SUPERLATIVE = ("ейше", "ейш")

for _group in (
    _PERFECTIVE_GERUND,
    _REFLEXIVE,
    _ADJECTIVE,
    _PARTICIPLE,
    _VERB,
    _NOUN,
    _DERIVATIONAL,
    _SUPERLATIVE,
):
    # Longest suffix must win: "ившись" before "ив", "ость" before "ь".
    _group = tuple(sorted(_group, key=len, reverse=True))


def _rv_start(word: str) -> int:
    """RV region: everything after the first vowel (Snowball's definition)."""
    for index, char in enumerate(word):
        if char in VOWELS:
            return index + 1
    return len(word)


def _strip(word: str, suffixes: tuple[str, ...], start: int) -> str | None:
    """Remove the first matching suffix that keeps the word inside RV."""
    for suffix in sorted(suffixes, key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= start:
            return word[: -len(suffix)]
    return None


def _step1(word: str, start: int) -> str:
    """Strip gerund / reflexive + adjectival / verb + noun / noun / trailing и."""
    stripped = _strip(word, _PERFECTIVE_GERUND, start)
    if stripped is not None:
        return stripped
    # Bare gerund "в" is valid only after "а"/"я" (Snowball rule), which keeps
    # word-final "в" in nouns such as "бонусов" intact for the noun step below.
    if word.endswith("в") and len(word) >= 2 and word[-2] in "ая" and len(word) - 1 >= start:
        return word[:-1]

    base = word
    reflexive = _strip(base, _REFLEXIVE, start)
    if reflexive is not None:
        base = reflexive

    adjective = _strip(base, _ADJECTIVE, start)
    if adjective is not None:
        return _strip(adjective, _PARTICIPLE, start) or adjective

    verb = _strip(base, _VERB, start)
    if verb is not None:
        return _strip(verb, _NOUN, start) or verb

    noun = _strip(base, _NOUN, start)
    if noun is not None:
        return noun

    if word.endswith("и") and len(word) - 1 >= start:
        return word[:-1]
    return word


def _step2(word: str, start: int) -> str:
    """Drop a trailing "и" left after a consonant (Snowball step 2)."""
    if word.endswith("и") and len(word) - 1 >= start and word[-2] not in VOWELS:
        return word[:-1]
    return word


def _step3(word: str, start: int) -> str:
    """Drop the derivational suffix (ость / ост)."""
    return _strip(word, _DERIVATIONAL, start) or word


def _step4(word: str) -> str:
    """Final cleanup: нн -> н, drop superlative, drop trailing soft sign."""
    if word.endswith("нн"):
        word = word[:-1]
    stripped = _strip(word, _SUPERLATIVE, 0)
    if stripped is not None:
        word = stripped
    if word.endswith("ь"):
        word = word[:-1]
    return word


def stem_word(word: str) -> str:
    """Return the stem of one lowercase word (digits and short words untouched)."""
    if len(word) <= 3 or word[0].isdigit() or not any(c in VOWELS for c in word):
        return word
    start = _rv_start(word)
    if start >= len(word):
        return word
    return _step4(_step3(_step2(_step1(word, start), start), start))
