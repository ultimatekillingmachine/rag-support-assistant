"""CLI: ``python -m scripts.gate_report`` — relevance-gate calibration report.

Prints the top-1 cosine score for every eval question, grouped into answerable
vs out-of-scope, and the best separating threshold together with its trade-off
statistics. This is the documented procedure behind ``MIN_RELEVANCE_SCORE``.
"""

from __future__ import annotations

import asyncio

from app.config import settings
from app.embeddings import FastEmbedEmbedder
from app.vectorstore import create_store
from scripts.run_eval import load_questions


async def run() -> None:
    store = create_store(settings)
    await store.init()
    embedder = FastEmbedEmbedder(settings.embedding_model)

    answerable: list[tuple[float, str]] = []
    out_of_scope: list[tuple[float, str]] = []
    for item in load_questions():
        query_vector = embedder.embed_query(item["question"])
        hits = await store.search(
            query_vector, 1, audience=item.get("audience", "customer")
        )
        top_score = hits[0].score if hits else 0.0
        row = f"{item['id']:>4}  {top_score:.3f}  {item['question'][:64]}"
        bucket = answerable if item["expected_slugs"] else out_of_scope
        bucket.append((top_score, row))

    print("== out-of-scope questions (higher score = harder to refuse) ==")
    for _, row in sorted(out_of_scope, reverse=True):
        print(row)

    print("\n== answerable questions with the weakest top-1 score ==")
    for _, row in sorted(answerable)[:10]:
        print(row)

    min_answerable = min(score for score, _ in answerable)
    max_out_of_scope = max(score for score, _ in out_of_scope)
    threshold = (min_answerable + max_out_of_scope) / 2
    print(f"\nmin answerable top-1:    {min_answerable:.3f}")
    print(f"max out-of-scope top-1:  {max_out_of_scope:.3f}")
    print(f"best separating threshold: {threshold:.3f}")
    keep = sum(score >= threshold for score, _ in answerable) / len(answerable)
    refuse = sum(score < threshold for score, _ in out_of_scope) / len(out_of_scope)
    print(f"  -> answerable kept: {keep:.0%} | out-of-scope refused: {refuse:.0%}")
    if min_answerable <= max_out_of_scope:
        print("NOTE: the classes overlap — no perfect threshold; choose by cost.")

    await store.dispose()


if __name__ == "__main__":
    asyncio.run(run())
