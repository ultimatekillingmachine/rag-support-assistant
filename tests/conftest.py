"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from app.vectorstore import SqliteVectorStore
from tests.helpers import FakeEmbedder, stored_chunk


@pytest.fixture
async def populated_store(tmp_path):
    """SQLite store seeded with three chunks (two customer, one operator)."""
    store = SqliteVectorStore(tmp_path / "test.db")
    await store.init()
    embedder = FakeEmbedder()
    chunks = [
        stored_chunk(
            "delivery-time",
            "Сроки доставки",
            "Доставка в Москву занимает 1-2 дня. В удалённые регионы 10-18 дней.",
        ),
        stored_chunk(
            "return-policy",
            "Возврат товара",
            "Вернуть товар можно в течение 7 дней после получения при сохранении упаковки.",
        ),
        stored_chunk(
            "internal-escalation",
            "Внутренний регламент эскалации",
            "Регламент эскалации обращений для операторов поддержки.",
            audience="operator",
        ),
    ]
    embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
    await store.upsert(chunks, embeddings)
    yield store, embedder
    await store.dispose()
