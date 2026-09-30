"""API tests with injected fakes (no model download, no network)."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.llm import MockLLM
from app.main import create_app
from app.retrieval import HybridRetriever, VectorRetriever
from app.vectorstore import SqliteVectorStore
from tests.helpers import FakeEmbedder, stored_chunk


def _build_app(tmp_path, *, with_data: bool = True, gated: bool = False):
    store = SqliteVectorStore(tmp_path / "api.db")
    embedder = FakeEmbedder()

    async def prepare() -> None:
        await store.init()
        if with_data:
            chunks = [
                stored_chunk(
                    "delivery-time",
                    "Сроки доставки",
                    "Доставка в Москву занимает 1-2 дня.",
                ),
                stored_chunk(
                    "return-policy",
                    "Возврат товара",
                    "Товар можно вернуть в течение 7 дней.",
                ),
            ]
            await store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
        # Release connections created in this loop; lifespan will use its own.
        await store.dispose()

    asyncio.run(prepare())
    retriever = (
        HybridRetriever(store, embedder, min_relevance_score=0.99)
        if gated
        else VectorRetriever(store, embedder)
    )
    return create_app(store=store, embedder=embedder, llm=MockLLM(), retriever=retriever)


def test_health_reports_chunk_count(tmp_path) -> None:
    app = _build_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks"] == 2


def test_ask_returns_answer_and_sources(tmp_path) -> None:
    app = _build_app(tmp_path)
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "Сколько идёт доставка в Москву?"})
    assert response.status_code == 200
    body = response.json()
    assert body["sources"][0]["slug"] == "delivery-time"
    assert body["retrieved"] >= 1
    assert "mock-ответ" in body["answer"]
    assert body["llm_provider"] == "mock"
    assert body["refused"] is False
    assert body["retrieval"] == "vector"


def test_ask_refuses_out_of_scope_question(tmp_path) -> None:
    app = _build_app(tmp_path, gated=True)
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "рецепт борща со сметаной"})
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["sources"] == []
    assert "нет информации" in body["answer"]


def test_ask_validates_question_length(tmp_path) -> None:
    app = _build_app(tmp_path)
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "ok"})
    assert response.status_code == 422

