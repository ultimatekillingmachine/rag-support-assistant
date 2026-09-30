"""Pipeline-level tests: retrieval + grounded answer with citations."""

from __future__ import annotations

from app.llm import MockLLM
from app.rag import REFUSAL_TEXT, RAGPipeline
from app.retrieval import HybridRetriever, VectorRetriever
from app.vectorstore import SqliteVectorStore
from tests.helpers import FakeEmbedder


async def test_pipeline_returns_answer_with_sources(populated_store):
    store, embedder = populated_store
    pipeline = RAGPipeline(VectorRetriever(store, embedder), MockLLM(), default_top_k=3)

    result = await pipeline.answer("Сколько дней идёт доставка в Москву?")

    assert result.hits, "retrieval должен вернуть хотя бы один фрагмент"
    assert result.hits[0].chunk.slug == "delivery-time"
    assert "mock-ответ" in result.answer
    assert result.refused is False
    assert result.latency_ms >= 0


async def test_pipeline_prioritizes_relevant_document(populated_store):
    store, embedder = populated_store
    pipeline = RAGPipeline(VectorRetriever(store, embedder), MockLLM(), default_top_k=3)

    result = await pipeline.answer("Как вернуть товар и сохранить упаковку?")

    assert result.hits[0].chunk.slug == "return-policy"


async def test_pipeline_audience_filter_hides_operator_chunks(populated_store):
    store, embedder = populated_store
    pipeline = RAGPipeline(VectorRetriever(store, embedder), MockLLM(), default_top_k=3)

    as_customer = await pipeline.answer("регламент эскалации операторов", audience="customer")
    assert all(hit.chunk.audience == "customer" for hit in as_customer.hits)

    as_operator = await pipeline.answer("регламент эскалации операторов", audience="operator")
    assert any(hit.chunk.slug == "internal-escalation" for hit in as_operator.hits)


async def test_pipeline_refuses_when_gate_blocks(populated_store):
    store, embedder = populated_store
    gated = HybridRetriever(store, embedder, min_relevance_score=0.99)
    pipeline = RAGPipeline(gated, MockLLM(), default_top_k=3)

    result = await pipeline.answer("рецепт борща со сметаной")

    assert result.refused is True
    assert result.hits == []
    assert result.answer == REFUSAL_TEXT


async def test_empty_store_returns_refusal(tmp_path):
    store = SqliteVectorStore(tmp_path / "empty.db")
    await store.init()
    try:
        pipeline = RAGPipeline(
            VectorRetriever(store, FakeEmbedder()), MockLLM(), default_top_k=3
        )
        result = await pipeline.answer("Любой вопрос")
        assert result.hits == []
        assert result.refused is True
    finally:
        await store.dispose()

