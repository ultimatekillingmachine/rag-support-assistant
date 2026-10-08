"""Tests for follow-up questions: query building, prompt assembly, API contract."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.cache import ResponseCache
from app.llm import MockLLM
from app.rag import ChatTurn, RAGPipeline, build_retrieval_query, build_user_prompt
from app.retrieval import VectorRetriever
from tests.test_api import CUSTOMER_AUTH, _build_app


def test_query_without_history_is_the_question_itself() -> None:
    assert build_retrieval_query("Какая гарантия на холодильники?") == (
        "Какая гарантия на холодильники?"
    )


def test_short_follow_up_inherits_the_previous_topic() -> None:
    history = [
        ChatTurn("user", "Какая гарантия на холодильники?"),
        ChatTurn("assistant", "24 месяца"),
    ]
    query = build_retrieval_query("А если брак?", history)
    assert query.startswith("Какая гарантия на холодильники?")
    assert query.endswith("А если брак?")


def test_anaphora_makes_a_long_question_a_follow_up() -> None:
    history = [ChatTurn("user", "Сколько стоит доставка холодильника?")]
    query = build_retrieval_query("А сколько это займёт по времени?", history)
    assert "Сколько стоит доставка холодильника?" in query


def test_independent_question_ignores_history() -> None:
    history = [ChatTurn("user", "Какая гарантия на холодильники?")]
    query = build_retrieval_query("Как вернуть стиральную машину?", history)
    assert query == "Как вернуть стиральную машину?"


def test_history_without_user_turns_does_not_break_the_query() -> None:
    history = [ChatTurn("assistant", "Здравствуйте!")]
    assert build_retrieval_query("А что с доставкой?", history) == "А что с доставкой?"


def test_prompt_contains_dialogue_and_numbered_fragments() -> None:
    from app.vectorstore import SearchHit
    from tests.helpers import stored_chunk

    sample = [
        SearchHit(chunk=stored_chunk("delivery-time", "Сроки доставки", "1-2 дня"), score=0.9)
    ]
    history = [ChatTurn("user", "Сколько идёт доставка?"), ChatTurn("assistant", "1-2 дня")]

    prompt = build_user_prompt("А в регионы?", sample, history)

    assert "ПРЕДЫДУЩИЙ ДИАЛОГ:" in prompt
    assert "Покупатель: Сколько идёт доставка?" in prompt
    assert "Ассистент: 1-2 дня" in prompt
    assert "[1] Сроки доставки" in prompt
    assert "ВОПРОС ПОКУПАТЕЛЯ: А в регионы?" in prompt


def test_prompt_without_history_has_no_dialogue_section() -> None:
    from app.vectorstore import SearchHit
    from tests.helpers import stored_chunk

    sample = [SearchHit(chunk=stored_chunk("a", "A", "текст"), score=0.5)]
    prompt = build_user_prompt("Вопрос?", sample)

    assert "ПРЕДЫДУЩИЙ ДИАЛОГ" not in prompt
    assert "ВОПРОС ПОКУПАТЕЛЯ: Вопрос?" in prompt


async def test_pipeline_flags_history_usage(populated_store) -> None:
    store, embedder = populated_store
    pipeline = RAGPipeline(VectorRetriever(store, embedder), MockLLM(), default_top_k=3)
    history = [ChatTurn("user", "Как вернуть товар?")]

    result = await pipeline.answer("А если упаковка порвана?", history=history)

    assert result.used_history is True


async def test_follow_up_retrieval_uses_previous_topic(populated_store) -> None:
    """A topic-less follow-up must still find the article discussed before."""
    store, embedder = populated_store
    pipeline = RAGPipeline(VectorRetriever(store, embedder), MockLLM(), default_top_k=1)
    history = [ChatTurn("user", "Сколько идёт доставка в Москву?")]

    with_history = await pipeline.answer("А это долго?", history=history)

    assert with_history.used_history is True
    # The previous question is prepended for retrieval, so the delivery article
    # is found even though the follow-up alone carries no topic.
    assert with_history.hits
    assert with_history.hits[0].chunk.slug == "delivery-time"


def test_api_reports_history_usage(tmp_path) -> None:
    app = _build_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Сколько идёт доставка в Москву?",
                "history": [
                    {"role": "user", "content": "Привет"},
                    {"role": "assistant", "content": "Здравствуйте!"},
                ],
            },
            headers=CUSTOMER_AUTH,
        )
    assert response.status_code == 200
    assert response.json()["used_history"] is True
    assert response.json()["role"] == "customer"


def test_api_rejects_too_long_history(tmp_path) -> None:
    app = _build_app(tmp_path)
    history = [{"role": "user", "content": f"вопрос {index}"} for index in range(20)]
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"question": "Сколько идёт доставка?", "history": history},
            headers=CUSTOMER_AUTH,
        )
    assert response.status_code == 422


def test_cache_key_separates_different_dialogues() -> None:
    first = [ChatTurn("user", "гарантия на холодильники")]
    second = [ChatTurn("user", "возврат стиральной машины")]
    digest_first = ResponseCache.history_digest(first)
    digest_second = ResponseCache.history_digest(second)

    key_a = ResponseCache.make_key("а если брак?", 5, "customer", digest_first)
    key_b = ResponseCache.make_key("а если брак?", 5, "customer", digest_second)

    assert key_a != key_b
    assert digest_first == ResponseCache.history_digest(first)
    assert ResponseCache.history_digest([]) == ""
