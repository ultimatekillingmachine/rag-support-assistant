"""Tests for streaming answers (pipeline level and SSE endpoint)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.llm import MockLLM
from app.rag import REFUSAL_TEXT, RAGPipeline
from app.retrieval import HybridRetriever, VectorRetriever
from tests.test_api import CUSTOMER_AUTH, _build_app


def _events(text: str) -> list[dict]:
    """Parse a server-sent events body into a list of payloads."""
    events = []
    for frame in text.split("\n\n"):
        frame = frame.strip()
        if frame.startswith("data:"):
            events.append(json.loads(frame[len("data:") :].strip()))
    return events


async def test_pipeline_stream_yields_sources_tokens_then_done(populated_store) -> None:
    store, embedder = populated_store
    pipeline = RAGPipeline(VectorRetriever(store, embedder), MockLLM(), default_top_k=3)

    events = [event async for event in pipeline.answer_stream("доставка в москву")]

    assert events[0]["type"] == "sources"
    assert events[0]["sources"], "sources must be sent before the text"
    assert events[0]["used_history"] is False
    assert events[-1]["type"] == "done"

    tokens = [event["text"] for event in events if event["type"] == "token"]
    assert len(tokens) > 1, "the answer must arrive in several pieces"
    assert events[-1]["answer"] == "".join(tokens)
    assert "[mock-ответ]" in events[-1]["answer"]
    assert events[-1]["refused"] is False


async def test_pipeline_stream_refusal_is_one_token(populated_store) -> None:
    store, embedder = populated_store
    gated = HybridRetriever(store, embedder, min_relevance_score=0.99)
    pipeline = RAGPipeline(gated, MockLLM(), default_top_k=3)

    events = [event async for event in pipeline.answer_stream("рецепт борща со сметаной")]

    assert events[0]["type"] == "sources"
    assert events[0]["sources"] == []
    tokens = [event["text"] for event in events if event["type"] == "token"]
    assert tokens == [REFUSAL_TEXT]
    assert events[-1]["refused"] is True


def test_stream_endpoint_emits_sse_frames(tmp_path) -> None:
    app = _build_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/ask/stream",
            json={"question": "Сколько идёт доставка в Москву?"},
            headers=CUSTOMER_AUTH,
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _events(response.text)
    assert events[0]["type"] == "sources"
    assert events[0]["sources"][0]["slug"] == "delivery-time"
    assert events[0]["role"] == "customer"
    assert events[-1]["type"] == "done"
    assert events[-1]["refused"] is False


def test_stream_endpoint_requires_token(tmp_path) -> None:
    app = _build_app(tmp_path)
    with TestClient(app) as client:
        response = client.post("/ask/stream", json={"question": "доставка в Москву"})
    assert response.status_code == 401


def test_chat_ui_and_script_are_served(tmp_path) -> None:
    app = _build_app(tmp_path)
    with TestClient(app) as client:
        page = client.get("/")
        script = client.get("/static/app.js")
    assert page.status_code == 200
    assert "Ассистент поддержки" in page.text
    assert script.status_code == 200
    assert "/ask/stream" in script.text
