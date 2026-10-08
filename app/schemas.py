"""Pydantic request/response models for the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.vectorstore import SearchHit

# Dialogue length accepted from a client (messages, not turns): keeps prompts
# bounded and prevents a caller from making us pay for a huge history.
HISTORY_LIMIT = 12


class ChatMessage(BaseModel):
    """One earlier message, used to resolve follow-up questions."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class AskRequest(BaseModel):
    """A question from a caller.

    Note what is *absent*: the client cannot ask for a different article
    audience. Visibility is decided server-side from the bearer token
    (see ``app.auth``); the body carries intent only.
    """

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["Сколько идёт доставка в регионы?"],
    )
    top_k: int | None = Field(default=None, ge=1, le=20)
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=HISTORY_LIMIT,
        description="Previous messages, oldest first. Used to resolve follow-ups.",
    )


class SourceItem(BaseModel):
    ref: int = Field(..., description="Citation number used in the answer, e.g. [1]")
    slug: str
    title: str
    category: str
    updated_at: str
    score: float


def to_source_items(hits: list[SearchHit]) -> list[SourceItem]:
    """Map retrieval hits to the public response shape (shared by both endpoints)."""
    return [
        SourceItem(
            ref=ref,
            slug=hit.chunk.slug,
            title=hit.chunk.title,
            category=hit.chunk.category,
            updated_at=hit.chunk.updated_at,
            score=round(hit.score, 4),
        )
        for ref, hit in enumerate(hits, start=1)
    ]


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    latency_ms: int
    llm_provider: str
    llm_model: str
    retrieved: int
    refused: bool = False
    retrieval: str = "vector"
    role: str = "customer"
    cached: bool = False
    used_history: bool = False


class StatsResponse(BaseModel):
    """Service counters: cache effectiveness and latency percentiles."""

    cache: dict[str, float | int]
    latency: dict[str, float | int]


class HealthResponse(BaseModel):
    status: str
    chunks: int
    embedding_model: str
    llm_provider: str
