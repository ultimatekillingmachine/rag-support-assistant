"""Pydantic request/response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class SourceItem(BaseModel):
    ref: int = Field(..., description="Citation number used in the answer, e.g. [1]")
    slug: str
    title: str
    category: str
    updated_at: str
    score: float


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


class StatsResponse(BaseModel):
    """Service counters: cache effectiveness and latency percentiles."""

    cache: dict[str, float | int]
    latency: dict[str, float | int]


class HealthResponse(BaseModel):
    status: str
    chunks: int
    embedding_model: str
    llm_provider: str
