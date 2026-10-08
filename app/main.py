"""FastAPI application.

Endpoints:

* ``GET  /health`` — liveness plus index size;
* ``GET  /stats`` — cache effectiveness and latency percentiles;
* ``POST /ask`` — one grounded answer (JSON);
* ``POST /ask/stream`` — the same, streamed as server-sent events;
* ``GET  /`` — a small chat page that uses the streaming endpoint.

``create_app`` accepts optional pre-built dependencies so tests can inject
lightweight fakes instead of downloading the embedding model.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.auth import Principal, require_principal
from app.cache import CachedPipeline, ResponseCache
from app.config import Settings
from app.config import settings as default_settings
from app.embeddings import Embedder, FastEmbedEmbedder
from app.llm import LLM, LLMError, get_llm
from app.metrics import LatencyRecorder
from app.rag import ChatTurn, RAGPipeline
from app.retrieval import Retriever, create_retriever
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    StatsResponse,
    to_source_items,
)
from app.vectorstore import VectorStore, create_store

UI_PATH = Path(__file__).parent / "static" / "index.html"
STATIC_DIR = Path(__file__).parent / "static"
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # tells nginx-style proxies not to buffer the stream
}


def to_chat_turns(request: AskRequest) -> list[ChatTurn]:
    """Convert the API dialogue into the pipeline's internal representation."""
    return [ChatTurn(role=message.role, content=message.content) for message in request.history]


def create_app(
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
    retriever: Retriever | None = None,
    config: Settings | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Any dependency left as ``None`` is created from settings inside lifespan
    (the production path). Tests pass fakes to keep model downloads out.
    """
    config = config or default_settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store or create_store(config)
        app.state.embedder = embedder or FastEmbedEmbedder(config.embedding_model)
        app.state.llm = llm or get_llm(config)
        await app.state.store.init()
        app.state.retriever = retriever or create_retriever(
            app.state.store, app.state.embedder, config
        )
        base_pipeline = RAGPipeline(
            app.state.retriever,
            app.state.llm,
            default_top_k=config.top_k,
        )
        app.state.cache = ResponseCache(
            max_entries=config.cache_max_entries,
            ttl_seconds=config.cache_ttl_seconds,
        )
        app.state.latency = LatencyRecorder()
        # Caching wraps the pipeline; when disabled the pipeline is used as-is.
        app.state.pipeline = (
            CachedPipeline(base_pipeline, app.state.cache, default_top_k=config.top_k)
            if config.cache_enabled
            else base_pipeline
        )
        app.state.cache_enabled = config.cache_enabled
        yield
        await app.state.store.dispose()

    app = FastAPI(
        title="Support RAG Assistant",
        description="Grounded customer-support answers over the «ТехноМаркет» knowledge base.",
        version="0.5.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            chunks=await app.state.store.count(),
            embedding_model=config.embedding_model,
            llm_provider=config.llm_provider,
        )

    @app.get("/stats", response_model=StatsResponse)
    async def stats() -> StatsResponse:
        """Cache effectiveness and latency percentiles (no authentication)."""
        return StatsResponse(
            cache=app.state.cache.stats(),
            latency=app.state.latency.snapshot(),
        )

    @app.get("/", include_in_schema=False)
    async def chat_ui() -> FileResponse:
        """Serve the small chat page (it talks to /ask/stream)."""
        return FileResponse(UI_PATH)

    @app.post("/ask", response_model=AskResponse)
    async def ask(
        request: AskRequest,
        principal: Annotated[Principal, Depends(require_principal)],
    ) -> AskResponse:
        pipeline: RAGPipeline = app.state.pipeline
        try:
            result = await pipeline.answer(
                request.question,
                top_k=request.top_k,
                # Visibility comes from the authenticated role, never from the
                # request body (see app/auth.py).
                audience=principal.audience_filter,
                history=to_chat_turns(request),
            )
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        # Latency 0 means "served from cache"; record it either way.
        app.state.latency.record(result.latency_ms)
        return AskResponse(
            answer=result.answer,
            sources=to_source_items(result.hits),
            latency_ms=result.latency_ms,
            llm_provider=getattr(app.state.llm, "name", config.llm_provider),
            llm_model=getattr(app.state.llm, "model", config.llm_model),
            retrieved=len(result.hits),
            refused=result.refused,
            retrieval=getattr(app.state.retriever, "name", "vector"),
            role=principal.role,
            cached=result.latency_ms == 0 and app.state.cache_enabled,
            used_history=result.used_history,
        )

    @app.post("/ask/stream")
    async def ask_stream(
        request: AskRequest,
        principal: Annotated[Principal, Depends(require_principal)],
    ) -> StreamingResponse:
        """Same answer as ``/ask``, delivered as server-sent events.

        Streaming intentionally bypasses the response cache: the value here is
        the token flow, and a cached answer would arrive instantly in one piece.
        """
        pipeline: RAGPipeline = app.state.pipeline
        history = to_chat_turns(request)

        async def event_stream():
            try:
                async for event in pipeline.answer_stream(
                    request.question,
                    top_k=request.top_k,
                    audience=principal.audience_filter,
                    history=history,
                ):
                    if event["type"] == "sources":
                        event = {
                            "type": "sources",
                            "sources": [
                                item.model_dump() for item in to_source_items(event["sources"])
                            ],
                            "role": principal.role,
                            "used_history": event.get("used_history", False),
                        }
                    elif event["type"] == "done":
                        app.state.latency.record(event["latency_ms"])
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except LLMError as exc:
                payload = {"type": "error", "message": str(exc)}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_stream(), media_type="text/event-stream", headers=SSE_HEADERS
        )

    return app


app = create_app()


