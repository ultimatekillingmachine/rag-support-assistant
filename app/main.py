"""FastAPI application: /health and /ask.

``create_app`` accepts optional pre-built dependencies so tests can inject
lightweight fakes instead of downloading the embedding model.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from app.auth import Principal, require_principal
from app.cache import CachedPipeline, ResponseCache
from app.config import Settings
from app.config import settings as default_settings
from app.embeddings import Embedder, FastEmbedEmbedder
from app.llm import LLM, LLMError, get_llm
from app.metrics import LatencyRecorder
from app.rag import RAGPipeline
from app.retrieval import Retriever, create_retriever
from app.schemas import AskRequest, AskResponse, HealthResponse, SourceItem, StatsResponse
from app.vectorstore import VectorStore, create_store


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
        version="0.4.0",
        lifespan=lifespan,
    )

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
            )
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        sources = [
            SourceItem(
                ref=ref,
                slug=hit.chunk.slug,
                title=hit.chunk.title,
                category=hit.chunk.category,
                updated_at=hit.chunk.updated_at,
                score=round(hit.score, 4),
            )
            for ref, hit in enumerate(result.hits, start=1)
        ]
        # Latency 0 means "served from cache"; record it either way.
        app.state.latency.record(result.latency_ms)
        return AskResponse(
            answer=result.answer,
            sources=sources,
            latency_ms=result.latency_ms,
            llm_provider=getattr(app.state.llm, "name", config.llm_provider),
            llm_model=getattr(app.state.llm, "model", config.llm_model),
            retrieved=len(result.hits),
            refused=result.refused,
            retrieval=getattr(app.state.retriever, "name", "vector"),
            role=principal.role,
            cached=result.latency_ms == 0 and app.state.cache_enabled,
        )

    return app


app = create_app()


