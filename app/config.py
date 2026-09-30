"""Application settings loaded from environment / .env file."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration in one place (see .env.example)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    storage_backend: Literal["sqlite", "postgres"] = "sqlite"
    sqlite_path: Path = Path("data/app.db")
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"

    # Knowledge base
    kb_dir: Path = Path("knowledge_base")

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-large"

    # LLM
    llm_provider: Literal["mock", "openai_compatible", "ollama"] = "mock"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0

    # Retrieval / chunking
    top_k: int = 5
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150

    # Retrieval mode (Phase 2b): hybrid = vector + BM25 fused with RRF.
    # Measured on data/eval (56 questions): weighted hybrid ties vector on
    # hit@1 but trails on hit@3/MRR — BM25's exact-token matching suffers on
    # Russian morphology. Default stays vector; enable hybrid when exact-term
    # recall matters more (weights are pre-tuned below). See README.
    hybrid_enabled: bool = False
    rrf_k: int = 60
    candidate_pool: int = 20
    # Refuse to answer when even the best cosine score is below this value
    # (0 disables the gate). Tuned on data/eval — see README.
    min_relevance_score: float = 0.80
    # RRF weights: the embedding model is the primary signal on this corpus;
    # BM25 adds exact-term recall as a secondary signal (tuned on data/eval).
    vector_weight: float = 1.0
    lexical_weight: float = 0.25
    # Diversity cap: at most this many chunks of the same article in results
    # (0 disables). Prevents one long article from flooding the context.
    max_chunks_per_slug: int = 1


settings = Settings()
