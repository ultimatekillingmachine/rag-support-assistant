"""Vector storage.

Phase 1 ships a SQLite backend: embeddings are stored as float32 blobs and
searched with brute-force cosine similarity (fine for thousands of chunks and
zero-setup local development on any OS). The Postgres + pgvector backend
(production path) plugs into the same :class:`VectorStore` interface in Phase 3;
``docker-compose.yml`` is already prepared for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sqlalchemy import Integer, LargeBinary, String, Text, delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import Settings


class Base(DeclarativeBase):
    pass


class ChunkRow(Base):
    __tablename__ = "kb_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(120), index=True)
    title: Mapped[str] = mapped_column(String(300))
    category: Mapped[str] = mapped_column(String(60), index=True)
    audience: Mapped[str] = mapped_column(String(20), index=True, default="customer")
    updated_at: Mapped[str] = mapped_column(String(20), default="")
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)


@dataclass(frozen=True)
class StoredChunk:
    slug: str
    title: str
    category: str
    audience: str
    updated_at: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: StoredChunk
    score: float


class VectorStore(Protocol):
    async def init(self) -> None: ...

    async def reset(self) -> None: ...

    async def upsert(self, chunks: list[StoredChunk], embeddings: np.ndarray) -> int: ...

    async def search(
        self, query_vector: np.ndarray, top_k: int, audience: str | None = None
    ) -> list[SearchHit]: ...

    async def list_chunks(self, audience: str | None = None) -> list[StoredChunk]: ...

    async def count(self) -> int: ...

    async def dispose(self) -> None: ...


def _row_to_stored(row: ChunkRow) -> StoredChunk:
    return StoredChunk(
        slug=row.slug,
        title=row.title,
        category=row.category,
        audience=row.audience,
        updated_at=row.updated_at,
        chunk_index=row.chunk_index,
        text=row.text,
    )


class SqliteVectorStore:
    """Brute-force cosine search over float32 blob embeddings."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def reset(self) -> None:
        async with self._session_factory() as session:
            await session.execute(delete(ChunkRow))
            await session.commit()

    async def upsert(self, chunks: list[StoredChunk], embeddings: np.ndarray) -> int:
        embeddings = np.atleast_2d(embeddings).astype(np.float32)
        rows = [
            ChunkRow(
                slug=chunk.slug,
                title=chunk.title,
                category=chunk.category,
                audience=chunk.audience,
                updated_at=chunk.updated_at,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                embedding=vector.tobytes(),
            )
            for chunk, vector in zip(chunks, embeddings, strict=True)
        ]
        async with self._session_factory() as session:
            session.add_all(rows)
            await session.commit()
        return len(rows)

    async def search(
        self, query_vector: np.ndarray, top_k: int, audience: str | None = None
    ) -> list[SearchHit]:
        statement = select(ChunkRow)
        if audience is not None:
            statement = statement.where(ChunkRow.audience == audience)
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).scalars().all()
        if not rows:
            return []
        matrix = np.vstack([np.frombuffer(row.embedding, dtype=np.float32) for row in rows])
        scores = matrix @ np.asarray(query_vector, dtype=np.float32)
        order = np.argsort(-scores)[:top_k]
        return [SearchHit(chunk=_row_to_stored(rows[i]), score=float(scores[i])) for i in order]

    async def list_chunks(self, audience: str | None = None) -> list[StoredChunk]:
        statement = select(ChunkRow)
        if audience is not None:
            statement = statement.where(ChunkRow.audience == audience)
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [_row_to_stored(row) for row in rows]

    async def count(self) -> int:
        async with self._session_factory() as session:
            rows = (await session.execute(select(ChunkRow.id))).scalars().all()
        return len(rows)

    async def dispose(self) -> None:
        await self._engine.dispose()


def create_store(settings: Settings) -> VectorStore:
    """Build the configured vector store backend."""
    if settings.storage_backend == "sqlite":
        return SqliteVectorStore(settings.sqlite_path)
    if settings.storage_backend == "postgres":
        raise NotImplementedError(
            "Postgres/pgvector backend arrives in Phase 3; "
            "set STORAGE_BACKEND=sqlite for now (docker-compose.yml is ready)."
        )
    raise ValueError(f"unknown storage backend: {settings.storage_backend}")
