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
from sqlalchemy import Integer, LargeBinary, String, Text, delete, inspect, select
from sqlalchemy.engine import Connection
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
    # Incremental re-index (Phase 3): lets ingestion skip unchanged articles.
    # ``revision`` is a content hash of the article's chunks; ``fingerprint``
    # also covers chunking and embedding settings, so changing the model or the
    # chunk size correctly invalidates the stored vectors.
    revision: Mapped[str] = mapped_column(String(64), index=True, default="")
    fingerprint: Mapped[str] = mapped_column(String(64), index=True, default="")


# Columns added after the first release. ``create_all`` only creates missing
# *tables*, so an existing database would otherwise lack them and every query
# would fail with "no such column". Keeping the list here is a deliberately
# small, transparent substitute for a full migration tool (documented in README).
_ADDED_COLUMNS: dict[str, str] = {
    "revision": "VARCHAR(64) DEFAULT ''",
    "fingerprint": "VARCHAR(64) DEFAULT ''",
}


def _migrate_schema(connection: Connection) -> None:
    """Add columns that are missing from an existing table (idempotent)."""
    inspector = inspect(connection)
    if "kb_chunks" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("kb_chunks")}
    for name, definition in _ADDED_COLUMNS.items():
        if name not in existing:
            connection.exec_driver_sql(
                f"ALTER TABLE kb_chunks ADD COLUMN {name} {definition}"  # noqa: S608 - fixed names
            )


@dataclass(frozen=True)
class StoredChunk:
    slug: str
    title: str
    category: str
    audience: str
    updated_at: str
    chunk_index: int
    text: str
    revision: str = ""
    fingerprint: str = ""


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

    async def article_versions(self) -> dict[str, tuple[str, str]]:
        """Map ``slug -> (revision, fingerprint)`` for the stored articles."""
        ...

    async def delete_slugs(self, slugs: list[str]) -> int: ...

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
        revision=row.revision,
        fingerprint=row.fingerprint,
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
            # Existing databases from earlier versions need the new columns.
            await connection.run_sync(_migrate_schema)

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
                revision=chunk.revision,
                fingerprint=chunk.fingerprint,
            )
            for chunk, vector in zip(chunks, embeddings, strict=True)
        ]
        async with self._session_factory() as session:
            session.add_all(rows)
            await session.commit()
        return len(rows)

    async def article_versions(self) -> dict[str, tuple[str, str]]:
        """Return ``slug -> (revision, fingerprint)`` taken from any chunk."""
        statement = select(
            ChunkRow.slug, ChunkRow.revision, ChunkRow.fingerprint
        ).distinct()
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        return {slug: (revision, fingerprint) for slug, revision, fingerprint in rows}

    async def delete_slugs(self, slugs: list[str]) -> int:
        """Remove every chunk of the given articles (used by incremental ingest)."""
        if not slugs:
            return 0
        async with self._session_factory() as session:
            result = await session.execute(delete(ChunkRow).where(ChunkRow.slug.in_(slugs)))
            await session.commit()
        return int(result.rowcount or 0)

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
