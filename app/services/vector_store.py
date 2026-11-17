import os

from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Integer,
    Text,
    UniqueConstraint,
    create_engine,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import EMBEDDING_DIM, EMBEDDING_MODEL, TABLE_NAME

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class DocumentChunkModel(Base):
    __tablename__ = TABLE_NAME
    __table_args__ = (
        UniqueConstraint(
            "source",
            "page",
            "chunk_index",
            name="uq_document_chunks_gemini_identity",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)
    embedding_model = Column(Text, nullable=False)
    source = Column(Text, nullable=False)
    page = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)


def ensure_schema():
    """Create the pgvector extension and the table if they are missing.

    Safe to call repeatedly. The CREATE EXTENSION statement needs a database
    user allowed to install extensions.
    """
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    except Exception as exc:
        raise RuntimeError(
            "Could not enable the PostgreSQL 'vector' extension. Install "
            "pgvector on the database server (e.g. "
            "`sudo apt install postgresql-16-pgvector`) and make sure your DB "
            "user may run CREATE EXTENSION."
        ) from exc

    Base.metadata.create_all(engine)


def _build_rows(embedded_chunks: list[dict]) -> list[dict]:
    rows = []

    for position, item in enumerate(embedded_chunks):
        if not item.get("content"):
            raise ValueError(f"Chunk at position {position} is missing content")

        embedding = item.get("embedding")
        if embedding is None:
            raise ValueError(f"Chunk at position {position} is missing an embedding")

        dim = len(embedding)
        if dim != EMBEDDING_DIM:
            raise ValueError(
                f"Chunk at position {position} has {dim} dimensions, "
                f"but the database column expects {EMBEDDING_DIM}. "
                f"Check EMBEDDING_MODEL/EMBEDDING_DIM in app/core/config.py."
            )

        if not item.get("embedding_model"):
            raise ValueError(
                f"Chunk at position {position} is missing embedding_model"
            )

        metadata = item.get("metadata") or {}
        missing = [
            key
            for key in ("source", "page", "chunk")
            if metadata.get(key) is None
        ]
        if missing:
            raise ValueError(
                f"Chunk at position {position} is missing metadata keys: {missing}"
            )

        rows.append({
            "content": item["content"],
            "embedding": embedding,
            "embedding_model": item["embedding_model"],
            "source": metadata["source"],
            "page": metadata["page"],
            "chunk_index": metadata["chunk"],
        })

    return rows


def save_chunks_to_postgres(embedded_chunks: list[dict]) -> int:
    """Persist embedded chunks, skipping ones already stored.

    Idempotent on (source, page, chunk_index), so re-running ingestion does not
    create duplicate rows. Returns the number of newly inserted chunks.
    """
    if not embedded_chunks:
        return 0

    rows = _build_rows(embedded_chunks)

    ensure_schema()

    session = SessionLocal()
    try:
        statement = (
            insert(DocumentChunkModel)
            .on_conflict_do_nothing(index_elements=["source", "page", "chunk_index"])
            .returning(DocumentChunkModel.id)
        )
        result = session.execute(statement, rows)
        inserted = len(result.fetchall())
        session.commit()
        print(
            f"Saved {inserted} new chunks to PostgreSQL "
            f"({len(rows) - inserted} already existed)."
        )
        return inserted
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def search_similar_chunks(query_embedding: list[float], limit: int = 3):
    """Return the closest chunks by pgvector cosine distance.

    Only vectors produced by the configured embedding model are considered, so
    switching embedding models cannot mix incomparable vector spaces.
    """
    distance = DocumentChunkModel.embedding.cosine_distance(query_embedding)

    session = SessionLocal()
    try:
        statement = (
            select(DocumentChunkModel, distance.label("distance"))
            .where(DocumentChunkModel.embedding_model == EMBEDDING_MODEL)
            .order_by(distance)
            .limit(limit)
        )
        results = session.execute(statement).all()

        return [
            {
                "content": row.content,
                "source": row.source,
                "page": row.page,
                "chunk_index": row.chunk_index,
                "distance": float(distance_value),
            }
            for row, distance_value in results
        ]
    finally:
        session.close()
