from sqlalchemy import Column, ForeignKey, Integer, Text

from app.services.vector_store import (
    Base,
    DocumentChunkModel,
    SessionLocal,
    ensure_schema,
)


class GraphFactModel(Base):
    """One row per fact extracted from one chunk.

    A row is either a plain entity mention or one relationship:

        entity="casual leave", relation=None, related_entity=None
        entity="sick leave", relation="CANNOT_BE_CLUBBED_WITH",
            related_entity="casual leave"

    Keeping the graph in one flat table means retrieval never needs a JOIN:
    an entity can be found by looking at two text columns.
    """

    __tablename__ = "graph_facts"

    id = Column(Integer, primary_key=True, index=True)

    chunk_id = Column(
        Integer,
        ForeignKey(f"{DocumentChunkModel.__tablename__}.id"),
        nullable=False,
        index=True,
    )

    entity = Column(Text, nullable=False, index=True)
    entity_type = Column(Text, nullable=False, default="UNKNOWN")

    # NULL for a plain mention; filled for a relationship.
    relation = Column(Text, nullable=True)
    related_entity = Column(Text, nullable=True)


def normalize_name(name: str) -> str:
    """Lowercase and collapse whitespace so names compare reliably."""
    return " ".join(str(name).strip().lower().split())


def save_graph_data(graph_records: list[dict]) -> int:
    """Replace the facts stored for each chunk in ``graph_records``.

    Each record must contain: source, page, chunk_index, entities,
    relationships. The facts for a chunk are deleted and rewritten, so
    re-running extraction never duplicates rows.
    """

    if not graph_records:
        return 0

    ensure_schema()

    session = SessionLocal()
    saved_records = 0

    try:
        for record in graph_records:

            chunk = (
                session.query(DocumentChunkModel)
                .filter_by(
                    source=record["source"],
                    page=record["page"],
                    chunk_index=record["chunk_index"],
                )
                .first()
            )

            if not chunk:
                continue

            # name -> entity type, for the entities in this chunk
            entity_types = {}

            for entity_data in record.get("entities", []):
                name = entity_data.get("name")

                if not isinstance(name, str) or not name.strip():
                    continue

                entity_types[normalize_name(name)] = (
                    entity_data.get("type") or "UNKNOWN"
                )

            # Rewrite this chunk's facts from scratch.
            (
                session.query(GraphFactModel)
                .filter_by(chunk_id=chunk.id)
                .delete()
            )

            added = set()

            def add_fact(entity, relation=None, related_entity=None):
                entity = normalize_name(entity)
                related_entity = (
                    normalize_name(related_entity)
                    if related_entity
                    else None
                )

                key = (entity, relation, related_entity)

                if not entity or key in added:
                    return

                added.add(key)

                session.add(
                    GraphFactModel(
                        chunk_id=chunk.id,
                        entity=entity,
                        entity_type=entity_types.get(entity, "UNKNOWN"),
                        relation=relation,
                        related_entity=related_entity,
                    )
                )

            # 1. Entity mentions.
            for entity_name in entity_types:
                add_fact(entity_name)

            # 2. Relationships.
            for relation in record.get("relationships", []):
                source = relation.get("source")
                target = relation.get("target")
                relation_type = relation.get("type")

                if not isinstance(source, str) or not source.strip():
                    continue

                if not isinstance(target, str) or not target.strip():
                    continue

                if not isinstance(relation_type, str):
                    continue

                relation_type = relation_type.strip().upper()

                if not relation_type:
                    continue

                add_fact(source, relation_type, target)

            saved_records += 1

        session.commit()
        return saved_records

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()
