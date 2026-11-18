from sqlalchemy import or_

from app.services.graph.extractor import extract_query_entities
from app.services.graph.graph_store import GraphFactModel, normalize_name
from app.services.vector_store import DocumentChunkModel, SessionLocal

# A chunk that contains a matched entity is the strongest signal; a chunk
# that contains one of its direct neighbours is the weakest.
DIRECT_MENTION_SCORE = 3
NEIGHBOUR_SCORE = 1


def _matching_entities(session, query_entities: list[str]) -> set[str]:
    """Find graph entity names that match the entities of the question."""

    matched = set()

    for raw_name in query_entities:
        name = normalize_name(raw_name)

        if not name:
            continue

        # Entities that match directly.
        rows = (
            session.query(GraphFactModel.entity)
            .filter(GraphFactModel.entity.ilike(f"%{name}%"))
            .distinct()
            .all()
        )

        matched.update(row[0] for row in rows)

        # Entities that only appear as the target of a relationship.
        rows = (
            session.query(GraphFactModel.related_entity)
            .filter(
                GraphFactModel.related_entity.isnot(None),
                GraphFactModel.related_entity.ilike(f"%{name}%"),
            )
            .distinct()
            .all()
        )

        matched.update(row[0] for row in rows)

    return matched


def retrieve_graph_chunks(
    question: str,
    limit: int = 5,
) -> list[dict]:
    """Retrieve chunks by matching entities and walking one hop."""

    if limit <= 0:
        raise ValueError("Retrieval limit must be greater than zero.")

    query_entities = extract_query_entities(question)

    if not query_entities:
        print("No entities extracted from the question.")
        return []

    session = SessionLocal()

    try:
        # 1. Match question entities to entity names in the graph.
        seed_entities = _matching_entities(session, query_entities)

        if not seed_entities:
            print("No matching entities found in the knowledge graph.")
            return []

        print("\nMatched graph entities:")

        for name in sorted(seed_entities):
            print(f"  - {name}")

        # 2. Chunks that mention a matched entity are the direct hits.
        #    A chunk that mentions several matched entities scores higher,
        #    because it is likely to be the chunk that talks about their
        #    relationship (e.g. "sick leave cannot be clubbed with casual
        #    leave").
        chunk_scores = {}

        direct_rows = (
            session.query(
                GraphFactModel.chunk_id,
                GraphFactModel.entity,
                GraphFactModel.related_entity,
            )
            .filter(
                or_(
                    GraphFactModel.entity.in_(seed_entities),
                    GraphFactModel.related_entity.in_(seed_entities),
                )
            )
            .all()
        )

        matched_per_chunk = {}

        for chunk_id, entity, related_entity in direct_rows:
            names = matched_per_chunk.setdefault(chunk_id, set())

            if entity in seed_entities:
                names.add(entity)

            if related_entity in seed_entities:
                names.add(related_entity)

        for chunk_id, names in matched_per_chunk.items():
            chunk_scores[chunk_id] = (
                DIRECT_MENTION_SCORE + len(names) - 1
            )

        # 3. One hop: follow relationships that touch a matched entity.
        edges = (
            session.query(
                GraphFactModel.chunk_id,
                GraphFactModel.entity,
                GraphFactModel.related_entity,
            )
            .filter(
                GraphFactModel.relation.isnot(None),
                or_(
                    GraphFactModel.entity.in_(seed_entities),
                    GraphFactModel.related_entity.in_(seed_entities),
                ),
            )
            .all()
        )

        neighbours = set()
        connecting_chunks = set()

        for chunk_id, entity, related_entity in edges:
            if entity in seed_entities and related_entity:
                neighbours.add(related_entity)

            if related_entity in seed_entities and entity:
                neighbours.add(entity)

            # A relationship between two matched entities is the strongest
            # signal: the chunk states how they are connected.
            if (
                entity in seed_entities
                and related_entity in seed_entities
            ):
                connecting_chunks.add(chunk_id)

        for chunk_id in connecting_chunks:
            chunk_scores[chunk_id] = (
                chunk_scores.get(chunk_id, DIRECT_MENTION_SCORE) + 1
            )

        neighbours -= seed_entities

        # 4. Chunks that mention a neighbour are weaker evidence.
        if neighbours:
            neighbour_rows = (
                session.query(GraphFactModel.chunk_id)
                .filter(
                    or_(
                        GraphFactModel.entity.in_(neighbours),
                        GraphFactModel.related_entity.in_(neighbours),
                    )
                )
                .all()
            )

            for row in neighbour_rows:
                chunk_scores.setdefault(row.chunk_id, NEIGHBOUR_SCORE)

        if not chunk_scores:
            print("No supporting chunks found in the graph.")
            return []

        # 5. Direct hits first, then neighbours; ties keep document order.
        ranked_chunk_ids = sorted(
            chunk_scores,
            key=lambda chunk_id: (-chunk_scores[chunk_id], chunk_id),
        )[:limit]

        # 6. Load the original chunks.
        chunks = (
            session.query(DocumentChunkModel)
            .filter(DocumentChunkModel.id.in_(ranked_chunk_ids))
            .all()
        )

        chunk_map = {chunk.id: chunk for chunk in chunks}

        results = []

        for chunk_id in ranked_chunk_ids:
            chunk = chunk_map.get(chunk_id)

            if not chunk:
                continue

            results.append({
                "content": chunk.content,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "distance": None,
                "graph_score": chunk_scores[chunk_id],
            })

        return results

    finally:
        session.close()
