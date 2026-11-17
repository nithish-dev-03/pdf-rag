from app.services.embeddings import generate_query_embedding
from app.services.generator import generate_hypothetical_document
from app.services.vector_store import search_similar_chunks


def retrieve_naive(query: str, limit: int):
    query_embedding = generate_query_embedding(query)

    return search_similar_chunks(
        query_embedding=query_embedding,
        limit=limit,
    )


def retrieve_hyde(query: str, limit: int):
    hypothetical_document = generate_hypothetical_document(query)

    query_embedding = generate_query_embedding(hypothetical_document)

    return search_similar_chunks(
        query_embedding=query_embedding,
        limit=limit,
    )


def retrieve_chunks(query: str, rag_type: str, limit: int):
    match rag_type:
        case "naive":
            return retrieve_naive(query, limit)

        case "hyde":
            return retrieve_hyde(query, limit)

        case _:
            raise ValueError(f"Unsupported RAG strategy: {rag_type}")