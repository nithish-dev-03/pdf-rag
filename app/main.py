import argparse
from pathlib import Path

from pypdf import PdfReader

from app.core.client import create_client
from app.core.config import (
    CHAT_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PDF_PATH,
    REFINE_QUERY,
    RETRIEVAL_LIMIT,
)
from app.services.embeddings import (
    generate_embeddings,
    generate_query_embedding,
)
from app.services.generator import (
    generate_answer,
    refine_user_query,
    generate_hypothetical_document,
)
from app.services.vector_store import (
    save_chunks_to_postgres,
    search_similar_chunks,
)
from app.services.graph.extractor import extract_graph_data
from app.services.graph.graph_store import save_graph_data
from app.services.graph.graph_retriever import retrieve_graph_chunks

DEFAULT_PDF = PDF_PATH


def get_context(pdf_path: Path) -> list[dict]:
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    documents = []

    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)

        for page_number, page in enumerate(reader.pages, start=1):
            content = page.extract_text()

            if content and content.strip():
                documents.append({
                    "content": content,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                })

    return documents


def chunk_text(documents: list[dict]) -> list[dict]:
    chunks = []

    for document in documents:
        content = document["content"]
        metadata = document["metadata"]

        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + CHUNK_SIZE

            chunk = content[start:end].strip()

            if chunk:
                chunk_index += 1
                chunks.append({
                    "content": chunk,
                    "metadata": {
                        **metadata,
                        # Per-page index so (source, page, chunk) stays stable
                        # across re-ingests and is safe to deduplicate on.
                        "chunk": chunk_index,
                    },
                })

            if end >= len(content):
                break

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks



def load_chunks(pdf_path: Path) -> list[dict]:
    """Extract and chunk a PDF, returning [] when it has no usable text."""
    documents = get_context(pdf_path)
    chunks = chunk_text(documents)

    if not chunks:
        print("No text chunks found in the PDF.")

    return chunks


def ingest_vector_store(chunks: list[dict]) -> int:
    """Embed chunks and upsert them into PostgreSQL/pgvector."""
    embedded_chunks = generate_embeddings(chunks)
    saved_count = save_chunks_to_postgres(embedded_chunks)

    print(f"Saved {saved_count} vector chunks.")

    return saved_count


def ingest_knowledge_graph(chunks: list[dict]) -> int:
    """Extract entities/relationships per chunk and store them in the graph tables."""
    graph_records = []

    for index, chunk in enumerate(chunks, start=1):
        print(f"Extracting graph data: {index}/{len(chunks)}")

        metadata = chunk["metadata"]

        extracted = extract_graph_data(chunk["content"])

        graph_records.append({
            "source": metadata["source"],
            "page": metadata["page"],
            # chunk_text() stores the per-page index under "chunk"; the
            # vector store renames it to "chunk_index" when inserting rows.
            "chunk_index": metadata["chunk"],
            "entities": extracted["entities"],
            "relationships": extracted["relationships"],
        })

    graph_saved_count = save_graph_data(graph_records)

    print(
        f"Graph ingestion completed. "
        f"Processed {graph_saved_count} chunks."
    )

    if graph_records and graph_saved_count == 0:
        print(
            "Warning: no chunks were matched in the vector store. "
            "Run ingestion without --skip-embeddings first."
        )

    return graph_saved_count


def ingestion(
    pdf_path: Path = DEFAULT_PDF,
    skip_embeddings: bool = False,
    skip_graph: bool = False,
) -> None:
    """Run the ingestion stages, each of which can be skipped independently."""
    chunks = load_chunks(pdf_path)

    if not chunks:
        return

    if skip_embeddings:
        print("Skipping embeddings: using the chunks already in PostgreSQL.")
    else:
        ingest_vector_store(chunks)

    if skip_graph:
        print("Skipping graph extraction.")
    else:
        ingest_knowledge_graph(chunks)



def retrieval_and_generation(
    question: str | None = None,
    limit: int = RETRIEVAL_LIMIT,
    rag_type: str = "naive",
    refine: bool = REFINE_QUERY,
) -> None:

    # 1. Get user question
    question = (question or input("Enter your question : \n")).strip()

    if not question:
        raise SystemExit("No question provided.")

    # 2. Refine the original query (optional)
    search_text = refine_user_query(question) if refine else question

    if refine and search_text != question:
        print(f"\nSearch query: {search_text}")

    print(f"\nQuery Refinement: {'Enabled' if refine else 'Disabled'}")

    # 3. Select RAG strategy
    
    # 3. Select RAG strategy
    match rag_type:

        case "naive":
            print("\nRAG Strategy: Naive RAG")

            query_vector = generate_query_embedding(search_text)

        case "hyde":
            print("\nRAG Strategy: HyDE RAG")

            hypothetical_document = generate_hypothetical_document(
                search_text
            )

            print("\nGenerated Hypothetical Document:")
            print(hypothetical_document)

            query_vector = generate_query_embedding(
                hypothetical_document
            )

        case "graph":
            print("\nRAG Strategy: Graph RAG")

            relevant_chunks = retrieve_graph_chunks(
                search_text,
                limit=limit,
            )

        case _:
            raise ValueError(
                f"Unsupported RAG strategy: {rag_type}"
            )

    # 4. Vector retrieval for Naive and HyDE.
    if rag_type in ("naive", "hyde"):
        relevant_chunks = search_similar_chunks(
            query_vector,
            limit=limit,
        )

    if not relevant_chunks:
        print("\nNo relevant chunks found.")
        return

    # 5. Display retrieved context.
    print("\nRetrieved context:")

    for chunk in relevant_chunks:
        distance = chunk.get("distance")

        retrieval_info = (
            f"(distance {distance:.4f})"
            if distance is not None
            else f"(graph score {chunk.get('graph_score', 0)})"
        )

        print(
            f"  - {chunk['source']} page {chunk['page']} "
            f"chunk {chunk['chunk_index']} "
            f"{retrieval_info}"
        )

    # 6. Generate answer using original user question.
    print("\nGenerating answer...")

    answer = generate_answer(question, relevant_chunks)

    print("\n--- Final Answer ---")
    print(answer)



def check_api_key() -> None:
    with create_client() as client:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents="Reply with the single word: OK",
        )

    print(response.text or "(empty response)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAG over a PDF using the Google Gen AI SDK",
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", help="chunk, embed and store a PDF")
    ingest.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    ingest.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="reuse the chunks already stored in PostgreSQL (graph extraction only)",
    )
    ingest.add_argument(
        "--skip-graph",
        action="store_true",
        help="only embed and store chunks, without extracting graph data",
    )

    ask = subparsers.add_parser("ask", help="ask a question about ingested data")
    ask.add_argument("question", nargs="?", help="question (prompted if omitted)")
    ask.add_argument("--limit", type=int, default=RETRIEVAL_LIMIT)
    ask.add_argument(
        "--refine",
        type=lambda value: value.lower() == "true",
        choices=[True, False],
        default=REFINE_QUERY,
        help="Enable or disable query refinement",
    )

    # RAG strategy selection
    ask.add_argument(
        "--rag",
        choices=["naive", "hyde", "graph"],
        default="naive",
        help="RAG retrieval strategy",
    )
    subparsers.add_parser("check", help="check that the API key and chat model work")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = args.command or "ask"

    try:
        if command == "ingest":
            ingestion(
                args.pdf,
                skip_embeddings=args.skip_embeddings,
                skip_graph=args.skip_graph,
            )

        elif command == "check":
            check_api_key()

        else:
            retrieval_and_generation(
                question=getattr(args, "question", None),
                limit=getattr(args, "limit", RETRIEVAL_LIMIT),
                rag_type=getattr(args, "rag", "naive"),
                refine=getattr(args, "refine", REFINE_QUERY),
            )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
