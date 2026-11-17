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
from app.services.generator import generate_answer, refine_user_query
from app.services.vector_store import (
    save_chunks_to_postgres,
    search_similar_chunks,
)

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


def ingestion(pdf_path: Path = DEFAULT_PDF) -> None:
    documents = get_context(pdf_path)
    chunks = chunk_text(documents)
    print(f"Extracted {len(documents)} pages and {len(chunks)} chunks from {pdf_path.name}.")

    embeddings = generate_embeddings(chunks)
    save_chunks_to_postgres(embeddings)


def retrieval_and_generation(
    question: str | None = None,
    limit: int = RETRIEVAL_LIMIT,
) -> None:
    question = (question or input("Enter your question : \n")).strip()
    if not question:
        raise SystemExit("No question provided.")

    search_text = refine_user_query(question) if REFINE_QUERY else question
    if REFINE_QUERY and search_text != question:
        print(f"\nSearch query: {search_text}")

    query_vector = generate_query_embedding(search_text)
    relevant_chunks = search_similar_chunks(query_vector, limit=limit)

    if not relevant_chunks:
        print(
            "\nNo chunks found. Run `python -m app.main ingest` first "
            "or lower the requested limit."
        )
        return

    print("\nRetrieved context:")
    for chunk in relevant_chunks:
        print(
            f"  - {chunk['source']} page {chunk['page']} "
            f"chunk {chunk['chunk_index']} (distance {chunk['distance']:.4f})"
        )

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

    ask = subparsers.add_parser("ask", help="ask a question about ingested data")
    ask.add_argument("question", nargs="?", help="question (prompted if omitted)")
    ask.add_argument("--limit", type=int, default=RETRIEVAL_LIMIT)

    subparsers.add_parser("check", help="check that the API key and chat model work")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = args.command or "ask"

    try:
        if command == "ingest":
            ingestion(args.pdf)
        elif command == "check":
            check_api_key()
        else:
            retrieval_and_generation(
                getattr(args, "question", None),
                getattr(args, "limit", RETRIEVAL_LIMIT),
            )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
