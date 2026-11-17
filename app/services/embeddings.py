import time

from google.genai import errors, types

from app.core.client import create_client
from app.core.config import EMBEDDING_DIM, EMBEDDING_MODEL

# Gemini Embedding 2 returns one aggregated embedding when several inputs are
# passed as plain strings. Wrapping each input in its own Content object asks
# the API for one embedding per input, so batches stay aligned with chunks.
EMBEDDING_BATCH_SIZE = 32
MAX_RATE_LIMIT_RETRIES = 3

# gemini-embedding-2 has no task_type parameter. Instead the task is expressed
# with a text prefix, and documents and queries must use matching formats.
DOCUMENT_PREFIX = "title: {title} | text: {text}"
QUERY_PREFIX = "task: question answering | query: {query}"


def _document_text(chunk: dict) -> str:
    metadata = chunk.get("metadata") or {}
    title = metadata.get("source") or "none"
    return DOCUMENT_PREFIX.format(title=title, text=chunk["content"])


def _embed_texts(client, texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[
            types.Content(parts=[types.Part.from_text(text=text)])
            for text in texts
        ],
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )

    embeddings = response.embeddings or []

    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding API returned {len(embeddings)} vectors for "
            f"{len(texts)} inputs"
        )

    vectors = []
    for position, embedding in enumerate(embeddings):
        values = embedding.values
        if values is None:
            raise RuntimeError(
                f"Embedding API returned no values for input {position}"
            )

        if len(values) != EMBEDDING_DIM:
            raise RuntimeError(
                f"Model {EMBEDDING_MODEL} returned {len(values)} dimensions, "
                f"expected {EMBEDDING_DIM}. Update EMBEDDING_DIM in "
                f"app/core/config.py."
            )

        # float() also guarantees plain JSON-friendly numbers for pgvector.
        vectors.append([float(value) for value in values])

    return vectors


def _status_code(exc: errors.APIError) -> int | None:
    return getattr(exc, "code", None)


def _retry_after_seconds(exc: errors.APIError, default: int = 30) -> int:
    try:
        value = exc.response.headers.get("retry-after")
        return max(1, min(65, int(float(value))))
    except (AttributeError, TypeError, ValueError):
        return default


def generate_embeddings(chunks: list[dict]) -> list[dict]:
    """Embed document chunks, shrinking batches if the API rejects them.

    Returns the input chunks with "embedding" and "embedding_model" added.
    """
    if not chunks:
        return []

    embedded_chunks = []
    total = len(chunks)
    batch_size = EMBEDDING_BATCH_SIZE
    start = 0
    rate_limit_retries = 0

    with create_client() as client:
        while start < total:
            batch = chunks[start:start + batch_size]

            try:
                vectors = _embed_texts(
                    client,
                    [_document_text(chunk) for chunk in batch],
                )
            except errors.APIError as exc:
                if (
                    _status_code(exc) == 429
                    and rate_limit_retries < MAX_RATE_LIMIT_RETRIES
                ):
                    delay = _retry_after_seconds(exc)
                    rate_limit_retries += 1
                    print(
                        f"Rate limited ({rate_limit_retries}/"
                        f"{MAX_RATE_LIMIT_RETRIES}); waiting {delay}s..."
                    )
                    time.sleep(delay)
                    continue

                # Free/limited tiers can reject large multi-input requests. Retry
                # the same chunks as smaller batches instead of failing the whole
                # ingest.
                status = _status_code(exc)
                if len(batch) == 1 or (status is not None and status < 500):
                    raise

                batch_size = max(1, len(batch) // 2)
                print(
                    f"Upstream error for a batch of {len(batch)} "
                    f"({status}); retrying with batch size {batch_size}"
                )
                continue

            for chunk, vector in zip(batch, vectors, strict=True):
                embedded_chunks.append({
                    **chunk,
                    "embedding": vector,
                    "embedding_model": EMBEDDING_MODEL,
                })

            start += len(batch)
            rate_limit_retries = 0
            print(f"Embedded {start} of {total} chunks.")

    return embedded_chunks


def generate_query_embedding(query: str) -> list[float]:
    """Embed a single search query using the matching retrieval task format."""
    text = QUERY_PREFIX.format(query=query)

    with create_client() as client:
        vectors = _embed_texts(client, [text])

    return vectors[0]
