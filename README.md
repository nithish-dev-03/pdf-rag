# pdf-rag-gemini — RAG over a PDF with the Google Gen AI SDK

A command-line Retrieval-Augmented Generation pipeline that ingests a PDF into
PostgreSQL/pgvector and answers questions about it with Gemini, using the
official `google-genai` SDK.

```
ingest:  PDF -> text per page -> 500-char chunks -> embeddings -> PostgreSQL
ask:     question -> (optional rewrite) -> query embedding -> cosine search -> Gemini answer
```

## Architecture at a glance

The system is two pipelines that meet at one table. Ingestion writes vectors;
asking reads them. Nothing about a question is stored, and nothing about the
document is re-embedded at query time.

```mermaid
flowchart LR
    subgraph ING["① Ingestion — python -m app.main ingest"]
        direction TB
        PDF["PDF<br/>app/data/*.pdf"]
        EXT["get_context()<br/>pypdf: text per page"]
        CHUNK["chunk_text()<br/>500 chars, 100 overlap"]
        EMB["generate_embeddings()<br/>gemini-embedding-2, batch 32"]
        UPS["save_chunks_to_postgres()<br/>idempotent upsert"]

        PDF --> EXT --> CHUNK --> EMB --> UPS
    end

    subgraph DB["② Storage — PostgreSQL 16 + pgvector"]
        TBL[("document_chunks_gemini<br/>id · content · embedding vector(1536)<br/>embedding_model · source · page · chunk_index")]
    end

    subgraph ASK["③ Ask — python -m app.main ask"]
        direction TB
        Q["Question"]
        REW["refine_user_query()<br/>optional standalone rewrite"]
        QEMB["generate_query_embedding()<br/>question-answering prefix"]
        SEARCH["search_similar_chunks()<br/>cosine distance, top-k"]
        GEN["generate_answer()<br/>gemini-3.5-flash-lite"]
        ANS["Answer + page citations"]

        Q --> REW --> QEMB --> SEARCH --> GEN --> ANS
    end

    UPS --> TBL
    TBL --> SEARCH
```

### Ingestion pipeline

```mermaid
flowchart TD
    A["PDF file"] --> B["open + PdfReader"]
    B --> C["page.extract_text()<br/>skip empty pages"]
    C --> D["metadata: source, page"]
    D --> E["sliding window<br/>CHUNK_SIZE=500, CHUNK_OVERLAP=100"]
    E --> F["metadata: + per-page chunk index"]
    F --> G["embedding text prefix<br/>title: source | text: chunk"]
    G --> H["embed_content()<br/>EMBEDDING_DIM=1536"]
    H --> I{"batch error?"}
    I -- "429 rate limit" --> J["sleep retry-after<br/>up to 3 times"]
    J --> H
    I -- "5xx / batch too large" --> K["halve batch size"]
    K --> H
    I -- "ok" --> L["INSERT ... ON CONFLICT DO NOTHING<br/>(source, page, chunk_index)"]
    L --> M[("PostgreSQL<br/>document_chunks_gemini")]
```

### Ask pipeline (retrieval → generation)

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant CLI as app.main
    participant G as generator.py
    participant E as embeddings.py
    participant V as vector_store.py
    participant PG as PostgreSQL/pgvector
    participant GM as Gemini API

    U->>CLI: ask "question" --limit k
    alt REFINE_QUERY = True
        CLI->>G: refine_user_query(question)
        G->>GM: rewrite prompt
        GM-->>G: standalone search query
        G-->>CLI: refined query
    end
    CLI->>E: generate_query_embedding(search_text)
    E->>GM: embed "task: question answering | query: ..."
    GM-->>E: vector(1536)
    E-->>CLI: query vector
    CLI->>V: search_similar_chunks(vector, limit=k)
    V->>PG: ORDER BY embedding <=> query<br/>WHERE embedding_model = ...
    PG-->>V: k nearest chunks + distance
    V-->>CLI: list of chunks
    CLI->>G: generate_answer(question, chunks)
    G->>GM: context + question
    GM-->>G: answer
    G-->>CLI: answer text
    CLI-->>U: retrieved context + answer
```

## What the retrieval limit does

`RETRIEVAL_LIMIT` (default `5`, set in `.env`, overridable per call with
`--limit`) is the **k in top-k retrieval**. It answers one question:

> How many of the nearest chunks are pasted into the prompt?

It is a query-time knob only. It does **not** change chunking, embeddings, or
what is stored in the database. It also does **not** act as a quality filter —
there is no similarity threshold, so search always returns exactly `k` chunks
(unless fewer rows exist), best-first by cosine distance.

The limit sets the ceiling on what the model can know. If the sentence that
answers the question lives in a chunk that ranks `k+1`, it is invisible to the
model, and the model will correctly say it cannot find the answer.

### Worked example from this repository

Question: *"I planned to take casual leave on Friday and Monday continuously —
will it be LOP?"* The rule is in **page 3, chunk 1** ("Weekends and any holiday
lying between the sanctioned leave periods will be included and be counted as
leave/LOP…"). Measured cosine distances:

| Rank | Chunk | Distance | Verdict with `--limit 2` |
|---|---|---|---|
| 1 | page 4, chunk 1 — Paternity Leave | 0.2928 | sent |
| 2 | page 4, chunk 3 — LOP pay rules | 0.3006 | sent |
| 3 | **page 3, chunk 1 — weekend rule** | **0.3023** | **dropped** |

Because rank 2 beat rank 3 by 0.0017, the only chunk containing the answer was
evicted, and the answer was *"I cannot find the answer in the provided
document."* With `--limit 4`, page 3 chunk 1 is included and the model answers
correctly, citing page 3.

### Trade-off

| Limit | Effect |
|---|---|
| Too low | Recall suffers: the answering chunk is cut off. Answers become "I cannot find…" even though the document covers it. |
| Balanced (`4–5` for this PDF) | Covers the neighbourhood around the question without flooding the prompt. |
| Too high | Context dilution: unrelated chunks compete for attention, token cost and latency rise, and answers can drift. |

Raising `k` is a blunt fix. Better recall comes from coherent sentence-based
chunks, whitespace normalisation, and **hybrid search** (pgvector + Postgres
full-text) so exact terms such as `LOP`, `Casual Leave`, and `weekend` are not
missed by dense similarity alone.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- PostgreSQL 16 with the `pgvector` extension (the project creates the
  extension and table automatically if the DB user is allowed to do so)
- A Google AI Studio / Gemini API key

The `DocumentChunkModel` table is created on the first ingest. It is named
`document_chunks_gemini`, so this project can share a database with other
ingest projects without vector-dimension conflicts.

> **Note:** if you get `Could not enable the PostgreSQL 'vector' extension`, the
> message may be masking a different failure. Verify the database itself exists:
>
> ```bash
> psql "postgresql://postgres:password@localhost:5432/postgres" \
>   -c "SELECT datname FROM pg_database;"
> psql "postgresql://postgres:password@localhost:5432/postgres" \
>   -c "CREATE DATABASE doc_analyser;"
> ```

## Setup

```bash
cd pdf-rag-gemini
uv sync
cp .env.example .env
# edit .env and set GOOGLE_API_KEY=...
```

`.env`:

```dotenv
GOOGLE_API_KEY=your-key-here
DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5432/doc_analyser

# Relative paths resolve against the project root; absolute paths also work.
PDF_PATH=app/data/ms-leave-policy.pdf

# Top-k chunks retrieved per question.
RETRIEVAL_LIMIT=5
```

`GEMINI_API_KEY` is also accepted if that is what you already have.

`PDF_PATH` is portable across operating systems. Use forward slashes on every
platform (`C:/Users/me/docs/policy.pdf` works on Windows), or an absolute POSIX
path (`/home/me/docs/policy.pdf`). `~` and environment variables such as
`$HOME` / `%USERPROFILE%` are expanded, and relative paths are anchored to the
project root, so the CLI behaves the same no matter which directory you launch
it from. The `--pdf` flag still overrides `PDF_PATH` for a single run.

## Usage

Every command has a plain-Python form and a uv form. Use whichever environment
you already have:

- **Python** — activate the project virtualenv first
  (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows).
- **uv** — resolves the virtualenv automatically; no activation needed.
  `uv run python -m app.main ...` also works if you prefer the module form.

```bash
# sanity-check the key and chat model
python -m app.main check
uv run pdf-rag check

# chunk + embed + store the PDF from PDF_PATH (default: ms-leave-policy.pdf)
python -m app.main ingest
uv run pdf-rag ingest

# ingest a different PDF for one run
python -m app.main ingest --pdf app/data/rag_chunking_sample.pdf
uv run pdf-rag ingest --pdf app/data/rag_chunking_sample.pdf

# ask a question (interactive prompt if omitted)
python -m app.main ask "How many annual leave days do employees get?"
uv run pdf-rag ask "How many annual leave days do employees get?"

python -m app.main ask
uv run pdf-rag ask

# retrieve more than the default 5 chunks
python -m app.main ask "What is the policy on probation?" --limit 4
uv run pdf-rag ask "What is the policy on probation?" --limit 4
```

Command reference:

| Task | Python | uv |
|---|---|---|
| Check API key / model | `python -m app.main check` | `uv run pdf-rag check` |
| Ingest default PDF | `python -m app.main ingest` | `uv run pdf-rag ingest` |
| Ingest a specific PDF | `python -m app.main ingest --pdf <path>` | `uv run pdf-rag ingest --pdf <path>` |
| Ask a question | `python -m app.main ask "<question>"` | `uv run pdf-rag ask "<question>"` |
| Ask interactively | `python -m app.main ask` | `uv run pdf-rag ask` |
| Override top-k | `python -m app.main ask "<question>" --limit 4` | `uv run pdf-rag ask "<question>" --limit 4` |

Ingestion is idempotent on `(source, page, chunk_index)`, so re-running it does
not duplicate rows. The bundled policy PDF produces 6 pages / 33 chunks. The
progress line `Embedded 32 of 33 chunks.` is just batch reporting, not an error.

## Configuration

All settings live in `app/core/config.py`.

| Setting | Value | Notes |
|---|---|---|
| `CHAT_MODEL` | `gemini-3.5-flash-lite` | stable, low-cost chat model |
| `EMBEDDING_MODEL` | `gemini-embedding-2` | multimodal embeddings model |
| `EMBEDDING_DIM` | `1536` | recommended size; must match the DB column |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | characters |
| `TABLE_NAME` | `document_chunks_gemini` | dedicated table, safe to share a database |
| `REFINE_QUERY` | `True` | rewrite the question with the chat model before embedding |

Environment knobs (read by `app/core/config.py`):

| Variable | Default | Notes |
|---|---|---|
| `PDF_PATH` | `app/data/ms-leave-policy.pdf` | ingest target; relative paths resolve against the project root, `~`/env vars expanded, forward slashes work on all OSes |
| `RETRIEVAL_LIMIT` | `5` | top-k chunks for `ask`; must be a positive integer; `--limit` overrides it per call |

`gemini-embedding-2` does not use a `task_type` parameter. Instead, documents
are prefixed with `title: <source> | text: <content>` and queries with
`task: question answering | query: <question>`, as recommended in the Gemini
embeddings docs. `output_dimensionality=1536` truncates and re-normalizes the
vector, so cosine distance remains valid.

The Gen AI client uses a 120 s timeout with bounded retries (4 attempts,
exponential backoff + jitter) on `429/500/502/503/504`. Embedding batches of 32
are halved automatically when the API rejects them, and 429s honour
`retry-after` up to three times.

## Data model

One row per chunk in `document_chunks_gemini`:

| Column | Type | Purpose |
|---|---|---|
| `id` | `integer` PK | surrogate key |
| `content` | `text` | raw chunk text (with PDF whitespace) |
| `embedding` | `vector(1536)` | document embedding |
| `embedding_model` | `text` | guards against mixing vector spaces |
| `source` | `text` | file name |
| `page` | `integer` | 1-based page number |
| `chunk_index` | `integer` | 1-based index within the page |

Unique constraint `(source, page, chunk_index)` makes re-ingest idempotent via
`ON CONFLICT DO NOTHING`. Search filters on `embedding_model` so vectors from
different models are never compared.

## Project layout

```
pdf-rag-gemini/
├── app/
│   ├── core/
│   │   ├── client.py      # Gen AI client + bounded retry/timeout config
│   │   └── config.py      # models, dimensions, chunking, table name
│   ├── services/
│   │   ├── embeddings.py  # batched embeddings, query embeddings
│   │   ├── generator.py   # query rewrite + answer generation
│   │   └── vector_store.py# pgvector schema, idempotent upsert, cosine search
│   ├── data/              # source PDFs
│   └── main.py            # CLI orchestration
├── .env.example
├── pyproject.toml
└── uv.lock
```

## Notes and limitations

- **No similarity threshold.** Search always returns `--limit` chunks. Lower the
  limit (or add a maximum cosine distance) if answers drift off-topic.
- **No ANN index.** Cosine distance is computed exactly; fine for a handful of
  PDFs, but add an HNSW/IVFFlat index before scaling up.
- **Chunking is character-based** and can split mid-word/mid-sentence (e.g.
  chunks starting `"anagement is mandatory"`). Swap in a token/sentence splitter
  for better retrieval quality.
- **PDF whitespace is not normalised.** `pypdf` emits one word per line for this
  document, which dilutes chunk density and adds embedding noise.
- **Dense-only retrieval.** Purely semantic search can miss literal terms; add
  hybrid (full-text/trigram) ranking for policy-style documents.
- **Embedding-model changes require a re-ingest.** Vectors from different
  models are not comparable; the search filters on `embedding_model`, but old
  rows should still be cleared or re-embedded.
- **`REFINE_QUERY`** adds one chat request per question. Set it to `False` to
  embed the user's wording verbatim if the rewrite ever hurts recall.
- The answer prompt instructs the model to reply "I cannot find the answer in
  the provided document." when the retrieved context is insufficient.
