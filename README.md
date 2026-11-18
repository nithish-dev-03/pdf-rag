# pdf-rag-gemini — RAG over a PDF with the Google Gen AI SDK

A command-line Retrieval-Augmented Generation pipeline that ingests a PDF into
PostgreSQL/pgvector, extracts a lightweight knowledge graph with Gemini, and
answers questions about it — using the official `google-genai` SDK.

```
ingest:  PDF -> text per page -> 500-char chunks -> embeddings ----------> document_chunks_gemini
                                              \-> entities + relationships -> graph_facts
ask:     question -> (optional rewrite) -> naive | HyDE | graph -> top-k chunks -> Gemini answer
```

Three retrieval strategies ship with the CLI: **naive RAG** (embed the
question), **HyDE** (embed a hypothetical answer passage instead of the
question), and **graph** (match entities and walk one relationship hop).

## Architecture at a glance

The system is two pipelines that meet at one database. Ingestion writes vectors
and graph rows; asking reads either the vectors or the graph. Nothing about a
question is stored, and nothing about the document is re-embedded at query time.

```mermaid
flowchart LR
    subgraph ING["① Ingestion — pdf-rag ingest"]
        direction TB
        PDF["PDF<br/>app/data/*.pdf"]
        EXT["get_context()<br/>pypdf: text per page"]
        CHUNK["chunk_text()<br/>500 chars, 100 overlap"]
        EMB["generate_embeddings()<br/>gemini-embedding-2, batch 32"]
        UPS["save_chunks_to_postgres()<br/>idempotent upsert"]
        EXTRACT["extract_graph_data()<br/>entities + relationships per chunk"]
        GSAVE["save_graph_data()<br/>delete + rewrite a chunk's facts"]

        PDF --> EXT --> CHUNK --> EMB --> UPS
        CHUNK --> EXTRACT --> GSAVE
    end

    subgraph DB["② Storage — PostgreSQL 16 + pgvector"]
        TBL[("document_chunks_gemini<br/>id · content · embedding vector(1536)<br/>embedding_model · source · page · chunk_index")]
        GTBL[("graph_facts<br/>chunk_id · entity · entity_type<br/>relation · related_entity")]
    end

    subgraph ASK["③ Ask — pdf-rag ask"]
        direction TB
        Q["Question"]
        REW["refine_user_query()<br/>optional standalone rewrite"]
        STRAT{"--rag"}
        NAIVE["naive<br/>embed the question"]
        HYDE["hyde<br/>embed a hypothetical passage"]
        GRAPH["graph<br/>match entity names<br/>+ one hop"]
        SEARCH["search_similar_chunks()<br/>cosine distance, top-k"]
        GEN["generate_answer()<br/>gemini-3.5-flash-lite"]
        ANS["Answer + page citations"]

        Q --> REW --> STRAT
        STRAT -->|naive| NAIVE --> SEARCH
        STRAT -->|hyde| HYDE --> SEARCH
        STRAT -->|graph| GRAPH --> SEARCH
        SEARCH --> GEN --> ANS
    end

    UPS --> TBL
    GSAVE --> GTBL
    GTBL --> GRAPH
    TBL --> SEARCH
```

The graph is stored in a single flat table, `graph_facts`, and is queried by
`--rag graph`: entity matching finds the chunks that mention the question's
entities, and one relationship hop adds their neighbours.

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

    F --> N["extract_graph_data()<br/>Gemini JSON extraction per chunk"]
    N --> O["entities + relationships<br/>deduplicated by canonical name"]
    O --> P[("graph_facts<br/>one row per fact")]
```

Both stages are independent: `--skip-embeddings` runs only the graph stage
against chunks already in the vector table, and `--skip-graph` runs only the
embedding stage.

### Ask pipeline (retrieval → generation)

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant CLI as app.main
    participant G as generator.py
    participant E as embeddings.py
    participant V as vector_store.py
    participant R as graph_retriever.py
    participant PG as PostgreSQL/pgvector
    participant GM as Gemini API

    U->>CLI: ask "question" --limit k --rag naive|hyde|graph --refine true|false
    alt refine = true
        CLI->>G: refine_user_query(question)
        G->>GM: rewrite prompt
        GM-->>G: standalone search query
        G-->>CLI: refined query
    end
    alt rag = graph
        CLI->>R: retrieve_graph_chunks(search_text, limit=k)
        R->>GM: extract_query_entities(question)
        GM-->>R: entity names
        R->>PG: match graph_facts<br/>direct mentions (score 3) + one hop (score 1)
        PG-->>R: ranked chunk ids
        R-->>CLI: chunks + graph_score
    else rag = hyde
        CLI->>G: generate_hypothetical_document(search_text)
        G->>GM: hypothetical passage prompt
        GM-->>G: hypothetical policy passage
        G-->>CLI: passage
        CLI->>E: generate_query_embedding(passage)
        E->>GM: embed "task: question answering | query: ..."
        GM-->>E: vector(1536)
        E-->>CLI: query vector
        CLI->>V: search_similar_chunks(vector, limit=k)
        V->>PG: ORDER BY embedding <=> query<br/>WHERE embedding_model = ...
        PG-->>V: k nearest chunks + distance
        V-->>CLI: list of chunks
    else rag = naive
        CLI->>E: generate_query_embedding(search_text)
        E->>GM: embed "task: question answering | query: ..."
        GM-->>E: vector(1536)
        E-->>CLI: query vector
        CLI->>V: search_similar_chunks(vector, limit=k)
        V->>PG: ORDER BY embedding <=> query<br/>WHERE embedding_model = ...
        PG-->>V: k nearest chunks + distance
        V-->>CLI: list of chunks
    end
    CLI->>G: generate_answer(question, chunks)
    G->>GM: context + question
    GM-->>G: answer
    G-->>CLI: answer text
    CLI-->>U: retrieved context + answer
```

### Retrieval strategies

| Strategy | Query representation | Retrieval signal | Extra chat calls | Best for |
|---|---|---|---|---|
| `naive` (default) | the question (optionally rewritten) | cosine distance | 0 | direct, keyword-like questions |
| `hyde` | a hypothetical policy passage the model writes about the question | cosine distance | 1 | vague or short questions whose wording differs from the document |
| `graph` | entity names extracted from the question | exact name match + one relationship hop | 1 | questions about named things and how they relate |

HyDE trades latency for recall: a passage-shaped query sits closer to document
passages in embedding space than a terse question does. It also trades accuracy
for recall — a hallucinated passage can pull retrieval toward the wrong topic,
so `--rag naive` remains the default.

Graph retrieval ignores embeddings entirely. Scores are simple to reason
about:

| Score | Meaning |
|---|---|
| `3` | the chunk mentions a matched entity (`+1` for every extra matched entity in the same chunk) |
| `+1` | the chunk contains a relationship that connects **two** matched entities |
| `1` | the chunk mentions a one-hop neighbour of a matched entity |

Ties keep document order.



## What the retrieval limit does

`RETRIEVAL_LIMIT` (default `5`, set in `.env`, overridable per call with
`--limit`) is the **k in top-k retrieval**. It answers one question:

> How many of the nearest chunks are pasted into the prompt?

It is a query-time knob only. It does **not** change chunking, embeddings, or
what is stored in the database. It also does **not** act as a quality filter —
there is no similarity threshold, so search always returns exactly `k` chunks
(unless fewer rows exist), best-first by cosine distance.

With `--rag graph` the same `--limit` still caps the number of chunks sent to
the model, but ranking is by `graph_score` (direct mention vs one hop) rather
than distance.


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
chunks, whitespace normalisation, **HyDE** (`--rag hyde`), and **hybrid search**
(pgvector + Postgres full-text) so exact terms such as `LOP`, `Casual Leave`, and
`weekend` are not missed by dense similarity alone.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- PostgreSQL 16 with the `pgvector` extension (the project creates the
  extension and table automatically if the DB user is allowed to do so)
- A Google AI Studio / Gemini API key

The `DocumentChunkModel` table is created on the first ingest. It is named
`document_chunks_gemini`, so this project can share a database with other
ingest projects without vector-dimension conflicts. The graph table,
`graph_facts`, is created on the first graph ingest, and every ingest also
creates the `vector` extension if it is missing.

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
# this also extracts entities/relationships and fills the graph tables
python -m app.main ingest
uv run pdf-rag ingest

# only re-extract the graph, reusing chunks already stored in PostgreSQL
python -m app.main ingest --skip-embeddings
uv run pdf-rag ingest --skip-embeddings

# only embed, without spending a chat call per chunk on graph extraction
python -m app.main ingest --skip-graph
uv run pdf-rag ingest --skip-graph

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

# HyDE retrieval: embed a hypothetical answer passage instead of the question
python -m app.main ask "Can I take Friday and Monday off?" --rag hyde
uv run pdf-rag ask "Can I take Friday and Monday off?" --rag hyde

# graph retrieval: match entity names and follow one relationship hop
python -m app.main ask "Can casual leave be clubbed with sick leave?" --rag graph
uv run pdf-rag ask "Can casual leave be clubbed with sick leave?" --rag graph

# disable query refinement for a single question
python -m app.main ask "What is LOP?" --refine false
uv run pdf-rag ask "What is LOP?" --refine false
```

Command reference:

| Task | Python | uv |
|---|---|---|
| Check API key / model | `python -m app.main check` | `uv run pdf-rag check` |
| Ingest default PDF | `python -m app.main ingest` | `uv run pdf-rag ingest` |
| Ingest a specific PDF | `python -m app.main ingest --pdf <path>` | `uv run pdf-rag ingest --pdf <path>` |
| Graph only (no re-embedding) | `python -m app.main ingest --skip-embeddings` | `uv run pdf-rag ingest --skip-embeddings` |
| Embeddings only (no graph) | `python -m app.main ingest --skip-graph` | `uv run pdf-rag ingest --skip-graph` |
| Ask a question | `python -m app.main ask "<question>"` | `uv run pdf-rag ask "<question>"` |
| Ask interactively | `python -m app.main ask` | `uv run pdf-rag ask` |
| Override top-k | `python -m app.main ask "<question>" --limit 4` | `uv run pdf-rag ask "<question>" --limit 4` |
| Choose retrieval strategy | `python -m app.main ask "<question>" --rag hyde` | `uv run pdf-rag ask "<question>" --rag hyde` |
| Graph retrieval | `python -m app.main ask "<question>" --rag graph` | `uv run pdf-rag ask "<question>" --rag graph` |
| Toggle query refinement | `python -m app.main ask "<question>" --refine false` | `uv run pdf-rag ask "<question>" --refine false` |

Vector ingestion is idempotent on `(source, page, chunk_index)`, so re-running it
does not duplicate rows. The bundled policy PDF produces 6 pages / 33 chunks. The
progress line `Embedded 32 of 33 chunks.` is just batch reporting, not an error.

Graph extraction is **not** free: every run makes one chat request per chunk
(33 for the bundled PDF). The writer deletes and rewrites a chunk's facts before
inserting, so re-running never duplicates rows — it only costs API calls. Use
`--skip-embeddings` when you only want to rebuild the graph, and `--skip-graph`
when you only want vectors.

## Configuration

All settings live in `app/core/config.py`.

| Setting | Value | Notes |
|---|---|---|
| `CHAT_MODEL` | `gemini-3.5-flash-lite` | stable, low-cost chat model |
| `EMBEDDING_MODEL` | `gemini-embedding-2` | multimodal embeddings model |
| `EMBEDDING_DIM` | `1536` | recommended size; must match the DB column |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | characters |
| `TABLE_NAME` | `document_chunks_gemini` | dedicated table, safe to share a database |
| `REFINE_QUERY` | `True` | default for query rewriting; `--refine true/false` overrides per call |

Graph settings are fixed in code (`app/services/graph/extractor.py`,
`graph_store.py`): `gemini-3.5-flash-lite` at `temperature=0` with JSON response
mode, entity names normalised to lowercase/collapsed whitespace, relationship
types upper-cased. `graph_facts` is created on the first graph ingest.

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

### Graph table

The whole knowledge graph lives in **one** flat table, `graph_facts`:

| Column | Type | Purpose |
|---|---|---|
| `id` | `integer` PK | surrogate key |
| `chunk_id` | `integer` FK → `document_chunks_gemini.id` | the chunk the fact came from |
| `entity` | `text` | subject of the fact, lowercased |
| `entity_type` | `text` | e.g. `LEAVE_TYPE`, `ROLE`, `UNKNOWN` |
| `relation` | `text NULL` | edge label; `NULL` when the row is a plain mention |
| `related_entity` | `text NULL` | edge target; `NULL` when the row is a plain mention |

Each row says *"this chunk talks about `entity`"*. When `relation` is filled it
also says *"and `entity` is connected to `related_entity`"*:

| chunk_id | entity | entity_type | relation | related_entity |
|---|---|---|---|---|
| 13 | casual leave | LEAVE_TYPE | NULL | NULL |
| 13 | sick leave | LEAVE_TYPE | CANNOT_BE_CLUBBED_WITH | casual leave |

Names are normalised (`strip().lower()` + collapsed whitespace), so
"Casual Leave" and "casual  leave" match. No joins are needed: retrieval just
filters the two text columns. The writer deletes a chunk's rows and rewrites
them, which keeps re-ingestion idempotent. A record is skipped if its
`(source, page, chunk_index)` is not in `document_chunks_gemini`.

That makes **two tables in total** for the whole project: `document_chunks_gemini`
for vectors and `graph_facts` for the graph.

## Project layout

```
pdf-rag-gemini/
├── app/
│   ├── core/
│   │   ├── client.py      # Gen AI client + bounded retry/timeout config
│   │   └── config.py      # models, dimensions, chunking, table name
│   ├── services/
│   │   ├── embeddings.py  # batched embeddings, query embeddings
│   │   ├── generator.py   # query rewrite, HyDE passages, answer generation
│   │   ├── retriever.py   # naive / HyDE retrieval strategies
│   │   ├── vector_store.py# pgvector schema, idempotent upsert, cosine search
│   │   └── graph/
│   │       ├── extractor.py      # per-chunk entity/relationship extraction
│   │       ├── graph_store.py    # graph_facts schema + writer
│   │       └── graph_retriever.py# entity match + one-hop retrieval
│   ├── data/              # source PDFs (git-ignored)
│   └── main.py            # CLI orchestration (ingest / ask / check)
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
- **`REFINE_QUERY`** adds one chat request per question. Set it to `False` (or
  pass `--refine false`) to embed the user's wording verbatim if the rewrite
  ever hurts recall.
- **HyDE adds another chat request** and can hallucinate a passage that pulls
  retrieval toward the wrong topic. It usually helps vague questions and can
  hurt precise ones; compare `--rag naive` against `--rag hyde` on your own
  questions before switching defaults.
- **Graph extraction costs one chat call per chunk** and is the slowest stage
  for large PDFs. It is skipped with `--skip-graph` and can be run alone with
  `--skip-embeddings`.
- **Graph retrieval has no vector fallback.** `--rag graph` makes one chat call
  to extract entities, then matches them by text. If nothing matches (a synonym,
  a paraphrase, an entity the extractor missed) it returns no chunks instead of
  falling back to similarity search.
- **Graph matches are exact substrings.** `CL` and `Casual Leave`, or `LOP` and
  `Loss of Pay`, are separate names. Add an alias map if synonyms matter.
- **Graph ties keep document order.** The score is built from entity overlap and
  edge evidence only, so many chunks can tie; a secondary relevance signal
  (embedding distance) is the obvious next improvement.
- **Relationship extraction is per chunk.** Edges spanning two chunks are not
  discovered, because both endpoints must appear in the same extraction result.
- The answer prompt instructs the model to reply "I cannot find the answer in
  the provided document." when the retrieved context is insufficient.
