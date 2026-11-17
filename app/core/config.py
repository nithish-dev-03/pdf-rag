import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# app/core/config.py -> app/
ROOT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ROOT_DIR.parent
DATA_DIR = ROOT_DIR / "data"


def _resolve_path(raw: str) -> Path:
    """Turn an env-supplied path into an absolute path.

    Handles ``~``, environment variables (``$HOME`` / ``%USERPROFILE%``) and
    both ``/`` and ``\\`` separators. Relative paths are anchored to the project
    root so the same ``.env`` works on Windows, macOS and Linux regardless of
    the directory the CLI is launched from.
    """
    path = Path(os.path.expandvars(raw)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer from the environment with a clear error."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


# Source PDF for ingest. Override with PDF_PATH in .env. Absolute paths work
# as-is; relative paths are resolved against the project root. Forward slashes
# are accepted on all platforms, including Windows.
_PDF_PATH = os.getenv("PDF_PATH", "").strip()
PDF_PATH = _resolve_path(_PDF_PATH) if _PDF_PATH else DATA_DIR / "ms-leave-policy.pdf"

# The SDK also accepts GEMINI_API_KEY; support either name.
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

CHAT_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "gemini-embedding-2"

# gemini-embedding-2 supports 128-3072 dimensions and recommends 768, 1536 or
# 3072. 1536 keeps the pgvector column (and any future index) reasonably small.
# It must match the width of the vector column in PostgreSQL.
EMBEDDING_DIM = 1536

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Number of chunks fetched per question (top-k). Override with RETRIEVAL_LIMIT
# in .env or per call with `--limit`. Too low drops the answering chunk; too
# high dilutes the prompt with unrelated context.
DEFAULT_RETRIEVAL_LIMIT = 5
RETRIEVAL_LIMIT = _env_int("RETRIEVAL_LIMIT", DEFAULT_RETRIEVAL_LIMIT)

# Dedicated table so this project's vectors can coexist with other tables in
# the same database. Vectors from different embedding models are not
# comparable, and their dimensions can differ.
TABLE_NAME = "document_chunks_gemini"

# Rewrite the user's question with the chat model before embedding it. Set to
# False to embed the question verbatim.
REFINE_QUERY = True
