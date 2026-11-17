import os

from google import genai
from google.genai import types

from app.core.config import API_KEY

REQUEST_TIMEOUT_MS = 120_000

# The SDK retries retryable HTTP statuses itself. Bounding attempts and delay
# keeps a dead endpoint from looking like a hang.
HTTP_OPTIONS = types.HttpOptions(
    timeout=REQUEST_TIMEOUT_MS,
    retry_options=types.HttpRetryOptions(
        attempts=4,
        initial_delay=1.0,
        max_delay=8.0,
        exp_base=2.0,
        jitter=1.0,
        http_status_codes=[429, 500, 502, 503, 504],
    ),
)


def create_client() -> genai.Client:
    """Build a Gen AI client, failing early if no API key is configured."""
    api_key = API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to .env "
            "(copy .env.example) and try again."
        )

    return genai.Client(api_key=api_key, http_options=HTTP_OPTIONS)
