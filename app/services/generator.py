from google.genai import types

from app.core.client import create_client
from app.core.config import CHAT_MODEL

SYSTEM_INSTRUCTION = (
    "You are a precise HR policy assistant. Answer the user's question using "
    "ONLY the provided context; never use outside knowledge and never invent "
    "policy. "
    "\n\n"
    "How to answer:\n"
    "1. Decide whether the context contains any rule relevant to the question. "
    "The context does not need to address the exact scenario verbatim: if a "
    "policy rule clearly governs the situation, apply it and reason from it.\n"
    "2. If the user asks for a yes/no answer, open with a direct verdict: "
    "\"Yes\", \"No\", or \"It depends\". Then explain.\n"
    "3. When the governing rule is conditional or discretionary, give a "
    "qualified answer (for example, \"No, not automatically; it depends on "
    "whether ...\") rather than refusing.\n"
    "4. Quote or closely paraphrase the relevant policy text and support each "
    "claim with the page number(s) it came from.\n"
    "5. Do not let holidays, weekends, or stated dates override an explicit "
    "policy rule; cite the rule that decides the case.\n"
    "6. Only reply \"I cannot find the answer in the provided document.\" when "
    "no rule in the context is relevant to the question at all.\n\n"
    "Keep the answer concise, structured, and grounded strictly in the context."
)

REFINE_INSTRUCTION = (
    "Rewrite the user's question as a clear, standalone search query for a "
    "document retrieval system. Keep every name, number, date and domain term "
    "exactly as written. Do not answer the question and do not add facts. "
    "Return only the rewritten question."
)


def generate_answer(user_query: str, retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return "I cannot find the answer in the provided document."

    context_text = "\n\n".join(
        f"Source: {chunk['source']} (page {chunk['page']}, "
        f"chunk {chunk['chunk_index']}):\n{chunk['content']}"
        for chunk in retrieved_chunks
    )

    prompt = f"""Context:
{context_text}

User Question: {user_query}"""

    with create_client() as client:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
        )

    return response.text or "(The model returned an empty response.)"


def refine_user_query(user_query: str) -> str:
    with create_client() as client:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=f"User question: {user_query}",
            config=types.GenerateContentConfig(
                system_instruction=REFINE_INSTRUCTION,
                temperature=0.0,
            ),
        )

    refined = (response.text or "").strip()
    # Never let a garbled rewrite turn into an empty search query.
    return refined or user_query
