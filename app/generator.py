"""
generator.py — RAG answer generation via Groq API.

Uses a strict grounded prompt: the model must only use the provided
context and must cite sources by [Source N] markers.
If the answer is not in the context, it says so explicitly.
"""

import os
from groq import Groq

MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are a precise document assistant. Answer questions using ONLY the provided context.

Rules:
- Cite sources as [Source N] inline when you use information from them
- If the answer is not in the context, say: "I could not find this information in the provided documents."
- Be concise and factual. Do not speculate or add information beyond the context.
- If multiple sources support the answer, cite all of them."""


def generate_answer(query: str, context: str) -> dict:
    """
    Generate a grounded answer from the LLM.
    Returns {answer, model, usage} dict.
    """
    client = _get_client()

    user_message = f"""Context:
{context}

Question: {query}

Answer (cite sources inline):"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1,   # low temp for factual, grounded answers
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    usage  = {
        "prompt_tokens":     response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens":      response.usage.total_tokens,
    }

    return {"answer": answer, "model": MODEL, "usage": usage}
