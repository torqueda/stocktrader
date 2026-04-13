"""Minimal Groq client helpers for environment verification."""

from __future__ import annotations

from groq import Groq

from .config import MODEL_ID, get_groq_api_key


def verify_groq_connection(model_id: str = MODEL_ID) -> dict[str, object]:
    """Run a tiny Groq chat-completions smoke test."""

    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        temperature=0,
        max_completion_tokens=8,
        messages=[
            {
                "role": "user",
                "content": "Reply with OK only.",
            }
        ],
    )

    message = response.choices[0].message.content or ""
    usage = response.usage

    return {
        "model_id": model_id,
        "reply": message.strip(),
        "prompt_tokens": int(usage.prompt_tokens) if usage is not None else None,
        "completion_tokens": int(usage.completion_tokens) if usage is not None else None,
    }
