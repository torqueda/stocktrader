"""Reusable Groq client helpers for raw, JSON, and validated calls."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, TypeVar

from groq import Groq

from .config import (
    LLM_MAX_RETRIES,
    MODEL_ID,
    STRATEGY_MAX_COMPLETION_TOKENS,
    VERIFY_MAX_COMPLETION_TOKENS,
    get_groq_api_key,
)

JSON_FENCE_PATTERN = re.compile(r"```(?:json)?", re.IGNORECASE)
JSON_OBJECT_RESPONSE_FORMAT = {"type": "json_object"}
T = TypeVar("T")


def create_groq_client() -> Groq:
    """Create a Groq client using the configured API key."""

    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    return Groq(api_key=api_key)


def request_chat_completion(
    messages: list[dict[str, str]],
    model_id: str = MODEL_ID,
    max_completion_tokens: int = STRATEGY_MAX_COMPLETION_TOKENS,
    temperature: float = 0,
    response_format: dict[str, str] | None = None,
) -> str:
    """Send a chat completion request and return the raw text content."""

    client = create_groq_client()
    request_kwargs: dict[str, object] = {
        "model": model_id,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "messages": messages,
    }
    if response_format is not None:
        request_kwargs["response_format"] = response_format

    response = client.chat.completions.create(**request_kwargs)

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Groq returned an empty response.")

    return content


def extract_json_text(response_text: str) -> str:
    """Extract a JSON object string from a model response."""

    stripped = JSON_FENCE_PATTERN.sub("", response_text).replace("```", "").strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start_index = stripped.find("{")
    end_index = stripped.rfind("}")
    if start_index == -1 or end_index == -1 or start_index >= end_index:
        raise ValueError("Model response did not contain a JSON object.")

    return stripped[start_index : end_index + 1].strip()


def parse_json_payload(json_text: str) -> Any:
    """Parse a JSON payload into native Python data."""

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON payload: {exc.msg}") from exc


def call_validated_json_completion(
    messages: list[dict[str, str]],
    validate_payload: Callable[[Any], T],
    model_id: str = MODEL_ID,
    max_completion_tokens: int = STRATEGY_MAX_COMPLETION_TOKENS,
    temperature: float = 0,
    max_retries: int = LLM_MAX_RETRIES,
) -> T:
    """Request a JSON response, validate it locally, and retry once on failure."""

    invalid_response: str | None = None
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        request_messages = [dict(message) for message in messages]
        if attempt > 0 and invalid_response is not None and last_error is not None:
            request_messages.extend(
                [
                    {"role": "assistant", "content": invalid_response},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was invalid because: "
                            f"{last_error}. Return only one valid JSON object and no extra text."
                        ),
                    },
                ]
            )

        response_text = request_chat_completion(
            messages=request_messages,
            model_id=model_id,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            response_format=JSON_OBJECT_RESPONSE_FORMAT,
        )

        try:
            json_text = extract_json_text(response_text)
            payload = parse_json_payload(json_text)
            return validate_payload(payload)
        except ValueError as exc:
            invalid_response = response_text
            last_error = exc

    attempt_count = max_retries + 1
    raise RuntimeError(
        f"Failed to produce a valid JSON response after {attempt_count} attempts: {last_error}"
    )


def verify_groq_connection(model_id: str = MODEL_ID) -> dict[str, object]:
    """Run a tiny Groq chat-completions smoke test."""

    client = create_groq_client()
    response = client.chat.completions.create(
        model=model_id,
        temperature=0,
        max_completion_tokens=VERIFY_MAX_COMPLETION_TOKENS,
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
