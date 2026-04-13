"""Tests for the reusable LLM client helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm_client import (
    call_validated_json_completion,
    extract_json_text,
    parse_json_payload,
    request_chat_completion,
    verify_groq_connection,
)


def test_extract_json_text_with_plain_json() -> None:
    assert extract_json_text('{"decision":"BUY"}') == '{"decision":"BUY"}'


def test_extract_json_text_with_fences_and_extra_text() -> None:
    response_text = 'Here is the result:\n```json\n{"decision":"HOLD"}\n```\nUse that.'
    assert extract_json_text(response_text) == '{"decision":"HOLD"}'


def test_extract_json_text_raises_on_missing_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        extract_json_text("No JSON here.")


def test_parse_json_payload_returns_python_data() -> None:
    payload = parse_json_payload('{"confidence": 7}')
    assert payload == {"confidence": 7}


def test_request_chat_completion_returns_text_from_mocked_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="raw content"))]
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )

    monkeypatch.setattr("src.llm_client.create_groq_client", lambda: fake_client)

    content = request_chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        model_id="test-model",
        max_completion_tokens=12,
    )

    assert content == "raw content"


def test_call_validated_json_completion_succeeds_on_first_try(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.llm_client.request_chat_completion",
        lambda messages, model_id, max_completion_tokens, temperature, response_format: '{"result":"ok","value":7}',
    )

    result = call_validated_json_completion(
        messages=[{"role": "user", "content": "Return JSON"}],
        validate_payload=lambda payload: payload["value"] if payload["result"] == "ok" else (_ for _ in ()).throw(ValueError("bad result")),
        max_retries=1,
    )

    assert result == 7


def test_call_validated_json_completion_retries_once_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(["not json", '{"result":"ok","value":9}'])
    captured_message_counts: list[int] = []

    def fake_request_chat_completion(
        messages: list[dict[str, str]],
        model_id: str,
        max_completion_tokens: int,
        temperature: float,
        response_format: dict[str, str],
    ) -> str:
        del model_id, max_completion_tokens, temperature, response_format
        captured_message_counts.append(len(messages))
        return next(responses)

    monkeypatch.setattr("src.llm_client.request_chat_completion", fake_request_chat_completion)

    result = call_validated_json_completion(
        messages=[{"role": "user", "content": "Return JSON"}],
        validate_payload=lambda payload: payload["value"] if payload["result"] == "ok" else (_ for _ in ()).throw(ValueError("bad result")),
        max_retries=1,
    )

    assert result == 9
    assert captured_message_counts == [1, 3]


def test_call_validated_json_completion_fails_after_two_bad_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(['{"result":"bad"}', '{"result":"bad"}'])

    monkeypatch.setattr(
        "src.llm_client.request_chat_completion",
        lambda messages, model_id, max_completion_tokens, temperature, response_format: next(responses),
    )

    with pytest.raises(RuntimeError, match="Failed to produce a valid JSON response after 2 attempts"):
        call_validated_json_completion(
            messages=[{"role": "user", "content": "Return JSON"}],
            validate_payload=lambda payload: payload["value"] if payload["result"] == "ok" else (_ for _ in ()).throw(ValueError("bad result")),
            max_retries=1,
        )


def test_verify_groq_connection_uses_mocked_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_response)
        )
    )

    monkeypatch.setattr("src.llm_client.create_groq_client", lambda: fake_client)

    payload = verify_groq_connection(model_id="test-model")

    assert payload == {
        "model_id": "test-model",
        "reply": "OK",
        "prompt_tokens": 3,
        "completion_tokens": 1,
    }
