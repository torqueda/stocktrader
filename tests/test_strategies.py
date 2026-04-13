"""Tests for strategy prompt loading and strategy-layer behavior."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.schemas import Decision, StrategyOutput
from src.strategies import (
    MOMENTUM_TRADER,
    STRATEGY_A_PROMPT_FILE,
    STRATEGY_B_PROMPT_FILE,
    VALUE_CONTRARIAN,
    _build_strategy_messages,
    load_strategy_prompt,
    run_momentum_trader,
    run_value_contrarian,
)


def make_market_context() -> dict[str, object]:
    """Build a compact but valid market context for strategy tests."""

    return {
        "ticker": "AAPL",
        "history_metadata": {
            "period_requested": "1y",
            "interval": "1d",
            "rows_fetched": 250,
            "latest_trading_date": "2026-04-13",
        },
        "price_summary": {
            "current_price": 259.2,
            "price_30d_ago": 264.2,
            "pct_change_30d": -1.89,
            "high_52w": 288.61,
            "low_52w": 189.81,
        },
        "momentum_features": {
            "moving_avg_20d": 253.74,
            "moving_avg_50d": 260.9,
            "price_vs_ma20_pct": 2.15,
            "price_vs_ma50_pct": -0.65,
            "avg_volume_30d": 55218234,
            "avg_volume_90d": 63194210,
            "volume_trend_ratio": 0.874,
            "daily_returns_30d": [0.45] * 30,
        },
        "value_contrarian_features": {
            "distance_from_52w_high_pct": -10.19,
            "distance_from_52w_low_pct": 36.56,
            "recent_drawdown_pct": -9.43,
            "rsi_14": 53.64,
        },
        "raw_window_summaries": {
            "last_5_closes": [255.2, 256.1, 257.0, 258.6, 259.2],
            "last_5_volumes": [51000000, 52000000, 53000000, 54000000, 55000000],
            "recent_peak_close_90d": 286.2,
            "recent_low_close_90d": 233.4,
        },
    }


def test_prompt_files_load_successfully() -> None:
    momentum_prompt = load_strategy_prompt(STRATEGY_A_PROMPT_FILE)
    value_prompt = load_strategy_prompt(STRATEGY_B_PROMPT_FILE)

    assert "Momentum Trader" in momentum_prompt
    assert "Value Contrarian" in value_prompt


def test_build_strategy_messages_includes_ticker_and_market_context() -> None:
    messages = _build_strategy_messages("Prompt text", "AAPL", make_market_context())

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Prompt text"
    assert messages[1]["role"] == "user"
    assert "TICKER: AAPL" in messages[1]["content"]
    assert '"ticker": "AAPL"' in messages[1]["content"]
    assert "MARKET_CONTEXT_JSON" in messages[1]["content"]


@pytest.mark.parametrize(
    ("strategy_runner", "expected_name"),
    [
        (run_momentum_trader, MOMENTUM_TRADER),
        (run_value_contrarian, VALUE_CONTRARIAN),
    ],
)
def test_strategy_runner_calls_generic_llm_helper(
    monkeypatch: pytest.MonkeyPatch,
    strategy_runner,
    expected_name: str,
) -> None:
    captured: dict[str, object] = {}

    def fake_call_validated_json_completion(
        messages: list[dict[str, str]],
        validate_payload,
        model_id: str = "unused",
        max_completion_tokens: int = 0,
        temperature: float = 0,
        max_retries: int = 0,
    ) -> StrategyOutput:
        captured["messages"] = messages
        captured["max_completion_tokens"] = max_completion_tokens
        payload = {
            "name": expected_name,
            "decision": "HOLD",
            "confidence": 5,
            "justification": (
                "Current price is 259.2. The signal mix is still moderate. "
                "A neutral stance fits the data."
            ),
        }
        return validate_payload(payload)

    monkeypatch.setattr("src.strategies.call_validated_json_completion", fake_call_validated_json_completion)

    result = strategy_runner("AAPL", make_market_context())

    assert result.name == expected_name
    assert result.decision is Decision.HOLD
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert expected_name in messages[0]["content"]
    assert '"ticker": "AAPL"' in messages[1]["content"]


def test_strategy_runner_rejects_unexpected_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_validated_json_completion(
        messages: list[dict[str, str]],
        validate_payload,
        model_id: str = "unused",
        max_completion_tokens: int = 0,
        temperature: float = 0,
        max_retries: int = 0,
    ) -> StrategyOutput:
        payload = {
            "name": "Unexpected Strategy",
            "decision": "BUY",
            "confidence": 8,
            "justification": "Price is 259.2. Trend is supportive. Momentum is positive.",
        }
        return validate_payload(payload)

    monkeypatch.setattr("src.strategies.call_validated_json_completion", fake_call_validated_json_completion)

    with pytest.raises(ValueError, match="Strategy output name must be"):
        run_momentum_trader("AAPL", make_market_context())


def test_strategy_functions_do_not_mutate_shared_market_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_context = make_market_context()
    original_context = deepcopy(market_context)
    captured_names: list[str] = []

    def fake_call_validated_json_completion(
        messages: list[dict[str, str]],
        validate_payload,
        model_id: str = "unused",
        max_completion_tokens: int = 0,
        temperature: float = 0,
        max_retries: int = 0,
    ) -> StrategyOutput:
        del messages, model_id, max_completion_tokens, temperature, max_retries
        payload = {
            "name": MOMENTUM_TRADER if not captured_names else VALUE_CONTRARIAN,
            "decision": "HOLD",
            "confidence": 5,
            "justification": "Current price is 259.2. The signal mix is still moderate. A neutral stance fits the data.",
        }
        captured_names.append(payload["name"])
        return validate_payload(payload)

    monkeypatch.setattr("src.strategies.call_validated_json_completion", fake_call_validated_json_completion)

    momentum_result = run_momentum_trader("AAPL", market_context)
    value_result = run_value_contrarian("AAPL", market_context)

    assert momentum_result.name == MOMENTUM_TRADER
    assert value_result.name == VALUE_CONTRARIAN
    assert market_context == original_context
