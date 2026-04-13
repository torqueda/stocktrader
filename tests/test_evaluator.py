"""Tests for evaluator prompt loading and evaluator-layer behavior."""

from __future__ import annotations

from copy import deepcopy

import pytest

from src.evaluator import (
    EVALUATOR_AGREEMENT_PROMPT_FILE,
    EVALUATOR_DISAGREEMENT_PROMPT_FILE,
    _build_evaluator_input_payload,
    evaluate_strategies,
    load_evaluator_prompt,
)
from src.schemas import Decision, EvaluatorOutput, StrategyOutput


def make_market_context() -> dict[str, object]:
    """Build a compact but valid market context for evaluator tests."""

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


def make_strategy_output(
    name: str,
    decision: Decision,
    justification: str,
) -> StrategyOutput:
    """Build a valid StrategyOutput for evaluator tests."""

    return StrategyOutput(
        name=name,
        decision=decision,
        confidence=7,
        justification=justification,
    )


def test_evaluator_prompt_files_load_successfully() -> None:
    agreement_prompt = load_evaluator_prompt(EVALUATOR_AGREEMENT_PROMPT_FILE)
    disagreement_prompt = load_evaluator_prompt(EVALUATOR_DISAGREEMENT_PROMPT_FILE)

    assert '"agents_agree" to true' in agreement_prompt
    assert '"agents_agree" to false' in disagreement_prompt


def test_build_evaluator_input_payload_is_compact_and_serializable() -> None:
    payload = _build_evaluator_input_payload(
        ticker="AAPL",
        market_context=make_market_context(),
        strategy_a=make_strategy_output(
            "Momentum Trader",
            Decision.BUY,
            "Price is 259.2. Trend evidence is constructive. Momentum remains supportive.",
        ),
        strategy_b=make_strategy_output(
            "Value Contrarian",
            Decision.HOLD,
            "Distance from the high is -10.19%. RSI is 53.64. The setup is mixed.",
        ),
    )

    assert payload["ticker"] == "AAPL"
    assert "market_data_summary" in payload
    assert "raw_window_summaries" not in payload["market_data_summary"]
    assert payload["strategy_a"]["name"] == "Momentum Trader"


def test_evaluator_routes_to_agreement_prompt_when_decisions_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_call_validated_json_completion(
        messages: list[dict[str, str]],
        validate_payload,
        model_id: str = "unused",
        max_completion_tokens: int = 0,
        temperature: float = 0,
        max_retries: int = 0,
    ) -> EvaluatorOutput:
        del model_id, temperature, max_retries
        captured["messages"] = messages
        captured["max_completion_tokens"] = max_completion_tokens
        payload = {
            "agents_agree": True,
            "analysis": (
                "Both strategies land on BUY because price is 259.2 and the market context is not deeply stressed. "
                "The momentum case leans on constructive trend evidence, while the contrarian case sees enough room from the 52-week high to avoid an overextended sell signal."
            ),
        }
        return validate_payload(payload)

    monkeypatch.setattr("src.evaluator.call_validated_json_completion", fake_call_validated_json_completion)

    result = evaluate_strategies(
        ticker="AAPL",
        market_context=make_market_context(),
        strategy_a=make_strategy_output(
            "Momentum Trader",
            Decision.BUY,
            "Price is 259.2. The 20-day average is 253.74. Momentum remains supportive.",
        ),
        strategy_b=make_strategy_output(
            "Value Contrarian",
            Decision.BUY,
            "Distance from the high is -10.19%. RSI is 53.64. The stock is not overextended.",
        ),
    )

    assert result.agents_agree is True
    messages = captured["messages"]
    assert 'Set "agents_agree" to true' in messages[0]["content"]
    assert '"market_data_summary"' in messages[1]["content"]
    assert '"raw_window_summaries"' not in messages[1]["content"]


def test_evaluator_routes_to_disagreement_prompt_when_decisions_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_call_validated_json_completion(
        messages: list[dict[str, str]],
        validate_payload,
        model_id: str = "unused",
        max_completion_tokens: int = 0,
        temperature: float = 0,
        max_retries: int = 0,
    ) -> EvaluatorOutput:
        del model_id, max_completion_tokens, temperature, max_retries
        captured["messages"] = messages
        payload = {
            "agents_agree": False,
            "analysis": (
                "The strategies diverge because the Momentum Trader emphasizes price relative to the moving averages, while the Value Contrarian emphasizes drawdown and RSI. "
                "Those inputs support different interpretations of the same market context."
            ),
        }
        return validate_payload(payload)

    monkeypatch.setattr("src.evaluator.call_validated_json_completion", fake_call_validated_json_completion)

    result = evaluate_strategies(
        ticker="AAPL",
        market_context=make_market_context(),
        strategy_a=make_strategy_output(
            "Momentum Trader",
            Decision.BUY,
            "Price is 259.2. The 20-day average is 253.74. Momentum remains supportive.",
        ),
        strategy_b=make_strategy_output(
            "Value Contrarian",
            Decision.SELL,
            "Distance from the high is -10.19%. Recent drawdown is -9.43%. Upside looks limited.",
        ),
    )

    assert result.agents_agree is False
    assert 'Set "agents_agree" to false' in captured["messages"][0]["content"]


def test_evaluator_rejects_wrong_agents_agree_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_validated_json_completion(
        messages: list[dict[str, str]],
        validate_payload,
        model_id: str = "unused",
        max_completion_tokens: int = 0,
        temperature: float = 0,
        max_retries: int = 0,
    ) -> EvaluatorOutput:
        del messages, model_id, max_completion_tokens, temperature, max_retries
        payload = {
            "agents_agree": False,
            "analysis": "This output uses the wrong agreement flag. It should be rejected cleanly.",
        }
        return validate_payload(payload)

    monkeypatch.setattr("src.evaluator.call_validated_json_completion", fake_call_validated_json_completion)

    with pytest.raises(ValueError, match="agents_agree value does not match"):
        evaluate_strategies(
            ticker="AAPL",
            market_context=make_market_context(),
            strategy_a=make_strategy_output(
                "Momentum Trader",
                Decision.BUY,
                "Price is 259.2. The 20-day average is 253.74. Momentum remains supportive.",
            ),
            strategy_b=make_strategy_output(
                "Value Contrarian",
                Decision.BUY,
                "Distance from the high is -10.19%. RSI is 53.64. The stock is not overextended.",
            ),
        )


def test_evaluator_does_not_mutate_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market_context = make_market_context()
    strategy_a = make_strategy_output(
        "Momentum Trader",
        Decision.BUY,
        "Price is 259.2. The 20-day average is 253.74. Momentum remains supportive.",
    )
    strategy_b = make_strategy_output(
        "Value Contrarian",
        Decision.HOLD,
        "Distance from the high is -10.19%. RSI is 53.64. The setup is mixed.",
    )
    original_context = deepcopy(market_context)
    original_strategy_a = strategy_a.model_dump(mode="json")
    original_strategy_b = strategy_b.model_dump(mode="json")

    def fake_call_validated_json_completion(
        messages: list[dict[str, str]],
        validate_payload,
        model_id: str = "unused",
        max_completion_tokens: int = 0,
        temperature: float = 0,
        max_retries: int = 0,
    ) -> EvaluatorOutput:
        del messages, model_id, max_completion_tokens, temperature, max_retries
        payload = {
            "agents_agree": False,
            "analysis": (
                "The strategies split because the trend signals and contrarian signals point in different directions. "
                "That divergence comes from different weighting of the same market summary."
            ),
        }
        return validate_payload(payload)

    monkeypatch.setattr("src.evaluator.call_validated_json_completion", fake_call_validated_json_completion)

    result = evaluate_strategies("AAPL", market_context, strategy_a, strategy_b)

    assert result.agents_agree is False
    assert market_context == original_context
    assert strategy_a.model_dump(mode="json") == original_strategy_a
    assert strategy_b.model_dump(mode="json") == original_strategy_b
