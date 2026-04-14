"""Debate-mode prompt loading and second-round strategy responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import STRATEGY_MAX_COMPLETION_TOKENS, get_prompt_path
from .llm_client import call_validated_json_completion
from .market_data import normalize_ticker_symbol
from .schemas import DebateChange, DebateOutput, StockAnalysisOutput, StrategyOutput
from .strategies import MOMENTUM_TRADER, VALUE_CONTRARIAN

DEBATE_A_PROMPT_FILE = "debate_a_momentum.txt"
DEBATE_B_PROMPT_FILE = "debate_b_value_contrarian.txt"


def get_debate_prompt_path(filename: str) -> Path:
    """Return the path to a debate prompt file."""

    return get_prompt_path(filename)


def load_debate_prompt(filename: str) -> str:
    """Load one debate prompt file from disk."""

    prompt_path = get_debate_prompt_path(filename)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Debate prompt file not found: {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Debate prompt file is empty: {prompt_path}")

    return prompt_text


def _build_debate_input_payload(
    ticker: str,
    market_data_summary: dict[str, object],
    self_strategy: StrategyOutput,
    opponent_strategy: StrategyOutput,
    original_evaluator_analysis: str,
) -> dict[str, object]:
    """Build a compact debate input payload for one strategy."""

    return {
        "ticker": ticker,
        "market_data_summary": market_data_summary,
        "your_first_round_output": self_strategy.model_dump(mode="json"),
        "opponent_first_round_output": opponent_strategy.model_dump(mode="json"),
        "evaluator_analysis": original_evaluator_analysis,
    }


def _build_debate_messages(
    prompt_text: str,
    debate_input: dict[str, object],
) -> list[dict[str, str]]:
    """Build debate messages using a compact JSON payload."""

    debate_input_json = json.dumps(debate_input, indent=2, sort_keys=True)
    user_message = (
        "Use only the following DEBATE_INPUT_JSON.\n"
        "Return exactly one JSON object that satisfies the prompt instructions.\n"
        f"DEBATE_INPUT_JSON:\n{debate_input_json}"
    )

    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": user_message},
    ]


def _validate_debate_strategy_payload(payload: Any, expected_name: str) -> StrategyOutput:
    """Validate one debate-round strategy payload."""

    try:
        strategy_output = StrategyOutput.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Debate strategy output validation failed: {exc}") from exc

    if strategy_output.name != expected_name:
        raise ValueError(
            f"Debate strategy output name must be '{expected_name}', received '{strategy_output.name}'."
        )

    return strategy_output


def _build_debate_change(before: StrategyOutput, after: StrategyOutput) -> DebateChange:
    """Build a structured comparison between rounds for one strategy."""

    decision_changed = before.decision != after.decision
    confidence_changed = before.confidence != after.confidence
    justification_changed = before.justification != after.justification

    return DebateChange(
        decision_before=before.decision,
        decision_after=after.decision,
        confidence_before=before.confidence,
        confidence_after=after.confidence,
        decision_changed=decision_changed,
        confidence_changed=confidence_changed,
        justification_changed=justification_changed,
        position_changed=decision_changed or confidence_changed,
    )


def _resolve_debate_ticker(output: StockAnalysisOutput) -> str:
    """Resolve the debate ticker from a validated saved output."""

    return normalize_ticker_symbol(output.ticker)


def run_debate_round(output: StockAnalysisOutput) -> DebateOutput:
    """Run a second-round debate using an existing saved disagreement artifact."""

    ticker = _resolve_debate_ticker(output)
    if output.evaluator.agents_agree:
        raise ValueError(
            f"Debate mode requires a disagreement. Saved output for ticker '{ticker}' already shows agreement."
        )

    evaluator_analysis = output.evaluator.analysis.strip()
    if not evaluator_analysis:
        raise ValueError("Debate mode requires a non-empty evaluator analysis from the saved output.")

    strategy_a_prompt = load_debate_prompt(DEBATE_A_PROMPT_FILE)
    strategy_b_prompt = load_debate_prompt(DEBATE_B_PROMPT_FILE)

    strategy_a_messages = _build_debate_messages(
        strategy_a_prompt,
        _build_debate_input_payload(
            ticker=ticker,
            market_data_summary=output.market_data_summary,
            self_strategy=output.strategy_a,
            opponent_strategy=output.strategy_b,
            original_evaluator_analysis=evaluator_analysis,
        ),
    )
    strategy_b_messages = _build_debate_messages(
        strategy_b_prompt,
        _build_debate_input_payload(
            ticker=ticker,
            market_data_summary=output.market_data_summary,
            self_strategy=output.strategy_b,
            opponent_strategy=output.strategy_a,
            original_evaluator_analysis=evaluator_analysis,
        ),
    )

    strategy_a_response = call_validated_json_completion(
        messages=strategy_a_messages,
        validate_payload=lambda payload: _validate_debate_strategy_payload(payload, MOMENTUM_TRADER),
        max_completion_tokens=STRATEGY_MAX_COMPLETION_TOKENS,
    )
    strategy_b_response = call_validated_json_completion(
        messages=strategy_b_messages,
        validate_payload=lambda payload: _validate_debate_strategy_payload(payload, VALUE_CONTRARIAN),
        max_completion_tokens=STRATEGY_MAX_COMPLETION_TOKENS,
    )

    return DebateOutput(
        strategy_a_response=strategy_a_response,
        strategy_b_response=strategy_b_response,
        strategy_a_change=_build_debate_change(output.strategy_a, strategy_a_response),
        strategy_b_change=_build_debate_change(output.strategy_b, strategy_b_response),
    )
