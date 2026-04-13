"""Evaluator prompt loading and evaluator-agent functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import EVALUATOR_MAX_COMPLETION_TOKENS, get_prompt_path
from .llm_client import call_validated_json_completion
from .market_data import build_market_data_summary, normalize_ticker_symbol
from .schemas import EvaluatorOutput, StrategyOutput

EVALUATOR_AGREEMENT_PROMPT_FILE = "evaluator_agreement.txt"
EVALUATOR_DISAGREEMENT_PROMPT_FILE = "evaluator_disagreement.txt"


def get_evaluator_prompt_path(filename: str) -> Path:
    """Return the path to an evaluator prompt file."""

    return get_prompt_path(filename)


def load_evaluator_prompt(filename: str) -> str:
    """Load one evaluator prompt file from disk."""

    prompt_path = get_evaluator_prompt_path(filename)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Evaluator prompt file not found: {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Evaluator prompt file is empty: {prompt_path}")

    return prompt_text


def _resolve_evaluator_ticker(ticker: str, market_context: dict[str, object]) -> str:
    """Ensure the requested ticker matches the supplied market context."""

    requested_ticker = normalize_ticker_symbol(ticker)
    context_ticker = str(market_context.get("ticker", "")).strip().upper()
    if context_ticker and context_ticker != requested_ticker:
        raise ValueError(
            f"Ticker '{requested_ticker}' does not match market context ticker '{context_ticker}'."
        )

    return context_ticker or requested_ticker


def _build_evaluator_input_payload(
    ticker: str,
    market_context: dict[str, object],
    strategy_a: StrategyOutput,
    strategy_b: StrategyOutput,
) -> dict[str, object]:
    """Build a compact evaluator input payload."""

    return {
        "ticker": ticker,
        "market_data_summary": build_market_data_summary(market_context),
        "strategy_a": strategy_a.model_dump(mode="json"),
        "strategy_b": strategy_b.model_dump(mode="json"),
    }


def _build_evaluator_messages(
    prompt_text: str,
    evaluator_input: dict[str, object],
) -> list[dict[str, str]]:
    """Build evaluator messages using a compact JSON payload."""

    evaluator_input_json = json.dumps(evaluator_input, indent=2, sort_keys=True)
    user_message = (
        "Use only the following EVALUATOR_INPUT_JSON.\n"
        "Return exactly one JSON object that satisfies the prompt instructions.\n"
        f"EVALUATOR_INPUT_JSON:\n{evaluator_input_json}"
    )

    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": user_message},
    ]


def _validate_evaluator_payload(payload: Any, expected_agents_agree: bool) -> EvaluatorOutput:
    """Validate parsed evaluator output and enforce the expected branch."""

    try:
        evaluator_output = EvaluatorOutput.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Evaluator output validation failed: {exc}") from exc

    analysis = evaluator_output.analysis.strip()
    if not analysis:
        raise ValueError("Evaluator analysis must be non-empty.")

    if evaluator_output.agents_agree != expected_agents_agree:
        raise ValueError(
            "Evaluator output agents_agree value does not match the agreement branch selected in Python."
        )

    return evaluator_output.model_copy(update={"analysis": analysis})


def evaluate_strategies(
    ticker: str,
    market_context: dict[str, object],
    strategy_a: StrategyOutput,
    strategy_b: StrategyOutput,
) -> EvaluatorOutput:
    """Compare two independent strategy outputs and produce evaluator analysis."""

    resolved_ticker = _resolve_evaluator_ticker(ticker, market_context)
    agents_agree = strategy_a.decision == strategy_b.decision
    prompt_filename = (
        EVALUATOR_AGREEMENT_PROMPT_FILE if agents_agree else EVALUATOR_DISAGREEMENT_PROMPT_FILE
    )
    prompt_text = load_evaluator_prompt(prompt_filename)
    evaluator_input = _build_evaluator_input_payload(
        ticker=resolved_ticker,
        market_context=market_context,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
    )
    messages = _build_evaluator_messages(prompt_text, evaluator_input)

    return call_validated_json_completion(
        messages=messages,
        validate_payload=lambda payload: _validate_evaluator_payload(payload, agents_agree),
        max_completion_tokens=EVALUATOR_MAX_COMPLETION_TOKENS,
    )
