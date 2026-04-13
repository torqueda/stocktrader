"""Strategy prompt loading and strategy-agent functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import STRATEGY_MAX_COMPLETION_TOKENS, get_prompt_path
from .llm_client import call_validated_json_completion
from .market_data import normalize_ticker_symbol
from .schemas import StrategyOutput

MOMENTUM_TRADER = "Momentum Trader"
VALUE_CONTRARIAN = "Value Contrarian"
STRATEGY_A_PROMPT_FILE = "strategy_a_momentum.txt"
STRATEGY_B_PROMPT_FILE = "strategy_b_value_contrarian.txt"


def get_strategy_prompt_path(filename: str) -> Path:
    """Return the path to a strategy prompt file."""

    return get_prompt_path(filename)


def load_strategy_prompt(filename: str) -> str:
    """Load one strategy prompt file from disk."""

    prompt_path = get_strategy_prompt_path(filename)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Strategy prompt file not found: {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        raise ValueError(f"Strategy prompt file is empty: {prompt_path}")

    return prompt_text


def _resolve_strategy_ticker(ticker: str, market_context: dict[str, object]) -> str:
    """Ensure the requested ticker matches the supplied market context."""

    requested_ticker = normalize_ticker_symbol(ticker)
    context_ticker = str(market_context.get("ticker", "")).strip().upper()
    if context_ticker and context_ticker != requested_ticker:
        raise ValueError(
            f"Ticker '{requested_ticker}' does not match market context ticker '{context_ticker}'."
        )

    return context_ticker or requested_ticker


def _build_strategy_messages(
    prompt_text: str,
    ticker: str,
    market_context: dict[str, object],
) -> list[dict[str, str]]:
    """Build strategy messages using the saved prompt and shared market context."""

    market_context_json = json.dumps(market_context, indent=2, sort_keys=True)
    user_message = (
        f"TICKER: {ticker}\n"
        "Use only the following MARKET_CONTEXT_JSON.\n"
        "Return exactly one JSON object that satisfies the prompt instructions.\n"
        f"MARKET_CONTEXT_JSON:\n{market_context_json}"
    )

    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": user_message},
    ]


def _validate_strategy_payload(payload: Any, expected_name: str) -> StrategyOutput:
    """Validate parsed model output against the StrategyOutput schema."""

    try:
        strategy_output = StrategyOutput.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Strategy output validation failed: {exc}") from exc

    if strategy_output.name != expected_name:
        raise ValueError(
            f"Strategy output name must be '{expected_name}', received '{strategy_output.name}'."
        )

    return strategy_output


def _run_strategy(
    ticker: str,
    market_context: dict[str, object],
    strategy_name: str,
    prompt_filename: str,
) -> StrategyOutput:
    """Load one strategy prompt and return a validated strategy result."""

    resolved_ticker = _resolve_strategy_ticker(ticker, market_context)
    prompt_text = load_strategy_prompt(prompt_filename)
    messages = _build_strategy_messages(prompt_text, resolved_ticker, market_context)
    return call_validated_json_completion(
        messages=messages,
        validate_payload=lambda payload: _validate_strategy_payload(payload, strategy_name),
        max_completion_tokens=STRATEGY_MAX_COMPLETION_TOKENS,
    )


def run_momentum_trader(ticker: str, market_context: dict[str, object]) -> StrategyOutput:
    """Run the Momentum Trader strategy against shared market data."""

    return _run_strategy(
        ticker=ticker,
        market_context=market_context,
        strategy_name=MOMENTUM_TRADER,
        prompt_filename=STRATEGY_A_PROMPT_FILE,
    )


def run_value_contrarian(ticker: str, market_context: dict[str, object]) -> StrategyOutput:
    """Run the Value Contrarian strategy against shared market data."""

    return _run_strategy(
        ticker=ticker,
        market_context=market_context,
        strategy_name=VALUE_CONTRARIAN,
        prompt_filename=STRATEGY_B_PROMPT_FILE,
    )
