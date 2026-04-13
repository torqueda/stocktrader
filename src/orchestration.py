"""Orchestration entry points for the stocktrader project."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .config import get_output_dir
from .evaluator import evaluate_strategies
from .market_data import build_market_context, build_market_data_summary, normalize_ticker_symbol
from .schemas import StockAnalysisOutput
from .strategies import run_momentum_trader, run_value_contrarian


def get_run_date() -> str:
    """Return the current run date as an ISO string."""

    return date.today().isoformat()


def get_output_path_for_ticker(ticker: str) -> Path:
    """Return the output JSON path for one ticker."""

    symbol = normalize_ticker_symbol(ticker)
    return get_output_dir() / f"{symbol}.json"


def save_stock_analysis(output: StockAnalysisOutput) -> Path:
    """Write one analyzed ticker result to a pretty-printed JSON file."""

    output_path = get_output_path_for_ticker(output.ticker)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_output = json.dumps(output.model_dump(mode="json"), indent=2)
    output_path.write_text(f"{serialized_output}\n", encoding="utf-8")
    return output_path


def analyze_ticker(ticker: str) -> StockAnalysisOutput:
    """Run the end-to-end analysis pipeline for one ticker."""

    symbol = normalize_ticker_symbol(ticker)
    market_context = build_market_context(symbol)

    strategy_a = run_momentum_trader(symbol, market_context)
    strategy_b = run_value_contrarian(symbol, market_context)
    evaluator = evaluate_strategies(symbol, market_context, strategy_a, strategy_b)
    market_data_summary = build_market_data_summary(market_context)

    return StockAnalysisOutput(
        ticker=symbol,
        run_date=get_run_date(),
        market_data_summary=market_data_summary,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        evaluator=evaluator,
    )
