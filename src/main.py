"""Command-line entry point for the stocktrader project."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .config import get_groq_api_key
from .llm_client import verify_groq_connection
from .market_data import (
    build_market_context,
    build_market_data_summary,
    verify_yfinance_connection,
)
from .orchestration import analyze_ticker, save_stock_analysis
from .strategies import run_momentum_trader, run_value_contrarian


def _print_json(payload: object) -> None:
    """Print a JSON payload with indentation."""

    print(json.dumps(payload, indent=2))


def check_groq_configuration() -> bool:
    """Return whether a Groq API key is present."""

    return bool(get_groq_api_key())


def run_verification(ticker: str, include_groq: bool = True) -> int:
    """Run the environment verification flow."""

    market_ready = False
    market_summary: dict[str, object] | None = None
    market_error: str | None = None

    try:
        market_summary = verify_yfinance_connection(ticker)
        market_ready = True
    except Exception as exc:  # pragma: no cover - exercised manually
        market_error = str(exc)

    groq_key_present = check_groq_configuration()
    groq_ready = False
    groq_summary: dict[str, object] | None = None
    groq_error: str | None = None
    groq_status = "SKIPPED"

    if include_groq:
        if not groq_key_present:
            groq_status = "FAIL"
            groq_error = "GROQ_API_KEY is not configured."
        else:
            try:
                groq_summary = verify_groq_connection()
                groq_ready = True
                groq_status = "PASS"
            except Exception as exc:  # pragma: no cover - exercised manually
                groq_error = str(exc)
                groq_status = "FAIL"

    print("Environment verification")
    print(f"yfinance check: {'PASS' if market_ready else 'FAIL'}")
    if market_summary is not None:
        _print_json(market_summary)
    if market_error is not None:
        print(f"yfinance error: {market_error}")

    if include_groq:
        print(f"Groq smoke test: {groq_status}")
        if groq_summary is not None:
            _print_json(groq_summary)
        if groq_error is not None:
            print(f"Groq error: {groq_error}")
    else:
        print("Groq smoke test: SKIPPED")

    print(f"GROQ_API_KEY present: {'YES' if groq_key_present else 'NO'}")

    if market_ready and (groq_ready if include_groq else True):
        print("Environment is ready for the next step.")
        return 0

    print("Environment is not fully ready yet. Fix the failed checks above and rerun verification.")
    return 1


def run_groq_verification_only() -> int:
    """Run only the Groq smoke test."""

    try:
        payload = verify_groq_connection()
    except Exception as exc:  # pragma: no cover - exercised manually
        print(f"Groq verification failed: {exc}", file=sys.stderr)
        return 1

    _print_json(payload)
    return 0


def run_market_data_command(ticker: str, summary_only: bool) -> int:
    """Build and print market-data output for one ticker."""

    try:
        context = build_market_context(ticker)
        payload = build_market_data_summary(context) if summary_only else context
    except Exception as exc:
        print(f"Market data error: {exc}", file=sys.stderr)
        return 1

    _print_json(payload)
    return 0


def run_strategy_command(ticker: str, use_strategy_a: bool) -> int:
    """Build market data, run one strategy, and print the validated output."""

    try:
        market_context = build_market_context(ticker)
        result = (
            run_momentum_trader(ticker, market_context)
            if use_strategy_a
            else run_value_contrarian(ticker, market_context)
        )
    except Exception as exc:
        print(f"Strategy error: {exc}", file=sys.stderr)
        return 1

    _print_json(result.model_dump(mode="json"))
    return 0


def run_analysis_command(ticker: str, save_output: bool) -> int:
    """Run the full single-ticker pipeline and optionally save the JSON artifact."""

    try:
        result = analyze_ticker(ticker)
        saved_path = save_stock_analysis(result) if save_output else None
    except Exception as exc:
        print(f"Analysis error: {exc}", file=sys.stderr)
        return 1

    _print_json(result.model_dump(mode="json"))
    if saved_path is not None:
        print(f"Saved output: {saved_path}", file=sys.stderr)

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description="stocktrader CLI")
    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument("--verify", action="store_true", help="Run yfinance and Groq verification checks.")
    command_group.add_argument("--verify-groq", action="store_true", help="Run only the Groq smoke test.")
    command_group.add_argument("--market-data", action="store_true", help="Build the full market-data context.")
    command_group.add_argument("--market-summary", action="store_true", help="Build the compact market-data summary.")
    command_group.add_argument("--strategy-a", action="store_true", help="Run the Momentum Trader on one ticker.")
    command_group.add_argument(
        "--strategy-b",
        action="store_true",
        help="Run the Value Contrarian on one ticker.",
    )
    command_group.add_argument("--analyze", action="store_true", help="Run the full single-ticker analysis pipeline.")
    parser.add_argument("--ticker", default="AAPL", help="Ticker to use for the yfinance verification fetch.")
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save single-ticker analysis JSON to outputs/{TICKER}.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.save_output and not args.analyze:
        print("--save-output requires --analyze.", file=sys.stderr)
        return 2

    if args.verify:
        return run_verification(args.ticker, include_groq=True)
    if args.verify_groq:
        return run_groq_verification_only()
    if args.market_data:
        return run_market_data_command(args.ticker, summary_only=False)
    if args.market_summary:
        return run_market_data_command(args.ticker, summary_only=True)
    if args.strategy_a:
        return run_strategy_command(args.ticker, use_strategy_a=True)
    if args.strategy_b:
        return run_strategy_command(args.ticker, use_strategy_a=False)
    if args.analyze:
        return run_analysis_command(args.ticker, save_output=args.save_output)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
