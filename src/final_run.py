"""Frozen stock selection for the final graded run."""

from __future__ import annotations

from .market_data import normalize_ticker_symbol

GRADED_RUN_MAPPING = {
    "steady_large_cap": "WMT",
    "volatile_momentum": "NVDA",
    "recent_decliner": "UNH",
    "sideways": "PG",
}


def get_graded_run_mapping() -> dict[str, str]:
    """Return the frozen graded stock mapping with normalized tickers."""

    return {
        category: normalize_ticker_symbol(ticker)
        for category, ticker in GRADED_RUN_MAPPING.items()
    }


def get_graded_run_tickers() -> list[str]:
    """Return the frozen graded ticker list in assignment order."""

    return list(get_graded_run_mapping().values())
