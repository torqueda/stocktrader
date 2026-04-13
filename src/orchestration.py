"""Orchestration entry points for the stocktrader project."""

from __future__ import annotations

from .schemas import StockAnalysisOutput


def analyze_ticker(ticker: str) -> StockAnalysisOutput:
    """Run the end-to-end analysis pipeline for one ticker."""

    raise NotImplementedError("Later phases will implement orchestration logic.")
