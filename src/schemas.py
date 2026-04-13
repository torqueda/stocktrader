"""Shared Pydantic models for assignment outputs."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Decision(str, Enum):
    """Supported trading decisions."""

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class StrategyOutput(BaseModel):
    """Structured output for a single strategy agent."""

    name: str = Field(..., min_length=1)
    decision: Decision
    confidence: int = Field(..., ge=1, le=10)
    justification: str = Field(..., min_length=1)


class EvaluatorOutput(BaseModel):
    """Structured output for the evaluator agent."""

    agents_agree: bool
    analysis: str = Field(..., min_length=1)


class StockAnalysisOutput(BaseModel):
    """Full per-stock analysis record."""

    ticker: str = Field(..., min_length=1)
    run_date: str = Field(..., min_length=1)
    market_data_summary: dict[str, object]
    strategy_a: StrategyOutput
    strategy_b: StrategyOutput
    evaluator: EvaluatorOutput


class SummaryRow(BaseModel):
    """Condensed summary row for one analyzed ticker."""

    ticker: str = Field(..., min_length=1)
    a_decision: Decision
    b_decision: Decision
    agree: bool


class SummaryOutput(BaseModel):
    """Batch summary across analyzed tickers."""

    strategies: list[str]
    stocks_analyzed: list[str]
    total_agreements: int = Field(..., ge=0)
    total_disagreements: int = Field(..., ge=0)
    results: list[SummaryRow]
