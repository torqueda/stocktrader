"""Shared Pydantic models for assignment outputs."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _count_sentences(text: str) -> int:
    """Count sentences using lightweight punctuation-based splitting."""

    parts = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text.strip()) if part.strip()]
    return len(parts)


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

    @field_validator("name", "justification", mode="before")
    @classmethod
    def strip_text_fields(cls, value: object) -> object:
        """Strip leading and trailing whitespace from text fields."""

        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Ensure the strategy name is not empty after stripping."""

        if not value:
            raise ValueError("name must be non-empty.")
        return value

    @field_validator("justification")
    @classmethod
    def validate_justification(cls, value: str) -> str:
        """Ensure the justification contains 3 to 5 sentences."""

        if not value:
            raise ValueError("justification must be non-empty.")

        sentence_count = _count_sentences(value)
        if sentence_count < 3 or sentence_count > 5:
            raise ValueError("justification must contain 3 to 5 sentences.")

        return value


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
