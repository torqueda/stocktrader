"""Tests for the shared Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    Decision,
    EvaluatorOutput,
    StockAnalysisOutput,
    StrategyOutput,
    SummaryOutput,
    SummaryRow,
)


def make_strategy_output(name: str = "Momentum Trader") -> StrategyOutput:
    """Build a valid StrategyOutput for tests."""

    return StrategyOutput(
        name=name,
        decision=Decision.BUY,
        confidence=7,
        justification="Price strength is improving. Trend conditions look constructive. Risk remains manageable.",
    )


def test_strategy_output_valid() -> None:
    strategy = make_strategy_output()
    assert strategy.decision is Decision.BUY
    assert strategy.confidence == 7


def test_strategy_output_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        StrategyOutput(
            name="Momentum Trader",
            decision=Decision.BUY,
            confidence=11,
            justification="This should fail because confidence is too high.",
        )


def test_stock_analysis_output_valid() -> None:
    output = StockAnalysisOutput(
        ticker="AAPL",
        run_date="2026-04-13",
        market_data_summary={"latest_close": 180.12, "rows_fetched": 5},
        strategy_a=make_strategy_output("Momentum Trader"),
        strategy_b=make_strategy_output("Value Contrarian"),
        evaluator=EvaluatorOutput(
            agents_agree=True,
            analysis="Both agents support a positive stance based on the same market context.",
        ),
    )

    assert output.ticker == "AAPL"
    assert output.strategy_b.name == "Value Contrarian"


def test_summary_output_valid() -> None:
    summary = SummaryOutput(
        strategies=["Momentum Trader", "Value Contrarian"],
        stocks_analyzed=["AAPL"],
        total_agreements=1,
        total_disagreements=0,
        results=[
            SummaryRow(
                ticker="AAPL",
                a_decision=Decision.BUY,
                b_decision=Decision.BUY,
                agree=True,
            )
        ],
    )

    assert summary.total_agreements == 1
    assert summary.results[0].agree is True
