"""Tests for orchestration quality-control helpers and artifact review flows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.main import run_review_output_command, run_review_summary_command
from src.orchestration import (
    build_summary_output,
    load_stock_analysis,
    load_summary_output,
    save_stock_analysis,
    save_summary_output,
    validate_stock_analysis_output,
    validate_summary_output,
)
from src.schemas import Decision, EvaluatorOutput, StockAnalysisOutput, SummaryOutput, SummaryRow, StrategyOutput


def make_strategy_output(name: str, decision: Decision) -> StrategyOutput:
    """Build a valid StrategyOutput for QC tests."""

    return StrategyOutput(
        name=name,
        decision=decision,
        confidence=7,
        justification="The evidence is numeric. The rationale is compact. The conclusion follows the supplied context.",
    )


def make_stock_analysis_output(
    ticker: str = "AAPL",
    a_decision: Decision = Decision.BUY,
    b_decision: Decision = Decision.HOLD,
    *,
    agents_agree: bool | None = None,
) -> StockAnalysisOutput:
    """Build a valid StockAnalysisOutput for QC tests."""

    if agents_agree is None:
        agents_agree = a_decision == b_decision

    return StockAnalysisOutput(
        ticker=ticker,
        run_date="2026-04-13",
        market_data_summary={
            "ticker": ticker,
            "latest_trading_date": "2026-04-13",
            "rows_fetched": 250,
            "current_price": 100.0,
            "pct_change_30d": 1.5,
            "moving_avg_20d": 99.4,
            "moving_avg_50d": 97.2,
            "volume_trend_ratio": 1.082,
            "distance_from_52w_high_pct": -6.5,
            "distance_from_52w_low_pct": 18.2,
            "recent_drawdown_pct": -4.1,
            "rsi_14": 56.4,
        },
        strategy_a=make_strategy_output("Momentum Trader", a_decision),
        strategy_b=make_strategy_output("Value Contrarian", b_decision),
        evaluator=EvaluatorOutput(
            agents_agree=agents_agree,
            analysis="The evaluator compares how both strategies interpret the same compact market evidence.",
        ),
    )


def test_validate_stock_analysis_output_rejects_agreement_mismatch() -> None:
    output = make_stock_analysis_output(a_decision=Decision.BUY, b_decision=Decision.SELL, agents_agree=True)

    with pytest.raises(ValueError, match="agents_agree"):
        validate_stock_analysis_output(output)


def test_validate_stock_analysis_output_rejects_market_summary_ticker_mismatch() -> None:
    output = make_stock_analysis_output()
    output.market_data_summary["ticker"] = "MSFT"

    with pytest.raises(ValueError, match="market_data_summary ticker"):
        validate_stock_analysis_output(output)


def test_validate_summary_output_rejects_invalid_totals() -> None:
    summary = SummaryOutput(
        strategies=["Momentum Trader", "Value Contrarian"],
        stocks_analyzed=["AAPL"],
        total_agreements=2,
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

    with pytest.raises(ValueError, match="Summary totals are inconsistent"):
        validate_summary_output(summary)


def test_validate_summary_output_rejects_invalid_row_agree_flag() -> None:
    summary = SummaryOutput(
        strategies=["Momentum Trader", "Value Contrarian"],
        stocks_analyzed=["AAPL"],
        total_agreements=0,
        total_disagreements=1,
        results=[
            SummaryRow(
                ticker="AAPL",
                a_decision=Decision.BUY,
                b_decision=Decision.BUY,
                agree=False,
            )
        ],
    )

    with pytest.raises(ValueError, match="agree flag"):
        validate_summary_output(summary)


def test_load_stock_analysis_validates_saved_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.orchestration.get_output_dir", lambda: tmp_path)
    output = make_stock_analysis_output()

    save_stock_analysis(output)
    loaded = load_stock_analysis(" aapl ")

    assert loaded.ticker == "AAPL"
    assert loaded.market_data_summary["ticker"] == "AAPL"
    assert loaded.evaluator.agents_agree is False


def test_load_stock_analysis_rejects_invalid_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.orchestration.get_output_dir", lambda: tmp_path)
    invalid_path = tmp_path / "AAPL.json"
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.write_text(json.dumps({"ticker": "AAPL"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Saved stock analysis artifact is invalid"):
        load_stock_analysis("AAPL")


def test_review_helpers_print_validated_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("src.orchestration.get_output_dir", lambda: tmp_path)
    output = make_stock_analysis_output()
    save_stock_analysis(output)
    summary = build_summary_output([output])
    save_summary_output(summary)

    output_status = run_review_output_command("AAPL")
    output_capture = capsys.readouterr()
    assert output_status == 0
    assert '"ticker": "AAPL"' in output_capture.out
    assert output_capture.err == ""

    summary_status = run_review_summary_command()
    summary_capture = capsys.readouterr()
    assert summary_status == 0
    assert '"stocks_analyzed": [' in summary_capture.out
    assert summary_capture.err == ""


def test_load_summary_output_validates_saved_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.orchestration.get_output_dir", lambda: tmp_path)
    summary = build_summary_output(
        [
            make_stock_analysis_output("AAPL", Decision.BUY, Decision.BUY),
            make_stock_analysis_output("MSFT", Decision.HOLD, Decision.SELL),
        ]
    )

    save_summary_output(summary)
    loaded = load_summary_output()

    assert loaded.total_agreements == 1
    assert loaded.total_disagreements == 1
    assert loaded.results[1].ticker == "MSFT"
