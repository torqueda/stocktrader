"""Tests for batch orchestration and summary output generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.orchestration import analyze_tickers, build_summary_output, save_summary_output
from src.schemas import Decision, EvaluatorOutput, StockAnalysisOutput, StrategyOutput


def make_stock_analysis_output(
    ticker: str,
    a_decision: Decision,
    b_decision: Decision,
) -> StockAnalysisOutput:
    """Build a valid StockAnalysisOutput for batch tests."""

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
        strategy_a=StrategyOutput(
            name="Momentum Trader",
            decision=a_decision,
            confidence=7,
            justification="Price is 100.0. Trend is stable. The decision follows the provided evidence.",
        ),
        strategy_b=StrategyOutput(
            name="Value Contrarian",
            decision=b_decision,
            confidence=6,
            justification="Price is 100.0. Range positioning matters. The decision follows the supplied signals.",
        ),
        evaluator=EvaluatorOutput(
            agents_agree=a_decision == b_decision,
            analysis="The evaluator reflects how the two strategies compare on the same ticker.",
        ),
    )


def test_analyze_tickers_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="At least one ticker is required"):
        analyze_tickers([])


def test_analyze_tickers_normalizes_deduplicates_and_reuses_single_ticker_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzed_tickers: list[str] = []
    saved_tickers: list[str] = []

    def fake_analyze_ticker(ticker: str) -> StockAnalysisOutput:
        analyzed_tickers.append(ticker)
        return make_stock_analysis_output(ticker, Decision.BUY, Decision.HOLD)

    def fake_save_stock_analysis(output: StockAnalysisOutput) -> Path:
        saved_tickers.append(output.ticker)
        return Path(f"/tmp/{output.ticker}.json")

    monkeypatch.setattr("src.orchestration.analyze_ticker", fake_analyze_ticker)
    monkeypatch.setattr("src.orchestration.save_stock_analysis", fake_save_stock_analysis)

    results = analyze_tickers([" aapl ", "MSFT", "AAPL", " nvda ", "MSFT"], save_outputs=True)

    assert [result.ticker for result in results] == ["AAPL", "MSFT", "NVDA"]
    assert analyzed_tickers == ["AAPL", "MSFT", "NVDA"]
    assert saved_tickers == ["AAPL", "MSFT", "NVDA"]


def test_analyze_tickers_skips_save_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.orchestration.analyze_ticker",
        lambda ticker: make_stock_analysis_output(ticker, Decision.BUY, Decision.BUY),
    )
    monkeypatch.setattr(
        "src.orchestration.save_stock_analysis",
        lambda output: (_ for _ in ()).throw(AssertionError("save_stock_analysis should not be called")),
    )

    results = analyze_tickers(["AAPL", "MSFT"], save_outputs=False)

    assert [result.ticker for result in results] == ["AAPL", "MSFT"]


def test_build_summary_output_produces_expected_counts_and_rows() -> None:
    results = [
        make_stock_analysis_output("AAPL", Decision.BUY, Decision.BUY),
        make_stock_analysis_output("MSFT", Decision.HOLD, Decision.SELL),
        make_stock_analysis_output("NVDA", Decision.SELL, Decision.SELL),
    ]

    summary = build_summary_output(results)

    assert summary.strategies == ["Momentum Trader", "Value Contrarian"]
    assert summary.stocks_analyzed == ["AAPL", "MSFT", "NVDA"]
    assert summary.total_agreements == 2
    assert summary.total_disagreements == 1
    assert summary.results[1].ticker == "MSFT"
    assert summary.results[1].agree is False


def test_save_summary_output_writes_summary_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    summary = build_summary_output(
        [
            make_stock_analysis_output("AAPL", Decision.BUY, Decision.BUY),
            make_stock_analysis_output("MSFT", Decision.HOLD, Decision.SELL),
        ]
    )
    monkeypatch.setattr("src.orchestration.get_output_dir", lambda: tmp_path)

    saved_path = save_summary_output(summary)

    assert saved_path == tmp_path / "summary.json"
    assert saved_path.is_file()
    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    assert payload["stocks_analyzed"] == ["AAPL", "MSFT"]
    assert payload["total_agreements"] == 1
    assert payload["total_disagreements"] == 1
