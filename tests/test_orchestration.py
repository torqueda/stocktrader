"""Tests for end-to-end single-ticker orchestration and saving."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.orchestration import analyze_ticker, get_output_path_for_ticker, save_stock_analysis
from src.schemas import Decision, EvaluatorOutput, StockAnalysisOutput, StrategyOutput


def make_market_context() -> dict[str, object]:
    """Build a deterministic market context for orchestration tests."""

    return {
        "ticker": "AAPL",
        "history_metadata": {
            "period_requested": "1y",
            "interval": "1d",
            "rows_fetched": 250,
            "latest_trading_date": "2026-04-13",
        },
        "price_summary": {
            "current_price": 259.2,
            "price_30d_ago": 264.2,
            "pct_change_30d": -1.89,
            "high_52w": 288.61,
            "low_52w": 189.81,
        },
        "momentum_features": {
            "moving_avg_20d": 253.74,
            "moving_avg_50d": 260.9,
            "price_vs_ma20_pct": 2.15,
            "price_vs_ma50_pct": -0.65,
            "avg_volume_30d": 55218234,
            "avg_volume_90d": 63194210,
            "volume_trend_ratio": 0.874,
            "daily_returns_30d": [0.45] * 30,
        },
        "value_contrarian_features": {
            "distance_from_52w_high_pct": -10.19,
            "distance_from_52w_low_pct": 36.56,
            "recent_drawdown_pct": -9.43,
            "rsi_14": 53.64,
        },
        "raw_window_summaries": {
            "last_5_closes": [255.2, 256.1, 257.0, 258.6, 259.2],
            "last_5_volumes": [51000000, 52000000, 53000000, 54000000, 55000000],
            "recent_peak_close_90d": 286.2,
            "recent_low_close_90d": 233.4,
        },
    }


def make_strategy_output(name: str, decision: Decision) -> StrategyOutput:
    """Build a valid StrategyOutput for orchestration tests."""

    return StrategyOutput(
        name=name,
        decision=decision,
        confidence=7,
        justification="Price is 259.2. Signal quality is moderate. The stance is grounded in the supplied context.",
    )


def make_analysis_output() -> StockAnalysisOutput:
    """Build a valid StockAnalysisOutput for save tests."""

    return StockAnalysisOutput(
        ticker="AAPL",
        run_date="2026-04-13",
        market_data_summary={
            "ticker": "AAPL",
            "latest_trading_date": "2026-04-13",
            "rows_fetched": 250,
            "current_price": 259.2,
            "pct_change_30d": -1.89,
            "moving_avg_20d": 253.74,
            "moving_avg_50d": 260.9,
            "volume_trend_ratio": 0.874,
            "distance_from_52w_high_pct": -10.19,
            "distance_from_52w_low_pct": 36.56,
            "recent_drawdown_pct": -9.43,
            "rsi_14": 53.64,
        },
        strategy_a=make_strategy_output("Momentum Trader", Decision.BUY),
        strategy_b=make_strategy_output("Value Contrarian", Decision.HOLD),
        evaluator=EvaluatorOutput(
            agents_agree=False,
            analysis="The strategies diverge because they prioritize different signals in the same market summary.",
        ),
    )


def test_analyze_ticker_runs_full_pipeline_once_and_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []
    shared_market_context = make_market_context()
    compact_summary = {
        "ticker": "AAPL",
        "latest_trading_date": "2026-04-13",
        "rows_fetched": 250,
        "current_price": 259.2,
        "pct_change_30d": -1.89,
        "moving_avg_20d": 253.74,
        "moving_avg_50d": 260.9,
        "volume_trend_ratio": 0.874,
        "distance_from_52w_high_pct": -10.19,
        "distance_from_52w_low_pct": 36.56,
        "recent_drawdown_pct": -9.43,
        "rsi_14": 53.64,
    }

    def fake_build_market_context(ticker: str) -> dict[str, object]:
        call_order.append("market")
        assert ticker == "AAPL"
        return shared_market_context

    def fake_run_momentum_trader(ticker: str, market_context: dict[str, object]) -> StrategyOutput:
        call_order.append("strategy_a")
        assert ticker == "AAPL"
        assert market_context is shared_market_context
        return make_strategy_output("Momentum Trader", Decision.BUY)

    def fake_run_value_contrarian(ticker: str, market_context: dict[str, object]) -> StrategyOutput:
        call_order.append("strategy_b")
        assert ticker == "AAPL"
        assert market_context is shared_market_context
        return make_strategy_output("Value Contrarian", Decision.HOLD)

    def fake_evaluate_strategies(
        ticker: str,
        market_context: dict[str, object],
        strategy_a: StrategyOutput,
        strategy_b: StrategyOutput,
    ) -> EvaluatorOutput:
        call_order.append("evaluator")
        assert ticker == "AAPL"
        assert market_context is shared_market_context
        assert strategy_a.name == "Momentum Trader"
        assert strategy_b.name == "Value Contrarian"
        return EvaluatorOutput(
            agents_agree=False,
            analysis="The strategies diverge because they prioritize different signals.",
        )

    def fake_build_market_data_summary(market_context: dict[str, object]) -> dict[str, object]:
        call_order.append("summary")
        assert market_context is shared_market_context
        return compact_summary

    monkeypatch.setattr("src.orchestration.build_market_context", fake_build_market_context)
    monkeypatch.setattr("src.orchestration.run_momentum_trader", fake_run_momentum_trader)
    monkeypatch.setattr("src.orchestration.run_value_contrarian", fake_run_value_contrarian)
    monkeypatch.setattr("src.orchestration.evaluate_strategies", fake_evaluate_strategies)
    monkeypatch.setattr("src.orchestration.build_market_data_summary", fake_build_market_data_summary)
    monkeypatch.setattr("src.orchestration.get_run_date", lambda: "2026-04-13")

    result = analyze_ticker(" aapl ")

    assert isinstance(result, StockAnalysisOutput)
    assert result.ticker == "AAPL"
    assert result.market_data_summary == compact_summary
    assert "raw_window_summaries" not in result.market_data_summary
    assert call_order == ["market", "strategy_a", "strategy_b", "evaluator", "summary"]


def test_get_output_path_for_ticker_normalizes_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.orchestration.get_output_dir", lambda: tmp_path)

    output_path = get_output_path_for_ticker(" aapl ")

    assert output_path == tmp_path / "AAPL.json"


def test_save_stock_analysis_creates_output_directory_and_writes_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "nested" / "outputs"
    monkeypatch.setattr("src.orchestration.get_output_dir", lambda: output_dir)
    analysis_output = make_analysis_output()

    saved_path = save_stock_analysis(analysis_output)

    assert saved_path == output_dir / "AAPL.json"
    assert saved_path.is_file()
    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    assert payload["ticker"] == "AAPL"
    assert payload["strategy_a"]["name"] == "Momentum Trader"
    assert payload["evaluator"]["agents_agree"] is False
