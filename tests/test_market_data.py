"""Tests for market-data shaping and serialization."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.market_data import build_market_context, build_market_data_summary


def make_history_frame(rows: int = 260) -> pd.DataFrame:
    """Build a deterministic daily OHLCV history frame."""

    dates = pd.date_range("2025-01-01", periods=rows, freq="B")
    close_values = np.linspace(100.0, 160.0, rows)
    if rows >= 30:
        close_values[-30:] -= np.linspace(0.0, 12.0, 30)

    close_series = pd.Series(close_values, index=dates)
    frame = pd.DataFrame(
        {
            "Open": close_series - 1.0,
            "High": close_series + 2.0,
            "Low": close_series - 2.0,
            "Close": close_series,
            "Volume": np.linspace(1_000_000, 1_600_000, rows).round(),
        },
        index=dates,
    )
    return frame


class FakeTicker:
    """Simple yfinance stub for tests."""

    def __init__(self, history_frame: pd.DataFrame) -> None:
        self._history_frame = history_frame

    def history(self, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
        assert interval == "1d"
        assert auto_adjust is False
        return self._history_frame.copy()


def patch_ticker(monkeypatch: pytest.MonkeyPatch, history_frame: pd.DataFrame) -> None:
    """Patch yfinance.Ticker to return a fake history frame."""

    monkeypatch.setattr(
        "src.market_data.yf.Ticker",
        lambda symbol: FakeTicker(history_frame),
    )


def test_build_market_context_returns_expected_top_level_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_ticker(monkeypatch, make_history_frame())

    context = build_market_context(" aapl ")

    assert context["ticker"] == "AAPL"
    assert set(context.keys()) == {
        "ticker",
        "history_metadata",
        "price_summary",
        "momentum_features",
        "value_contrarian_features",
        "raw_window_summaries",
    }
    assert len(context["momentum_features"]["daily_returns_30d"]) == 30


def test_build_market_context_raises_on_insufficient_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_ticker(monkeypatch, make_history_frame(rows=60))

    with pytest.raises(ValueError, match="Insufficient history"):
        build_market_context("AAPL")


def test_build_market_context_is_json_serializable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_ticker(monkeypatch, make_history_frame())

    context = build_market_context("AAPL")
    serialized = json.dumps(context)

    assert isinstance(serialized, str)


def test_build_market_data_summary_returns_compact_core_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_ticker(monkeypatch, make_history_frame())

    context = build_market_context("AAPL")
    summary = build_market_data_summary(context)

    assert summary["ticker"] == "AAPL"
    assert "current_price" in summary
    assert "volume_trend_ratio" in summary
    assert "rsi_14" in summary
