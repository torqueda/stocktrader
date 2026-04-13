"""Tests for pure market-data indicator helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.indicators import (
    compute_daily_returns,
    compute_distance_from_52_week_high,
    compute_distance_from_52_week_low,
    compute_drawdown_from_recent_peak,
    compute_moving_average,
    compute_rsi,
    compute_volume_trend,
)


def test_compute_moving_average_on_simple_series() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compute_moving_average(series, window=3)
    assert result == pytest.approx(4.0)


def test_compute_daily_returns_returns_recent_percentages() -> None:
    series = pd.Series([100.0, 105.0, 110.25, 99.225])
    result = compute_daily_returns(series, periods=3)
    assert result == pytest.approx([5.0, 5.0, -10.0], abs=0.01)


def test_compute_rsi_returns_bounded_value() -> None:
    series = pd.Series([44, 45, 46, 45, 47, 48, 47, 49, 50, 51, 50, 52, 54, 53, 55, 56], dtype=float)
    result = compute_rsi(series, window=14)
    assert 0.0 <= result <= 100.0
    assert result > 50.0


def test_compute_volume_trend_returns_short_long_ratio() -> None:
    series = pd.Series([100.0] * 60 + [200.0] * 30)
    result = compute_volume_trend(series, short_window=30, long_window=90)

    assert result["avg_volume_short"] == pytest.approx(200.0)
    assert result["avg_volume_long"] == pytest.approx(133.33, abs=0.01)
    assert result["volume_trend_ratio"] == pytest.approx(1.5, abs=0.01)


def test_compute_drawdown_from_recent_peak_behaves_as_expected() -> None:
    series = pd.Series([100.0, 110.0, 105.0, 90.0])
    result = compute_drawdown_from_recent_peak(series, lookback_window=4)
    assert result == pytest.approx(-18.18, abs=0.01)


def test_distance_from_52_week_helpers_return_expected_percentages() -> None:
    high_series = pd.Series([80.0, 120.0, 110.0])
    low_series = pd.Series([80.0, 90.0, 85.0])

    distance_from_high = compute_distance_from_52_week_high(high_series, current_price=100.0)
    distance_from_low = compute_distance_from_52_week_low(low_series, current_price=100.0)

    assert distance_from_high == pytest.approx(-16.67, abs=0.01)
    assert distance_from_low == pytest.approx(25.0, abs=0.01)
