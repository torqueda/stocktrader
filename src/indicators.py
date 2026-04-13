"""Pure indicator helpers for market-data processing."""

from __future__ import annotations

import pandas as pd


def _clean_series(values: pd.Series, label: str) -> pd.Series:
    """Return a cleaned numeric series or raise a helpful error."""

    cleaned = values.dropna().astype(float)
    if cleaned.empty:
        raise ValueError(f"{label} requires at least 1 observation.")
    return cleaned


def _require_length(values: pd.Series, required: int, label: str) -> pd.Series:
    """Validate that a series contains the required number of observations."""

    cleaned = _clean_series(values, label)
    if len(cleaned) < required:
        raise ValueError(
            f"{label} requires at least {required} observations; received {len(cleaned)}."
        )
    return cleaned


def compute_daily_returns(prices: pd.Series, periods: int = 30) -> list[float]:
    """Return the most recent daily percentage returns."""

    if periods < 1:
        raise ValueError("periods must be at least 1.")

    cleaned = _require_length(prices, periods + 1, "Daily returns")
    returns = cleaned.pct_change().dropna().tail(periods) * 100
    if len(returns) < periods:
        raise ValueError(
            f"Daily returns requires at least {periods + 1} observations; received {len(cleaned)}."
        )
    return [float(value) for value in returns.tolist()]


def compute_moving_average(values: pd.Series, window: int) -> float:
    """Return the trailing moving average for a series."""

    if window < 1:
        raise ValueError("window must be at least 1.")

    cleaned = _require_length(values, window, "Moving average")
    return float(cleaned.tail(window).mean())


def compute_volume_trend(
    volumes: pd.Series,
    short_window: int = 30,
    long_window: int = 90,
) -> dict[str, float]:
    """Return short/long average volumes and the short-to-long ratio."""

    if short_window < 1 or long_window < 1:
        raise ValueError("Volume windows must be at least 1.")
    if short_window > long_window:
        raise ValueError("short_window cannot be greater than long_window.")

    cleaned = _require_length(volumes, long_window, "Volume trend")
    avg_short = float(cleaned.tail(short_window).mean())
    avg_long = float(cleaned.tail(long_window).mean())
    if avg_long == 0:
        raise ValueError("Volume trend cannot be computed when long-window average volume is zero.")

    return {
        "avg_volume_short": avg_short,
        "avg_volume_long": avg_long,
        "volume_trend_ratio": avg_short / avg_long,
    }


def compute_rsi(prices: pd.Series, window: int = 14) -> float:
    """Return the latest RSI value using Wilder-style smoothing."""

    if window < 1:
        raise ValueError("window must be at least 1.")

    cleaned = _require_length(prices, window + 1, "RSI")
    deltas = cleaned.diff().dropna()
    gains = deltas.clip(lower=0)
    losses = -deltas.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    latest_gain = float(avg_gain.iloc[-1])
    latest_loss = float(avg_loss.iloc[-1])

    if latest_loss == 0:
        return 100.0 if latest_gain > 0 else 50.0

    relative_strength = latest_gain / latest_loss
    return float(100 - (100 / (1 + relative_strength)))


def compute_drawdown_from_recent_peak(prices: pd.Series, lookback_window: int = 90) -> float:
    """Return the percentage drawdown from the recent peak to the latest price."""

    if lookback_window < 1:
        raise ValueError("lookback_window must be at least 1.")

    cleaned = _require_length(prices, lookback_window, "Recent drawdown")
    recent_window = cleaned.tail(lookback_window)
    recent_peak = float(recent_window.max())
    current_price = float(recent_window.iloc[-1])
    if recent_peak == 0:
        raise ValueError("Recent drawdown cannot be computed when the recent peak is zero.")

    return ((current_price - recent_peak) / recent_peak) * 100


def compute_distance_from_52_week_high(
    high_prices: pd.Series,
    current_price: float,
    lookback_window: int = 252,
) -> float:
    """Return the current price distance from the trailing 52-week high in percent."""

    if lookback_window < 1:
        raise ValueError("lookback_window must be at least 1.")

    cleaned = _clean_series(high_prices, "52-week high distance")
    trailing_high = float(cleaned.tail(min(len(cleaned), lookback_window)).max())
    if trailing_high == 0:
        raise ValueError("52-week high distance cannot be computed when the trailing high is zero.")

    return ((float(current_price) - trailing_high) / trailing_high) * 100


def compute_distance_from_52_week_low(
    low_prices: pd.Series,
    current_price: float,
    lookback_window: int = 252,
) -> float:
    """Return the current price distance from the trailing 52-week low in percent."""

    if lookback_window < 1:
        raise ValueError("lookback_window must be at least 1.")

    cleaned = _clean_series(low_prices, "52-week low distance")
    trailing_low = float(cleaned.tail(min(len(cleaned), lookback_window)).min())
    if trailing_low == 0:
        raise ValueError("52-week low distance cannot be computed when the trailing low is zero.")

    return ((float(current_price) - trailing_low) / trailing_low) * 100
