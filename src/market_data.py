"""Market data utilities for the stocktrader project."""

from __future__ import annotations

from typing import Final

import pandas as pd
import yfinance as yf

from .config import DEFAULT_HISTORY_PERIOD
from .indicators import (
    compute_daily_returns,
    compute_distance_from_52_week_high,
    compute_distance_from_52_week_low,
    compute_drawdown_from_recent_peak,
    compute_moving_average,
    compute_rsi,
    compute_volume_trend,
)

DAILY_INTERVAL: Final[str] = "1d"
MIN_HISTORY_ROWS: Final[int] = 90
TRADING_DAYS_52W: Final[int] = 252
RECENT_DRAWDOWN_WINDOW: Final[int] = 90
SHORT_MOVING_AVERAGE_WINDOW: Final[int] = 20
LONG_MOVING_AVERAGE_WINDOW: Final[int] = 50
SHORT_VOLUME_WINDOW: Final[int] = 30
LONG_VOLUME_WINDOW: Final[int] = 90
RSI_WINDOW: Final[int] = 14
PRICE_LOOKBACK_30D: Final[int] = 30
REQUIRED_COLUMNS: Final[tuple[str, ...]] = ("Open", "High", "Low", "Close", "Volume")


def normalize_ticker_symbol(ticker: str) -> str:
    """Return a normalized ticker symbol."""

    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker symbol cannot be empty.")
    return symbol


def _round_float(value: float, digits: int = 2) -> float:
    """Round a numeric value and return a plain Python float."""

    return round(float(value), digits)


def _round_int(value: float) -> int:
    """Round a numeric value and return a plain Python int."""

    return int(round(float(value)))


def _round_list(values: list[float], digits: int = 2) -> list[float]:
    """Round a list of numeric values."""

    return [_round_float(value, digits) for value in values]


def _format_trading_date(index_value: object) -> str:
    """Convert a pandas index value into an ISO date string."""

    return pd.Timestamp(index_value).strftime("%Y-%m-%d")


def _fetch_history(symbol: str, period: str) -> pd.DataFrame:
    """Fetch daily OHLCV history for a ticker."""

    try:
        history = yf.Ticker(symbol).history(
            period=period,
            interval=DAILY_INTERVAL,
            auto_adjust=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch market data for ticker '{symbol}': {exc}") from exc

    if history.empty:
        raise RuntimeError(f"No market data returned for ticker '{symbol}'.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in history.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise RuntimeError(f"Market data for ticker '{symbol}' is missing required columns: {missing}.")

    cleaned = history.loc[:, list(REQUIRED_COLUMNS)].dropna(subset=["Close", "Volume"]).sort_index()
    if cleaned.empty:
        raise RuntimeError(f"Market data returned for ticker '{symbol}' did not contain usable rows.")

    return cleaned


def _validate_history_length(history: pd.DataFrame, symbol: str) -> None:
    """Validate that the fetched history is long enough for Phase 2 features."""

    row_count = len(history)
    if row_count < MIN_HISTORY_ROWS:
        raise ValueError(
            f"Insufficient history for ticker '{symbol}': expected at least "
            f"{MIN_HISTORY_ROWS} daily rows, received {row_count}."
        )


def _compute_pct_change(current_value: float, prior_value: float, label: str) -> float:
    """Compute a percentage change with explicit zero-division handling."""

    if prior_value == 0:
        raise ValueError(f"{label} cannot be computed because the prior value is zero.")
    return ((current_value - prior_value) / prior_value) * 100


def build_market_context(ticker: str) -> dict[str, object]:
    """Build the Phase 2 market-data context for a ticker."""

    symbol = normalize_ticker_symbol(ticker)
    history = _fetch_history(symbol, period=DEFAULT_HISTORY_PERIOD)
    _validate_history_length(history, symbol)

    close_prices = history["Close"]
    high_prices = history["High"]
    low_prices = history["Low"]
    volumes = history["Volume"]

    current_price = float(close_prices.iloc[-1])
    price_30d_ago = float(close_prices.iloc[-(PRICE_LOOKBACK_30D + 1)])
    moving_avg_20d = compute_moving_average(close_prices, SHORT_MOVING_AVERAGE_WINDOW)
    moving_avg_50d = compute_moving_average(close_prices, LONG_MOVING_AVERAGE_WINDOW)
    volume_trend = compute_volume_trend(volumes, SHORT_VOLUME_WINDOW, LONG_VOLUME_WINDOW)
    daily_returns_30d = compute_daily_returns(close_prices, PRICE_LOOKBACK_30D)
    high_52w = float(high_prices.tail(min(len(high_prices), TRADING_DAYS_52W)).max())
    low_52w = float(low_prices.tail(min(len(low_prices), TRADING_DAYS_52W)).min())

    context = {
        "ticker": symbol,
        "history_metadata": {
            "period_requested": DEFAULT_HISTORY_PERIOD,
            "interval": DAILY_INTERVAL,
            "rows_fetched": int(len(history)),
            "latest_trading_date": _format_trading_date(history.index[-1]),
        },
        "price_summary": {
            "current_price": _round_float(current_price),
            "price_30d_ago": _round_float(price_30d_ago),
            "pct_change_30d": _round_float(
                _compute_pct_change(current_price, price_30d_ago, "30-day percentage change")
            ),
            "high_52w": _round_float(high_52w),
            "low_52w": _round_float(low_52w),
        },
        "momentum_features": {
            "moving_avg_20d": _round_float(moving_avg_20d),
            "moving_avg_50d": _round_float(moving_avg_50d),
            "price_vs_ma20_pct": _round_float(
                _compute_pct_change(current_price, moving_avg_20d, "Price vs 20-day moving average")
            ),
            "price_vs_ma50_pct": _round_float(
                _compute_pct_change(current_price, moving_avg_50d, "Price vs 50-day moving average")
            ),
            "avg_volume_30d": _round_int(volume_trend["avg_volume_short"]),
            "avg_volume_90d": _round_int(volume_trend["avg_volume_long"]),
            "volume_trend_ratio": _round_float(volume_trend["volume_trend_ratio"], 3),
            "daily_returns_30d": _round_list(daily_returns_30d, 2),
        },
        "value_contrarian_features": {
            "distance_from_52w_high_pct": _round_float(
                compute_distance_from_52_week_high(high_prices, current_price, TRADING_DAYS_52W)
            ),
            "distance_from_52w_low_pct": _round_float(
                compute_distance_from_52_week_low(low_prices, current_price, TRADING_DAYS_52W)
            ),
            "recent_drawdown_pct": _round_float(
                compute_drawdown_from_recent_peak(close_prices, RECENT_DRAWDOWN_WINDOW)
            ),
            "rsi_14": _round_float(compute_rsi(close_prices, RSI_WINDOW)),
        },
        "raw_window_summaries": {
            "last_5_closes": _round_list([float(value) for value in close_prices.tail(5).tolist()]),
            "last_5_volumes": [_round_int(value) for value in volumes.tail(5).tolist()],
            "recent_peak_close_90d": _round_float(float(close_prices.tail(RECENT_DRAWDOWN_WINDOW).max())),
            "recent_low_close_90d": _round_float(float(close_prices.tail(RECENT_DRAWDOWN_WINDOW).min())),
        },
    }

    return context


def build_market_data_summary(market_context: dict[str, object]) -> dict[str, object]:
    """Extract a compact summary from a full market-data context."""

    history_metadata = market_context["history_metadata"]
    price_summary = market_context["price_summary"]
    momentum_features = market_context["momentum_features"]
    value_features = market_context["value_contrarian_features"]

    return {
        "ticker": market_context["ticker"],
        "latest_trading_date": history_metadata["latest_trading_date"],
        "rows_fetched": history_metadata["rows_fetched"],
        "current_price": price_summary["current_price"],
        "pct_change_30d": price_summary["pct_change_30d"],
        "moving_avg_20d": momentum_features["moving_avg_20d"],
        "moving_avg_50d": momentum_features["moving_avg_50d"],
        "volume_trend_ratio": momentum_features["volume_trend_ratio"],
        "distance_from_52w_high_pct": value_features["distance_from_52w_high_pct"],
        "distance_from_52w_low_pct": value_features["distance_from_52w_low_pct"],
        "recent_drawdown_pct": value_features["recent_drawdown_pct"],
        "rsi_14": value_features["rsi_14"],
    }


def verify_yfinance_connection(ticker: str = "AAPL") -> dict[str, object]:
    """Fetch a tiny slice of market data to confirm yfinance is working."""

    symbol = normalize_ticker_symbol(ticker)
    history = _fetch_history(symbol, period="5d")

    latest_row = history.iloc[-1]

    return {
        "ticker": symbol,
        "rows_fetched": int(len(history)),
        "latest_close": _round_float(float(latest_row["Close"])),
        "latest_date": _format_trading_date(history.index[-1]),
    }
