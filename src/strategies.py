"""Strategy interfaces for the stocktrader project."""

from __future__ import annotations

from .schemas import StrategyOutput

MOMENTUM_TRADER = "Momentum Trader"
VALUE_CONTRARIAN = "Value Contrarian"


def run_momentum_trader(ticker: str, market_context: dict[str, object]) -> StrategyOutput:
    """Run the Momentum Trader strategy against shared market data in a later phase."""

    raise NotImplementedError("A later phase will implement the Momentum Trader strategy.")


def run_value_contrarian(ticker: str, market_context: dict[str, object]) -> StrategyOutput:
    """Run the Value Contrarian strategy against shared market data in a later phase."""

    raise NotImplementedError("A later phase will implement the Value Contrarian strategy.")
