"""Evaluator interfaces for the stocktrader project."""

from __future__ import annotations

from .schemas import EvaluatorOutput, StrategyOutput


def evaluate_strategies(
    ticker: str,
    strategy_a: StrategyOutput,
    strategy_b: StrategyOutput,
) -> EvaluatorOutput:
    """Compare independent strategy outputs and produce evaluator analysis."""

    raise NotImplementedError("Phase 3 will implement evaluator logic.")
