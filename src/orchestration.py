"""Orchestration entry points for the stocktrader project."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .config import get_output_dir
from .debate import run_debate_round
from .evaluator import evaluate_strategies
from .market_data import build_market_context, build_market_data_summary, normalize_ticker_symbol
from .schemas import DebateChange, DebateOutput, StockAnalysisOutput, SummaryOutput, SummaryRow
from .strategies import (
    MOMENTUM_TRADER,
    VALUE_CONTRARIAN,
    run_momentum_trader,
    run_value_contrarian,
)

REQUIRED_MARKET_SUMMARY_KEYS = (
    "ticker",
    "latest_trading_date",
    "rows_fetched",
    "current_price",
    "pct_change_30d",
    "moving_avg_20d",
    "moving_avg_50d",
    "volume_trend_ratio",
    "distance_from_52w_high_pct",
    "distance_from_52w_low_pct",
    "recent_drawdown_pct",
    "rsi_14",
)
DISALLOWED_MARKET_SUMMARY_KEYS = (
    "history_metadata",
    "price_summary",
    "momentum_features",
    "value_contrarian_features",
    "raw_window_summaries",
)


def get_run_date() -> str:
    """Return the current run date as an ISO string."""

    return date.today().isoformat()


def get_output_path_for_ticker(ticker: str) -> Path:
    """Return the output JSON path for one ticker."""

    symbol = normalize_ticker_symbol(ticker)
    return get_output_dir() / f"{symbol}.json"


def save_stock_analysis(output: StockAnalysisOutput) -> Path:
    """Write one analyzed ticker result to a pretty-printed JSON file."""

    if output.debate is not None:
        raise ValueError("Use save_debate_stock_analysis() for artifacts that include a debate field.")

    validated_output = validate_stock_analysis_output(output)
    output_path = get_output_path_for_ticker(validated_output.ticker)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_output = json.dumps(validated_output.model_dump(mode="json"), indent=2)
    output_path.write_text(f"{serialized_output}\n", encoding="utf-8")
    return output_path


def get_summary_output_path() -> Path:
    """Return the output JSON path for the batch summary."""

    return get_output_dir() / "summary.json"


def get_debate_output_path_for_ticker(ticker: str) -> Path:
    """Return the debate artifact JSON path for one ticker."""

    symbol = normalize_ticker_symbol(ticker)
    return get_output_dir() / f"{symbol}.debate.json"


def _load_json_file(path: Path) -> Any:
    """Load one JSON artifact from disk."""

    if not path.is_file():
        raise FileNotFoundError(f"Output file not found: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Output file is not valid JSON: {path}") from exc


def _require_non_empty_text(value: str, field_name: str) -> str:
    """Ensure a narrative field is not empty after stripping whitespace."""

    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be non-empty.")
    return stripped


def _validate_market_data_summary(summary: dict[str, object], ticker: str) -> None:
    """Validate that the compact market summary is aligned and grader-friendly."""

    missing_keys = [key for key in REQUIRED_MARKET_SUMMARY_KEYS if key not in summary]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"market_data_summary is missing required keys: {missing}.")

    disallowed_keys = [key for key in DISALLOWED_MARKET_SUMMARY_KEYS if key in summary]
    if disallowed_keys:
        disallowed = ", ".join(disallowed_keys)
        raise ValueError(
            "market_data_summary must remain compact and must not include raw market-context blocks: "
            f"{disallowed}."
        )

    summary_ticker = normalize_ticker_symbol(str(summary["ticker"]))
    if summary_ticker != ticker:
        raise ValueError(
            f"market_data_summary ticker '{summary_ticker}' does not match top-level ticker '{ticker}'."
        )


def _validate_debate_change(
    change: DebateChange,
    before_name: str,
    before_decision: object,
    before_confidence: int,
    before_justification: str,
    after_decision: object,
    after_confidence: int,
    after_justification: str,
) -> None:
    """Validate one structured change block in debate output."""

    if change.decision_before != before_decision or change.decision_after != after_decision:
        raise ValueError(f"Debate change block for {before_name} has inconsistent decision values.")
    if change.confidence_before != before_confidence or change.confidence_after != after_confidence:
        raise ValueError(f"Debate change block for {before_name} has inconsistent confidence values.")

    expected_decision_changed = before_decision != after_decision
    expected_confidence_changed = before_confidence != after_confidence
    expected_justification_changed = before_justification != after_justification
    expected_position_changed = expected_decision_changed or expected_confidence_changed

    if change.decision_changed != expected_decision_changed:
        raise ValueError(f"Debate change block for {before_name} has an inconsistent decision_changed flag.")
    if change.confidence_changed != expected_confidence_changed:
        raise ValueError(f"Debate change block for {before_name} has an inconsistent confidence_changed flag.")
    if change.justification_changed != expected_justification_changed:
        raise ValueError(
            f"Debate change block for {before_name} has an inconsistent justification_changed flag."
        )
    if change.position_changed != expected_position_changed:
        raise ValueError(f"Debate change block for {before_name} has an inconsistent position_changed flag.")


def _validate_debate_output(output: StockAnalysisOutput, debate: DebateOutput) -> None:
    """Validate the debate field against the original first-round output."""

    if output.evaluator.agents_agree:
        raise ValueError("Debate output may only be attached to a saved disagreement artifact.")

    if debate.strategy_a_response.name != MOMENTUM_TRADER:
        raise ValueError(
            f"debate.strategy_a_response.name must be '{MOMENTUM_TRADER}', received '{debate.strategy_a_response.name}'."
        )
    if debate.strategy_b_response.name != VALUE_CONTRARIAN:
        raise ValueError(
            "debate.strategy_b_response.name must be "
            f"'{VALUE_CONTRARIAN}', received '{debate.strategy_b_response.name}'."
        )

    _validate_debate_change(
        change=debate.strategy_a_change,
        before_name=MOMENTUM_TRADER,
        before_decision=output.strategy_a.decision,
        before_confidence=output.strategy_a.confidence,
        before_justification=output.strategy_a.justification,
        after_decision=debate.strategy_a_response.decision,
        after_confidence=debate.strategy_a_response.confidence,
        after_justification=debate.strategy_a_response.justification,
    )
    _validate_debate_change(
        change=debate.strategy_b_change,
        before_name=VALUE_CONTRARIAN,
        before_decision=output.strategy_b.decision,
        before_confidence=output.strategy_b.confidence,
        before_justification=output.strategy_b.justification,
        after_decision=debate.strategy_b_response.decision,
        after_confidence=debate.strategy_b_response.confidence,
        after_justification=debate.strategy_b_response.justification,
    )


def validate_stock_analysis_output(output: StockAnalysisOutput) -> StockAnalysisOutput:
    """Run lightweight sanity checks on the final per-ticker analysis output."""

    normalized_ticker = normalize_ticker_symbol(output.ticker)
    if output.ticker != normalized_ticker:
        raise ValueError(
            f"StockAnalysisOutput ticker must be normalized to '{normalized_ticker}', received '{output.ticker}'."
        )

    if output.strategy_a.name != MOMENTUM_TRADER:
        raise ValueError(
            f"strategy_a.name must be '{MOMENTUM_TRADER}', received '{output.strategy_a.name}'."
        )
    if output.strategy_b.name != VALUE_CONTRARIAN:
        raise ValueError(
            f"strategy_b.name must be '{VALUE_CONTRARIAN}', received '{output.strategy_b.name}'."
        )
    if output.strategy_a.name == output.strategy_b.name:
        raise ValueError("Strategy outputs must have distinct strategy names.")

    _require_non_empty_text(output.strategy_a.justification, "strategy_a.justification")
    _require_non_empty_text(output.strategy_b.justification, "strategy_b.justification")
    evaluator_analysis = _require_non_empty_text(output.evaluator.analysis, "evaluator.analysis")

    expected_agreement = output.strategy_a.decision == output.strategy_b.decision
    if output.evaluator.agents_agree != expected_agreement:
        raise ValueError(
            "evaluator.agents_agree does not match the two strategy decisions in the final output."
        )

    _validate_market_data_summary(output.market_data_summary, normalized_ticker)
    if output.debate is not None:
        _validate_debate_output(output, output.debate)

    return output.model_copy(
        update={
            "evaluator": output.evaluator.model_copy(update={"analysis": evaluator_analysis}),
        }
    )


def validate_summary_output(summary: SummaryOutput) -> SummaryOutput:
    """Run lightweight sanity checks on the batch summary output."""

    result_count = len(summary.results)
    if summary.total_agreements + summary.total_disagreements != result_count:
        raise ValueError(
            "Summary totals are inconsistent: total_agreements + total_disagreements "
            f"must equal {result_count}."
        )

    if len(summary.stocks_analyzed) != result_count:
        raise ValueError("stocks_analyzed must contain one ticker per summary row.")

    for expected_ticker, row in zip(summary.stocks_analyzed, summary.results, strict=True):
        normalized_expected = normalize_ticker_symbol(expected_ticker)
        normalized_row_ticker = normalize_ticker_symbol(row.ticker)
        if normalized_row_ticker != normalized_expected:
            raise ValueError(
                f"Summary row ticker '{normalized_row_ticker}' does not align with "
                f"stocks_analyzed entry '{normalized_expected}'."
            )

        expected_agree = row.a_decision == row.b_decision
        if row.agree != expected_agree:
            raise ValueError(
                f"Summary row agree flag for ticker '{normalized_row_ticker}' does not match its decisions."
            )

    return summary


def load_stock_analysis(ticker: str) -> StockAnalysisOutput:
    """Load and validate one saved per-ticker analysis artifact."""

    payload = _load_json_file(get_output_path_for_ticker(ticker))
    try:
        analysis_output = StockAnalysisOutput.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"Saved stock analysis artifact is invalid for ticker '{ticker}'.") from exc

    return validate_stock_analysis_output(analysis_output)


def load_debate_stock_analysis(ticker: str) -> StockAnalysisOutput:
    """Load and validate one saved debate artifact."""

    payload = _load_json_file(get_debate_output_path_for_ticker(ticker))
    try:
        analysis_output = StockAnalysisOutput.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"Saved debate artifact is invalid for ticker '{ticker}'.") from exc

    validated_output = validate_stock_analysis_output(analysis_output)
    if validated_output.debate is None:
        raise ValueError(f"Saved debate artifact for ticker '{ticker}' does not contain a debate field.")

    return validated_output


def save_debate_stock_analysis(output: StockAnalysisOutput) -> Path:
    """Write a debate-extended artifact without overwriting the original first-pass output."""

    validated_output = validate_stock_analysis_output(output)
    if validated_output.debate is None:
        raise ValueError("Debate artifacts must include a populated debate field.")

    output_path = get_debate_output_path_for_ticker(validated_output.ticker)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_output = json.dumps(validated_output.model_dump(mode="json"), indent=2)
    output_path.write_text(f"{serialized_output}\n", encoding="utf-8")
    return output_path


def generate_debate_for_saved_output(ticker: str) -> StockAnalysisOutput:
    """Load one saved disagreement artifact, run debate mode, and attach the debate field."""

    original_output = load_stock_analysis(ticker)
    debate_output = run_debate_round(original_output)
    debated_output = original_output.model_copy(update={"debate": debate_output})
    return validate_stock_analysis_output(debated_output)


def load_summary_output() -> SummaryOutput:
    """Load and validate the saved batch summary artifact."""

    payload = _load_json_file(get_summary_output_path())
    try:
        summary_output = SummaryOutput.model_validate(payload)
    except Exception as exc:
        raise ValueError("Saved summary artifact is invalid.") from exc

    return validate_summary_output(summary_output)


def _normalize_unique_tickers(tickers: list[str]) -> list[str]:
    """Normalize and deduplicate tickers while preserving input order."""

    if not tickers:
        raise ValueError("At least one ticker is required for batch analysis.")

    normalized_tickers: list[str] = []
    seen_tickers: set[str] = set()

    for ticker in tickers:
        symbol = normalize_ticker_symbol(ticker)
        if symbol not in seen_tickers:
            seen_tickers.add(symbol)
            normalized_tickers.append(symbol)

    if not normalized_tickers:
        raise ValueError("At least one valid ticker is required for batch analysis.")

    return normalized_tickers


def analyze_ticker(ticker: str) -> StockAnalysisOutput:
    """Run the end-to-end analysis pipeline for one ticker."""

    symbol = normalize_ticker_symbol(ticker)
    market_context = build_market_context(symbol)

    strategy_a = run_momentum_trader(symbol, market_context)
    strategy_b = run_value_contrarian(symbol, market_context)
    evaluator = evaluate_strategies(symbol, market_context, strategy_a, strategy_b)
    market_data_summary = build_market_data_summary(market_context)

    output = StockAnalysisOutput(
        ticker=symbol,
        run_date=get_run_date(),
        market_data_summary=market_data_summary,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        evaluator=evaluator,
    )
    return validate_stock_analysis_output(output)


def analyze_tickers(tickers: list[str], save_outputs: bool = True) -> list[StockAnalysisOutput]:
    """Run the full single-ticker pipeline for multiple tickers."""

    normalized_tickers = _normalize_unique_tickers(tickers)
    results: list[StockAnalysisOutput] = []

    for ticker in normalized_tickers:
        result = analyze_ticker(ticker)
        results.append(result)
        if save_outputs:
            save_stock_analysis(result)

    return results


def build_summary_output(results: list[StockAnalysisOutput]) -> SummaryOutput:
    """Aggregate per-ticker results into the assignment-style summary output."""

    if not results:
        raise ValueError("At least one stock analysis result is required to build summary output.")

    validated_results = [validate_stock_analysis_output(result) for result in results]
    summary_rows: list[SummaryRow] = []
    total_agreements = 0

    for result in validated_results:
        agree = result.strategy_a.decision == result.strategy_b.decision
        if agree:
            total_agreements += 1

        summary_rows.append(
            SummaryRow(
                ticker=result.ticker,
                a_decision=result.strategy_a.decision,
                b_decision=result.strategy_b.decision,
                agree=agree,
            )
        )

    summary = SummaryOutput(
        strategies=[MOMENTUM_TRADER, VALUE_CONTRARIAN],
        stocks_analyzed=[result.ticker for result in validated_results],
        total_agreements=total_agreements,
        total_disagreements=len(validated_results) - total_agreements,
        results=summary_rows,
    )
    return validate_summary_output(summary)


def save_summary_output(summary: SummaryOutput) -> Path:
    """Write the batch summary to outputs/summary.json."""

    validated_summary = validate_summary_output(summary)
    output_path = get_summary_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_summary = json.dumps(validated_summary.model_dump(mode="json"), indent=2)
    output_path.write_text(f"{serialized_summary}\n", encoding="utf-8")
    return output_path
