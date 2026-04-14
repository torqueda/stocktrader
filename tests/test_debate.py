"""Tests for Debate Mode and separate debate artifact saving."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.debate import (
    DEBATE_A_PROMPT_FILE,
    DEBATE_B_PROMPT_FILE,
    _build_debate_input_payload,
    load_debate_prompt,
    run_debate_round,
)
from src.main import run_debate_output_command, run_review_debate_command
from src.orchestration import (
    generate_debate_for_saved_output,
    get_debate_output_path_for_ticker,
    load_debate_stock_analysis,
    save_debate_stock_analysis,
    validate_stock_analysis_output,
)
from src.schemas import DebateChange, DebateOutput, Decision, EvaluatorOutput, StockAnalysisOutput, StrategyOutput


def make_strategy_output(name: str, decision: Decision, confidence: int = 7) -> StrategyOutput:
    """Build a valid StrategyOutput for debate tests."""

    return StrategyOutput(
        name=name,
        decision=decision,
        confidence=confidence,
        justification=(
            "The evidence is numeric. The rationale is compact. "
            "The conclusion follows the supplied context."
        ),
    )


def make_saved_output(
    *,
    agents_agree: bool = False,
    a_decision: Decision = Decision.BUY,
    b_decision: Decision = Decision.SELL,
) -> StockAnalysisOutput:
    """Build a valid saved first-round artifact for debate tests."""

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
        strategy_a=make_strategy_output("Momentum Trader", a_decision, confidence=7),
        strategy_b=make_strategy_output("Value Contrarian", b_decision, confidence=6),
        evaluator=EvaluatorOutput(
            agents_agree=agents_agree,
            analysis="The evaluator explains why the two strategies disagree on the same market summary.",
        ),
    )


def make_debate_output() -> DebateOutput:
    """Build a valid DebateOutput for save/load tests."""

    strategy_a_response = StrategyOutput(
        name="Momentum Trader",
        decision=Decision.BUY,
        confidence=7,
        justification="Price remains above the 20-day average. The trend case still holds. The opposing argument does not outweigh the momentum evidence.",
    )
    strategy_b_response = StrategyOutput(
        name="Value Contrarian",
        decision=Decision.HOLD,
        confidence=5,
        justification="The stock is still below the 52-week high by -10.19%. RSI at 53.64 is not deeply stretched. The rebuttal weakens the prior sell case, so a neutral stance fits better.",
    )

    return DebateOutput(
        strategy_a_response=strategy_a_response,
        strategy_b_response=strategy_b_response,
        strategy_a_change=DebateChange(
            decision_before=Decision.BUY,
            decision_after=Decision.BUY,
            confidence_before=7,
            confidence_after=7,
            decision_changed=False,
            confidence_changed=False,
            justification_changed=True,
            position_changed=False,
        ),
        strategy_b_change=DebateChange(
            decision_before=Decision.SELL,
            decision_after=Decision.HOLD,
            confidence_before=6,
            confidence_after=5,
            decision_changed=True,
            confidence_changed=True,
            justification_changed=True,
            position_changed=True,
        ),
    )


def test_debate_prompt_files_load_successfully() -> None:
    momentum_prompt = load_debate_prompt(DEBATE_A_PROMPT_FILE)
    value_prompt = load_debate_prompt(DEBATE_B_PROMPT_FILE)

    assert "Debate Mode" in momentum_prompt
    assert "Debate Mode" in value_prompt


def test_build_debate_input_payload_is_compact_and_serializable() -> None:
    payload = _build_debate_input_payload(
        ticker="AAPL",
        market_data_summary=make_saved_output().market_data_summary,
        self_strategy=make_saved_output().strategy_a,
        opponent_strategy=make_saved_output().strategy_b,
        original_evaluator_analysis="The strategies disagree because they prioritize different signals.",
    )

    assert payload["ticker"] == "AAPL"
    assert payload["your_first_round_output"]["name"] == "Momentum Trader"
    assert payload["opponent_first_round_output"]["name"] == "Value Contrarian"
    assert "evaluator_analysis" in payload


def test_run_debate_round_rejects_agreement_artifact() -> None:
    with pytest.raises(ValueError, match="requires a disagreement"):
        run_debate_round(make_saved_output(agents_agree=True, a_decision=Decision.BUY, b_decision=Decision.BUY))


def test_run_debate_round_calls_llm_twice_and_builds_change_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[dict[str, str]]] = []

    def fake_call_validated_json_completion(
        messages: list[dict[str, str]],
        validate_payload,
        model_id: str = "unused",
        max_completion_tokens: int = 0,
        temperature: float = 0,
        max_retries: int = 0,
    ) -> StrategyOutput:
        del model_id, max_completion_tokens, temperature, max_retries
        captured_messages.append(messages)
        payload = (
            {
                "name": "Momentum Trader",
                "decision": "BUY",
                "confidence": 7,
                "justification": (
                    "Price remains above the 20-day average. The trend case still holds. "
                    "The opposing argument does not outweigh the momentum evidence."
                ),
            }
            if len(captured_messages) == 1
            else {
                "name": "Value Contrarian",
                "decision": "HOLD",
                "confidence": 5,
                "justification": (
                    "The stock is still below the 52-week high by -10.19%. RSI at 53.64 is not deeply stretched. "
                    "The rebuttal weakens the prior sell case, so a neutral stance fits better."
                ),
            }
        )
        return validate_payload(payload)

    monkeypatch.setattr("src.debate.call_validated_json_completion", fake_call_validated_json_completion)

    result = run_debate_round(make_saved_output())

    assert result.strategy_a_response.name == "Momentum Trader"
    assert result.strategy_b_response.name == "Value Contrarian"
    assert result.strategy_a_change.position_changed is False
    assert result.strategy_a_change.justification_changed is True
    assert result.strategy_b_change.position_changed is True
    assert '"opponent_first_round_output"' in captured_messages[0][1]["content"]
    assert '"evaluator_analysis"' in captured_messages[1][1]["content"]


def test_generate_debate_for_saved_output_reuses_saved_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved_output = make_saved_output()
    debate_output = make_debate_output()

    monkeypatch.setattr("src.orchestration.load_stock_analysis", lambda ticker: saved_output)
    monkeypatch.setattr("src.orchestration.run_debate_round", lambda output: debate_output)

    result = generate_debate_for_saved_output("AAPL")

    assert result.ticker == "AAPL"
    assert result.debate == debate_output


def test_save_and_load_debate_stock_analysis_use_separate_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("src.orchestration.get_output_dir", lambda: tmp_path)
    debated_output = make_saved_output().model_copy(update={"debate": make_debate_output()})

    saved_path = save_debate_stock_analysis(debated_output)
    loaded_output = load_debate_stock_analysis(" aapl ")

    assert saved_path == tmp_path / "AAPL.debate.json"
    assert get_debate_output_path_for_ticker("aapl") == tmp_path / "AAPL.debate.json"
    assert not (tmp_path / "AAPL.json").exists()
    assert loaded_output.debate is not None
    assert loaded_output.debate.strategy_b_change.position_changed is True


def test_validate_stock_analysis_output_rejects_inconsistent_debate_change_flags() -> None:
    debated_output = make_saved_output().model_copy(update={"debate": make_debate_output()})
    debated_output.debate.strategy_b_change.position_changed = False

    with pytest.raises(ValueError, match="position_changed"):
        validate_stock_analysis_output(debated_output)


def test_debate_cli_commands_use_saved_artifacts_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    debated_output = make_saved_output().model_copy(update={"debate": make_debate_output()})

    monkeypatch.setattr("src.main.generate_debate_for_saved_output", lambda ticker: debated_output)
    monkeypatch.setattr(
        "src.main.save_debate_stock_analysis",
        lambda output: Path("/tmp/AAPL.debate.json"),
    )

    debate_status = run_debate_output_command("AAPL")
    debate_capture = capsys.readouterr()

    assert debate_status == 0
    assert '"debate"' in debate_capture.out
    assert "Saved debate output: /tmp/AAPL.debate.json" in debate_capture.err

    monkeypatch.setattr("src.main.load_debate_stock_analysis", lambda ticker: debated_output)

    review_status = run_review_debate_command("AAPL")
    review_capture = capsys.readouterr()

    assert review_status == 0
    assert '"strategy_a_change"' in review_capture.out
    assert review_capture.err == ""
