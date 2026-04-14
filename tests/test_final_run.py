"""Tests for the frozen graded stock set helpers and CLI command."""

from __future__ import annotations

import pytest

from src.final_run import get_graded_run_mapping, get_graded_run_tickers
from src.main import main, run_graded_set_analysis_command


def test_get_graded_run_tickers_returns_expected_ordered_set() -> None:
    assert get_graded_run_tickers() == ["WMT", "NVDA", "UNH", "PG"]


def test_get_graded_run_mapping_returns_expected_categories() -> None:
    assert get_graded_run_mapping() == {
        "steady_large_cap": "WMT",
        "volatile_momentum": "NVDA",
        "recent_decliner": "UNH",
        "sideways": "PG",
    }


def test_run_graded_set_analysis_command_reuses_batch_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_tickers: list[str] = []

    def fake_run_batch_analysis_with_tickers(tickers: list[str]) -> int:
        seen_tickers.extend(tickers)
        return 0

    monkeypatch.setattr("src.main._run_batch_analysis_with_tickers", fake_run_batch_analysis_with_tickers)

    exit_code = run_graded_set_analysis_command()

    assert exit_code == 0
    assert seen_tickers == ["WMT", "NVDA", "UNH", "PG"]


def test_main_routes_analyze_graded_set_to_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.main.run_graded_set_analysis_command", lambda: 0)

    assert main(["--analyze-graded-set"]) == 0
