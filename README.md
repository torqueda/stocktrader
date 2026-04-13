# stocktrader

`stocktrader` is a CMU-style assignment repo for a multi-agent stock analysis system built in plain Python.

## Current Scope

Phase 1 through single-ticker orchestration are in place:
- project scaffolding and shared contracts
- environment verification for `yfinance` and Groq
- a market-data component that builds compact JSON-ready context for later strategy agents
- reusable prompt files for both strategy agents
- single-strategy Groq calls that return validated JSON outputs
- a reusable LLM wrapper that separates raw calls, JSON parsing, and validated structured output
- evaluator prompt files and evaluator logic for agreement vs disagreement analysis
- end-to-end one-ticker orchestration with optional JSON saving to `outputs/`

Chosen strategies:
- Momentum Trader
- Value Contrarian

Chosen LLM provider:
- Groq

Chosen orchestration approach:
- Plain Python modules and functions

Runtime market data source:
- `yfinance`

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
.venv/bin/pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your Groq key if needed.

## Verification

Run:

```bash
.venv/bin/python -m src.main --verify
```

This command:
- performs a minimal `yfinance` fetch for a ticker
- runs a tiny Groq smoke test when a key is available
- prints whether the environment is ready for the next phase

You can also run only the Groq check:

```bash
.venv/bin/python -m src.main --verify-groq
```

## Market Data Commands

Build the full market context for one ticker:

```bash
.venv/bin/python -m src.main --market-data --ticker AAPL
```

Build the compact grader-friendly market summary:

```bash
.venv/bin/python -m src.main --market-summary --ticker AAPL
```

The market-data component uses daily `yfinance` history, computes the Phase 2 indicators, and returns a compact JSON-serializable structure for later strategy prompts.

## Strategy Commands

Run the Momentum Trader on one ticker:

```bash
.venv/bin/python -m src.main --strategy-a --ticker AAPL
```

Run the Value Contrarian on one ticker:

```bash
.venv/bin/python -m src.main --strategy-b --ticker AAPL
```

These commands:
- build the shared market context from Phase 2
- load the saved prompt template from `prompts/`
- call Groq for one strategy only
- validate the returned JSON into the required strategy-output shape

## Single-Ticker Analysis

Run the full pipeline for one ticker:

```bash
.venv/bin/python -m src.main --analyze --ticker AAPL
```

Run the full pipeline and save the result to `outputs/AAPL.json`:

```bash
.venv/bin/python -m src.main --analyze --ticker AAPL --save-output
```

This command:
- builds market data once
- runs both strategies independently on the same market context
- runs the evaluator after both strategies finish
- prints the validated final JSON result
- optionally writes a pretty-printed per-ticker artifact to `outputs/`

## Notes for Grading

Multi-ticker batch orchestration, `summary.json`, and report artifacts are still upcoming. Pre-generated outputs will be included so the project can be reviewed without requiring graders to provide API keys.
