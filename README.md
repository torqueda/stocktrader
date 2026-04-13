# stocktrader

`stocktrader` is a CMU-style assignment repo for a multi-agent stock analysis system built in plain Python.

## Current Scope

Phase 1 and Phase 2 are in place:
- project scaffolding and shared contracts
- environment verification for `yfinance` and Groq
- a market-data component that builds compact JSON-ready context for later strategy agents

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

## Notes for Grading

Later phases will add prompt files, generated JSON outputs, and report artifacts. Pre-generated outputs will be included so the project can be reviewed without requiring graders to provide API keys.
