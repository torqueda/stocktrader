# stocktrader

`stocktrader` is a CMU-style assignment repo for a multi-agent stock analysis system built in plain Python.

## Current Scope

Phase 1 through batch orchestration are in place:
- project scaffolding and shared contracts
- environment verification for `yfinance` and Groq
- a market-data component that builds compact JSON-ready context for later strategy agents
- reusable prompt files for both strategy agents
- single-strategy Groq calls that return validated JSON outputs
- a reusable LLM wrapper that separates raw calls, JSON parsing, and validated structured output
- evaluator prompt files and evaluator logic for agreement vs disagreement analysis
- end-to-end one-ticker orchestration with optional JSON saving to `outputs/`
- multi-ticker analysis with per-stock outputs plus `outputs/summary.json`
- lightweight output review helpers for validating saved artifacts without rerunning models
- a frozen graded four-stock set with a dedicated final-run command

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

## Batch Analysis

Run a batch analysis and save per-stock outputs plus `summary.json`:

```bash
.venv/bin/python -m src.main --analyze-many --tickers AAPL,MSFT,NVDA,PFE
```

This command:
- normalizes and deduplicates the ticker list while preserving order
- reuses the single-ticker pipeline for each ticker
- saves per-stock JSON files to `outputs/`
- writes `outputs/summary.json`
- prints the validated summary JSON to stdout

## Graded Set Run

The frozen graded stock set is:

- `steady_large_cap`: `WMT`
- `volatile_momentum`: `NVDA`
- `recent_decliner`: `UNH`
- `sideways`: `PG`

Run the recommended final deliverable path:

```bash
.venv/bin/python -m src.main --analyze-graded-set
```

This command reuses the existing batch pipeline, saves all four per-stock outputs plus `outputs/summary.json`, and prints the validated summary JSON to stdout.

## Reviewing Saved Outputs

Review one saved per-ticker artifact:

```bash
.venv/bin/python -m src.main --review-output --ticker AAPL
```

Review the saved batch summary:

```bash
.venv/bin/python -m src.main --review-summary
```

These commands validate the saved JSON structure locally and print the artifact only if it passes schema and quality-control checks.

## Notes for Grading

Report artifacts are still upcoming. Pre-generated outputs will be included so the project can be reviewed without requiring graders to provide API keys.
For live Groq-backed runs, make sure `GROQ_API_KEY` is visible to the terminal or VS Code process that launches the CLI.
