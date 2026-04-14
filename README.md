# stocktrader

`stocktrader` is an educational multi-agent stock-analysis project built for a CMU-style assignment. It uses plain Python orchestration, `yfinance` for market data, and Groq for LLM-backed reasoning.

The project implements two distinct strategy agents:

- `Momentum Trader`
- `Value Contrarian`

Its purpose is to analyze a stock from two different viewpoints, compare the results, and save structured JSON artifacts for review and grading. It is not a production trading system.

## Project Overview

At a high level, the program:

- fetches daily market data for a ticker with `yfinance`
- builds a compact market context and summary
- sends the same market context to two strategy agents independently
- validates each strategy result as structured JSON
- runs an evaluator after both strategies finish
- saves per-stock results and a batch summary for later review

This repository is designed for an academic assignment and reproducible grading, not for live trading.

## Scope And Non-Goals

The project is designed to:

- analyze one or more stock tickers using two LLM-driven strategy perspectives
- compare agreement versus disagreement between those strategies
- save JSON outputs that can be reviewed without rerunning the model
- support a frozen graded stock set for a final deliverable run

The project is explicitly not designed to be:

- a real trading execution system
- a portfolio optimizer
- a backtesting engine
- a brokerage integration
- a guarantee of investment performance or financial advice

## Architecture Summary

The analysis pipeline is:

1. `market_data`: fetch daily OHLCV history from `yfinance` and compute indicators.
2. `strategy_a`: run `Momentum Trader` on the shared market context.
3. `strategy_b`: run `Value Contrarian` on the same shared market context.
4. `evaluator`: compare the two validated strategy outputs after both strategies are complete.
5. `per-stock output`: save a `StockAnalysisOutput` JSON artifact for each analyzed ticker.
6. `batch summary`: save `outputs/summary.json` with aggregate agreement and disagreement counts.

Both strategy agents receive the same market context independently before evaluation. Neither strategy sees the other strategy's output before the evaluator step.

## Repository Structure

- `src/`: core application code for configuration, market data, strategies, evaluator, orchestration, CLI, and the frozen graded stock set
- `prompts/`: reusable prompt templates for the two strategies and evaluator branches
- `outputs/`: generated per-stock JSON files and `summary.json`
- `report/`: short write-up artifacts and notes for final deliverables
- `tests/`: mocked and deterministic test coverage for schemas, market data, strategies, evaluator, orchestration, QC, and final-run helpers

## Requirements

- Python `3.10+`
- Dependencies from `requirements.txt`:

```text
groq
yfinance
pandas
numpy
pydantic
python-dotenv
pytest
```

- A Groq API key is required for live LLM-backed strategy and evaluator runs.
- Market data is fetched from `yfinance` only.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Important setup notes:

- `.env` must stay local to your machine.
- `.env` is already ignored by Git in `.gitignore`.
- Do not commit `.env` or store real secrets anywhere else in the repo.
- Live commands expect `.env` to exist in the repo root or the same variables to be available in your terminal environment.

## Environment Verification

Run the full environment check:

```bash
.venv/bin/python -m src.main --verify
```

This command:

- performs a small `yfinance` fetch
- checks whether Groq is reachable for a tiny smoke test
- reports whether the environment is ready for a live run

Run only the Groq smoke test:

```bash
.venv/bin/python -m src.main --verify-groq
```

How to interpret results:

- If `yfinance` passes and Groq passes, the environment is ready for live analysis.
- If `yfinance` fails, the issue is usually network/data availability or the requested ticker.
- If Groq fails, the issue is usually that `GROQ_API_KEY` is not visible to the current terminal or VS Code session.

## How To Run The Program

Build the full market-data context for one ticker:

```bash
.venv/bin/python -m src.main --market-data --ticker AAPL
```

Build the compact market-data summary for one ticker:

```bash
.venv/bin/python -m src.main --market-summary --ticker AAPL
```

Run only the Momentum Trader:

```bash
.venv/bin/python -m src.main --strategy-a --ticker AAPL
```

Run only the Value Contrarian:

```bash
.venv/bin/python -m src.main --strategy-b --ticker AAPL
```

Run the full single-ticker pipeline and print the result:

```bash
.venv/bin/python -m src.main --analyze --ticker AAPL
```

Run the full single-ticker pipeline and also save `outputs/AAPL.json`:

```bash
.venv/bin/python -m src.main --analyze --ticker AAPL --save-output
```

Run a custom multi-ticker batch and save per-stock outputs plus `summary.json`:

```bash
.venv/bin/python -m src.main --analyze-many --tickers AAPL,MSFT,NVDA,PFE
```

Run the frozen graded set and save all final artifacts:

```bash
.venv/bin/python -m src.main --analyze-graded-set
```

Review one saved per-stock artifact:

```bash
.venv/bin/python -m src.main --review-output --ticker AAPL
```

Review the saved batch summary:

```bash
.venv/bin/python -m src.main --review-summary
```

Successful analysis commands print machine-readable JSON to stdout. Operational messages such as saved file paths and runtime failures are printed separately.

## Frozen Graded Stock Set

The final frozen graded set is:

- `WMT` = steady large-cap
- `NVDA` = volatile momentum
- `UNH` = recent decliner
- `PG` = sideways

The recommended final deliverable path is:

```bash
.venv/bin/python -m src.main --analyze-graded-set
```

This command reuses the same batch pipeline as `--analyze-many`, but with the fixed four-stock set used for the graded run.

## Outputs

Per-stock outputs are saved in `outputs/` as:

- `outputs/WMT.json`
- `outputs/NVDA.json`
- `outputs/UNH.json`
- `outputs/PG.json`

Each per-stock file follows the `StockAnalysisOutput` shape and includes:

- `ticker`
- `run_date`
- `market_data_summary`
- `strategy_a`
- `strategy_b`
- `evaluator`

Batch runs also write:

- `outputs/summary.json`

The summary follows the `SummaryOutput` shape and includes:

- `strategies`
- `stocks_analyzed`
- `total_agreements`
- `total_disagreements`
- `results`

The review commands validate these saved artifacts against the current schemas and quality-control checks before printing them. That makes them useful as a final sanity check before submission.

For grading and public review, the repository is intended to include pre-generated output artifacts so reviewers do not need to rerun live Groq calls.

## Potential Issues And Troubleshooting

### `GROQ_API_KEY` not visible

If a live command fails with a Groq configuration error:

- confirm that a local repo-root `.env` file exists
- confirm you are running the command from the same repo and terminal session that should load it
- confirm VS Code is using the terminal session where the environment is available
- rerun:

```bash
.venv/bin/python -m src.main --verify-groq
```

### Running from the wrong terminal or session

If the project works in one shell but not another, the most common cause is that the `.env` file is local but the current session is not loading it or is running from the wrong working directory.

### Missing `.env`

If `.env` is missing, live Groq-backed commands will fail even though tests may still pass. Copy `.env.example` to `.env` locally before running live analysis.

### `yfinance` returns no data or insufficient history

Possible causes:

- invalid ticker symbol
- temporary network or provider issue
- too little recent data returned for the required indicator windows

The market-data layer requires enough daily history to compute the moving averages, RSI, drawdown, and 30-day return features.

### Malformed or weak LLM outputs

The strategy and evaluator layers validate JSON structure locally. If the model returns malformed JSON or invalid fields, the program retries once and then fails with a clear validation error instead of saving a weak artifact.

### Why review commands are useful

Use the review commands before submission to confirm that:

- the saved JSON files exist
- the schema is still valid
- the per-stock outputs and batch summary remain consistent with the current QC rules

## Testing

Run the full test suite with:

```bash
.venv/bin/python -m pytest
```

The test suite is designed to be deterministic and does not require live network access or a real API key.

## Submission And Grading Notes

- The repo is organized so saved JSON outputs can be reviewed without rerunning live model calls.
- Graders should not need Groq credentials if pre-generated outputs are included in `outputs/`.
- Report writing and any AI appendix are separate deliverables from the core program.
- The frozen graded stock set and review commands are intended to support reproducible final submission artifacts.
