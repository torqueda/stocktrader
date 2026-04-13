# Project Rules

- Use plain Python orchestration only.
- Use Groq for LLM calls and `yfinance` for market data.
- Keep strategy agents independent and prevent cross-agent leakage before evaluation.
- Favor small, testable functions.
- Do not add extra frameworks unless explicitly requested.
- Keep outputs aligned with the assignment-required JSON structure.
- Avoid bonus extensions unless explicitly requested later.
