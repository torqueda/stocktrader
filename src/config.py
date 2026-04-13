"""Configuration helpers for the stocktrader project."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PROMPTS_DIR = "prompts"


def _get_int_env(name: str, default: int) -> int:
    """Read an integer environment variable with a sane fallback."""

    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


MODEL_ID = os.getenv("MODEL_ID", "llama-3.1-8b-instant")
FINAL_MODEL_ID = os.getenv("FINAL_MODEL_ID", "llama-3.3-70b-versatile")
DEFAULT_HISTORY_PERIOD = os.getenv("DEFAULT_HISTORY_PERIOD", "1y")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
LLM_MAX_RETRIES = _get_int_env("LLM_MAX_RETRIES", 1)
STRATEGY_MAX_COMPLETION_TOKENS = _get_int_env("STRATEGY_MAX_COMPLETION_TOKENS", 250)
EVALUATOR_MAX_COMPLETION_TOKENS = _get_int_env("EVALUATOR_MAX_COMPLETION_TOKENS", 220)
VERIFY_MAX_COMPLETION_TOKENS = _get_int_env("VERIFY_MAX_COMPLETION_TOKENS", 8)


def get_project_root() -> Path:
    """Return the repository root."""

    return PROJECT_ROOT


def resolve_project_path(*parts: str) -> Path:
    """Resolve a path relative to the repository root."""

    return PROJECT_ROOT.joinpath(*parts)


def get_output_dir() -> Path:
    """Return the configured output directory."""

    return resolve_project_path(OUTPUT_DIR)


def get_prompts_dir() -> Path:
    """Return the repository prompts directory."""

    return resolve_project_path(PROMPTS_DIR)


def get_prompt_path(filename: str) -> Path:
    """Return the path to a prompt file by filename."""

    return get_prompts_dir() / filename


def ensure_output_dir() -> Path:
    """Create the output directory if it does not already exist."""

    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_groq_api_key() -> str | None:
    """Return the configured Groq API key, if present."""

    return os.getenv("GROQ_API_KEY")
