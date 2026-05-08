"""Runtime configuration: env, model IDs, pricing, safety limits."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return float(val)


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return int(val)


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_ROUTINE = "anthropic/claude-haiku-4.5"
MODEL_PLAN = "anthropic/claude-opus-4.7"

# OpenRouter pricing (USD per 1M tokens) — used for live cost estimate.
# Update if OpenRouter changes pricing.
PRICING = {
    MODEL_ROUTINE: {"in": 1.00, "out": 5.00},
    MODEL_PLAN:    {"in": 15.00, "out": 75.00},
}

MAX_STEPS      = _env_int("MAX_STEPS", 200)
MAX_BUDGET_USD = _env_float("MAX_BUDGET_USD", 5.0)
MAX_RUNTIME_S  = _env_int("MAX_RUNTIME_S", 1800)

DOM_CHAR_LIMIT = 12_000
DOM_LINK_LIMIT = 150
ACTION_DELAY_MIN_S = 1.0
ACTION_DELAY_MAX_S = 3.0
PAGE_LOAD_TIMEOUT_S = 25
NO_PROGRESS_STEPS_BEFORE_RECOVERY = 3


@dataclass
class RunConfig:
    task: str
    count: int
    out_path: str
    headed: bool = False
    log_path: str = "run.log"


def require_api_key() -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and paste your key."
        )
    return OPENROUTER_API_KEY
