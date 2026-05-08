"""Thin OpenRouter client with JSON-mode enforcement and running cost tally."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

import config

log = logging.getLogger("ai")


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or config.require_api_key()
        self.total_cost_usd = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.call_count = 0

    # ---- public helpers ----------------------------------------------------

    def routine(self, messages: list[dict], **kw) -> dict:
        """Cheap per-step call (Haiku 4.5)."""
        return self._call(config.MODEL_ROUTINE, messages, **kw)

    def plan(self, messages: list[dict], **kw) -> dict:
        """Expensive planning / recovery call (Opus 4.7)."""
        return self._call(config.MODEL_PLAN, messages, **kw)

    # ---- core --------------------------------------------------------------

    def _call(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 1500,
        temperature: float = 0.2,
        retries: int = 3,
        prefill: str = "{",
    ) -> dict:
        """Call OpenRouter with Anthropic-style assistant prefill to force JSON.

        Prefilling the assistant turn with `{` makes the model continue from
        there — it cannot emit prose before the brace, so we get parseable
        JSON even when the model would otherwise narrate. We prepend `prefill`
        to the reply before parsing."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/local/ai-lead-scraper",
            "X-Title": "AI Lead Scraper",
        }

        effective = list(messages)
        if prefill:
            effective.append({"role": "assistant", "content": prefill})

        payload = {
            "model": model,
            "messages": effective,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                r = requests.post(
                    config.OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    wait = 2 ** attempt
                    log.warning("OpenRouter %s — retrying in %ss", r.status_code, wait)
                    time.sleep(wait)
                    continue
                if r.status_code != 200:
                    raise OpenRouterError(
                        f"OpenRouter {r.status_code}: {r.text[:300]}"
                    )
                data = r.json()
                self._tally(model, data.get("usage", {}))
                content = data["choices"][0]["message"]["content"] or ""
                return self._parse_json(prefill + content if prefill else content)
            except requests.RequestException as e:
                last_err = e
                log.warning("Network error: %s (attempt %d)", e, attempt + 1)
                time.sleep(2 ** attempt)
        raise OpenRouterError(f"OpenRouter unreachable after {retries} retries: {last_err}")

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Best-effort JSON extraction.

        Order of attempts:
          1. Literal parse after trimming code fences.
          2. Extract the first balanced {...} block and parse that.
        """
        text = (text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        if start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
        raise OpenRouterError(f"Non-JSON reply: {text[:300]}")

    def _tally(self, model: str, usage: dict) -> None:
        tin = int(usage.get("prompt_tokens", 0))
        tout = int(usage.get("completion_tokens", 0))
        price = config.PRICING.get(model, {"in": 0, "out": 0})
        cost = (tin * price["in"] + tout * price["out"]) / 1_000_000
        self.total_tokens_in += tin
        self.total_tokens_out += tout
        self.total_cost_usd += cost
        self.call_count += 1
        log.info(
            "ai call #%d model=%s in=%d out=%d cost=$%.4f total=$%.4f",
            self.call_count, model, tin, tout, cost, self.total_cost_usd,
        )

    def over_budget(self) -> bool:
        return self.total_cost_usd >= config.MAX_BUDGET_USD

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.call_count,
            "tokens_in": self.total_tokens_in,
            "tokens_out": self.total_tokens_out,
            "cost_usd": round(self.total_cost_usd, 4),
        }
