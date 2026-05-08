"""CLI entry point.

Examples:
  python scraper.py --task "find 50 real estate agents in Mumbai" --count 50
  python scraper.py --task "dentists with email in Delhi" --count 30 --out dentists.xlsx
  python scraper.py --task "SaaS founders in Bangalore" --count 100 --headed
"""
from __future__ import annotations

import argparse
import logging
import sys

import config
from agent import Agent
from ai_client import OpenRouterClient
from browser import Browser


def _setup_logging(log_path: str) -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Autonomous AI lead scraper (OpenRouter + Selenium)."
    )
    p.add_argument("--task", required=True,
                   help='Natural-language task, e.g. "find 50 real estate agents in Mumbai".')
    p.add_argument("--count", type=int, default=50,
                   help="Target number of unique leads (default 50).")
    p.add_argument("--out", default="leads.xlsx",
                   help="Excel output path (default leads.xlsx).")
    p.add_argument("--headed", action="store_true",
                   help="Show the browser window (debugging).")
    p.add_argument("--log", default="run.log", help="Log file path.")
    args = p.parse_args(argv)

    _setup_logging(args.log)
    log = logging.getLogger("main")

    try:
        config.require_api_key()
    except RuntimeError as e:
        log.error("%s", e)
        return 2

    ai = OpenRouterClient()
    browser = Browser(headless=not args.headed)
    agent = Agent(ai, browser)

    try:
        summary = agent.run(args.task, args.count, args.out)
    except KeyboardInterrupt:
        log.warning("interrupted — partial leads are already saved to %s", args.out)
        return 130

    log.info("DONE: %s", summary)
    print(
        f"\nCollected {summary['leads']}/{summary['target']} leads in "
        f"{summary['steps']} steps, {summary['elapsed_s']}s, "
        f"${summary['cost_usd']:.4f}. File: {summary['out']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
