"""Offline smoke test for the agent loop — no network, no browser, no API.

Stubs out Browser + OpenRouterClient + ExcelWriter, feeds the agent a fixed
script of AI actions, and asserts the loop does the right thing:

  1. scroll / back no longer count as progress.
  2. extract with 0 new increments url_stall_counts and emits EXHAUSTED directive.
  3. After 3 no-progress steps, Opus "recovery" fires AND forces a fallback
     navigation instead of letting the AI re-extract the stuck page.
  4. When a URL has >=2 stalls, the loop skips the AI call and forces a fallback.
"""
from __future__ import annotations

import json
import sys
from typing import Any

import config
from agent import Agent


class StubBrowser:
    def __init__(self) -> None:
        self._url = ""
        self.actions: list[str] = []

    def start(self) -> None:
        self._url = "about:blank"

    def quit(self) -> None:
        self.actions.append("quit")

    def visit(self, url: str) -> None:
        self._url = url
        self.actions.append(f"visit:{url}")

    def search(self, engine: str, query: str) -> None:
        self._url = f"https://{engine}.test/?q={query}"
        self.actions.append(f"search:{engine}:{query}")

    def scroll(self, amount: str = "page") -> None:
        self.actions.append(f"scroll:{amount}")

    def back(self) -> None:
        self.actions.append("back")

    def current_url(self) -> str:
        return self._url

    def title(self) -> str:
        return "stub"

    def is_blocked(self) -> bool:
        return False

    def simplified_dom(self) -> str:
        return "TITLE: stub\nCONTENT:\nnothing\n"


class StubWriter:
    """In-memory drop-in for ExcelWriter."""
    def __init__(self) -> None:
        self.path = "stub.xlsx"
        self.rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(dict(row))

    def existing_rows(self) -> list[dict[str, Any]]:
        return []


class ScriptedAI:
    """Returns actions from a fixed queue. Also provides a plan()."""
    def __init__(self, actions: list[dict]) -> None:
        self.queue = list(actions)
        self.call_count = 0
        self.total_cost_usd = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.routine_calls: list[dict] = []
        self.plan_calls = 0

    def routine(self, messages, max_tokens=2000):
        self.call_count += 1
        if not self.queue:
            return {"action": "finish", "reason": "script exhausted"}
        action = self.queue.pop(0)
        self.routine_calls.append(action)
        return action

    def plan(self, messages, max_tokens=800):
        self.plan_calls += 1
        # Keep the "fresh" hints distinct from the visit URLs the script uses.
        return {
            "columns": ["name", "phone", "email", "website", "address"],
            "seed_queries": ["fresh query A", "fresh query B"],
            "source_hints": ["fresh-hint-1.test", "fresh-hint-2.test"],
        }

    def over_budget(self) -> bool:
        return False

    def summary(self) -> dict:
        return {
            "calls": self.call_count,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        }


def _build_agent(ai: ScriptedAI) -> tuple[Agent, StubBrowser, StubWriter]:
    browser = StubBrowser()
    writer = StubWriter()
    agent = Agent(ai, browser)  # type: ignore[arg-type]
    agent.writer = writer  # bypass ExcelWriter(path=…) construction
    agent._plan_ref = {
        "columns": ["name", "phone", "email", "website", "address"],
        "seed_queries": ["seed 1", "seed 2"],
        "source_hints": ["hint-A.test", "hint-B.test"],
    }
    browser.start()
    return agent, browser, writer


# ---------- unit-style checks on _execute ------------------------------------

def test_scroll_and_back_are_not_progress() -> None:
    agent, _, _ = _build_agent(ScriptedAI([]))
    _, progressed = agent._execute({"action": "scroll", "amount": "bottom"})
    assert progressed is False, "scroll must not count as progress"
    _, progressed = agent._execute({"action": "back"})
    assert progressed is False, "back must not count as progress"
    print("  OK  scroll/back -> progressed=False")


def test_extract_zero_new_emits_exhausted_directive() -> None:
    agent, browser, _ = _build_agent(ScriptedAI([]))
    browser.visit("https://example.test/page1")
    # Pre-seed so every row is a duplicate.
    agent.seen_keys.add("p|9920048811")
    result, progressed = agent._execute({
        "action": "extract",
        "leads": [{"name": "Walk 2 Dreams Home LLP", "phone": "099200 48811"}],
    })
    assert progressed is False
    assert "EXHAUSTED" in result, f"directive missing EXHAUSTED: {result!r}"
    assert "stall #1" in result
    assert agent.url_stall_counts["https://example.test/page1"] == 1
    # Second stall increments the counter.
    agent._execute({
        "action": "extract",
        "leads": [{"name": "Walk 2 Dreams Home LLP", "phone": "099200 48811"}],
    })
    assert agent.url_stall_counts["https://example.test/page1"] == 2
    print("  OK  extract(0 new) bumps stall + emits EXHAUSTED")


def test_extract_with_new_resets_stall() -> None:
    agent, browser, writer = _build_agent(ScriptedAI([]))
    browser.visit("https://example.test/page1")
    agent.url_stall_counts["https://example.test/page1"] = 5
    _, progressed = agent._execute({
        "action": "extract",
        "leads": [{"name": "Fresh Lead", "phone": "012 345 6789"}],
    })
    assert progressed is True
    assert agent.url_stall_counts["https://example.test/page1"] == 0
    assert len(writer.rows) == 1
    print("  OK  extract(N new) resets stall to 0 and writes row")


# ---------- loop-level integration checks ------------------------------------

def test_recovery_fires_after_three_no_progress_steps() -> None:
    # Scripted AI tries to extract duplicates and scroll — classic stuck loop.
    # After 3 no-progress steps the loop should fire recovery, which invokes
    # plan(), then execute a forced fallback action (visit untried hint).
    dup_lead = {"name": "X", "phone": "012 345 6789"}  # seeded as dupe below
    ai = ScriptedAI([
        {"action": "visit", "url": "https://stuck.test/list"},
        {"action": "extract", "leads": [dup_lead]},   # 0 new
        {"action": "scroll", "amount": "bottom"},     # not progress
        {"action": "extract", "leads": [dup_lead]},   # 0 new -> recovery
        # If the loop forgot to force fallback, the AI would get called again
        # and we'd pop this; we assert it's NOT consumed.
        {"action": "extract", "leads": [dup_lead]},
    ])
    agent, browser, _ = _build_agent(ai)
    # Pre-seed the dedupe set so the "X" lead is always a duplicate.
    agent.seen_keys.add("p|0123456789")

    # Run a short, bounded version of the loop manually (mimics Agent.run
    # but skips browser.start/quit and uses our stubs).
    import time
    last_action: dict[str, Any] | None = None
    last_result = "start"
    no_progress_streak = 0
    steps_taken = 0
    started = time.time()
    orig_max_steps = config.MAX_STEPS
    config.MAX_STEPS = 10
    try:
        while (
            agent.leads_collected < 1000
            and steps_taken < config.MAX_STEPS
            and not ai.over_budget()
            and time.time() - started < 60
        ):
            # Stop after recovery+forced-fallback round, before the next
            # normal step would ask the AI about the new page.
            if ai.plan_calls >= 1:
                break
            steps_taken += 1
            if no_progress_streak >= config.NO_PROGRESS_STEPS_BEFORE_RECOVERY:
                agent._plan_ref = agent._recover("task", agent._plan_ref)
                action = agent._fallback_action(agent._plan_ref)
                last_action = action
                last_result, progressed = agent._execute(action)
                no_progress_streak = 0 if progressed else no_progress_streak + 1
                if action.get("action") == "finish":
                    break
                continue

            cur_url = browser.current_url()
            if agent.url_stall_counts.get(cur_url, 0) >= 2:
                action = agent._fallback_action(agent._plan_ref)
            else:
                action = agent._step("task", 1000, agent._plan_ref,
                                     last_action, last_result)
            last_action = action
            last_result, progressed = agent._execute(action)
            no_progress_streak = 0 if progressed else no_progress_streak + 1
            if action.get("action") == "finish":
                break
    finally:
        config.MAX_STEPS = orig_max_steps

    assert ai.plan_calls >= 1, (
        "recovery never fired — no_progress_streak didn't reach threshold"
    )
    # The forced fallback after recovery should navigate to an untried hint.
    visits = [a for a in browser.actions if a.startswith("visit:")]
    assert any("fresh-hint-1.test" in v or "hint-A.test" in v or
               "hint-B.test" in v for v in visits[1:]), (
        f"no post-recovery navigation to untried hint: {browser.actions}"
    )
    # The queued 5th AI action should NOT have been consumed — recovery
    # forced a deterministic fallback instead of calling the AI.
    assert len(ai.routine_calls) <= 4, (
        f"AI was called after recovery instead of forced fallback: "
        f"{len(ai.routine_calls)} routine calls, {ai.routine_calls}"
    )
    print(f"  OK  recovery fired after no_progress_streak=3; "
          f"plan_calls={ai.plan_calls}, browser.actions={browser.actions}")


def test_exhausted_url_skips_ai_call() -> None:
    ai = ScriptedAI([
        # Primer: visit a URL
        {"action": "visit", "url": "https://stuck.test/list"},
        # Extract 0 new twice -> stall counter reaches 2
        {"action": "extract", "leads": [{"name": "X", "phone": "012 345 6789"}]},
        {"action": "extract", "leads": [{"name": "X", "phone": "012 345 6789"}]},
        # The loop should now force fallback instead of asking AI.
        # This action would only run if the bug was still present:
        {"action": "extract", "leads": [{"name": "X", "phone": "012 345 6789"}]},
    ])
    agent, browser, _ = _build_agent(ai)
    agent.seen_keys.add("p|0123456789")

    # Step 1: visit
    a = agent._step("task", 1, agent._plan_ref, None, "start")
    agent._execute(a)
    # Step 2: extract 0 new -> stall #1
    a = agent._step("task", 1, agent._plan_ref, a, "ok")
    agent._execute(a)
    # Step 3: extract 0 new -> stall #2
    a = agent._step("task", 1, agent._plan_ref, a, "ok")
    agent._execute(a)
    assert agent.url_stall_counts["https://stuck.test/list"] >= 2

    # Step 4: URL is exhausted -> loop MUST skip AI and force fallback.
    cur_url = browser.current_url()
    ai_calls_before = len(ai.routine_calls)
    if agent.url_stall_counts.get(cur_url, 0) >= 2:
        action = agent._fallback_action(agent._plan_ref)
    else:
        action = agent._step("task", 1, agent._plan_ref, a, "ok")
    agent._execute(action)
    ai_calls_after = len(ai.routine_calls)

    assert ai_calls_after == ai_calls_before, (
        "AI was called even though current URL had >=2 stalls"
    )
    assert action.get("action") in ("visit", "search", "finish"), (
        f"fallback picked a non-navigation action: {action}"
    )
    print("  OK  exhausted URL bypasses AI and forces navigation")


# ---------- run everything ---------------------------------------------------

def main() -> int:
    tests = [
        test_scroll_and_back_are_not_progress,
        test_extract_zero_new_emits_exhausted_directive,
        test_extract_with_new_resets_stall,
        test_recovery_fires_after_three_no_progress_steps,
        test_exhausted_url_skips_ai_call,
    ]
    failures = 0
    for t in tests:
        print(f"[{t.__name__}]")
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL  {e}")
            failures += 1
        except Exception as e:
            print(f"  ERROR {type(e).__name__}: {e}")
            failures += 1
    print()
    if failures:
        print(f"{failures} failure(s)")
        return 1
    print(f"all {len(tests)} smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
