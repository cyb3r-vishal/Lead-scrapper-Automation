"""The autonomous control loop.

Flow:
  1. Opus 4.7 reads the task, returns { columns, seed_queries, source_hints }.
  2. Haiku 4.5 drives each step: given task + state + DOM, emits one action.
  3. On 3 consecutive no-progress steps, Opus is re-invoked for a strategy reset.
  4. Deduped leads are streamed into the Excel file as they arrive.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from typing import Any

import config
from ai_client import OpenRouterClient, OpenRouterError
from browser import Browser
from excel_writer import ExcelWriter

log = logging.getLogger("agent")

RECENT_MEMORY = 12  # how many past queries/URLs the stepper sees and must avoid


PLANNER_SYSTEM = """You are a lead-generation strategist. Given a user task, \
return a JSON plan.

Respond with ONLY this JSON shape:
{
  "columns": ["name", "phone", "address", "email", "website", "rating", "reviews"],
  "seed_queries": ["...", "...", "..."],
  "source_hints": ["site1.com", "site2.com"]
}

Rules:
- "columns" = the exact Excel columns that best fit the task. Always include \
"name". Include "phone" and "email" when the task involves contactable leads. \
Add task-specific fields (e.g. "specialty" for doctors, "company" for founders).
- "seed_queries" = 3 to 5 Google-style queries that will surface leads. \
Vary them (directory sites, map queries, '"email" OR "contact"' patterns).
- "source_hints" = 3 to 8 domains likely to list such leads \
(directories, marketplaces, professional registries). Location-aware.
- Do NOT output prose, only JSON.
"""


STEPPER_SYSTEM = """You are an autonomous web-scraping agent.

OUTPUT CONTRACT — CRITICAL:
Respond with a single JSON object and NOTHING else. No prose, no explanation,
no markdown, no code fences. Your reply MUST start with `{` and end with `}`.

The JSON object must have an "action" key with one of these shapes:

  {"action":"search","engine":"google","query":"real estate agents mumbai"}
  {"action":"search","engine":"duckduckgo","query":"..."}
  {"action":"search","engine":"bing","query":"..."}
  {"action":"visit","url":"https://..."}
  {"action":"extract","leads":[{"name":"...","phone":"...","email":"..."}]}
  {"action":"scroll","amount":"page"}
  {"action":"scroll","amount":"bottom"}
  {"action":"back"}
  {"action":"finish","reason":"target reached"}

BEHAVIORAL RULES:
- Prefer "extract" whenever the current page clearly lists leads matching the
  task. Emit as many rows as you can read in one shot; missing fields -> "".
- If the state says blocked:true, immediately search on a different engine.
- On a SERP, pick the most promising result and "visit" it. Prefer
  directory/listing sites (justdial, indiamart, 99acres) over blog posts.
- Never repeat a query or URL already listed in state.recent_queries or
  state.recent_urls — pick something new.
- If state.source_hints lists sites you haven't tried, prefer visiting one of
  them directly over searching again.
- "scroll" when more leads are likely below the fold.
- "back" when the current page is a dead end.
- "finish" only when target_count is reached or no further progress is possible.
- NEVER fabricate leads. Only extract what is clearly on the page.
- Keep phone numbers with country/area codes when shown.
- If state.current_page_exhausted is true OR last_result says the current URL
  is EXHAUSTED / added 0 new, the next action MUST be "search" or "visit" to
  a URL NOT already in recent_urls. Do NOT emit "scroll" or "extract" on an
  exhausted page — it will just repeat duplicates.
"""


class Agent:
    def __init__(
        self,
        ai: OpenRouterClient,
        browser: Browser,
        writer: ExcelWriter | None = None,
    ) -> None:
        self.ai = ai
        self.browser = browser
        self.writer = writer  # created after planning, when columns are known
        self.seen_keys: set[str] = set()  # dedupe fingerprints
        self.leads_collected = 0
        self.recent_queries: deque[str] = deque(maxlen=RECENT_MEMORY)
        self.recent_urls: deque[str] = deque(maxlen=RECENT_MEMORY)
        self.tried_source_hints: set[str] = set()
        # Per-URL count of consecutive extracts that produced 0 new leads.
        # Lets us detect "this page is fully scraped / already in the file"
        # and force the agent to navigate away instead of re-extracting.
        self.url_stall_counts: dict[str, int] = {}

    # ---- public API --------------------------------------------------------

    def run(self, task: str, count: int, out_path: str) -> dict[str, Any]:
        started = time.time()
        log.info("=== run start === task=%r count=%d out=%s", task, count, out_path)

        self._plan_ref = self._plan(task)
        columns = self._plan_ref.get("columns") or ["name", "phone", "email", "website", "address"]
        self.writer = ExcelWriter(out_path, columns)
        log.info("plan columns=%s seed_queries=%s", columns, self._plan_ref.get("seed_queries"))
        self._hydrate_seen_from_file()

        self.browser.start()

        last_action: dict[str, Any] | None = None
        last_result: str = "Session starting. Use your seed queries to begin."
        no_progress_streak = 0
        step = 0

        try:
            while (
                self.leads_collected < count
                and step < config.MAX_STEPS
                and not self.ai.over_budget()
                and time.time() - started < config.MAX_RUNTIME_S
            ):
                step += 1
                log.info("--- step %d --- leads=%d/%d", step, self.leads_collected, count)

                if no_progress_streak >= config.NO_PROGRESS_STEPS_BEFORE_RECOVERY:
                    self._plan_ref = self._recover(task, self._plan_ref)
                    # After reset, force a deterministic navigation so we don't
                    # re-extract the stuck page. The AI otherwise tends to
                    # repeat the same extract on the same DOM.
                    action = self._fallback_action(self._plan_ref)
                    log.info("post-recovery forced action: %s",
                             json.dumps(action, ensure_ascii=False)[:200])
                    last_action = action
                    try:
                        last_result, progressed = self._execute(action)
                    except Exception as e:
                        log.exception("post-recovery action failed: %s", e)
                        last_result = f"Action error: {e}"
                        progressed = False
                    no_progress_streak = 0 if progressed else no_progress_streak + 1
                    if action.get("action") == "finish":
                        log.info("fallback exhausted after recovery: %s",
                                 action.get("reason"))
                        break
                    continue

                # If the current URL is already known-exhausted, don't waste
                # an AI call — just navigate away deterministically.
                cur_url = self.browser.current_url()
                if self.url_stall_counts.get(cur_url, 0) >= 2:
                    log.info("URL is exhausted (%d stalls); forcing fallback",
                             self.url_stall_counts[cur_url])
                    action = self._fallback_action(self._plan_ref)
                else:
                    action = self._step(task, count, self._plan_ref, last_action, last_result)
                last_action = action

                try:
                    last_result, progressed = self._execute(action)
                except Exception as e:
                    log.exception("action failed: %s", e)
                    last_result = f"Action error: {e}. Try something different."
                    progressed = False

                no_progress_streak = 0 if progressed else no_progress_streak + 1

                if action.get("action") == "finish":
                    log.info("agent chose finish: %s", action.get("reason"))
                    break

        finally:
            self.browser.quit()

        elapsed = time.time() - started
        summary = {
            "leads": self.leads_collected,
            "target": count,
            "steps": step,
            "elapsed_s": round(elapsed, 1),
            **self.ai.summary(),
            "out": out_path,
        }
        log.info("=== run end === %s", summary)
        return summary

    # ---- planning ----------------------------------------------------------

    def _plan(self, task: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"Task: {task}"},
        ]
        try:
            return self.ai.plan(messages, max_tokens=800)
        except OpenRouterError as e:
            log.warning("planner failed (%s); using generic defaults", e)
            return {
                "columns": ["name", "phone", "email", "address", "website"],
                "seed_queries": [task, f'"{task}" contact', f"{task} directory"],
                "source_hints": ["justdial.com", "indiamart.com", "google.com/maps"],
            }

    def _recover(self, task: str, prior: dict[str, Any]) -> dict[str, Any]:
        log.warning("no progress for %d steps — invoking Opus for recovery",
                    config.NO_PROGRESS_STEPS_BEFORE_RECOVERY)
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content":
                f"Task: {task}\n"
                f"Prior plan had these columns: {prior.get('columns')} and "
                f"seed_queries: {prior.get('seed_queries')}.\n"
                f"These queries are not producing leads. Propose a FRESH set of "
                f"seed_queries targeting different directories or angles. Keep the "
                f"same 'columns' unless you have a strong reason to change."
            },
        ]
        try:
            new_plan = self.ai.plan(messages, max_tokens=800)
            new_plan.setdefault("columns", prior.get("columns"))
            # Preserve prior hints if Opus omits them — _fallback_action needs
            # them to navigate away when the AI gets stuck again.
            prior_hints = prior.get("source_hints") or []
            new_hints = new_plan.get("source_hints") or []
            merged = list(dict.fromkeys([*new_hints, *prior_hints]))
            if merged:
                new_plan["source_hints"] = merged
            return new_plan
        except OpenRouterError as e:
            log.warning("recovery planner failed (%s); keeping prior plan", e)
            return prior

    # ---- per-step ----------------------------------------------------------

    def _step(
        self,
        task: str,
        count: int,
        plan: dict[str, Any],
        last_action: dict[str, Any] | None,
        last_result: str,
    ) -> dict[str, Any]:
        hints = plan.get("source_hints") or []
        current_url = self.browser.current_url()
        stalls_here = self.url_stall_counts.get(current_url, 0)
        state = {
            "task": task,
            "target_count": count,
            "leads_collected": self.leads_collected,
            "columns": plan.get("columns"),
            "seed_queries": plan.get("seed_queries"),
            "source_hints": hints,
            "untried_source_hints": [h for h in hints if h not in self.tried_source_hints],
            "current_url": current_url,
            "page_title": self.browser.title(),
            "blocked": self.browser.is_blocked(),
            "recent_queries": list(self.recent_queries),
            "recent_urls": list(self.recent_urls),
            "current_page_stalls": stalls_here,
            "current_page_exhausted": stalls_here >= 1,
            "exhausted_urls": [u for u, n in self.url_stall_counts.items() if n >= 2],
            "last_action": last_action,
            "last_result": last_result,
        }
        dom = self.browser.simplified_dom()
        user = (
            "STATE:\n" + json.dumps(state, ensure_ascii=False, indent=2)
            + "\n\nPAGE:\n" + (dom or "(no page loaded yet)")
        )
        messages = [
            {"role": "system", "content": STEPPER_SYSTEM},
            {"role": "user", "content": user},
        ]
        try:
            return self.ai.routine(messages, max_tokens=2000)
        except OpenRouterError as e:
            log.warning("stepper failed (%s); using deterministic fallback", e)
            return self._fallback_action(plan)

    def _fallback_action(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Pick something new when the AI can't. Priority:
        1) visit an untried source_hint, 2) try an unused seed_query,
        3) finish (we've exhausted obvious options)."""
        hints = plan.get("source_hints") or []
        for h in hints:
            if h in self.tried_source_hints:
                continue
            url = h if h.startswith("http") else f"https://{h}"
            if url not in self.recent_urls:
                return {"action": "visit", "url": url}

        for q in plan.get("seed_queries") or []:
            if q not in self.recent_queries:
                return {"action": "search", "engine": "duckduckgo", "query": q}

        return {"action": "finish", "reason": "fallback exhausted all hints and seed queries"}

    # ---- action execution --------------------------------------------------

    def _execute(self, action: dict[str, Any]) -> tuple[str, bool]:
        """Return (result_summary, progressed). progressed=True when we either
        added leads or successfully moved to a new page. Repeating a recent
        query or URL counts as no progress."""
        a = action.get("action")
        log.info("action: %s", json.dumps(action, ensure_ascii=False)[:400])

        if a == "search":
            eng = action.get("engine", "duckduckgo")
            q = (action.get("query") or "").strip()
            if not q:
                return "Empty query; pick another action.", False
            if q in self.recent_queries:
                return (
                    f"Query {q!r} was already tried (see recent_queries). "
                    "Pick a different query or visit a source_hint.",
                    False,
                )
            self.recent_queries.append(q)
            self.browser.search(eng, q)
            if self.browser.is_blocked():
                return f"{eng} blocked us. Pivot to a different engine.", False
            return f"Searched {eng} for '{q}'. Review results and pick one to visit.", True

        if a == "visit":
            url = action.get("url", "")
            if not url or not re.match(r"^https?://", url):
                return "Invalid URL; try another result.", False
            if url in self.recent_urls:
                return (
                    f"URL {url} was already visited (see recent_urls). "
                    "Pick a different result.",
                    False,
                )
            self.recent_urls.append(url)
            self._mark_hint_if_matches(url)
            self.browser.visit(url)
            return f"Visited {url}.", True

        if a == "scroll":
            self.browser.scroll(action.get("amount", "page"))
            # Scrolling alone isn't progress — only actual new leads count.
            # Otherwise extract/scroll loops on a stuck page reset the
            # no-progress streak forever and recovery never fires.
            return "Scrolled.", False

        if a == "back":
            self.browser.back()
            return "Went back.", False

        if a == "extract":
            leads = action.get("leads") or []
            added = self._ingest(leads)
            url = self.browser.current_url()
            if added == 0:
                self.url_stall_counts[url] = self.url_stall_counts.get(url, 0) + 1
                msg = (
                    f"Extracted {len(leads)} rows but added 0 NEW leads "
                    f"(all duplicates — already in the output file). "
                    f"URL {url!r} is EXHAUSTED (stall #{self.url_stall_counts[url]}). "
                    "You MUST navigate away now: emit 'visit' with a NEW url "
                    "(not in recent_urls) or 'search' with a fresh query. "
                    "Do NOT 'scroll' or 'extract' on this page again."
                )
            else:
                self.url_stall_counts[url] = 0
                msg = (
                    f"Extracted {len(leads)} candidate rows, added {added} new "
                    f"(total {self.leads_collected})."
                )
            return msg, added > 0

        if a == "finish":
            return f"Finished: {action.get('reason','')}", False

        return f"Unknown action {a!r}; try again.", False

    # ---- lead bookkeeping --------------------------------------------------

    def _ingest(self, leads: list[dict[str, Any]]) -> int:
        """Add leads that don't collide with anything already seen.

        Each row yields MULTIPLE fingerprint keys (phone, email, website,
        name+phone). A row is a duplicate if ANY of its keys is already in
        seen_keys. Accepted rows register ALL their keys so later rows with
        only-partially-overlapping data (same phone, different website) are
        still caught."""
        assert self.writer is not None
        added = 0
        dupes = 0
        for row in leads:
            if not isinstance(row, dict):
                continue
            keys = self._fingerprints(row)
            if not keys:
                continue  # not enough identifying info
            if any(k in self.seen_keys for k in keys):
                dupes += 1
                continue
            row.setdefault("source_url", self.browser.current_url())
            for k in keys:
                self.seen_keys.add(k)
            self.writer.append(row)
            self.leads_collected += 1
            added += 1
        if dupes:
            log.info("skipped %d duplicate row(s) in this extract", dupes)
        return added

    def _hydrate_seen_from_file(self) -> int:
        """Pre-fill seen_keys with everything already in the output file, so
        a re-run against an existing workbook never re-adds prior leads."""
        assert self.writer is not None
        loaded = 0
        for existing in self.writer.existing_rows():
            for k in self._fingerprints(existing):
                self.seen_keys.add(k)
            loaded += 1
        if loaded:
            log.info("hydrated %d prior lead(s) from %s into dedupe set",
                     loaded, self.writer.path)
        return loaded

    def _mark_hint_if_matches(self, url: str) -> None:
        plan = getattr(self, "_plan_ref", None) or {}
        for hint in plan.get("source_hints") or []:
            host = hint.replace("https://", "").replace("http://", "").strip("/")
            if host and host.split("/")[0] in url:
                self.tried_source_hints.add(hint)

    # ---- fingerprinting ----------------------------------------------------

    _NAME_SUFFIXES = re.compile(
        r"\b(pvt\.?|private|ltd\.?|limited|llp|llc|inc\.?|co\.?|corp\.?|"
        r"corporation|company|realtors?|realty|properties|property|estates?|"
        r"developers?|group|and|&)\b",
        flags=re.IGNORECASE,
    )

    @classmethod
    def _norm_name(cls, name: Any) -> str:
        s = re.sub(r"\s+", " ", str(name or "").strip().lower())
        s = cls._NAME_SUFFIXES.sub("", s)
        s = re.sub(r"[^\w\s]", "", s)
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _norm_phone(phone: Any) -> str:
        """Keep only the trailing 10 digits so country-code / separator
        variants collapse to the same key. E.g. '+91 98765 43210',
        '09876543210', '98765-43210' all become '9876543210'."""
        digits = re.sub(r"\D", "", str(phone or ""))
        return digits[-10:] if len(digits) >= 10 else digits

    @staticmethod
    def _norm_website(url: Any) -> str:
        s = str(url or "").strip().lower()
        s = re.sub(r"^https?://", "", s)
        s = re.sub(r"^www\.", "", s)
        return s.split("?")[0].rstrip("/")

    @staticmethod
    def _norm_email(email: Any) -> str:
        return str(email or "").strip().lower()

    @classmethod
    def _fingerprints(cls, row: dict[str, Any]) -> list[str]:
        """All dedupe keys applicable to this row. ANY match = duplicate."""
        name    = cls._norm_name(row.get("name"))
        phone   = cls._norm_phone(row.get("phone"))
        email   = cls._norm_email(row.get("email"))
        website = cls._norm_website(row.get("website"))

        keys: list[str] = []
        if phone and len(phone) >= 7:
            keys.append(f"p|{phone}")
        if email and "@" in email:
            keys.append(f"e|{email}")
        if website and "." in website:
            keys.append(f"w|{website}")
        if name and phone:
            keys.append(f"np|{name}|{phone}")
        # Name-only is the weakest key — use it only when no contact info
        # exists, so two legitimately distinct businesses with identical
        # names don't collide when they *do* have different contacts.
        if name and not keys:
            keys.append(f"n|{name}")
        return keys
