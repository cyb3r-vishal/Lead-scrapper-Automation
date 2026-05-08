"""Selenium wrapper: stealth Chrome, DOM simplifier, search-engine helpers,
block detection."""
from __future__ import annotations

import logging
import random
import time
import urllib.parse as up
from typing import Literal

from bs4 import BeautifulSoup, NavigableString
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options

import config

log = logging.getLogger("browser")

SearchEngine = Literal["google", "duckduckgo", "bing"]

_USER_AGENTS = [
    # Recent desktop Chrome on Windows. Rotated per session.
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
]

_BLOCK_URL_MARKERS = ("/sorry/", "captcha", "blocked")
_BLOCK_TEXT_MARKERS = (
    "unusual traffic",
    "our systems have detected",
    "not a robot",
    "verify you are human",
    "access denied",
)


class Browser:
    def __init__(self, headless: bool = True) -> None:
        self.headless = headless
        self.driver: webdriver.Chrome | None = None
        self._ua = random.choice(_USER_AGENTS)

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        opts = Options()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1366,900")
        opts.add_argument(f"--user-agent={self._ua}")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)

        try:
            self.driver = webdriver.Chrome(options=opts)
        except WebDriverException as e:
            # Fall back to undetected-chromedriver if plain Selenium can't launch.
            log.warning("Plain Chromedriver failed (%s); trying undetected-chromedriver", e)
            import undetected_chromedriver as uc  # type: ignore
            uc_opts = uc.ChromeOptions()
            if self.headless:
                uc_opts.add_argument("--headless=new")
            uc_opts.add_argument(f"--user-agent={self._ua}")
            uc_opts.add_argument("--window-size=1366,900")
            self.driver = uc.Chrome(options=uc_opts)

        self.driver.set_page_load_timeout(config.PAGE_LOAD_TIMEOUT_S)
        # Hide the webdriver flag via CDP.
        try:
            self.driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
            )
        except Exception:
            pass

    def quit(self) -> None:
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    # ---- actions -----------------------------------------------------------

    def visit(self, url: str) -> None:
        assert self.driver is not None
        try:
            self.driver.get(url)
        except TimeoutException:
            log.warning("Page load timed out on %s (continuing with partial DOM)", url)
        self._sleep()

    def search(self, engine: SearchEngine, query: str) -> None:
        q = up.quote_plus(query)
        url = {
            "google":     f"https://www.google.com/search?q={q}",
            "duckduckgo": f"https://duckduckgo.com/?q={q}",
            "bing":       f"https://www.bing.com/search?q={q}",
        }[engine]
        self.visit(url)

    def scroll(self, amount: str = "page") -> None:
        assert self.driver is not None
        if amount == "bottom":
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        else:
            self.driver.execute_script("window.scrollBy(0, window.innerHeight * 0.9);")
        self._sleep(0.5, 1.2)

    def back(self) -> None:
        assert self.driver is not None
        self.driver.back()
        self._sleep()

    # ---- introspection -----------------------------------------------------

    def current_url(self) -> str:
        assert self.driver is not None
        try:
            return self.driver.current_url
        except WebDriverException:
            return ""

    def title(self) -> str:
        assert self.driver is not None
        try:
            return self.driver.title or ""
        except WebDriverException:
            return ""

    def is_blocked(self) -> bool:
        """Heuristic: did Google (or anyone) serve us a block page?"""
        url = self.current_url().lower()
        if any(m in url for m in _BLOCK_URL_MARKERS):
            return True
        title = self.title().lower()
        if any(m in title for m in _BLOCK_TEXT_MARKERS):
            return True
        try:
            body = self.driver.find_element("tag name", "body").text.lower()[:2000]  # type: ignore
        except Exception:
            body = ""
        return any(m in body for m in _BLOCK_TEXT_MARKERS)

    def simplified_dom(self) -> str:
        """Return a text+links skeleton of the current page, capped to keep
        AI prompts cheap. This is what the agent 'sees'."""
        assert self.driver is not None
        try:
            html = self.driver.page_source
        except WebDriverException:
            return ""
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "noscript", "svg", "iframe", "meta", "link"]):
            tag.decompose()

        parts: list[str] = []
        title = (soup.title.string if soup.title and soup.title.string else "").strip()
        if title:
            parts.append(f"TITLE: {title}")

        body = soup.body or soup
        text_lines: list[str] = []
        links: list[str] = []
        contacts: list[str] = []

        for el in body.descendants:
            if isinstance(el, NavigableString):
                continue
            name = getattr(el, "name", None)
            if name in ("h1", "h2", "h3", "h4"):
                txt = el.get_text(" ", strip=True)
                if txt:
                    text_lines.append(f"[{name.upper()}] {txt}")
            elif name in ("p", "li", "td", "th", "dd", "dt", "span", "div"):
                # Only grab leaf-like elements with direct text to avoid dupes.
                direct = "".join(
                    c for c in el.strings if c.parent is el
                ).strip()
                if direct and len(direct) > 3:
                    text_lines.append(direct)
            elif name == "a":
                href = el.get("href", "")
                txt = el.get_text(" ", strip=True)
                if href.startswith("tel:") or href.startswith("mailto:"):
                    contacts.append(f"{href} ({txt})")
                elif href and txt:
                    links.append(f"{txt} -> {href}")

        if contacts:
            parts.append("CONTACTS:\n" + "\n".join(dict.fromkeys(contacts))[:2000])

        if text_lines:
            joined = "\n".join(dict.fromkeys(text_lines))
            parts.append("CONTENT:\n" + joined)

        if links:
            uniq = list(dict.fromkeys(links))[: config.DOM_LINK_LIMIT]
            parts.append("LINKS:\n" + "\n".join(uniq))

        out = "\n\n".join(parts)
        if len(out) > config.DOM_CHAR_LIMIT:
            # Keep head + tail around the middle cut; leads often live in both.
            keep = config.DOM_CHAR_LIMIT // 2 - 40
            out = out[:keep] + "\n...[TRUNCATED]...\n" + out[-keep:]
        return out

    # ---- internal ----------------------------------------------------------

    @staticmethod
    def _sleep(lo: float | None = None, hi: float | None = None) -> None:
        lo = lo if lo is not None else config.ACTION_DELAY_MIN_S
        hi = hi if hi is not None else config.ACTION_DELAY_MAX_S
        time.sleep(random.uniform(lo, hi))
