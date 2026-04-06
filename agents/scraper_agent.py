"""
agents/scraper_agent.py — Web scraping agent.

Responsibilities:
  1. Pop URLs from state.pending_sources one at a time.
  2. Scrape each source (browser → requests → PDF fallback).
  3. Apply a quality gate: skip abstract-only or thin content.
  4. Try open-access fallbacks for paywalled pages.
  5. Save raw scraped text back into the state for the Extractor agent.
  6. Mark sources as processed or dead.

The agent does NOT extract structured data — that is the Extractor's job.
Keeping scraping and extraction separate makes each piece testable and
replaceable independently.

Output:
  state.pending_sources is drained (each URL becomes processed or dead).
  Returns a list of ScrapedDocument objects ready for extraction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rich.console import Console

import config as cfg
from state import SessionState
from tools.browser import BrowserPool, extract_pdf_text
from tools.pdf_tools import (
    extract_tables_from_html,
    extract_tables_from_pdf,
    find_data_availability_section,
    find_dataset_links,
)

console = Console()


# ── Data container returned to Extractor ─────────────────────────────────────

@dataclass
class ScrapedDocument:
    """
    Everything the Extractor agent needs about one scraped source.

    text           — cleaned plain text (primary extraction target)
    tables         — list of CSV strings from tables found in the source
    dataset_links  — DOIs, GitHub repos, etc. found in the text
    da_section     — Data Availability section if found (high value for papers)
    source_url     — original URL / file path
    method         — how it was scraped: playwright | requests | pdf | local_pdf
    """
    text:          str
    tables:        list[str] = field(default_factory=list)
    dataset_links: list[dict] = field(default_factory=list)
    da_section:    str = ""
    source_url:    str = ""
    method:        str = ""

    def full_context(self, max_chars: int = cfg.MAX_TEXT_CHARS) -> str:
        """
        Build the full text block to send to the Extractor.

        Order: tables first (most structured), then da_section, then text.
        This ordering ensures the LLM sees the most information-dense content
        early in its context window.
        """
        parts = []

        if self.da_section:
            parts.append(f"[DATA AVAILABILITY SECTION]\n{self.da_section}")

        if self.tables:
            for i, tbl in enumerate(self.tables):
                parts.append(f"[TABLE {i+1}]\n{tbl}")

        parts.append(f"[MAIN TEXT]\n{self.text}")

        combined = "\n\n---\n\n".join(parts)
        return combined[:max_chars]


# ═══════════════════════════════════════════════════════════════════════════════
# Quality gate
# ═══════════════════════════════════════════════════════════════════════════════

# LLM-generated topic keywords injected by Orchestrator
_topic_content_keywords: list[str] = []

# Universal content-quality keywords (present on real pages, absent on paywalls)
_DEFAULT_CONTENT_KEYWORDS = [
    "method", "material", "experiment", "protocol", "procedure",
    "result", "finding", "conclusion", "discussion", "abstract",
    "figure", "table", "data", "analysis", "study", "sample",
    "participants", "patient", "treatment", "control", "outcome",
]


def configure_content_keywords(keywords: list[str]) -> None:
    """Inject LLM-generated topic-specific content keywords."""
    global _topic_content_keywords
    _topic_content_keywords = [k.lower() for k in keywords]


def _check_quality(text: str, method: str) -> tuple[bool, str]:
    """
    Decide whether scraped content is rich enough to extract data from.

    Returns:
        (is_good, reason_string)

    Reasons for failure:
        "too_short"    — fewer than MIN_FULLTEXT_CHARS chars
        "landing_page" — no topic-relevant content keywords detected (HTML only)
    """
    length = len(text.strip())

    if length < cfg.MIN_FULLTEXT_CHARS:
        return False, f"too_short ({length:,} chars, need {cfg.MIN_FULLTEXT_CHARS:,}+)"

    # For HTML-scraped pages only: check for content keywords
    # PDFs that are short were already handled by the text extractor
    if method in ("playwright", "requests"):
        text_lower = text.lower()
        all_keywords = _DEFAULT_CONTENT_KEYWORDS + _topic_content_keywords
        if not any(w in text_lower for w in all_keywords):
            return False, "landing_page (no content keywords detected)"

    return True, "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# Scraper agent
# ═══════════════════════════════════════════════════════════════════════════════

class ScraperAgent:
    """
    Drains state.pending_sources and returns ScrapedDocument objects.

    Usage:
        agent = ScraperAgent(browser_pool)
        docs  = agent.run(state, max_docs=10)
        # docs is a list of ScrapedDocument
    """

    def __init__(self, browser: BrowserPool) -> None:
        self.browser = browser

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        state: SessionState,
        max_docs: int | None = None,
    ) -> list[ScrapedDocument]:
        """
        Process up to max_docs sources from state.pending_sources.

        Args:
            state    — the shared session state
            max_docs — cap on how many sources to process (default: state.source_limit)

        Returns:
            List of ScrapedDocument objects ready for the Extractor agent.
        """
        if max_docs is None:
            max_docs = state.source_limit

        console.print(f"\n[bold cyan]🌐 Scraper Agent[/bold cyan]"
                      f" — {len(state.pending_sources)} pending, "
                      f"target {max_docs} good docs")

        docs:       list[ScrapedDocument] = []
        good_count: int = 0
        skipped:    int = 0

        # Work through the queue until we have enough good docs
        while state.pending_sources and good_count < max_docs:
            url = state.pending_sources[0]
            state.pending_sources.pop(0)

            console.print(f"\n[cyan]► {url[:80]}[/cyan]")

            doc = self._process_one(url, state)

            if doc is not None:
                docs.append(doc)
                good_count += 1
                state.mark_processed(url)
                console.print(
                    f"  [green]✓ Good ({len(doc.text):,} chars, "
                    f"{len(doc.tables)} tables, "
                    f"{len(doc.dataset_links)} links)[/green]"
                )
            else:
                skipped += 1
                state.mark_processed(url)

        console.print(
            f"\n[dim]Scraper done: {good_count} good, {skipped} skipped[/dim]"
        )
        return docs

    # ── Single-source processing ────────────────────────────────────────────

    def _process_one(
        self,
        url: str,
        state: SessionState,
    ) -> ScrapedDocument | None:
        """
        Attempt to scrape one source and return a ScrapedDocument.

        Returns None if:
          - All scrape attempts fail
          - Content is too thin (abstract-only, paywall, landing page)
          - No open-access fallback was found
        """
        is_local_pdf = url.endswith(".pdf") and not url.startswith("http")

        # ── Scrape with retry ────────────────────────────────────────────────
        scraped = self._scrape_with_retry(url, is_local_pdf)

        if not scraped or not scraped.get("text"):
            console.print(f"  [red]✗ SKIPPED — no text extracted[/red]")
            state.mark_dead(url)
            return None

        full_text = scraped["text"]
        method    = scraped.get("method", "unknown")

        # ── Quality gate ─────────────────────────────────────────────────────
        is_good, reason = _check_quality(full_text, method)

        if not is_good:
            console.print(f"  [yellow]⚠ Thin content ({reason})[/yellow]")

            # Try open-access fallback for paywalled academic URLs
            if not is_local_pdf:
                fallback_doc = self._try_oa_fallback(url, state)
                if fallback_doc:
                    return fallback_doc

            state.mark_dead(url)
            return None

        # ── Build document ───────────────────────────────────────────────────
        return self._build_document(scraped, url)

    def _scrape_with_retry(
        self,
        url: str,
        is_local_pdf: bool,
    ) -> dict | None:
        """
        Attempt scraping with exponential backoff retry.

        Local PDFs: single attempt (filesystem is deterministic).
        URLs: up to MAX_SCRAPE_RETRIES attempts with 2s/4s waits.
        """
        if is_local_pdf:
            return self.browser.scrape_local_pdf(url)

        for attempt in range(1, cfg.MAX_SCRAPE_RETRIES + 1):
            result = self.browser.scrape(url)
            if result and result.get("text"):
                return result

            if attempt < cfg.MAX_SCRAPE_RETRIES:
                wait = 2 ** attempt
                console.print(
                    f"  [yellow]⚠ Attempt {attempt}/{cfg.MAX_SCRAPE_RETRIES} failed"
                    f" — retrying in {wait}s…[/yellow]"
                )
                time.sleep(wait)

        return None

    def _try_oa_fallback(
        self,
        original_url: str,
        state: SessionState,
    ) -> ScrapedDocument | None:
        """
        For a paywalled URL, try to find an open-access version.

        Uses the browser's fallback URL generator (domain swaps + Semantic Scholar).
        """
        fallbacks = self.browser.find_open_access_fallback(original_url)

        for fb_url in fallbacks[:3]:   # try at most 3 fallbacks
            if fb_url in state.processed_sources or fb_url in state.dead_sources:
                continue

            console.print(f"  [cyan]↩ OA fallback: {fb_url[:70]}[/cyan]")

            # Special case: Semantic Scholar returns JSON, not HTML
            if "api.semanticscholar.org" in fb_url:
                pdf_url = self._resolve_semantic_scholar(fb_url)
                if pdf_url:
                    scraped = self._scrape_with_retry(pdf_url, is_local_pdf=False)
                else:
                    continue
            else:
                scraped = self._scrape_with_retry(fb_url, is_local_pdf=False)

            if not scraped or not scraped.get("text"):
                continue

            is_good, reason = _check_quality(scraped["text"], scraped.get("method", ""))
            if is_good:
                console.print(f"  [green]✓ OA fallback succeeded[/green]")
                return self._build_document(scraped, fb_url)
            else:
                console.print(f"  [dim]Fallback also thin: {reason}[/dim]")

        return None

    def _resolve_semantic_scholar(self, api_url: str) -> str | None:
        """
        Call the Semantic Scholar API and extract a direct PDF URL.
        Returns None if no open-access PDF is found.
        """
        try:
            import requests
            resp = requests.get(api_url, timeout=10)
            data = resp.json()
            papers = data.get("data", [])
            for paper in papers:
                oa = paper.get("openAccessPdf")
                if oa and oa.get("url"):
                    return oa["url"]
        except Exception:
            pass
        return None

    def _build_document(self, scraped: dict, url: str) -> ScrapedDocument:
        """
        Enrich a raw scraped result with tables, links, and DA section.
        """
        text     = scraped["text"]
        html     = scraped.get("html", "")
        method   = scraped.get("method", "unknown")
        pdf_path = scraped.get("pdf_path")

        # Extract tables
        tables: list[str] = []
        if pdf_path:
            tables = extract_tables_from_pdf(pdf_path)
        elif html:
            tables = extract_tables_from_html(html)

        # Find dataset DOIs / repo links
        dataset_links = find_dataset_links(text)

        # Find data availability section (most valuable for academic papers)
        da_section = find_data_availability_section(text)

        return ScrapedDocument(
            text          = text,
            tables        = tables,
            dataset_links = dataset_links,
            da_section    = da_section,
            source_url    = url,
            method        = method,
        )