"""
agents/search_agent.py — Web search agent.

Responsibilities:
  1. Ask the LLM to generate diverse, targeted search queries for the topic.
  2. Execute those queries against DuckDuckGo (no API key required).
  3. Filter, deduplicate, and prioritise results using LLM-generated domain config.
  4. Write approved URLs into state.pending_sources.

Design notes:
  • Multiple queries are used so we cover different angles of the topic
    (overview, data, methods, specific sub-topics the user mentioned).
  • Domain priority tiers are LLM-generated per topic so a biology query
    prioritises PubMed/bioRxiv while a finance query prioritises SEC/Bloomberg.
  • Junk URL patterns (tag pages, login pages, etc.) are filtered universally.
  • Sources are collected into a buffer 3× the target limit, then the best
    ones are selected — this way skipped/paywalled sources have replacements.
"""

from __future__ import annotations

import random
import time
from typing import Optional

from ddgs import DDGS
from rich.console import Console

import config as cfg
from state import SessionState
from tools.llm_tools import LLMClient

console = Console()

# ── URL junk patterns — always filtered regardless of topic ──────────────────
_JUNK_URL_PATTERNS = [
    "/tag/", "/category/", "/blog/page/", "/news/page/",
    "/about/", "/contact/", "/login", "/signup", "/cart",
    "/product/", "/shop/", "/search?", "/topic/", "/author/",
    "/press-release/", "/advertise", "/subscribe/",
]

# Mutable per-session — set by _configure_domains()
_relevant_domains:   list[str]        = []
_blocked_domains:    list[str]        = []
_priority_tiers:     list[list[str]]  = []


def _configure_domains(domain_config: dict) -> None:
    """Apply LLM-generated domain configuration to the module globals."""
    global _relevant_domains, _blocked_domains, _priority_tiers
    _relevant_domains = [d.lower() for d in domain_config.get("relevant_domains", [])]
    _blocked_domains  = [d.lower() for d in domain_config.get("blocked_domains", [])]
    raw_tiers         = domain_config.get("priority_tiers", [])
    _priority_tiers   = [
        [d.lower() for d in tier]
        for tier in raw_tiers
        if isinstance(tier, list)
    ]


def _is_blocked(url: str) -> bool:
    u = url.lower()
    if any(d in u for d in _blocked_domains):
        return True
    return any(p in u for p in _JUNK_URL_PATTERNS)


def _priority(url: str) -> int:
    """Lower = higher priority. LLM tier 0 = best."""
    u = url.lower()
    for tier_idx, tier_domains in enumerate(_priority_tiers):
        if any(d in u for d in tier_domains):
            return tier_idx
    if any(d in u for d in _relevant_domains):
        return len(_priority_tiers) + 1
    return 99


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════

_QUERY_GENERATION_SYSTEM = """\
You are an expert research librarian and data analyst.

Your task is to generate highly effective web search queries that will find
the best sources for structured data extraction on the given topic.

Rules:
- Generate exactly 5 queries targeting DIFFERENT angles of the topic:
  * Query 1: Broad overview / list-type sources (e.g., "top X", "list of Y")
  * Query 2: Structured data / dataset / database angle
  * Query 3: Academic / research study angle (if applicable)
  * Query 4: Specific sub-aspect the user mentioned or implied
  * Query 5: Comparison / review / meta-analysis angle
- Do NOT use site: operators — keep queries generic.
- Do NOT add quotation marks around phrases unless essential.
- Each query should be 3-8 words, specific, and directly actionable.
- Prioritise queries that would lead to pages with TABLES or LISTS of data,
  not just narrative articles.

Return ONLY a JSON array of 5 query strings. No explanation."""


_DOMAIN_CONFIG_SYSTEM = """\
You are a web research expert specialising in domain quality assessment.

Given a research topic, generate a domain configuration for a web scraper.
Think carefully about which websites typically host the best quality, freely
accessible, structured data for this specific topic.

Return a JSON object with exactly these keys:
  "relevant_domains": list of 15-25 domain strings (no https://) that are
    likely to have high-quality, detailed content for this topic.
  "blocked_domains": list of 10-20 domain strings to always skip:
    social media, generic wikis, forums, ad-heavy sites, known paywalls.
  "priority_tiers": list of 3 lists, each containing domains, ordered:
    Tier 0 = best free full-text sources for this topic
    Tier 1 = good secondary sources
    Tier 2 = marginal / acceptable sources
  "paywall_domains": list of 5-15 URL fragment patterns for paywalled sites.
  "paywall_signals": list of 8-12 lowercase phrases that appear in paywalled
    HTML (e.g., "subscribe to read", "purchase access").

Be SPECIFIC to the topic — a biomedical topic should list PubMed, bioRxiv,
Nature Open Access, etc. A finance topic should list SEC, Bloomberg, Yahoo
Finance, etc. Generic lists are useless.

Return ONLY the JSON object. No explanation."""


# ═══════════════════════════════════════════════════════════════════════════════
# Search agent
# ═══════════════════════════════════════════════════════════════════════════════

class SearchAgent:
    """
    Generates search queries and populates state.pending_sources.

    Call run() once at the start of a pipeline, and again whenever the user
    asks for more sources ("find 5 more papers on this").
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self, state: SessionState) -> None:
        """
        Main entry point.

        1. Generate topic-specific domain config (once per session)
        2. Generate search queries
        3. Execute DuckDuckGo searches
        4. Filter, rank, and add URLs to state.pending_sources
        """
        console.print(f"\n[bold cyan]🔍 Search Agent[/bold cyan] — topic: '{state.topic[:60]}'")

        # Step 1: Generate domain config (skip if already done this session)
        if not _relevant_domains:
            self._setup_domain_config(state.topic)

        # Step 2: Generate search queries
        console.print("[dim]Generating search queries…[/dim]")
        queries = self._generate_queries(state)
        console.print(f"[dim]Generated {len(queries)} queries[/dim]")

        # Step 3: Execute searches
        # Fetch 3× target so we have buffer for paywalled/empty sources
        target = state.source_limit
        buffer = target * 3

        results = self._search_all(queries, limit=buffer)

        # Step 4: Filter already-processed sources
        new_results = [
            r for r in results
            if r["url"] not in state.processed_sources
            and r["url"] not in state.dead_sources
        ]

        # Add to pending (avoid duplicates)
        existing = set(state.pending_sources)
        added = 0
        for r in new_results:
            if r["url"] not in existing:
                state.pending_sources.append(r["url"])
                existing.add(r["url"])
                added += 1

        console.print(
            f"[green]✓ Search complete — {added} new URLs queued "
            f"({len(state.pending_sources)} total pending)[/green]"
        )

    # ── Domain configuration ────────────────────────────────────────────────

    def _setup_domain_config(self, topic: str) -> dict:
        """
        Ask the LLM for topic-specific domain config and apply it.
        Also configures paywall detection in the browser tool.
        """
        console.print("[dim]Generating domain & paywall config…[/dim]")
        try:
            config = self.llm.complete_json(
                system=_DOMAIN_CONFIG_SYSTEM,
                user=f'Topic: "{topic}"',
                model=cfg.FAST_MODEL,
                max_tokens=1500,
            )
            if isinstance(config, dict):
                _configure_domains(config)

                # Configure paywall detection in the browser pool
                from tools.browser import configure_paywall_detection
                configure_paywall_detection(
                    domain_patterns=config.get("paywall_domains", []),
                    html_signals=config.get("paywall_signals", []),
                )
                console.print(
                    f"  [dim]Domain config: {len(_relevant_domains)} relevant, "
                    f"{len(_blocked_domains)} blocked, {len(_priority_tiers)} tiers[/dim]"
                )
                return config
        except Exception as exc:
            console.print(f"  [yellow]⚠ Domain config failed ({exc}) — using defaults[/yellow]")

        return {}

    # ── Query generation ────────────────────────────────────────────────────

    def _generate_queries(self, state: SessionState) -> list[str]:
        """
        Ask the LLM to generate 5 diverse search queries for the topic.
        Falls back to simple variants if the LLM call fails.
        """
        # Include column names in the prompt so the LLM can tailor queries
        # to find pages that likely have those specific fields
        col_context = ""
        if state.columns:
            col_context = (
                f"\nThe user wants to extract these fields: "
                + ", ".join(f'"{c.name}"' for c in state.columns[:8])
                + "\nTailor queries to find sources containing these specific data points."
            )

        try:
            raw = self.llm.complete_json(
                system=_QUERY_GENERATION_SYSTEM,
                user=f'Topic: "{state.topic}"{col_context}\n\nGenerate 5 search queries:',
                model=cfg.FAST_MODEL,
                max_tokens=512,
            )
            if isinstance(raw, list) and raw:
                queries = [str(q) for q in raw if isinstance(q, str)]
                if queries:
                    for i, q in enumerate(queries):
                        console.print(f"  [dim]{i+1}. {q}[/dim]")
                    return queries
        except Exception as exc:
            console.print(f"  [yellow]⚠ Query generation failed ({exc}) — using fallback[/yellow]")

        # Fallback queries — topic only, no LLM
        return [
            state.topic,
            f"{state.topic} research study data",
            f"{state.topic} review article results",
            f"{state.topic} dataset statistics",
            f"{state.topic} list overview",
        ]

    # ── DuckDuckGo search ────────────────────────────────────────────────────

    def _search_all(self, queries: list[str], limit: int) -> list[dict]:
        """
        Execute all queries on DuckDuckGo, deduplicate, and rank results.
        """
        results:   list[dict] = []
        seen_urls: set[str]   = set()
        per_query = max(limit // len(queries) + 2, 3)
        cap       = limit * 2

        for q in queries:
            if len(results) >= cap:
                break

            console.print(f"  [dim]🔍 {q[:90]}[/dim]")

            # Polite delay between queries
            time.sleep(random.uniform(
                cfg.REQUEST_DELAY[0], cfg.REQUEST_DELAY[1]
            ))

            try:
                with DDGS() as ddgs:
                    hits = list(ddgs.text(q, max_results=per_query))

                for hit in hits:
                    url = hit.get("href", "")
                    if not url or url in seen_urls:
                        continue
                    if _is_blocked(url):
                        continue
                    seen_urls.add(url)
                    results.append({
                        "title":    hit.get("title", ""),
                        "url":      url,
                        "snippet":  hit.get("body", ""),
                        "priority": _priority(url),
                    })

            except Exception as exc:
                console.print(f"  [yellow]⚠ Query failed: {exc}[/yellow]")

        # Sort best first, cap at limit
        results.sort(key=lambda r: r["priority"])
        return results[:limit]

    # ── Additional sources from user-provided URLs ──────────────────────────

    def add_custom_urls(self, urls: list[str], state: SessionState) -> None:
        """
        Add user-specified URLs directly to pending_sources.
        Called when the user types something like "--urls https://..."
        """
        existing = set(state.pending_sources) | state.processed_sources
        added = 0
        for url in urls:
            if url not in existing:
                state.pending_sources.append(url)
                existing.add(url)
                added += 1
        console.print(f"[green]✓ Added {added} custom URL(s) to queue[/green]")

    def add_local_pdfs(self, paths: list[str], state: SessionState) -> None:
        """
        Add local PDF file paths to pending_sources.
        Called when the user provides --pdfs or --pdf-folder.
        """
        existing = set(state.pending_sources) | state.processed_sources
        added = 0
        for path in paths:
            if path not in existing:
                state.pending_sources.append(path)
                existing.add(path)
                added += 1
        console.print(f"[green]✓ Added {added} local PDF(s) to queue[/green]")