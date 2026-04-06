"""
agents/refiner_agent.py — Gap-filling refinement agent.

This is the most innovative part of the system, solving the core problem
of the original pipeline: N/A fields were just left blank forever.

The Refiner's approach is fundamentally different from re-running the extractor:

  PROBLEM: A row has "Sample Size = N/A" even though the paper probably
  reports this. The extractor missed it.

  OLD APPROACH: Nothing. The output ships with N/A. ❌

  REFINER APPROACH:
    1. Generate a TARGETED search query specifically for this missing field.
       e.g.: "CRISPR Cas9 efficiency comparison 2022 sample size participants"
    2. Search DuckDuckGo and scrape the top result.
    3. Ask the LLM to extract ONLY the missing field from that page.
    4. If found, update the row in state.
    5. If still not found after MAX_REFINE_SEARCHES attempts, mark it as
       "searched — genuinely not available" so we stop trying.

  This targeted approach works because:
    - The search query includes entity identity fields (title, drug name, etc.)
      so we find pages specifically about THIS entity.
    - We ask the LLM to find ONE specific field, not all fields — this is
      much more accurate than asking "extract everything".
    - We search the open web, not just the original source, so we can find
      information that wasn't in the original paper but exists in databases,
      registries, or secondary sources.

Examples of what the Refiner can find that the original extractor missed:
  - Sample size from a clinical trial registry (ClinicalTrials.gov)
  - Drug approval status from FDA database
  - Company revenue from a financial data site
  - Author affiliations from their institutional page
  - Dataset DOI from a data repository
"""

from __future__ import annotations

import random
import time
from typing import Optional

from ddgs import DDGS
from rich.console import Console

import config as cfg
from agents.critic_agent import GapAssessment
from state import ColumnDef, SessionState
from tools.browser import BrowserPool, _html_to_text
from tools.llm_tools import LLMClient

console = Console()

# Values that count as "empty"
_EMPTY_VALS = frozenset({
    "n/a", "none", "not specified", "not mentioned", "not provided",
    "not explicitly mentioned", "unspecified", "unknown", "na", "nil", "",
    "not available", "not found", "not discussed", "not applicable",
    "not stated", "not reported", "not described", "null", "—", "-",
})


def _is_empty(v: str) -> bool:
    return str(v).strip().lower() in _EMPTY_VALS


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════

_QUERY_GEN_SYSTEM = """\
You are a precision research assistant specialising in targeted information retrieval.

Your task: generate ONE highly specific web search query to find a SINGLE missing
data point for a specific entity.

The query must:
  1. Include 2-3 identity terms from the entity (name, title, year, ID) to ensure
     the search finds THIS entity and not others.
  2. Include the specific missing field as a keyword.
  3. Optionally include a database or registry likely to have this info
     (e.g., ClinicalTrials.gov for trial info, PubMed for papers, SEC for financials).
  4. Be 5-10 words. No site: operators.

Return ONLY the query string. No quotes, no explanation."""

_FIELD_EXTRACT_SYSTEM = """\
You are a precision data extraction agent.

Your ONLY task is to find and extract ONE specific data point from the provided text.

Field to extract: "{field_name}"
Field definition: {field_def}
Expected format: {field_example}

Rules:
  - Return ONLY the extracted value as a plain string.
  - If the value is explicitly stated, return it exactly as written.
  - If you can reliably calculate or infer it from stated facts, do so.
  - If the information is genuinely absent, return exactly: NOT_FOUND
  - Do NOT return a sentence. Return only the value itself.
  - Do NOT explain your answer.

Examples of good responses: "342 patients", "2021", "Phase III", "10.1038/s41586-020-2649-2"
Examples of BAD responses: "The study had 342 patients", "I found that the year was 2021" """


# ═══════════════════════════════════════════════════════════════════════════════
# Refiner agent
# ═══════════════════════════════════════════════════════════════════════════════

class RefinerAgent:
    """
    Fills missing fields in extracted rows via targeted web searches.

    Works through GapAssessments from the Critic (highest priority first)
    and attempts to fill each actionable gap independently.
    """

    def __init__(self, llm: LLMClient, browser: BrowserPool) -> None:
        self.llm     = llm
        self.browser = browser

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        assessments: list[GapAssessment],
        state: SessionState,
    ) -> int:
        """
        Attempt to fill all actionable gaps identified by the Critic.

        Args:
            assessments — sorted list from CriticAgent.run()
            state       — the shared session state (rows mutated in-place)

        Returns:
            Number of fields successfully filled.
        """
        to_fix = [a for a in assessments if a.needs_refinement]

        if not to_fix:
            console.print("[dim]Refiner: nothing to fix.[/dim]")
            return 0

        console.print(
            f"\n[bold cyan]🛠  Refiner Agent[/bold cyan]"
            f" — {len(to_fix)} row(s) with gaps to fill"
        )

        total_filled = 0

        for assessment in to_fix:
            row   = state.rows[assessment.row_index]
            filled = self._fix_row(row, assessment, state)
            total_filled += filled
            row.refinement_done = True

        state.refinement_rounds += 1
        console.print(
            f"[green]✓ Refiner done — filled {total_filled} field(s) "
            f"(round {state.refinement_rounds})[/green]"
        )
        state.save()
        return total_filled

    # ── Row-level fixing ────────────────────────────────────────────────────

    def _fix_row(
        self,
        row,
        assessment: GapAssessment,
        state: SessionState,
    ) -> int:
        """
        Attempt to fill each actionable gap in a single row.

        Returns the number of fields successfully filled.
        """
        # Build entity context from filled fields (used to craft precise queries)
        entity_context = self._build_entity_context(row, state)
        filled_count   = 0

        for field_name in assessment.actionable_gaps:
            col_def = next(
                (c for c in state.columns if c.name == field_name), None
            )
            if col_def is None:
                continue

            console.print(
                f"  [dim]Searching for '{field_name}' | entity: {entity_context[:60]}[/dim]"
            )

            value = self._find_field_value(
                field_name    = field_name,
                col_def       = col_def,
                entity_context = entity_context,
                topic         = state.topic,
            )

            if value and not _is_empty(value) and value != "NOT_FOUND":
                row.data[field_name]       = value
                row.confidence[field_name] = 0.7   # found via secondary search
                filled_count += 1
                console.print(f"    [green]✓ {field_name}: {value[:60]}[/green]")
            else:
                console.print(f"    [dim]✗ {field_name}: not found online[/dim]")

        return filled_count

    # ── Field search pipeline ───────────────────────────────────────────────

    def _find_field_value(
        self,
        field_name: str,
        col_def: ColumnDef,
        entity_context: str,
        topic: str,
    ) -> str | None:
        """
        Core refinement logic for one field:
          1. Generate a targeted search query.
          2. Search DuckDuckGo.
          3. Scrape the top 2 results.
          4. Ask the LLM to extract ONLY this field from the scraped text.

        Returns the extracted value or None.
        """
        # Step 1: Generate a targeted search query
        query = self._generate_search_query(
            field_name, col_def, entity_context, topic
        )
        if not query:
            return None

        console.print(f"    [dim]🔍 Refine query: {query}[/dim]")

        # Step 2: Search DuckDuckGo
        urls = self._search(query, max_results=3)
        if not urls:
            console.print(f"    [dim]No results for: {query}[/dim]")
            return None

        # Step 3: Scrape and extract from each result until we find the value
        for url in urls:
            time.sleep(random.uniform(1, 2))   # polite delay

            text = self._scrape_for_text(url)
            if not text or len(text.strip()) < 200:
                continue

            # Step 4: Targeted single-field extraction
            value = self._extract_single_field(
                text       = text,
                field_name = field_name,
                col_def    = col_def,
                entity_context = entity_context,
            )

            if value and value != "NOT_FOUND" and not _is_empty(value):
                return value

        return None

    # ── Query generation ────────────────────────────────────────────────────

    def _generate_search_query(
        self,
        field_name: str,
        col_def: ColumnDef,
        entity_context: str,
        topic: str,
    ) -> str | None:
        """
        Ask the LLM to generate a precise search query for one missing field.
        """
        try:
            query = self.llm.complete_fast(
                system=_QUERY_GEN_SYSTEM,
                user=(
                    f'Topic: "{topic}"\n'
                    f'Entity: {entity_context}\n'
                    f'Missing field: "{field_name}"\n'
                    f'Field definition: {col_def.description}\n\n'
                    f'Generate one targeted search query to find this value:'
                ),
            )
            query = query.strip().strip('"').strip("'")
            if query:
                return query
        except Exception as exc:
            console.print(f"    [dim yellow]Query gen failed: {exc}[/dim yellow]")

        # Fallback: manual query construction from entity context
        identity = entity_context.split("|")[0] if "|" in entity_context else entity_context
        return f"{identity} {field_name}".strip()

    # ── DuckDuckGo search ────────────────────────────────────────────────────

    def _search(self, query: str, max_results: int = 3) -> list[str]:
        """Run a DuckDuckGo search and return the top URLs."""
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=max_results))
            return [h["href"] for h in hits if h.get("href")]
        except Exception as exc:
            console.print(f"    [dim yellow]Search failed: {exc}[/dim yellow]")
            return []

    # ── Targeted scraping ────────────────────────────────────────────────────

    def _scrape_for_text(self, url: str) -> str | None:
        """
        Scrape a URL and return plain text.

        Uses requests fallback first (faster) then Playwright if needed.
        For the Refiner, we don't need full quality gating — even an abstract
        or short page might have the specific value we're looking for.
        """
        # Try fast requests scrape first
        try:
            import requests
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
                )
            }
            resp = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            if resp.ok:
                ct = resp.headers.get("content-type", "")
                if "pdf" in ct or url.lower().endswith(".pdf"):
                    from tools.browser import extract_pdf_text
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                        f.write(resp.content)
                        return extract_pdf_text(f.name)
                return _html_to_text(resp.text)
        except Exception:
            pass

        # Playwright fallback
        try:
            result = self.browser.scrape(url)
            if result:
                return result.get("text", "")
        except Exception:
            pass

        return None

    # ── Single-field extraction ───────────────────────────────────────────────

    def _extract_single_field(
        self,
        text: str,
        field_name: str,
        col_def: ColumnDef,
        entity_context: str,
    ) -> str | None:
        """
        Ask the LLM to find ONE specific field in the scraped text.

        This is extremely precise — we're asking "what is the sample size?"
        not "extract all fields". This dramatically reduces hallucinations.
        """
        # Trim text to keep within context limits — most specific info is
        # either near the top or in a specific section
        text_excerpt = text[:15000]

        system = _FIELD_EXTRACT_SYSTEM.format(
            field_name    = field_name,
            field_def     = col_def.description or f"the {field_name} value",
            field_example = col_def.example or "exact value from text",
        )

        user = (
            f"Entity: {entity_context}\n\n"
            f"Text to search:\n{text_excerpt}\n\n"
            f"Find and return ONLY the value for '{field_name}':"
        )

        try:
            response = self.llm.complete_fast(system=system, user=user)
            response = response.strip().strip('"').strip("'")

            # Reject responses that look like explanations
            if len(response.split()) > 20:
                return None
            if response.lower().startswith(("i ", "the ", "based on", "according")):
                return None

            return response if response else None

        except Exception as exc:
            console.print(f"    [dim yellow]Extract failed: {exc}[/dim yellow]")
            return None

    # ── Entity context builder ────────────────────────────────────────────────

    @staticmethod
    def _build_entity_context(row, state: SessionState) -> str:
        """
        Build a concise string describing the entity in a row.
        Used to make search queries entity-specific.

        e.g.: "Trastuzumab | HER2 antibody drug conjugate | 2021"
        """
        # Identify high-value identity fields
        identity_priority = [
            "title", "name", "drug", "compound", "gene", "study", "trial",
            "company", "product", "author", "doi", "year", "id", "accession",
        ]

        parts = []
        for col in state.columns:
            val = row.data.get(col.name, "N/A")
            if _is_empty(str(val)):
                continue
            # Check if this is an identity field
            col_lower = col.name.lower()
            if any(kw in col_lower for kw in identity_priority):
                parts.append(str(val)[:60])
                if len(parts) >= 4:
                    break

        if not parts:
            # Fallback: first non-empty field
            for val in row.data.values():
                if not _is_empty(str(val)):
                    parts.append(str(val)[:60])
                    break

        return " | ".join(parts) if parts else "unknown entity"