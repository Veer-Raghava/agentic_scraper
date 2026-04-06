"""
agents/orchestrator.py — The orchestrating brain of the entire system.

The Orchestrator does three things:

  1. CONVERSATION: Maintains the chat loop with the user. It greets them,
     asks clarifying questions, and understands what they want — including
     refinement requests after the first pipeline run.

  2. COLUMN NEGOTIATION: This is a critical UX feature.
     - If the user provides explicit column names in their first message →
       use them exactly, no questions asked.
     - If the user doesn't mention columns → suggest 8-12 LLM-generated
       columns tailored to their topic and ASK the user to confirm/modify.
     This ensures the user always knows what data they're getting.

  3. PIPELINE COORDINATION: Once the session is set up, the Orchestrator
     calls each agent in the correct order:
       Search → Scrape → Extract → Critic → Refine → Export
     
     After the initial run, it handles follow-up commands:
       "add column X"          → ExtractorAgent on existing rows
       "find more sources"     → SearchAgent + ScraperAgent + Extractor
       "fix row 3"             → RefinerAgent on specific row
       "export to xlsx"        → ExportAgent
       "show me the results"   → print_preview()

Intent parsing:
  A lightweight LLM call interprets every user message into a structured
  intent dict. This avoids fragile regex parsing and handles natural language
  like "actually, can you also grab the publication year?" correctly.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from typing import Any

from ddgs import DDGS
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config as cfg
from agents.critic_agent import CriticAgent
from agents.extractor_agent import ExtractorAgent
from agents.refiner_agent import RefinerAgent
from agents.scraper_agent import ScraperAgent
from agents.search_agent import SearchAgent
from state import ColumnDef, SessionState
from tools.browser import BrowserPool
from tools.export import print_fill_report, print_preview, save_csv, save_xlsx
from tools.llm_tools import LLMClient

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════

_ORCHESTRATOR_SYSTEM = """\
You are ARIA (Agentic Research & Intelligence Assistant), an expert data
extraction assistant. You help users collect structured data from the web
on any topic they choose.

Your personality:
  - Direct and efficient — don't pad responses with filler.
  - Enthusiastic about data quality — you genuinely care about filling N/A fields.
  - Honest about limitations — if something can't be found, say so clearly.
  - Proactive — suggest improvements the user hasn't thought of.

Current session state:
  Topic: {topic}
  Columns: {columns}
  Rows collected: {row_count}
  Sources processed: {source_count}
  Refinement rounds: {refinement_rounds}

Respond conversationally to the user. Keep responses under 150 words unless
the user asks for a detailed explanation."""

_INTENT_SYSTEM = """\
You are an intent classifier for a data extraction assistant.

Parse the user's message and return a JSON object with these keys:
  "intent": one of:
    "start_pipeline"       — user is giving a new topic to research
    "confirm_columns"      — user is confirming/modifying suggested columns
    "add_column"           — user wants to add a new extraction column
    "more_sources"         — user wants to find and process more sources
    "fix_row"              — user wants to fix a specific row
    "export"               — user wants to export data (csv/xlsx)
    "show_results"         — user wants to see current results / fill report
    "change_topic"         — user wants to start a completely new topic
    "custom_urls"          — user is providing specific URLs to scrape
    "local_pdfs"           — user is providing local PDF file paths
    "stop"                 — user wants to stop / exit
    "question"             — user is asking a question (no pipeline action needed)
    "chat"                 — general conversation

  "topic": extracted topic string (if intent is start_pipeline/change_topic, else null)
  "columns": list of column name strings (if user specified them, else null)
  "source_limit": integer (if user mentioned a number of sources, else null)
  "new_columns": list of new column names to add (if intent is add_column)
  "row_index": integer row number (1-indexed) if intent is fix_row, else null
  "urls": list of URL strings (if intent is custom_urls, else null)
  "export_format": "csv" or "xlsx" (if intent is export, else null)

Return ONLY the JSON object. Be generous in parsing — if the user says
"look at 5 papers" interpret source_limit as 5."""

_COLUMN_SUGGESTION_SYSTEM = """\
You are a data analyst and domain expert.

Given a research topic, suggest 8-12 specific, valuable columns to extract
from web sources about that topic.

Requirements:
  - Columns must be directly relevant to the topic, not generic.
  - Never change the domain (e.g. do not switch a biotech topic to gaming/startups).
  - Include a mix of: identity fields (name, ID, title), quantitative fields
    (numbers, dates, metrics), and qualitative fields (description, status, type).
  - Column names should be concise (2-4 words), title-cased.
  - Think about what a researcher or analyst would actually WANT in a spreadsheet.
  - Always include at least one identity column (something that uniquely identifies
    each entry) and at least one date/year column.
  - If topic is about scientific/medical papers or molecules, prefer technical
    columns over business/consumer columns.
  - Put "Source URL" as the final column.

Return ONLY a JSON array of column name strings.

Examples for different topics:
  "FDA approved cancer drugs":
    ["Drug Name", "Target", "Indication", "Approval Year", "Mechanism of Action",
     "Clinical Trial Phase", "Efficacy Rate", "Developer", "Administration Route", "DOI"]

  "Tech startup funding rounds 2024":
    ["Company Name", "Funding Amount", "Round Type", "Lead Investor", "Date",
     "Valuation", "Sector", "HQ Location", "Total Raised", "Source URL"]"""


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class Orchestrator:
    """
    Central coordinator for the interactive session.

    One instance lives for the entire session. It owns references to all agents
    and coordinates their execution in response to user messages.
    """

    def __init__(
        self,
        llm:     LLMClient,
        browser: BrowserPool,
        state:   SessionState,
    ) -> None:
        self.llm     = llm
        self.browser = browser
        self.state   = state

        # Instantiate all agents once
        self.search_agent    = SearchAgent(llm)
        self.scraper_agent   = ScraperAgent(browser)
        self.extractor_agent = ExtractorAgent(llm)
        self.critic_agent    = CriticAgent(llm)
        self.refiner_agent   = RefinerAgent(llm, browser)

        # Track whether we've asked the user to confirm columns
        self._awaiting_column_confirmation = False
        self._pending_topic = ""
        self._stage_order = ["Search", "Scrape", "Extract", "Critic", "Refine", "Export"]
        self._stage_state = {s: ("pending", "") for s in self._stage_order}

    def _live_status(self, message: str) -> None:
        """Print a timestamped live status line in the CLI."""
        console.print(f"[bold cyan]⏱[/bold cyan] [dim]{message}[/dim]")

    def _render_pipeline_status(self) -> None:
        """Render a compact status table for current pipeline state."""
        icon = {
            "pending": "○",
            "running": "▶",
            "done": "✓",
            "failed": "✗",
        }
        table = Table(title="Pipeline Status", show_header=True, header_style="bold cyan")
        table.add_column("Stage", style="bold")
        table.add_column("State")
        table.add_column("Details", overflow="fold")
        for stage in self._stage_order:
            state, details = self._stage_state.get(stage, ("pending", ""))
            table.add_row(stage, f"{icon.get(state, '○')} {state}", details)
        console.print(table)

    def _set_stage(self, stage: str, state: str, details: str = "") -> None:
        self._stage_state[stage] = (state, details)
        self._render_pipeline_status()

    def _run_step_with_timeout(self, fn, timeout_s: int, step_name: str):
        """
        Run a blocking step with a timeout to avoid long silent hangs.
        Returns the function result or None if timed out/failed.
        """
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(fn)
                return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            self._live_status(f"{step_name} timed out after {timeout_s}s")
            return None
        except Exception as exc:
            self._live_status(f"{step_name} failed: {exc}")
            return None

    @staticmethod
    def _topic_is_pdf_heavy(topic: str) -> bool:
        """Best-effort heuristic to detect PDF/paper-heavy requests."""
        t = (topic or "").lower()
        signals = (
            "pdf", "paper", "papers", "study", "studies", "journal",
            "publication", "publications", "doi", "clinical trial", "review",
        )
        return any(s in t for s in signals)

    def _fallback_pdf_search(self, topic: str, limit: int = 12) -> list[str]:
        """
        Fallback web search that prefers direct PDF links when primary search
        returns nothing useful.
        """
        queries = [
            f"{topic} filetype:pdf",
            f"{topic} pdf",
            f"{topic} research paper pdf",
        ]
        found: list[str] = []
        seen: set[str] = set()
        try:
            with DDGS() as ddgs:
                for q in queries:
                    for r in ddgs.text(q, max_results=limit):
                        url = str(r.get("href") or r.get("url") or "").strip()
                        if not url or url in seen:
                            continue
                        is_pdf = (
                            ".pdf" in url.lower()
                            or "/pdf/" in url.lower()
                            or "arxiv.org" in url.lower()
                        )
                        if is_pdf:
                            seen.add(url)
                            found.append(url)
                        if len(found) >= limit:
                            return found
        except Exception as exc:
            self._live_status(f"PDF fallback search failed: {exc}")
        return found

    # ── Main entry point ────────────────────────────────────────────────────

    def greet(self) -> str:
        """Return the greeting message shown on startup."""
        return (
            "Hi! I'm ARIA — your Agentic Research & Intelligence Assistant.\n\n"
            "Tell me what you want to research. For example:\n"
            "  • 'Collect data on FDA-approved cancer drugs'\n"
            "  • 'Scrape info on EV companies: Name, Revenue, Country, Founded'\n"
            "  • 'Analyze 10 papers on CRISPR gene editing efficiency'\n\n"
            "If you specify column names in your message, I'll use them directly.\n"
            "Otherwise, I'll suggest columns and ask you to confirm.\n\n"
            "Type 'help' at any time for commands."
        )

    def handle(self, user_message: str) -> str:
        """
        Process one user message and return the assistant's response.

        This is the main dispatch method called by the chat loop in main.py.
        """
        self.state.add_message("user", user_message)

        # Help shortcut
        if user_message.strip().lower() in ("help", "/help"):
            return self._help_text()

        # Parse intent
        intent = self._parse_intent(user_message)

        # If we're waiting for column confirmation, prioritize that flow.
        # This avoids LLM intent misclassification loops (e.g. "go" treated as
        # a brand new topic instead of confirming suggested columns).
        if self._awaiting_column_confirmation and self._should_force_column_confirm(user_message, intent):
            intent = {"intent": "confirm_columns"}

        if cfg.DEBUG:
            console.print(f"[dim]Intent: {json.dumps(intent, indent=2)}[/dim]")

        # ── Route to handler ─────────────────────────────────────────────────
        handler_map = {
            "start_pipeline":    self._handle_start,
            "confirm_columns":   self._handle_column_confirm,
            "add_column":        self._handle_add_column,
            "more_sources":      self._handle_more_sources,
            "fix_row":           self._handle_fix_row,
            "export":            self._handle_export,
            "show_results":      self._handle_show_results,
            "change_topic":      self._handle_change_topic,
            "custom_urls":       self._handle_custom_urls,
            "local_pdfs":        self._handle_local_pdfs,
            "stop":              self._handle_stop,
        }

        handler = handler_map.get(intent.get("intent", "chat"))

        if handler:
            response = handler(intent, user_message)
        else:
            response = self._handle_chat(user_message)

        self.state.add_message("assistant", response)
        return response

    # ── Intent parsing ──────────────────────────────────────────────────────

    def _should_force_column_confirm(self, message: str, intent: dict) -> bool:
        """Return True when input should be treated as column confirmation."""
        text = (message or "").strip().lower()
        parsed_intent = (intent or {}).get("intent", "")

        # Allow explicit control intents to pass through.
        if parsed_intent in {"stop", "change_topic"}:
            return False

        # Strong confirmation signals.
        if text in {"go", "yes", "ok", "all", "use all", "y"}:
            return True
        if re.match(r"^[\d,\s]+$", text):
            return True
        if text.startswith("use:") or text.startswith("use "):
            return True
        if text.startswith("add "):
            return True

        # In pending-confirmation mode, default to confirmation unless user is
        # clearly asking for a new topic.
        if any(phrase in text for phrase in ("new topic", "change topic", "start over", "different topic")):
            return False
        return True

    def _parse_intent(self, message: str) -> dict:
        """
        Use a fast LLM call to parse the user's message into a structured intent.
        Falls back to "chat" if parsing fails.
        """
        try:
            raw = self.llm.complete_json(
                system=_INTENT_SYSTEM,
                user=f"User message: {message}",
                model=cfg.FAST_MODEL,
                max_tokens=400,
            )
            if isinstance(raw, dict):
                return raw
        except Exception as exc:
            if cfg.DEBUG:
                console.print(f"[dim yellow]Intent parse failed: {exc}[/dim yellow]")
        return {"intent": "chat"}

    # ── Pipeline start ──────────────────────────────────────────────────────

    def _handle_start(self, intent: dict, raw_message: str) -> str:
        """
        Handle a new pipeline start.

        If the user specified columns → use them.
        If not → suggest columns and ask for confirmation.
        """
        topic = intent.get("topic") or raw_message.strip()
        if not topic:
            return "I didn't catch the topic. What do you want to research?"

        self.state.topic        = topic
        self.state.source_limit = intent.get("source_limit") or cfg.DEFAULT_SOURCE_LIMIT

        user_columns = intent.get("columns")

        # ── Case 1: User explicitly provided column names ────────────────────
        if user_columns and isinstance(user_columns, list) and len(user_columns) >= 2:
            cols = self._finalise_columns(user_columns)
            self.state.columns = cols
            console.print(f"[dim]Using user-specified columns: {[c.name for c in cols]}[/dim]")
            # Kick off the pipeline immediately
            return self._run_pipeline()

        # ── Case 2: Suggest columns and wait for confirmation ────────────────
        console.print("[dim]Generating column suggestions…[/dim]")
        suggested = self._suggest_columns(topic)

        if not suggested:
            return (
                "I'm having trouble suggesting columns right now. "
                "Please tell me what fields you want to extract, "
                "e.g. 'Name, Year, Method, Result, DOI'"
            )

        self._pending_topic = topic
        self._awaiting_column_confirmation = True

        # Store suggestions in state for reference
        self.state.pending_column_defs = [
            ColumnDef(name=s) for s in suggested
        ]

        col_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(suggested))
        return (
            f"Great topic! Here are my suggested columns for '{topic}':\n\n"
            f"{col_list}\n\n"
            f"Reply with:\n"
            f"  • **'go'** or **'yes'** to use all of these\n"
            f"  • **'1,3,5'** to select specific columns by number\n"
            f"  • **'add X, Y'** to add extra columns\n"
            f"  • **'use: Column A, Column B, Column C'** to specify your own\n"
            f"\nI'll process up to {self.state.source_limit} sources. "
            f"Type 'limit 20' to change."
        )

    def _handle_column_confirm(self, intent: dict, raw_message: str) -> str:
        """
        Handle the user's response to a column suggestion.
        Parses 'go', '1,3,5', 'use: X, Y, Z', etc.
        """
        if not self._awaiting_column_confirmation:
            return self._handle_chat(raw_message)

        pending = self.state.pending_column_defs
        msg     = raw_message.strip().lower()

        # "go" / "yes" / "ok" → use all suggestions
        if msg in ("go", "yes", "ok", "all", "use all", "y"):
            selected = pending

        # Numeric selection: "1,3,5"
        elif re.match(r'^[\d,\s]+$', msg):
            indices = [int(x.strip()) - 1 for x in msg.split(",")
                       if x.strip().isdigit()]
            selected = [pending[i] for i in indices if 0 <= i < len(pending)]
            if not selected:
                return "I couldn't parse those numbers. Try 'go' for all, or '1,3,5'."

        # "use: X, Y, Z" → custom columns
        elif "use:" in msg or "use " in msg:
            custom_part = re.sub(r'^use:?\s*', '', msg, flags=re.I).strip()
            names = [n.strip().title() for n in custom_part.split(",") if n.strip()]
            selected = [ColumnDef(name=n) for n in names]

        # Unrecognised — treat as chat
        else:
            return self._handle_chat(raw_message)

        if not selected:
            return "Please select at least one column."

        self.state.columns = self._finalise_columns([c.name for c in selected])
        self._awaiting_column_confirmation = False
        self.state.pending_column_defs     = []

        col_names = ", ".join(c.name for c in self.state.columns)
        console.print(f"[dim]Confirmed columns: {col_names}[/dim]")

        # Kick off the pipeline
        return self._run_pipeline()

    # ── Pipeline execution ──────────────────────────────────────────────────

    def _run_pipeline(self) -> str:
        """
        Execute the full pipeline: Search → Scrape → Extract → Critic → Refine.
        Returns a summary message for the chat.
        """
        self.state.pipeline_running = True
        self._live_status("Pipeline started")
        self._stage_state = {s: ("pending", "") for s in self._stage_order}
        console.print(Panel.fit(
            f"[bold]Pipeline starting[/bold]\n"
            f"Topic: {self.state.topic}\n"
            f"Columns: {', '.join(c.name for c in self.state.columns[:6])}"
            + (" …" if len(self.state.columns) > 6 else ""),
            border_style="cyan"
        ))

        # 1. Search
        self._live_status(f"Searching sources for topic: {self.state.topic}")
        self._set_stage("Search", "running", f"Topic: {self.state.topic[:60]}")
        search_result = self._run_step_with_timeout(
            lambda: self.search_agent.run(self.state),
            timeout_s=120,
            step_name="Source search",
        )
        if search_result is None and not self.state.pending_sources:
            self.state.pipeline_running = False
            self._set_stage("Search", "failed", "Timed out or failed")
            return (
                "Search step timed out or failed. "
                "Try 'limit 5', provide direct URLs, or retry."
            )
        self._live_status(f"Search finished. Pending sources: {len(self.state.pending_sources)}")
        self._set_stage("Search", "done", f"Pending sources: {len(self.state.pending_sources)}")

        # Fallback for PDF-heavy topics when no usable source was found.
        if not self.state.pending_sources and self._topic_is_pdf_heavy(self.state.topic):
            self._live_status("No sources yet. Running PDF-priority fallback search")
            pdf_urls = self._fallback_pdf_search(self.state.topic, limit=max(8, self.state.source_limit))
            for url in pdf_urls:
                if url not in self.state.processed_sources and url not in self.state.dead_sources:
                    self.state.pending_sources.append(url)
            self._live_status(f"PDF fallback added {len(self.state.pending_sources)} source(s)")

        if not self.state.pending_sources:
            self.state.pipeline_running = False
            self._live_status("No sources found. Pipeline stopped")
            self._set_stage("Search", "failed", "No sources found")
            return (
                "I couldn't find any sources for this topic. "
                "Try rephrasing the topic or providing specific URLs."
            )

        # 2. Scrape
        self._live_status(f"Scraping up to {self.state.source_limit} source(s)")
        self._set_stage("Scrape", "running", f"Limit: {self.state.source_limit}")
        docs = self._run_step_with_timeout(
            lambda: self.scraper_agent.run(self.state, max_docs=self.state.source_limit),
            timeout_s=240,
            step_name="Scraping",
        ) or []
        self._live_status(f"Scrape finished. Readable documents: {len(docs)}")
        self._set_stage("Scrape", "done", f"Readable docs: {len(docs)}")
        if not docs:
            self.state.pipeline_running = False
            self._live_status("No readable documents found. Pipeline stopped")
            self._set_stage("Scrape", "failed", "No readable documents")
            return (
                "I found URLs but couldn't extract readable content from them. "
                "This often happens with heavily paywalled topics. "
                "Try providing specific URLs with --urls, or local PDFs."
            )

        # 3. Extract
        self._live_status(f"Extracting rows from {len(docs)} document(s)")
        self._set_stage("Extract", "running", f"Documents: {len(docs)}")
        self.extractor_agent.run(docs, self.state)
        self._live_status(f"Extraction finished. Rows collected: {len(self.state.rows)}")
        self._set_stage("Extract", "done", f"Rows: {len(self.state.rows)}")

        # 4. Critic + Refine (if we have rows and haven't exceeded limits)
        if self.state.rows and self.state.refinement_rounds < cfg.MAX_REFINEMENT_ROUNDS:
            self._live_status("Running critic pass")
            self._set_stage("Critic", "running", f"Rows: {len(self.state.rows)}")
            assessments = self.critic_agent.run(self.state)
            if any(a.needs_refinement for a in assessments):
                self._live_status("Refining missing/low-confidence fields")
                self._set_stage("Critic", "done", "Needs refinement")
                self._set_stage("Refine", "running", "Filling gaps")
                self.refiner_agent.run(assessments, self.state)
                self._live_status("Refinement finished")
                self._set_stage("Refine", "done", "Gap fill completed")
            else:
                self._live_status("Critic finished. No refinement needed")
                self._set_stage("Critic", "done", "No refinement needed")
                self._set_stage("Refine", "done", "Skipped")

        # 5. Export
        self._live_status("Saving CSV and rendering preview")
        self._set_stage("Export", "running", "Writing CSV")
        csv_path = save_csv(self.state)
        print_preview(self.state)
        self._set_stage("Export", "done", csv_path)

        self.state.pipeline_running = False
        self._live_status("Pipeline completed")

        fill_rates = [r.fill_rate(self.state.columns) for r in self.state.rows]
        avg_fill   = sum(fill_rates) / len(fill_rates) if fill_rates else 0.0

        return (
            f"✅ Done! Collected **{len(self.state.rows)} rows** from "
            f"{len(self.state.processed_sources)} sources.\n"
            f"Average field fill rate: **{avg_fill:.0%}**\n"
            f"Saved to: `{csv_path}`\n\n"
            f"What next? You can:\n"
            f"  • 'find more sources' — process more URLs\n"
            f"  • 'add column X' — extract a new field\n"
            f"  • 'export xlsx' — save as Excel\n"
            f"  • 'show fill report' — see which fields have gaps"
        )

    # ── Add column ──────────────────────────────────────────────────────────

    def _handle_add_column(self, intent: dict, raw_message: str) -> str:
        """Add new columns and re-extract from existing rows."""
        new_col_names = intent.get("new_columns", [])
        if not new_col_names:
            # Try parsing from raw message
            match = re.search(r'add\s+(?:column\s+)?(.+)', raw_message, re.I)
            if match:
                new_col_names = [n.strip().title()
                                 for n in match.group(1).split(",") if n.strip()]

        if not new_col_names:
            return "What column(s) do you want to add? e.g. 'add Publication Year, Country'"

        # Avoid duplicates
        existing = {c.name.lower() for c in self.state.columns}
        to_add   = [n for n in new_col_names if n.lower() not in existing]

        if not to_add:
            return f"Those columns already exist: {new_col_names}"

        # Add to state columns
        new_defs = self._finalise_columns(to_add)
        self.state.columns.extend(new_defs)

        console.print(f"[dim]Added columns: {to_add}[/dim]")

        # Re-extract from processed sources if we have rows
        if not self.state.rows:
            return (
                f"Added column(s): {', '.join(to_add)}. "
                "Run the pipeline first to collect data."
            )

        # Re-run extraction on existing docs is complex — instead, run the
        # refiner to fill the new columns specifically for all existing rows
        console.print("[dim]Re-running Critic to fill new columns…[/dim]")
        assessments = self.critic_agent.run(self.state)
        filled = self.refiner_agent.run(assessments, self.state)
        save_csv(self.state)

        return (
            f"Added: {', '.join(to_add)}. "
            f"Refiner found values for {filled} field(s) across existing rows. "
            "Use 'find more sources' to process new documents with these columns too."
        )

    # ── More sources ────────────────────────────────────────────────────────

    def _handle_more_sources(self, intent: dict, raw_message: str) -> str:
        """Search for and process additional sources."""
        extra = intent.get("source_limit") or 5
        console.print(f"[dim]Adding {extra} more sources…[/dim]")

        self._live_status(f"Finding {extra} additional source(s)")
        self.search_agent.run(self.state)
        docs = self.scraper_agent.run(self.state, max_docs=extra)
        self._live_status(f"Additional scrape finished. New docs: {len(docs)}")

        if not docs:
            return "Couldn't find additional sources. The topic may be well-covered already."

        self._live_status("Extracting rows from additional sources")
        self.extractor_agent.run(docs, self.state)
        assessments = self.critic_agent.run(self.state)
        if any(a.needs_refinement for a in assessments):
            self._live_status("Refining rows from additional sources")
            self.refiner_agent.run(assessments, self.state)

        csv_path = save_csv(self.state)
        self._live_status("Additional source pass completed")
        return (
            f"Processed {len(docs)} more source(s). "
            f"Total rows: {len(self.state.rows)}. "
            f"Saved → {csv_path}"
        )

    # ── Fix specific row ────────────────────────────────────────────────────

    def _handle_fix_row(self, intent: dict, raw_message: str) -> str:
        """Manually trigger refinement for a specific row."""
        row_idx = intent.get("row_index")
        if row_idx is None:
            # Parse from message: "fix row 3"
            m = re.search(r'\d+', raw_message)
            row_idx = int(m.group()) if m else None

        if row_idx is None or row_idx < 1 or row_idx > len(self.state.rows):
            return f"Invalid row number. I have {len(self.state.rows)} rows."

        idx = row_idx - 1   # convert to 0-indexed
        row = self.state.rows[idx]
        row.refinement_done = False  # allow re-refinement

        assessments = self.critic_agent.run(self.state)
        target = [a for a in assessments if a.row_index == idx]

        if not target or not target[0].needs_refinement:
            return f"Row {row_idx} looks complete — no actionable gaps found."

        filled = self.refiner_agent.run(target, self.state)
        save_csv(self.state)
        return f"Row {row_idx}: filled {filled} additional field(s)."

    # ── Export ───────────────────────────────────────────────────────────────

    def _handle_export(self, intent: dict, raw_message: str) -> str:
        """Export data to CSV or XLSX."""
        fmt = (intent.get("export_format") or "csv").lower()
        if not self.state.rows:
            return "No data to export yet. Run the pipeline first."

        if fmt == "xlsx":
            path = save_xlsx(self.state)
        else:
            path = save_csv(self.state)

        return f"Exported {len(self.state.rows)} rows → `{path}`"

    # ── Show results ─────────────────────────────────────────────────────────

    def _handle_show_results(self, intent: dict, raw_message: str) -> str:
        """Print preview and fill report."""
        if not self.state.rows:
            return "No results yet. Start a pipeline first."
        print_preview(self.state)
        if "fill" in raw_message.lower() or "report" in raw_message.lower():
            print_fill_report(self.state)
        return f"Showing {len(self.state.rows)} rows. Fill report above."

    # ── Change topic ─────────────────────────────────────────────────────────

    def _handle_change_topic(self, intent: dict, raw_message: str) -> str:
        """Reset the session for a new topic."""
        new_topic = intent.get("topic") or raw_message.strip()

        # Save existing session first
        if self.state.rows:
            csv_path = save_csv(self.state)
            console.print(f"[dim]Saved previous session → {csv_path}[/dim]")

        # Reset state
        from state import SessionState
        old_id   = self.state.session_id
        new_state = SessionState()
        new_state.topic = new_topic

        # Copy over the new state (in-place mutation so the reference stays valid)
        self.state.__dict__.update(new_state.__dict__)
        self._awaiting_column_confirmation = False

        return self._handle_start(
            {"intent": "start_pipeline", "topic": new_topic},
            raw_message,
        )

    # ── Custom URLs ───────────────────────────────────────────────────────────

    def _handle_custom_urls(self, intent: dict, raw_message: str) -> str:
        """Add user-provided URLs to the pipeline."""
        urls = intent.get("urls", [])
        if not urls:
            # Parse from message
            urls = re.findall(r'https?://\S+', raw_message)
        if not urls:
            return "I didn't find any URLs in your message. Paste them in full (https://...)."

        if not self.state.topic:
            return "Please set a topic first before providing URLs."

        self.search_agent.add_custom_urls(urls, self.state)
        self._live_status(f"Processing {len(urls)} custom URL(s)")
        docs = self.scraper_agent.run(self.state, max_docs=len(urls))
        self._live_status(f"Custom URL scrape finished. Docs: {len(docs)}")
        if not docs:
            return "Couldn't extract content from those URLs."

        self._live_status("Extracting rows from custom URLs")
        self.extractor_agent.run(docs, self.state)
        assessments = self.critic_agent.run(self.state)
        if any(a.needs_refinement for a in assessments):
            self._live_status("Refining custom URL rows")
            self.refiner_agent.run(assessments, self.state)

        save_csv(self.state)
        self._live_status("Custom URL run completed")
        return f"Processed {len(docs)} URL(s). {len(self.state.rows)} total rows."

    # ── Local PDFs ────────────────────────────────────────────────────────────

    def _handle_local_pdfs(self, intent: dict, raw_message: str) -> str:
        """Process user-provided local PDF file paths."""
        import os, glob

        # Extract paths from intent or raw message
        paths = intent.get("urls", [])
        if not paths:
            # Find file paths in the message
            paths = re.findall(r'[\w\./\\-]+\.pdf', raw_message, re.I)
            paths = [p for p in paths if os.path.isfile(p)]

        if not paths:
            return "I didn't find any valid PDF paths in your message."

        if not self.state.topic:
            return "Please set a topic first."

        self.search_agent.add_local_pdfs(paths, self.state)
        self._live_status(f"Processing {len(paths)} local PDF(s)")
        docs = self.scraper_agent.run(self.state, max_docs=len(paths))
        self._live_status(f"Local PDF parse finished. Docs: {len(docs)}")
        if not docs:
            return "Couldn't extract content from those PDFs."

        self._live_status("Extracting rows from local PDFs")
        self.extractor_agent.run(docs, self.state)
        assessments = self.critic_agent.run(self.state)
        if any(a.needs_refinement for a in assessments):
            self._live_status("Refining local PDF rows")
            self.refiner_agent.run(assessments, self.state)

        save_csv(self.state)
        self._live_status("Local PDF run completed")
        return f"Processed {len(docs)} PDF(s). {len(self.state.rows)} total rows."

    # ── Stop ──────────────────────────────────────────────────────────────────

    def _handle_stop(self, intent: dict, raw_message: str) -> str:
        self.state.stop_requested = True
        if self.state.rows:
            path = save_csv(self.state)
            return f"Goodbye! Final save → {path}"
        return "Goodbye! No data was collected this session."

    # ── General chat ──────────────────────────────────────────────────────────

    def _handle_chat(self, user_message: str) -> str:
        """
        Handle general questions and conversation using the full history.
        """
        # Build message history for multi-turn context
        history = [
            {"role": m.role, "content": m.content[:800]}
            for m in self.state.history[-10:]   # last 10 turns for context
        ]

        system = _ORCHESTRATOR_SYSTEM.format(
            topic            = self.state.topic or "(not set)",
            columns          = ", ".join(c.name for c in self.state.columns) or "(not set)",
            row_count        = len(self.state.rows),
            source_count     = len(self.state.processed_sources),
            refinement_rounds = self.state.refinement_rounds,
        )

        try:
            return self.llm.chat(
                system   = system,
                messages = history,
                model    = cfg.FAST_MODEL,
                max_tokens = 300,
            )
        except Exception:
            return "I'm here! What would you like to do?"

    # ── Column generation utilities ───────────────────────────────────────────

    def _suggest_columns(self, topic: str) -> list[str]:
        """Ask the LLM to suggest extraction columns for the topic."""
        try:
            raw = self.llm.complete_json(
                system=_COLUMN_SUGGESTION_SYSTEM,
                user=f'Topic: "{topic}"\n\nSuggest 8-12 extraction columns:',
                model=cfg.FAST_MODEL,
                max_tokens=512,
            )
            if isinstance(raw, list):
                cleaned = self._clean_column_suggestions(
                    topic,
                    [str(x) for x in raw if isinstance(x, str) and x.strip()],
                )
                return cleaned
        except Exception as exc:
            console.print(f"[yellow]⚠ Column suggestion failed: {exc}[/yellow]")
        return []

    def _clean_column_suggestions(self, topic: str, cols: list[str]) -> list[str]:
        """Normalize/de-noise LLM suggestions and inject topic-aware defaults."""
        def norm(name: str) -> str:
            n = re.sub(r"\s+", " ", name).strip().title()
            return n.replace("Url", "URL").replace("Doi", "DOI").replace("Dar", "DAR")

        # Filter common off-topic noise we've seen in misfires.
        blocked_tokens = {
            "game", "player", "publisher", "download", "price", "startup",
            "investor", "funding", "employee", "business model",
        }
        cleaned: list[str] = []
        seen: set[str] = set()
        for c in cols:
            n = norm(c)
            low = n.lower()
            if any(tok in low for tok in blocked_tokens):
                continue
            if n.lower() in seen:
                continue
            seen.add(n.lower())
            cleaned.append(n)

        topic_low = (topic or "").lower()
        if "antibody drug conjugate" in topic_low or re.search(r"\badc\b", topic_low):
            preferred = [
                "Antibody Name",
                "Target Antigen",
                "Linker",
                "Payload",
                "DAR",
                "Indication",
                "Approval Year",
            ]
            for p in reversed(preferred):
                if p.lower() not in seen:
                    cleaned.insert(0, p)
                    seen.add(p.lower())

        # Ensure identity and source columns exist.
        if not any("name" in c.lower() or "title" in c.lower() for c in cleaned):
            cleaned.insert(0, "Entity Name")
        if "source url" not in [c.lower() for c in cleaned]:
            cleaned.append("Source URL")

        # Keep concise list.
        cleaned = cleaned[:12]
        if len(cleaned) < 8:
            defaults = ["Type", "Status", "Evidence", "Reference Year"]
            for d in defaults:
                if d.lower() not in [c.lower() for c in cleaned]:
                    cleaned.insert(-1, d)
                if len(cleaned) >= 8:
                    break
        return cleaned

    def _finalise_columns(self, names: list[str]) -> list[ColumnDef]:
        """
        Convert a list of column name strings into ColumnDef objects.

        Marks common identity columns as required (Critic will always try to fill these).
        Does NOT generate descriptions here — ExtractorAgent will do that lazily
        on first use, with topic context.
        """
        identity_signals = {"name", "title", "id", "doi", "drug", "compound",
                            "gene", "study", "trial", "company", "product"}
        cols = []
        for name in names:
            is_identity = any(sig in name.lower() for sig in identity_signals)
            cols.append(ColumnDef(
                name     = name.strip(),
                required = is_identity,
            ))
        return cols

    # ── Help text ─────────────────────────────────────────────────────────────

    @staticmethod
    def _help_text() -> str:
        return """
**ARIA commands:**

**Starting a research session:**
  • Just describe your topic: 'Collect data on CRISPR papers 2020-2024'
  • With columns: 'Research EVs: Name, Revenue, Country, Founded, CEO'
  • With source limit: 'Find 20 sources on blockchain startups'

**After a pipeline run:**
  • 'find more sources' / 'get 5 more'
  • 'add column X' / 'add columns A, B, C'
  • 'fix row 3' — manually retry refinement for a row
  • 'show results' / 'show fill report'
  • 'export csv' / 'export xlsx'

**Providing sources:**
  • 'scrape https://example.com https://other.com'
  • 'process paper.pdf report.pdf'

**Other:**
  • 'change topic' / 'new topic: ...'
  • 'stop' / 'exit' / 'quit'
""".strip()
