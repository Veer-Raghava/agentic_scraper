"""
agents/critic_agent.py — Quality critic agent.

The Critic is what fundamentally separates this system from the original
pipeline. Instead of accepting whatever the Extractor produced and moving on,
the Critic:

  1. Reviews EVERY row in the session state.
  2. Identifies fields that are N/A or suspiciously vague.
  3. Uses the LLM to assess whether an N/A is acceptable (the info genuinely
     doesn't exist for this entity) or problematic (likely missed by the extractor).
  4. Flags rows and specific fields for the Refiner agent to fix.
  5. Provides a diagnostic report so the user knows what's happening.

The key insight: not all N/As are equal.
  - "Sample Size = N/A" for a blog post is acceptable.
  - "Sample Size = N/A" for a clinical trial is a red flag that must be fixed.
The Critic uses the source type and topic context to make this distinction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

import config as cfg
from state import ColumnDef, ExtractedRow, SessionState
from tools.llm_tools import LLMClient

console = Console()

# Values that count as "empty" for critic assessment
_EMPTY_VALS = frozenset({
    "n/a", "none", "not specified", "not mentioned", "not provided",
    "not explicitly mentioned", "unspecified", "unknown", "na", "nil", "",
    "not available", "not found", "not discussed", "not applicable",
    "not stated", "not reported", "not described", "null", "—", "-",
})


def _is_empty(v: str) -> bool:
    return str(v).strip().lower() in _EMPTY_VALS


# ── Gap assessment result ─────────────────────────────────────────────────────

@dataclass
class GapAssessment:
    """
    The Critic's verdict on one row.

    row_index   — index into state.rows
    row         — the ExtractedRow itself
    actionable_gaps — fields the Refiner SHOULD try to fill
    acceptable_gaps — fields where N/A is genuinely expected
    priority    — 0 (urgent) to 3 (low) — used to order refinement
    reason      — brief explanation of the verdict
    """
    row_index:       int
    row:             ExtractedRow
    actionable_gaps: list[str] = field(default_factory=list)
    acceptable_gaps: list[str] = field(default_factory=list)
    priority:        int = 1
    reason:          str = ""

    @property
    def needs_refinement(self) -> bool:
        return bool(self.actionable_gaps)


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════

_GAP_TRIAGE_SYSTEM = """\
You are a rigorous data quality critic reviewing structured data extracted
from web sources about "{topic}".

Your job is to assess which N/A fields are:
  - ACTIONABLE: The information almost certainly exists somewhere and the
    extractor missed it. The Refiner should search for it.
  - ACCEPTABLE: The information genuinely may not exist for this entity type
    (e.g., a news article won't have a "DOI", a blog won't have "Sample Size").

For each missing field, decide: is it reasonable to expect this information
exists for this type of source? Consider:
  - Source type: is it a paper, news article, product page, company report?
  - Entity type: is it a drug, study, company, person, event?
  - Field type: some fields are simply not applicable to all entities.

Return a JSON object with:
  "actionable": list of field names the Refiner should try to fill
  "acceptable": list of field names where N/A is understandable
  "priority": integer 0-3 (0=urgent: >60% empty; 1=high: 40-60%; 2=medium: 20-40%; 3=low: <20%)
  "reason": one sentence explaining the overall verdict

Return ONLY the JSON object."""


# ═══════════════════════════════════════════════════════════════════════════════
# Critic agent
# ═══════════════════════════════════════════════════════════════════════════════

class CriticAgent:
    """
    Reviews all extracted rows and produces GapAssessment objects.

    Usage:
        critic = CriticAgent(llm)
        assessments = critic.run(state)
        # assessments is sorted by priority (most urgent first)
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self, state: SessionState) -> list[GapAssessment]:
        """
        Assess all rows and return sorted GapAssessments.

        Rows that already had a refinement pass are re-evaluated only if
        their fill rate is still below 50%.
        """
        if not state.rows:
            console.print("[dim]Critic: no rows to review.[/dim]")
            return []

        console.print(f"\n[bold cyan]🔍 Critic Agent[/bold cyan]"
                      f" — reviewing {len(state.rows)} row(s)")

        assessments: list[GapAssessment] = []

        for i, row in enumerate(state.rows):
            missing = row.missing_fields(state.columns)

            # Skip rows that are already fully filled
            if not missing:
                console.print(f"  [dim]Row {i+1}: ✓ fully filled[/dim]")
                continue

            # For already-refined rows, only re-assess if still very empty
            if row.refinement_done and row.fill_rate(state.columns) >= 0.5:
                console.print(
                    f"  [dim]Row {i+1}: refined, fill={row.fill_rate(state.columns):.0%} — skip[/dim]"
                )
                continue

            assessment = self._assess_row(i, row, missing, state)
            assessments.append(assessment)

            status = (
                f"[red]{len(assessment.actionable_gaps)} actionable gaps[/red]"
                if assessment.actionable_gaps
                else "[green]all gaps acceptable[/green]"
            )
            console.print(
                f"  Row {i+1}: fill={row.fill_rate(state.columns):.0%}, {status}"
            )

        # Sort by priority (0 = most urgent)
        assessments.sort(key=lambda a: (a.priority, -len(a.actionable_gaps)))

        self._print_report(assessments, state)

        return assessments

    # ── Single-row assessment ───────────────────────────────────────────────

    def _assess_row(
        self,
        idx: int,
        row: ExtractedRow,
        missing: list[str],
        state: SessionState,
    ) -> GapAssessment:
        """
        Use the LLM to triage missing fields as actionable or acceptable.

        For rows with many missing fields, we use the full LLM triage.
        For rows missing just 1-2 fields, we use a simple heuristic
        (required columns are always actionable) to save API calls.
        """
        # Quick path for minor gaps — no LLM call needed
        if len(missing) <= 2:
            actionable = [
                f for f in missing
                if any(c.name == f and c.required for c in state.columns)
            ]
            acceptable = [f for f in missing if f not in actionable]
            fill_rate  = row.fill_rate(state.columns)
            priority   = 3 if fill_rate >= 0.8 else 2

            return GapAssessment(
                row_index       = idx,
                row             = row,
                actionable_gaps = actionable,
                acceptable_gaps = acceptable,
                priority        = priority,
                reason          = "Minor gaps — required fields flagged only.",
            )

        # Full LLM triage for significant gaps
        return self._llm_triage(idx, row, missing, state)

    def _llm_triage(
        self,
        idx: int,
        row: ExtractedRow,
        missing: list[str],
        state: SessionState,
    ) -> GapAssessment:
        """
        Ask the LLM to classify missing fields as actionable or acceptable.
        """
        # Build a summary of what we DO have (gives context to the LLM)
        filled_fields = {
            k: v for k, v in row.data.items()
            if not _is_empty(str(v)) and k in state.column_names()
        }
        filled_summary = "\n".join(
            f"  {k}: {str(v)[:80]}" for k, v in list(filled_fields.items())[:8]
        )

        # Column definitions for missing fields only
        col_defs = "\n".join(
            f'  "{c.name}": {c.description}'
            for c in state.columns if c.name in missing
        )

        try:
            result = self.llm.complete_json(
                system=_GAP_TRIAGE_SYSTEM.format(topic=state.topic),
                user=(
                    f"Source URL: {row.source_url[:120]}\n\n"
                    f"Fields successfully extracted:\n{filled_summary or '  (none)'}\n\n"
                    f"Missing fields to triage:\n{col_defs}\n\n"
                    f"Missing field names: {missing}\n\n"
                    "Classify each missing field as actionable or acceptable:"
                ),
                model=cfg.FAST_MODEL,
                max_tokens=512,
            )

            if isinstance(result, dict):
                actionable = [f for f in result.get("actionable", []) if f in missing]
                acceptable = [f for f in result.get("acceptable", []) if f in missing]
                # Any field not classified goes to actionable (conservative)
                classified = set(actionable) | set(acceptable)
                unclassified = [f for f in missing if f not in classified]
                actionable.extend(unclassified)

                return GapAssessment(
                    row_index       = idx,
                    row             = row,
                    actionable_gaps = actionable,
                    acceptable_gaps = acceptable,
                    priority        = int(result.get("priority", 1)),
                    reason          = str(result.get("reason", "")),
                )

        except Exception as exc:
            console.print(f"    [dim yellow]⚠ LLM triage failed ({exc}) — using heuristic[/dim yellow]")

        # Fallback: all missing fields are actionable
        fill_rate = row.fill_rate(state.columns)
        priority  = 0 if fill_rate < 0.4 else (1 if fill_rate < 0.6 else 2)
        return GapAssessment(
            row_index       = idx,
            row             = row,
            actionable_gaps = missing,
            acceptable_gaps = [],
            priority        = priority,
            reason          = "LLM triage unavailable — all gaps flagged.",
        )

    # ── Report ──────────────────────────────────────────────────────────────

    def _print_report(
        self,
        assessments: list[GapAssessment],
        state: SessionState,
    ) -> None:
        """Print a summary table of the critic's findings."""
        if not assessments:
            console.print("[green]✓ Critic: all rows pass quality check![/green]")
            return

        needs_work = [a for a in assessments if a.needs_refinement]
        console.print(
            f"\n[bold]Critic report:[/bold] "
            f"{len(needs_work)}/{len(assessments)} rows need refinement"
        )

        if not needs_work:
            return

        tbl = Table(show_header=True, header_style="bold", show_lines=False)
        tbl.add_column("Row", width=4)
        tbl.add_column("Fill", width=6)
        tbl.add_column("Actionable gaps")
        tbl.add_column("Priority", width=8)

        for a in needs_work[:10]:   # show first 10
            fill    = a.row.fill_rate(state.columns)
            color   = "red" if fill < 0.4 else ("yellow" if fill < 0.7 else "green")
            gaps_str = ", ".join(a.actionable_gaps[:5])
            if len(a.actionable_gaps) > 5:
                gaps_str += f" +{len(a.actionable_gaps)-5} more"
            prio_str = ["🔴 urgent", "🟠 high", "🟡 medium", "🟢 low"][a.priority]
            tbl.add_row(
                str(a.row_index + 1),
                f"[{color}]{fill:.0%}[/{color}]",
                gaps_str,
                prio_str,
            )

        console.print(tbl)