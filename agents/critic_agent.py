"""Critic agent: assess row completeness."""

from __future__ import annotations

from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table

from state import SessionState

console = Console()


@dataclass
class RowAssessment:
    row_index: int
    fill_rate: float
    actionable_gaps: list[str] = field(default_factory=list)

    @property
    def needs_refinement(self) -> bool:
        return len(self.actionable_gaps) > 0


class CriticAgent:
    def __init__(self, llm=None) -> None:
        self.llm = llm

    def run(self, state: SessionState) -> list[RowAssessment]:
        console.print(f"\n[cyan]🔍 Critic Agent[/cyan] — reviewing {len(state.rows)} row(s)")
        out: list[RowAssessment] = []
        for i, r in enumerate(state.rows):
            missing = r.missing_fields(state.columns)
            fill = r.fill_rate(state.columns)
            gaps = missing if fill < 0.8 else []
            out.append(RowAssessment(row_index=i, fill_rate=fill, actionable_gaps=gaps))

        needs = [a for a in out if a.needs_refinement]
        console.print(f"\nCritic report: {len(needs)}/{len(out)} rows need refinement")
        if needs:
            t = Table()
            t.add_column("Row")
            t.add_column("Fill")
            t.add_column("Actionable gaps")
            for a in needs[:10]:
                t.add_row(str(a.row_index + 1), f"{a.fill_rate*100:.0f}%", ", ".join(a.actionable_gaps[:6]))
            console.print(t)
        return out
