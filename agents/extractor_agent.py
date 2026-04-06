"""Extractor agent: convert scraped docs into structured rows."""

from __future__ import annotations

import re
from rich.console import Console

from state import ExtractedRow, SessionState
from agents.scraper_agent import ScrapedDocument

console = Console()


class ExtractorAgent:
    def __init__(self, llm=None) -> None:
        self.llm = llm

    def _extract_value(self, column: str, text: str) -> str:
        c = column.lower()
        if "source url" in c:
            return "N/A"
        # Tiny heuristic extractor for ADC demos
        if "dar" in c:
            m = re.search(r"\bDAR\b[^\d]{0,8}(\d+(?:\.\d+)?)", text, re.I)
            return m.group(1) if m else "N/A"
        if "antigen" in c:
            for x in ["HER2", "CD30", "CD22", "CD79b", "TROP2", "CD19"]:
                if x.lower() in text.lower():
                    return x
            return "N/A"
        if "antibody" in c or "name" in c:
            m = re.search(r"(trastuzumab|brentuximab vedotin|inotuzumab ozogamicin|polatuzumab vedotin)", text, re.I)
            return m.group(1).title() if m else "N/A"
        if "payload" in c:
            for x in ["MMAE", "DM1", "SN-38", "Calicheamicin"]:
                if x.lower() in text.lower():
                    return x
            return "N/A"
        if "linker" in c:
            if "maleimidocaproyl" in text.lower() or "mcc" in text.lower():
                return "Maleimidocaproyl (MCC)"
            return "N/A"
        return "N/A"

    def run(self, docs: list[ScrapedDocument], state: SessionState) -> int:
        console.print(f"\n[cyan]🧠 Extractor Agent[/cyan] — {len(docs)} document(s) to process")
        added = 0
        cols = state.column_names()
        for i, doc in enumerate(docs, start=1):
            console.print(f"\n  Document {i}/{len(docs)}: {doc.url[:85]}")
            data = {}
            for c in cols:
                data[c] = doc.url if c.lower() == "source url" else self._extract_value(c, doc.text)
            # only add if at least one non-N/A field other than source url
            real_vals = [v for k, v in data.items() if k.lower() != "source url" and str(v).strip().lower() not in {"", "n/a"}]
            if real_vals:
                state.add_row(ExtractedRow(data=data, source_url=doc.url))
                added += 1
                console.print("  [green]✓ Extracted 1 row[/green]")
            else:
                console.print("  [yellow]✓ Extracted 0 row(s)[/yellow]")

        console.print(f"\n[green]✓ Extractor done[/green] — {added} new row(s) added")
        return added
