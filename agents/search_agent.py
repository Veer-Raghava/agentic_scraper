"""Search agent: find candidate URLs for a topic."""

from __future__ import annotations

from ddgs import DDGS
from rich.console import Console

import config as cfg
from state import SessionState

console = Console()


class SearchAgent:
    def __init__(self, llm=None) -> None:
        self.llm = llm

    def _queries(self, topic: str) -> list[str]:
        t = topic.strip()
        return [
            f"{t} dataset",
            f"{t} research paper",
            f"{t} review pdf",
            f"{t} clinical data",
            f"{t} table",
        ]

    def run(self, state: SessionState) -> None:
        console.print(f"[cyan]🔍 Search Agent[/cyan] — topic: '{state.topic}'")
        queries = self._queries(state.topic)
        seen = set(state.pending_sources) | state.processed_sources | state.dead_sources
        added = 0
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=max(10, state.source_limit * 3)):
                    url = (r.get("href") or r.get("url") or "").strip()
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    state.pending_sources.append(url)
                    added += 1
                    if added >= state.source_limit * 4:
                        break
                if added >= state.source_limit * 4:
                    break
        console.print(f"[green]✓ Search complete[/green] — {added} new URLs queued ({len(state.pending_sources)} total pending)")

    def add_custom_urls(self, urls: list[str], state: SessionState) -> None:
        seen = set(state.pending_sources) | state.processed_sources | state.dead_sources
        for u in urls:
            if u not in seen:
                state.pending_sources.append(u)

    def add_local_pdfs(self, paths: list[str], state: SessionState) -> None:
        seen = set(state.pending_sources) | state.processed_sources | state.dead_sources
        for p in paths:
            if p not in seen:
                state.pending_sources.append(p)
