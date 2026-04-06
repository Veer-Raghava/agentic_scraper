"""Scraper agent: download/read sources and return normalized documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from rich.console import Console

import config as cfg
from state import SessionState
from tools.pdf_tools import (
    is_pdf_url,
    download_pdf,
    extract_pdf_text,
    extract_tables_from_pdf,
    extract_tables_from_html,
    extract_links_from_html,
)

console = Console()


@dataclass
class ScrapedDocument:
    url: str
    text: str
    tables: list = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    local_pdf_path: str = ""


class ScraperAgent:
    def __init__(self, browser=None) -> None:
        self.browser = browser
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        })

    def _fetch_html(self, url: str) -> str:
        r = self.session.get(url, timeout=30)
        r.raise_for_status()
        return r.text

    def run(self, state: SessionState, max_docs: int = 10) -> list[ScrapedDocument]:
        console.print(f"\n[cyan]🌐 Scraper Agent[/cyan] — {len(state.pending_sources)} pending, target {max_docs} good docs")
        docs: list[ScrapedDocument] = []

        while state.pending_sources and len(docs) < max_docs:
            src = state.pending_sources[0]
            console.print(f"\n► {src}")
            try:
                if src.lower().endswith('.pdf') or is_pdf_url(src):
                    out_dir = state.pdf_dir or cfg.PDF_TEMP_DIR
                    local_pdf = download_pdf(src, out_dir)
                    if not local_pdf:
                        raise RuntimeError("PDF download failed")
                    text = extract_pdf_text(local_pdf)
                    tables = extract_tables_from_pdf(local_pdf)
                    doc = ScrapedDocument(
                        url=src,
                        text=text,
                        tables=tables,
                        links=[],
                        local_pdf_path=local_pdf,
                    )
                    docs.append(doc)
                    console.print(f"  [green]✓ PDF downloaded & extracted[/green] ({len(text):,} chars)")
                    for i, t in enumerate(tables[:6], start=1):
                        rows = len(t)
                        cols = len(t[0]) if t else 0
                        console.print(f"  Table {i}: {rows}×{cols}")
                else:
                    html = self._fetch_html(src)
                    soup = BeautifulSoup(html, "lxml")
                    for tag in soup(["script", "style", "noscript"]):
                        tag.decompose()
                    text = soup.get_text(" ", strip=True)
                    tables = extract_tables_from_html(html)
                    links = extract_links_from_html(html)
                    if len(text) < 500:
                        raise RuntimeError("no text extracted")
                    docs.append(ScrapedDocument(url=src, text=text, tables=tables, links=links))
                    console.print(f"  [green]✓ Good[/green] ({len(text):,} chars, {len(tables)} tables, {len(links)} links)")

                state.mark_processed(src)
            except Exception as exc:
                console.print(f"  [yellow]⚠ SKIPPED[/yellow] — {exc}")
                state.mark_dead(src)

        console.print(f"\nScraper done: {len(docs)} good")
        return docs
