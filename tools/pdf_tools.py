"""
PDF utilities:
- Download PDF files locally with requests
- Extract text, tables, and page images for downstream extraction/refinement
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber
import requests
from bs4 import BeautifulSoup


def is_pdf_url(url: str) -> bool:
    low = (url or "").lower()
    return low.endswith(".pdf") or ".pdf?" in low or "/pdf/" in low


def _safe_pdf_name(url: str) -> str:
    tail = re.sub(r"[^a-zA-Z0-9._-]+", "_", url.split("/")[-1] or "document.pdf")
    if not tail.lower().endswith(".pdf"):
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
        tail = f"{tail}_{digest}.pdf"
    return tail[:140]


def download_pdf(url: str, out_dir: str, timeout: int = 45) -> str | None:
    """
    Download one PDF to out_dir and return local path.
    Returns None if download failed or content wasn't a PDF.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    target = Path(out_dir) / _safe_pdf_name(url)
    try:
        r = requests.get(url, timeout=timeout, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "pdf" not in ctype and not is_pdf_url(url):
            return None
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)
        if target.stat().st_size < 1024:
            return None
        return str(target)
    except Exception:
        return None


def extract_pdf_artifacts(
    pdf_path: str,
    tables_dir: str,
    images_dir: str,
    supplementary_dir: str,
) -> dict[str, Any]:
    """
    Extract lightweight artifacts from a PDF:
      - full text (saved in supplementary dir)
      - tables (CSV files)
      - page images (PNG files)
    """
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    Path(supplementary_dir).mkdir(parents=True, exist_ok=True)

    pdf = Path(pdf_path)
    stem = pdf.stem
    meta: dict[str, Any] = {
        "pdf_path": str(pdf),
        "tables": [],
        "images": [],
        "text_path": "",
        "num_pages": 0,
    }

    # Text + images via PyMuPDF
    text_parts: list[str] = []
    try:
        doc = fitz.open(pdf_path)
        meta["num_pages"] = len(doc)
        for i, page in enumerate(doc, start=1):
            text_parts.append(page.get_text("text") or "")
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_path = Path(images_dir) / f"{stem}_p{i}.png"
            pix.save(str(img_path))
            meta["images"].append(str(img_path))
    except Exception:
        pass

    text_out = Path(supplementary_dir) / f"{stem}.txt"
    text_out.write_text("\n\n".join(text_parts), encoding="utf-8")
    meta["text_path"] = str(text_out)

    # Tables via pdfplumber
    try:
        with pdfplumber.open(pdf_path) as p:
            for pidx, page in enumerate(p.pages, start=1):
                for tidx, table in enumerate(page.extract_tables() or [], start=1):
                    rows = [[("" if c is None else str(c)) for c in row] for row in table]
                    tpath = Path(tables_dir) / f"{stem}_p{pidx}_t{tidx}.csv"
                    with open(tpath, "w", encoding="utf-8") as tf:
                        for row in rows:
                            tf.write(",".join(cell.replace(",", " ") for cell in row) + "\n")
                    meta["tables"].append(str(tpath))
    except Exception:
        pass

    meta_path = Path(supplementary_dir) / f"{stem}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Backward-compat helpers (for older scraper_agent implementations)
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize whitespace and strip null-ish artifacts."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_links_from_html(html: str, base_url: str = "") -> list[str]:
    """Extract unique absolute-ish links from raw HTML."""
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href or href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        if href in seen:
            continue
        seen.add(href)
        links.append(href)
    return links


def extract_tables_from_html(html: str) -> list[list[list[str]]]:
    """
    Extract HTML tables as nested rows:
      [
        [["h1","h2"], ["v1","v2"]],
        ...
      ]
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    out: list[list[list[str]]] = []
    for table in soup.find_all("table"):
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            row = [clean_text(c.get_text(" ", strip=True)) for c in cells]
            if any(cell for cell in row):
                rows.append(row)
        if rows:
            out.append(rows)
    return out


def extract_pdf_text(pdf_path: str) -> str:
    """Extract plain text from all PDF pages."""
    parts: list[str] = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            parts.append(page.get_text("text") or "")
    except Exception:
        return ""
    return clean_text("\n\n".join(parts))


def extract_pdf_tables(pdf_path: str) -> list[list[list[str]]]:
    """Extract all tables from a PDF using pdfplumber."""
    tables: list[list[list[str]]] = []
    try:
        with pdfplumber.open(pdf_path) as p:
            for page in p.pages:
                for table in page.extract_tables() or []:
                    rows = [[clean_text("" if c is None else str(c)) for c in row] for row in table]
                    if rows:
                        tables.append(rows)
    except Exception:
        return []
    return tables
