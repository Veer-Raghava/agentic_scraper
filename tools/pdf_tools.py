"""
tools/pdf_tools.py — PDF-specific extraction utilities.

Covers:
  • OCR for scanned PDFs (Tesseract via OpenCV preprocessing)
  • Table extraction from PDFs (pdfplumber) and HTML (pandas)
  • Data availability section detection
  • DOI / repository / accession number pattern matching
  • Smart section-aware text chunking with embedding-based chunk selection

These functions are called by the Extractor agent when processing PDFs.
They are designed to be composable — each does exactly one thing.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from rich.console import Console

import config as cfg
from tools.llm_tools import LLMClient

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# OCR
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text_ocr(pdf_path: str) -> str:
    """
    Extract text from a scanned PDF using Tesseract OCR.

    Pipeline per page:
      1. Render PDF page as a PIL image (pdf2image / poppler)
      2. Convert to numpy array → grayscale
      3. Binarize with Otsu's method (improves Tesseract accuracy)
      4. Run Tesseract

    Returns:
        Concatenated OCR text from all pages, or "" on failure.
    """
    if not cfg.ENABLE_OCR:
        return ""

    try:
        import cv2
        import pytesseract
        from pdf2image import convert_from_path

        pytesseract.pytesseract.tesseract_cmd = cfg.TESSERACT_CMD
        console.print("  [yellow]Starting OCR…[/yellow]")
        images = convert_from_path(pdf_path)
        parts  = []

        for i, img in enumerate(images):
            console.print(f"    [dim]OCR page {i+1}/{len(images)}[/dim]")
            arr   = np.array(img)
            gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            # Otsu binarization — automatically finds the best threshold
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            text = pytesseract.image_to_string(thresh)
            parts.append(text)

        return "\n\n".join(parts)

    except ImportError as exc:
        console.print(f"  [red]OCR dependencies missing: {exc}[/red]")
        return ""
    except Exception as exc:
        console.print(f"  [red]OCR failed: {exc}[/red]")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Table extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_tables_from_pdf(pdf_path: str) -> list[str]:
    """
    Extract tables from a PDF as CSV strings using pdfplumber.

    Returns a list of strings, each being a CSV representation of one table.
    Tables with fewer than 2 rows are skipped (likely just headers or noise).
    """
    results = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for pnum, page in enumerate(pdf.pages):
                for tbl in (page.extract_tables() or []):
                    if not tbl or len(tbl) < 2:
                        continue
                    # Clean whitespace in every cell
                    cleaned = [
                        [(cell.strip() if cell else "") for cell in row]
                        for row in tbl
                    ]
                    # Convert to CSV string
                    lines = [",".join(f'"{c}"' for c in row) for row in cleaned]
                    results.append(f"[Table from page {pnum+1}]\n" + "\n".join(lines))
                    console.print(
                        f"  [dim]Table p{pnum+1}: {len(cleaned)}×{len(cleaned[0])}[/dim]"
                    )
    except ImportError:
        console.print("  [dim yellow]pdfplumber not installed — no table extraction[/dim yellow]")
    except Exception as exc:
        console.print(f"  [dim yellow]Table extraction failed: {exc}[/dim yellow]")
    return results


def extract_tables_from_html(html: str) -> list[str]:
    """
    Extract tables from HTML using pandas.read_html.

    Returns a list of CSV strings.
    """
    results = []
    try:
        soup   = BeautifulSoup(html, "lxml")
        for tbl in soup.find_all("table"):
            try:
                df = pd.read_html(io.StringIO(str(tbl)))[0]
                results.append(df.to_csv(index=False))
            except Exception:
                continue
    except Exception:
        pass
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset URL / identifier pattern matching
# ═══════════════════════════════════════════════════════════════════════════════

DOI_PATTERN = re.compile(r"10\.\d{4,}/[^\s,;\"'<>]+")

REPO_PATTERNS: dict[str, re.Pattern] = {
    "Zenodo":   re.compile(r"zenodo\.org/record[s]?/\d+", re.I),
    "Figshare": re.compile(r"figshare\.com/\S+", re.I),
    "GitHub":   re.compile(r"github\.com/[\w\-]+/[\w\-]+", re.I),
    "Dryad":    re.compile(r"datadryad\.org/\S+", re.I),
    "Kaggle":   re.compile(r"kaggle\.com/\S+", re.I),
    "OSF":      re.compile(r"osf\.io/\S+", re.I),
    "HuggingFace": re.compile(r"huggingface\.co/\S+", re.I),
}

ACCESSION_PATTERNS: dict[str, re.Pattern] = {
    "GEO":        re.compile(r"GSE\d{4,}"),
    "SRA":        re.compile(r"SR[APRX]\d{6,}"),
    "BioProject": re.compile(r"PRJNA?\d+"),
    "ClinicalTrials": re.compile(r"NCT\d{8}"),
    "ArrayExpress": re.compile(r"E-[A-Z]{4}-\d+"),
}

# Mutable — LLM-generated topic-specific patterns injected by Extractor agent
_extra_url_patterns: dict[str, re.Pattern] = {}


def register_url_patterns(patterns: dict[str, str]) -> None:
    """
    Inject topic-specific URL patterns generated by the LLM.
    patterns — {label: regex_string}
    """
    global _extra_url_patterns
    compiled = {}
    for label, pat_str in patterns.items():
        try:
            compiled[str(label)] = re.compile(str(pat_str), re.I)
        except re.error as exc:
            console.print(f"  [dim yellow]⚠ Invalid pattern '{pat_str}': {exc}[/dim yellow]")
    _extra_url_patterns = compiled


def find_dataset_links(text: str) -> list[dict]:
    """
    Scan text for dataset DOIs, repository URLs, and accession numbers.

    Returns a list of {type, value} dicts, deduplicated.
    """
    found, seen = [], set()

    def _add(type_: str, val: str) -> None:
        v = val.rstrip(".")
        if v not in seen:
            seen.add(v)
            found.append({"type": type_, "value": v})

    for m in DOI_PATTERN.finditer(text):
        _add("DOI", m.group())
    for name, pat in REPO_PATTERNS.items():
        for m in pat.finditer(text):
            _add(name, m.group())
    for name, pat in ACCESSION_PATTERNS.items():
        for m in pat.finditer(text):
            _add(name, m.group())
    for name, pat in _extra_url_patterns.items():
        for m in pat.finditer(text):
            _add(name, m.group())

    return found


# ═══════════════════════════════════════════════════════════════════════════════
# Data availability section extraction
# ═══════════════════════════════════════════════════════════════════════════════

# Mutable — topic-specific headers injected by Extractor agent
_extra_section_headers: list[str] = []

BUILTIN_HEADERS = [
    r"data\s+availab", r"availability\s+of\s+data", r"data\s+access",
    r"code\s+availab", r"supplementary\s+(?:information|materials?|data)",
    r"supporting\s+information", r"open\s+science",
]


def register_section_headers(headers: list[str]) -> None:
    """Inject LLM-generated topic-specific section headers."""
    global _extra_section_headers
    _extra_section_headers = [h.lower() for h in headers]


def find_data_availability_section(text: str) -> str:
    """
    Extract the 'Data Availability' section (or equivalent) from a paper.

    Uses a two-pass approach:
      1. Search in the main body (before References)
      2. Fallback: search the last 20 % of the document (some journals put
         the section after references)

    Returns up to 2000 characters of the section text.
    """
    all_headers = BUILTIN_HEADERS + [
        h for h in _extra_section_headers if h not in BUILTIN_HEADERS
    ]
    header_re = "|".join(all_headers)

    # Pass 1: section before References/Acknowledgments
    pat = re.compile(
        r"(?:^|\n)\s*(?:" + header_re + r").*?"
        r"(?=\n\s*(?:references|acknowledgment|author|conflict|funding|appendix)|\Z)",
        re.I | re.DOTALL,
    )
    m = pat.search(text)
    if m:
        return m.group().strip()[:2000]

    # Pass 2: tail of document
    tail_start = max(0, int(len(text) * 0.80))
    tail_text  = text[tail_start:]
    tail_pat   = re.compile(
        r"(?:^|\n)\s*(?:" + header_re + r").*", re.I | re.DOTALL
    )
    m2 = tail_pat.search(tail_text)
    return m2.group().strip()[:2000] if m2 else ""


# ═══════════════════════════════════════════════════════════════════════════════
# Text chunking with smart selection
# ═══════════════════════════════════════════════════════════════════════════════

# Section headers that indicate irrelevant content — skip these chunks
_JUNK_HEADERS = frozenset({
    "references", "bibliography", "acknowledgment", "acknowledgements",
    "funding", "conflict of interest", "author contribution",
    "abbreviation", "list of figures", "list of tables",
    "ethical approval", "declaration",
})


def _structural_score(chunk: str, idx: int, total: int) -> float:
    """
    Lightweight 0–1 score based on structural cues (no LLM needed).
    Returns -1.0 for chunks that should be hard-skipped (junk headers).
    """
    cl      = chunk.lower()
    first300 = cl[:300]

    # Hard disqualify junk sections
    if any(j in first300 for j in _JUNK_HEADERS):
        return -1.0

    score   = 0.0
    raw_len = len(chunk.strip())

    # Table-like formatting
    if sum(1 for s in ["||", "|", "\t", "---"] if s in chunk) >= 2:
        score += 0.2

    # Numeric density
    nums = len(re.findall(r'\b\d+(?:\.\d+)?\b', cl))
    score += min(nums * 0.01, 0.15)

    # Key-value patterns (common in structured papers)
    kv = len(re.findall(r'(?:^|\n)\s*[A-Z][^:\n]{2,30}:\s*\S', chunk))
    if kv >= 2:
        score += 0.1

    # List items
    li = len(re.findall(r'(?:^|\n)\s*(?:[-*]|\d{1,3}[.)]\s)', chunk))
    if li >= 3:
        score += 0.1

    # Positional: first chunk has title/abstract context
    if idx == 0:
        score += 0.15
    elif idx == total - 1 and total > 2:
        score += 0.05

    # Length penalty
    if raw_len < 150:
        score -= 0.3
    elif raw_len < 500:
        score -= 0.1

    return max(min(score, 1.0), 0.0)


def chunk_text(
    text: str,
    max_chars: int | None = None,
    overlap: int = 500,
) -> list[str]:
    """
    Split text into chunks, respecting section boundaries.

    Strategy:
      1. Detect section headers (numbered, ALL CAPS, markdown #)
      2. Split at section boundaries first
      3. Merge small sections; sub-split large sections at paragraph boundaries
      4. Apply overlap only at sub-splits (not at natural section boundaries)

    Args:
        text      — input text (will be cleaned first)
        max_chars — max characters per chunk (default: cfg.MAX_TEXT_CHARS)
        overlap   — characters of overlap when sub-splitting large sections

    Returns:
        List of text chunks.
    """
    max_chars = max_chars or cfg.MAX_TEXT_CHARS
    text      = _clean_text(text)

    if len(text) <= max_chars:
        return [text]

    # Detect section boundaries
    section_pattern = re.compile(
        r'\n(?='
        r'(?:\d{1,3}(?:\.\d{1,3})*\.?\s+[A-Z])'
        r'|(?:[A-Z][A-Z\s]{4,}(?:\n|$))'
        r'|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*\n[-=]{3,})'
        r'|(?:#{1,4}\s+\S)'
        r')'
    )
    section_starts = [0] + [m.start() + 1 for m in section_pattern.finditer(text)]
    sections = []
    for i, start in enumerate(section_starts):
        end = section_starts[i + 1] if i + 1 < len(section_starts) else len(text)
        sec = text[start:end].strip()
        if sec:
            sections.append(sec)

    chunks: list[str] = []
    current = ""

    for section in sections:
        if current and len(current) + len(section) + 2 > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = ""

        if len(section) > max_chars:
            if current.strip():
                chunks.append(current.strip())
                current = ""

            pos = 0
            while pos < len(section):
                end = pos + max_chars
                sub = section[pos:end]
                if end < len(section):
                    pb = sub.rfind("\n\n")
                    if pb > max_chars * 0.4:
                        sub = sub[:pb]
                        end = pos + pb
                    else:
                        sb = sub.rfind(". ")
                        if sb > max_chars * 0.4:
                            sub = sub[:sb + 1]
                            end = pos + sb + 1
                if sub.strip():
                    chunks.append(sub.strip())
                pos = end - overlap if end < len(section) else end
        else:
            current = (current + "\n\n" + section) if current else section

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text[:max_chars]]


def select_best_chunks(
    chunks: list[str],
    columns: list[str],
    max_chunks: int,
    llm_client: Optional["LLMClient"] = None,
) -> list[str]:
    """
    Select the most relevant chunks for extraction.

    Uses a hybrid scoring approach:
      65% semantic similarity (embeddings, if available)
      35% structural score (table density, numeric richness, position)

    Falls back to structural-only scoring when embeddings are unavailable.

    Adjacency boost: if chunk i is selected, neighbours i±1 are boosted
    (they often contain continuation rows from the same table).

    Duplicate penalty: non-adjacent chunks with >85% embedding similarity
    are de-duplicated (same table printed twice, common in some journals).

    Args:
        chunks      — list of text chunks from chunk_text()
        columns     — column names to extract (used to build the query)
        max_chunks  — maximum number of chunks to return
        llm_client  — LLMClient instance for embeddings (optional)

    Returns:
        Ordered list of selected chunks (original document order preserved).
    """
    n = len(chunks)
    if n <= max_chunks:
        return chunks

    struct_scores = [_structural_score(chunks[i], i, n) for i in range(n)]

    # ── Embedding-based scoring ──────────────────────────────────────────────
    embeddings = None
    if llm_client is not None:
        query = "Extract structured data for these fields: " + ", ".join(columns)
        all_texts = [query] + chunks
        embeddings = llm_client.embed(all_texts)

    if embeddings is not None:
        query_emb  = embeddings[0]
        chunk_embs = embeddings[1:]
        sem_scores = [
            LLMClient.cosine_similarity(query_emb, ce) for ce in chunk_embs
        ]

        hybrid: list[float] = []
        for i in range(n):
            if struct_scores[i] == -1.0:
                hybrid.append(-1.0)
            else:
                hybrid.append(0.65 * sem_scores[i] + 0.35 * struct_scores[i])

        # Dynamic threshold: 70% of median positive score
        positives = [s for s in hybrid if s > 0]
        threshold = (sorted(positives)[len(positives) // 2] * 0.70) if positives else 0.0

        selected: set[int] = {0}  # always include first chunk
        for i in range(n):
            if hybrid[i] >= threshold and hybrid[i] != -1.0:
                selected.add(i)

        # Adjacency boost
        for i in list(selected):
            for nb in [i - 1, i + 1]:
                if 0 <= nb < n and nb not in selected and hybrid[nb] != -1.0:
                    if hybrid[nb] >= threshold * 0.5:
                        selected.add(nb)

        # De-duplicate non-adjacent chunks with high embedding similarity
        if len(selected) > max_chunks:
            sel_list = sorted(selected)
            to_remove: set[int] = set()
            for ai, ia in enumerate(sel_list):
                if ia in to_remove:
                    continue
                for ib in sel_list[ai + 1:]:
                    if ib in to_remove:
                        continue
                    if abs(ia - ib) > 1:
                        sim = LLMClient.cosine_similarity(chunk_embs[ia], chunk_embs[ib])
                        if sim > 0.85:
                            if hybrid[ia] < hybrid[ib]:
                                to_remove.add(ia)
                            else:
                                to_remove.add(ib)
            selected -= to_remove

        # Trim to budget
        if len(selected) > max_chunks:
            selected = set(sorted(selected, key=lambda i: hybrid[i], reverse=True)[:max_chunks])

        # Fill budget from remaining
        if len(selected) < max_chunks:
            remaining = sorted(
                [(i, hybrid[i]) for i in range(n)
                 if i not in selected and hybrid[i] != -1.0],
                key=lambda x: x[1], reverse=True
            )
            for idx, _ in remaining:
                if len(selected) >= max_chunks:
                    break
                selected.add(idx)

    else:
        # Structural fallback
        selected = {0}
        for i in range(n):
            if struct_scores[i] > 0.1 and struct_scores[i] != -1.0:
                selected.add(i)
        for i in list(selected):
            for nb in [i - 1, i + 1]:
                if 0 <= nb < n and struct_scores[nb] != -1.0:
                    selected.add(nb)
        if len(selected) > max_chunks:
            selected = set(sorted(selected, key=lambda i: struct_scores[i], reverse=True)[:max_chunks])

    return [chunks[i] for i in sorted(selected)]


# ── Private helpers ───────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Clean text before chunking."""
    text = re.sub(r'\[\s*\n\s*([A-Za-z0-9]+)\s*\n\s*\]', r'[\1]', text)
    text = re.sub(r'\[\s*\n\s*\]', '', text)
    text = re.sub(r'\[\s*\d+(?:[\s,\-]*\d+)*\s*\]', '', text)
    text = re.sub(r'\(\s*[A-Za-z]+ et al\.,?\s*\d{4}\s*\)', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'doi:\s*10\.\S+', '', text, flags=re.I)
    lines = [l for l in text.split('\n')
             if not (len(l.strip()) < 3 and l.strip().isdigit())]
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()