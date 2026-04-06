"""
tools/browser.py — Web scraping engine.

Strategy (tried in order):
  1. Playwright (headless Chromium with stealth settings) — handles JS-rendered
     pages, cookie banners, and most soft paywalls.
  2. requests + BeautifulSoup — fast fallback for simple HTML pages.
  3. Direct PDF download — if the URL ends in .pdf or Content-Type is PDF.

Paywall detection:
  Hard paywalls (Nature, Science, Elsevier full-text, etc.) are detected via
  URL patterns and HTML signals injected by the Orchestrator.  Paywalled pages
  trigger the open-access fallback logic (PubMed Central, bioRxiv, etc.).

Deduplication:
  PDF content is hashed (SHA-256) on download.  Duplicate papers reachable
  from two different URLs are skipped the second time.

Usage:
    from tools.browser import BrowserPool
    pool = BrowserPool()
    pool.start()
    result = pool.scrape("https://example.com/paper.html")
    pool.stop()
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from rich.console import Console

import config as cfg

console = Console()

# ── PDF deduplication registry (per-run) ─────────────────────────────────────
_SEEN_PDF_HASHES: set[str] = set()


# ── Open-access fallback domains ──────────────────────────────────────────────
# When a page is paywalled, we try to find the same paper on these free hosts.
OA_FALLBACK_PATTERNS = [
    # PubMed Central
    ("europepmc.org", "https://europepmc.org/search?query={title}"),
    ("ncbi.nlm.nih.gov/pmc", "https://www.ncbi.nlm.nih.gov/pmc/search/?term={title}"),
    # Preprint servers
    ("arxiv.org", "https://arxiv.org/search/?searchtype=all&query={title}"),
    ("biorxiv.org", "https://www.biorxiv.org/search/{title}"),
    ("medrxiv.org", "https://www.medrxiv.org/search/{title}"),
    # Institutional
    ("semanticscholar.org", "https://api.semanticscholar.org/graph/v1/paper/search?query={title}&fields=openAccessPdf"),
]

# URL fragments that indicate non-content pages — always skip
JUNK_URL_PATTERNS = [
    "/tag/", "/category/", "/blog/", "/news/page/",
    "/about", "/contact", "/login", "/signup", "/cart",
    "/product/", "/shop/", "/search?", "/topic/", "/author/",
    "/press-release/", "/advertise", "/subscribe",
]

# Default paywall HTML signals (supplemented by Orchestrator's LLM-generated list)
DEFAULT_PAYWALL_SIGNALS = [
    "subscribe to read", "purchase access", "buy this article",
    "log in to read", "get full access", "institutional login",
    "full text unavailable", "view access options",
    "purchase pdf", "rent this article",
]

# Mutable — updated by configure_paywall_detection()
_PAYWALL_DOMAIN_PATTERNS: list[str] = []
_PAYWALL_HTML_SIGNALS: list[str] = list(DEFAULT_PAYWALL_SIGNALS)


def configure_paywall_detection(
    domain_patterns: list[str],
    html_signals: list[str],
) -> None:
    """
    Inject LLM-generated paywall config.
    Called once per session by the Orchestrator after topic analysis.
    """
    global _PAYWALL_DOMAIN_PATTERNS, _PAYWALL_HTML_SIGNALS
    _PAYWALL_DOMAIN_PATTERNS = [p.lower() for p in domain_patterns]
    _PAYWALL_HTML_SIGNALS = list(DEFAULT_PAYWALL_SIGNALS) + [
        s.lower() for s in html_signals
    ]
    console.print(
        f"  [dim]Paywall config: {len(_PAYWALL_DOMAIN_PATTERNS)} domain patterns, "
        f"{len(_PAYWALL_HTML_SIGNALS)} HTML signals[/dim]"
    )


def _is_junk_url(url: str) -> bool:
    u = url.lower()
    return any(p in u for p in JUNK_URL_PATTERNS)


def _is_hard_paywalled(url: str, html: str = "") -> bool:
    """True if the URL is on a known paywall domain OR the HTML shows a wall."""
    u = url.lower()
    if any(p in u for p in _PAYWALL_DOMAIN_PATTERNS):
        return True
    if html:
        h = html.lower()
        if any(s in h for s in _PAYWALL_HTML_SIGNALS):
            return True
    return False


def _register_pdf_hash(pdf_path: str) -> bool:
    """Return True if this PDF is a duplicate (already seen this session)."""
    try:
        with open(pdf_path, "rb") as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        if h in _SEEN_PDF_HASHES:
            console.print("  [dim]⊘ Duplicate PDF (identical content) — skipping[/dim]")
            return True
        _SEEN_PDF_HASHES.add(h)
        return False
    except Exception:
        return False


# ── Playwright browser pool ───────────────────────────────────────────────────

class BrowserPool:
    """
    Manages a single Playwright browser instance shared across all scrape calls.

    The browser is launched once and reused to avoid the overhead of launching
    a new Chrome instance for every URL.  Pages are opened and closed per scrape.
    """

    def __init__(self) -> None:
        self._playwright = None
        self._browser    = None
        self._started    = False

    def start(self) -> None:
        """Launch the browser.  Safe to call multiple times."""
        if self._started:
            return
        try:
            from playwright.sync_api import sync_playwright
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                ],
            )
            self._started = True
            console.print("[dim green]✓ Browser ready[/dim green]")
        except Exception as exc:
            console.print(f"[yellow]⚠ Browser launch failed ({exc}) — will use requests fallback[/yellow]")

    def stop(self) -> None:
        """Close the browser cleanly."""
        try:
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._started = False

    def _stealth_page(self):
        """Create a new page with stealth settings to avoid bot detection."""
        ctx = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            timezone_id="America/New_York",
        )
        page = ctx.new_page()
        # Remove navigator.webdriver flag (main bot-detection vector)
        page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )
        return ctx, page

    def scrape(self, url: str) -> Optional[dict]:
        """
        Scrape a URL and return a result dict or None.

        Result dict keys:
            url     — final URL (may differ from input if redirected)
            text    — extracted plain text
            html    — raw HTML (empty for PDFs)
            method  — "playwright" | "requests" | "pdf" | "local_pdf"
            pdf_path — path to downloaded PDF (if applicable)
        """
        if _is_junk_url(url):
            console.print(f"  [dim]✗ Junk URL pattern — skipping[/dim]")
            return None

        # ── Try Playwright ───────────────────────────────────────────────────
        if self._started and self._browser:
            result = self._scrape_playwright(url)
            if result:
                return result

        # ── Requests fallback ────────────────────────────────────────────────
        result = self._scrape_requests(url)
        return result

    def _scrape_playwright(self, url: str) -> Optional[dict]:
        """Playwright-based scrape with human-like delays."""
        ctx = page = None
        try:
            ctx, page = self._stealth_page()
            page.set_default_timeout(cfg.BROWSER_TIMEOUT_MS)

            response = page.goto(url, wait_until="domcontentloaded")
            if response is None:
                return None

            # Handle PDF served as HTTP response
            content_type = response.headers.get("content-type", "")
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                return self._download_pdf(url)

            # Human-like random pause
            time.sleep(random.uniform(0.5, 1.5))

            # Scroll to trigger lazy-loading
            page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            time.sleep(random.uniform(0.3, 0.8))

            html = page.content()

            # Paywall check
            if _is_hard_paywalled(url, html):
                console.print(f"  [yellow]⚠ Paywall detected: {url[:70]}[/yellow]")
                return None

            text = _html_to_text(html)
            if not text or len(text.strip()) < 200:
                return None

            return {"url": url, "text": text, "html": html,
                    "method": "playwright", "pdf_path": None}

        except Exception as exc:
            if cfg.DEBUG:
                console.print(f"  [dim red]Playwright error: {exc}[/dim red]")
            return None
        finally:
            try:
                if page:
                    page.close()
                if ctx:
                    ctx.close()
            except Exception:
                pass

    def _scrape_requests(self, url: str) -> Optional[dict]:
        """Lightweight requests-based scrape."""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            resp.raise_for_status()

            ct = resp.headers.get("content-type", "")
            if "pdf" in ct or url.lower().endswith(".pdf"):
                return self._download_pdf(url, content=resp.content)

            if _is_hard_paywalled(url, resp.text):
                return None

            text = _html_to_text(resp.text)
            if not text or len(text.strip()) < 200:
                return None

            return {"url": url, "text": text, "html": resp.text,
                    "method": "requests", "pdf_path": None}

        except Exception as exc:
            if cfg.DEBUG:
                console.print(f"  [dim red]requests error ({url[:60]}): {exc}[/dim red]")
            return None

    def _download_pdf(
        self,
        url: str,
        content: bytes | None = None,
    ) -> Optional[dict]:
        """Download a PDF, deduplicate by hash, and extract its text."""
        try:
            if content is None:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                content = resp.content

            # Save to temp dir
            stem = re.sub(r"[^\w\-]", "_", url[-40:])
            pdf_path = str(Path(cfg.PDF_TEMP_DIR) / f"{stem}.pdf")
            Path(pdf_path).write_bytes(content)

            # Deduplicate by content hash
            if _register_pdf_hash(pdf_path):
                return None

            text = extract_pdf_text(pdf_path)
            if not text or len(text.strip()) < 200:
                return None

            console.print(f"  [green]✓ PDF downloaded & extracted ({len(text):,} chars)[/green]")
            return {"url": url, "text": text, "html": "",
                    "method": "pdf", "pdf_path": pdf_path}

        except Exception as exc:
            console.print(f"  [dim red]PDF download failed: {exc}[/dim red]")
            return None

    def scrape_local_pdf(self, path: str) -> Optional[dict]:
        """Process a local PDF file (not from the web)."""
        if not os.path.isfile(path):
            console.print(f"  [red]✗ File not found: {path}[/red]")
            return None
        console.print(f"  [cyan]📄 Local PDF: {os.path.basename(path)}[/cyan]")
        text = extract_pdf_text(path)
        if not text or len(text.strip()) < 100:
            console.print("  [yellow]Text too short — trying OCR[/yellow]")
            from tools.pdf_tools import extract_pdf_text_ocr
            text = extract_pdf_text_ocr(path)
        if not text:
            return None
        console.print(f"  [green]✓ Local PDF extracted ({len(text):,} chars)[/green]")
        return {"url": path, "text": text, "html": "",
                "method": "local_pdf", "pdf_path": path}

    def find_open_access_fallback(self, original_url: str, title: str = "") -> list[str]:
        """
        Given a paywalled URL (and optionally the paper title), return
        a list of potential open-access URLs to try.

        Strategy:
          1. Replace known publisher domains with PubMed Central equivalent
          2. Search Semantic Scholar for open-access PDF link
          3. Try unpaywall.org API (no key needed, email required)
        """
        fallbacks: list[str] = []

        # Direct domain swaps
        domain_map = {
            "nature.com":       "europepmc.org",
            "science.org":      "europepmc.org",
            "cell.com":         "europepmc.org",
            "thelancet.com":    "europepmc.org",
            "nejm.org":         "europepmc.org",
            "springer.com":     "link.springer.com",
        }
        parsed = urlparse(original_url)
        for publisher, oa_host in domain_map.items():
            if publisher in parsed.netloc:
                doi_match = re.search(r"10\.\d{4,}/\S+", original_url)
                if doi_match:
                    doi = doi_match.group()
                    fallbacks.append(f"https://europepmc.org/search?query={doi}")
                break

        # Semantic Scholar open-access PDF
        if title:
            safe_title = requests.utils.quote(title[:100])
            fallbacks.append(
                f"https://api.semanticscholar.org/graph/v1/paper/search"
                f"?query={safe_title}&fields=openAccessPdf,title&limit=3"
            )

        return fallbacks


# ── HTML → plain text ─────────────────────────────────────────────────────────

def _html_to_text(html: str) -> str:
    """
    Convert HTML to clean plain text.

    Removes:
      - Script, style, nav, header, footer, aside tags
      - Inline citations [1], (Smith et al., 2020)
      - URLs and DOI noise
      - Standalone page/ref numbers
      - Excessive whitespace
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove noisy structural elements
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "noscript", "iframe", "figure"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Clean up extracted text
    text = re.sub(r'\[\s*\d+(?:[\s,\-]*\d+)*\s*\]', '', text)   # [1] [1,2]
    text = re.sub(r'\(\s*[A-Za-z]+ et al\.,?\s*\d{4}\s*\)', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'doi:\s*10\.\S+', '', text, flags=re.I)
    text = re.sub(r'\[\s*(?:DOI|PMC free article|PubMed|Google Scholar)\s*\]', '', text, flags=re.I)

    lines, cleaned = text.split('\n'), []
    for line in lines:
        s = line.strip()
        if len(s) < 3 and s.isdigit():   # standalone page number
            continue
        cleaned.append(line)

    text = '\n'.join(cleaned)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz).

    Falls back to OCR (via tools/pdf_tools.py) if the extracted text
    is suspiciously short (scanned PDF).
    """
    try:
        doc = fitz.open(pdf_path)
        parts = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        text = "\n\n".join(parts)

        # If text is too short, it's probably a scanned PDF
        if cfg.ENABLE_OCR and len(text.strip()) < 500:
            console.print("  [dim]Text sparse — attempting OCR[/dim]")
            from tools.pdf_tools import extract_pdf_text_ocr
            ocr_text = extract_pdf_text_ocr(pdf_path)
            if len(ocr_text) > len(text):
                return ocr_text

        return text.strip()
    except Exception as exc:
        console.print(f"  [red]PDF text extraction failed: {exc}[/red]")
        return ""


# ── Supplement link finder ────────────────────────────────────────────────────

def find_supplement_links(html: str, base_url: str) -> list[dict]:
    """
    Find links to supplementary data files in a scraped HTML page.

    Returns a list of dicts: {url, filename, type, text}
    """
    soup = BeautifulSoup(html, "lxml")
    results, seen = [], set()
    DATA_EXTS    = {".csv", ".xlsx", ".xls", ".zip", ".tsv", ".json", ".xml"}
    SUP_KEYWORDS = {"supplement", "supporting", "additional", "appendix",
                    "data", "table s", "figure s", "dataset", "supplementary"}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        ext  = Path(href).suffix.lower()

        if (any(k in text for k in SUP_KEYWORDS) or ext in DATA_EXTS) and href not in seen:
            seen.add(href)
            if not href.startswith("http"):
                href = urljoin(base_url, href)
            results.append({
                "url":      href,
                "filename": Path(href).name,
                "type":     ext,
                "text":     a.get_text(strip=True)[:100],
            })
    return results