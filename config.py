"""
config.py — Central configuration for the Agentic Scraper.

All settings are loaded from environment variables or a .env file.
Sensible defaults are provided for every setting so the system works
out-of-the-box with just an OPENAI_API_KEY.

Hierarchy (highest → lowest priority):
  1. Environment variable already set in the shell
  2. Value in .env file
  3. Default coded here
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Load .env silently — no error if the file is missing
load_dotenv(override=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM  (OpenAI is the only provider in this version)
# ═══════════════════════════════════════════════════════════════════════════════

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Model used for all extraction, orchestration, and critic tasks.
# gpt-4o gives the best structured-output quality; gpt-4o-mini is 10× cheaper
# but occasionally misses nuance in complex prompts.
EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "gpt-4o")

# Cheaper model for lighter tasks: query generation, column suggestion, etc.
FAST_MODEL: str = os.getenv("FAST_MODEL", "gpt-4o-mini")

# Embedding model for semantic chunk ranking
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Sampling temperature for all LLM calls.
# 0.0 = fully deterministic (best for extraction).
# Raise to ~0.3 if you want more creative column suggestions.
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

# Hard cap on tokens the LLM may return per extraction call.
MAX_TOKENS_EXTRACTION: int = int(os.getenv("MAX_TOKENS_EXTRACTION", "4096"))
MAX_TOKENS_FAST: int = int(os.getenv("MAX_TOKENS_FAST", "1024"))


# ═══════════════════════════════════════════════════════════════════════════════
# Scraping
# ═══════════════════════════════════════════════════════════════════════════════

# Max sources to scrape per pipeline run (can be overridden per-session)
DEFAULT_SOURCE_LIMIT: int = int(os.getenv("SOURCE_LIMIT", "10"))

# Workers for parallel scraping (keep ≤ 3 to avoid rate-limiting)
DEFAULT_WORKERS: int = int(os.getenv("SCRAPE_WORKERS", "2"))

# Seconds to wait between requests (min, max) — polite crawling
REQUEST_DELAY_MIN: int = int(os.getenv("REQUEST_DELAY_MIN", "2"))
REQUEST_DELAY_MAX: int = int(os.getenv("REQUEST_DELAY_MAX", "5"))
REQUEST_DELAY: tuple[int, int] = (REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)

# A scraped page shorter than this is assumed to be abstract-only / paywalled
MIN_FULLTEXT_CHARS: int = int(os.getenv("MIN_FULLTEXT_CHARS", "3000"))

# Max characters sent to the LLM per extraction call
MAX_TEXT_CHARS: int = int(os.getenv("MAX_TEXT_CHARS", "80000"))

# Max retry attempts when a scrape fails
MAX_SCRAPE_RETRIES: int = int(os.getenv("MAX_SCRAPE_RETRIES", "2"))

# Playwright browser timeout in milliseconds
BROWSER_TIMEOUT_MS: int = int(os.getenv("BROWSER_TIMEOUT_MS", "30000"))


# ═══════════════════════════════════════════════════════════════════════════════
# Critic / Refiner thresholds
# ═══════════════════════════════════════════════════════════════════════════════

# A row is flagged for refinement when this fraction of its fields are N/A.
# 0.4 = "if 40 % of fields are empty, try to fill them"
REFINEMENT_THRESHOLD: float = float(os.getenv("REFINEMENT_THRESHOLD", "0.4"))

# Max targeted web searches the Refiner may fire per row
MAX_REFINE_SEARCHES_PER_ROW: int = int(os.getenv("MAX_REFINE_SEARCHES_PER_ROW", "3"))

# After this many full refinement passes, stop (prevents infinite loops)
MAX_REFINEMENT_ROUNDS: int = int(os.getenv("MAX_REFINEMENT_ROUNDS", "2"))


# ═══════════════════════════════════════════════════════════════════════════════
# OCR
# ═══════════════════════════════════════════════════════════════════════════════

ENABLE_OCR: bool = os.getenv("ENABLE_OCR", "true").lower() in ("true", "1", "yes")
TESSERACT_CMD: str = (
    os.getenv("TESSERACT_CMD")
    or shutil.which("tesseract")
    or "tesseract"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR: str    = os.getenv("OUTPUT_DIR", "data/outputs")
PDF_TEMP_DIR: str  = os.getenv("PDF_DIR",    "data/pdfs")
CHUNK_DIR: str     = os.getenv("CHUNK_DIR",  "data/chunks")
SESSION_DIR: str   = os.getenv("SESSION_DIR","data/sessions")

# Auto-create all directories on import so nothing crashes at runtime
for _d in [OUTPUT_DIR, PDF_TEMP_DIR, CHUNK_DIR, SESSION_DIR]:
    Path(_d).mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Debug
# ═══════════════════════════════════════════════════════════════════════════════

DEBUG: bool = os.getenv("DEBUG", "").lower() in ("true", "1", "yes")


# ═══════════════════════════════════════════════════════════════════════════════
# Validation helper
# ═══════════════════════════════════════════════════════════════════════════════

def validate() -> None:
    """
    Call at startup to catch misconfiguration early.
    Raises ValueError with a clear message if the API key is missing.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set.\n"
            "  1. Create a .env file in the project root\n"
            "  2. Add:  OPENAI_API_KEY=sk-...\n"
            "  3. Re-run main.py"
        )