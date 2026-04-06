"""
state.py — Shared session state passed between all agents.

Every agent receives a reference to ONE SessionState object that lives for the
entire interactive session.  This is the single source of truth for:

  • What the user asked for
  • Which columns to extract and what they mean
  • Every row of data extracted so far
  • Which sources have been processed
  • The conversation history (for the chat loop)
  • The gap registry (fields still marked N/A after the first pass)

Design principles:
  - Plain Python dataclasses — no framework dependency.
  - Agents mutate state in-place; the orchestrator orchestrates, not coordinates.
  - State can be serialized to JSON at any point for resumability.
"""

from __future__ import annotations

import json
import time
import uuid
import re
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config as cfg


# ── Column definition ──────────────────────────────────────────────────────────

@dataclass
class ColumnDef:
    """
    Represents one column the user wants to extract.

    name        – display name as the user stated it (e.g. "Sample Size")
    description – precise, agent-written definition of exactly what value to
                  pull from a source.  This goes directly into every LLM prompt
                  so it must be specific and unambiguous.
    example     – a realistic example value (e.g. "n=342 participants").
                  Shown to the LLM as a few-shot hint.
    required    – if True, Critic will always try to fill this even when it
                  is N/A (use for identity columns like title, DOI).
    """
    name: str
    description: str = ""
    example: str = ""
    required: bool = False

    def to_prompt_line(self) -> str:
        """
        Format for injection into LLM extraction prompts.
        Example output:
            "Sample Size": number of subjects/samples in the study (e.g. n=342).
        """
        parts = [f'"{self.name}": {self.description}']
        if self.example:
            parts.append(f"(e.g. {self.example})")
        return " ".join(parts)


# ── Per-row data ───────────────────────────────────────────────────────────────

@dataclass
class ExtractedRow:
    """
    One row of structured data extracted from a source.

    data            – dict mapping column name → extracted value string.
    source_url      – where the data came from.
    confidence      – dict mapping column name → float 0–1.
                      1.0 = LLM was confident; 0.0 = guessed or N/A.
    refinement_done – True once the Refiner has attempted to fill gaps.
    """
    data: dict[str, str]
    source_url: str = ""
    confidence: dict[str, float] = field(default_factory=dict)
    refinement_done: bool = False

    def missing_fields(self, columns: list[ColumnDef]) -> list[str]:
        """Return column names that are still N/A or empty."""
        empty = {"n/a", "none", "not specified", "not mentioned",
                 "not provided", "unspecified", "unknown", "na", "",
                 "not available", "not found", "not applicable",
                 "not stated", "not reported", "not described", "nil"}
        return [
            col.name for col in columns
            if str(self.data.get(col.name, "N/A")).strip().lower() in empty
        ]

    def fill_rate(self, columns: list[ColumnDef]) -> float:
        """Fraction of columns that have a real (non-N/A) value."""
        if not columns:
            return 0.0
        missing = len(self.missing_fields(columns))
        return 1.0 - missing / len(columns)

    def needs_refinement(self, columns: list[ColumnDef]) -> bool:
        """True if enough fields are missing to warrant a refinement pass."""
        return (
            not self.refinement_done
            and self.fill_rate(columns) < (1.0 - cfg.REFINEMENT_THRESHOLD)
        )


# ── Chat message ───────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    """One turn in the conversation history."""
    role: str       # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)


# ── Session state ──────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """
    The complete mutable state of one research session.

    One instance is created at startup and passed to every agent.
    Agents mutate it directly — there is no message-passing overhead.

    Lifecycle:
        1. main.py creates SessionState()
        2. Orchestrator fills topic, columns, sources
        3. Agents append to rows, mark sources as seen
        4. Critic/Refiner update rows in-place
        5. Export agent reads rows and writes CSV/XLSX
        6. State is auto-saved after every mutation so nothing is lost
    """

    # ── Identity ────────────────────────────────────────────────────────────
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    dataset_id: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))

    # ── What the user wants ─────────────────────────────────────────────────
    topic: str = ""                         # e.g. "antibody drug conjugates approved by FDA"
    columns: list[ColumnDef] = field(default_factory=list)
    source_limit: int = cfg.DEFAULT_SOURCE_LIMIT

    # ── Sources ─────────────────────────────────────────────────────────────
    # URLs/paths queued for scraping
    pending_sources: list[str] = field(default_factory=list)
    # URLs/paths already processed (don't re-scrape)
    processed_sources: set[str] = field(default_factory=set)
    # URLs known to be paywalled/empty — skip on retry
    dead_sources: set[str] = field(default_factory=set)

    # ── Extracted data ───────────────────────────────────────────────────────
    rows: list[ExtractedRow] = field(default_factory=list)

    # ── Conversation history ─────────────────────────────────────────────────
    history: list[ChatMessage] = field(default_factory=list)

    # ── Pipeline control ─────────────────────────────────────────────────────
    # Number of refinement rounds completed so far
    refinement_rounds: int = 0
    # Set to True by the orchestrator when the pipeline is running
    pipeline_running: bool = False
    # Set to True when the user asks to stop
    stop_requested: bool = False

    # ── Runtime flags ────────────────────────────────────────────────────────
    # The orchestrator sets these after interpreting the user's message
    user_wants_more_sources: bool = False
    user_wants_new_columns: list[str] = field(default_factory=list)
    user_wants_fix_row: int | None = None   # row index to fix

    # ── Column definitions generated/confirmed by the orchestrator ───────────
    # This is separate from `columns` so the orchestrator can propose
    # definitions without committing them until the user approves.
    pending_column_defs: list[ColumnDef] = field(default_factory=list)

    # ── Misc ─────────────────────────────────────────────────────────────────
    created_at: float = field(default_factory=time.time)
    last_saved: float = 0.0
    dataset_dir: str = ""
    pdf_dir: str = ""
    tables_dir: str = ""
    images_dir: str = ""
    supplementary_dir: str = ""
    live_csv_path: str = ""

    # ── Convenience methods ──────────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        self.history.append(ChatMessage(role=role, content=content))

    def add_row(self, row: ExtractedRow) -> None:
        """Append an extracted row."""
        self.rows.append(row)
        # Best-effort live append so users can observe progress in realtime.
        if self.live_csv_path:
            try:
                with open(self.live_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        int(time.time()),
                        row.source_url,
                        json.dumps(row.data, ensure_ascii=False),
                    ])
            except Exception:
                # Do not block extraction if live-write fails.
                pass

    def mark_processed(self, url: str) -> None:
        """Move a URL from pending to processed."""
        self.processed_sources.add(url)
        if url in self.pending_sources:
            self.pending_sources.remove(url)

    def mark_dead(self, url: str) -> None:
        """Mark a URL as permanently unusable."""
        self.dead_sources.add(url)
        self.mark_processed(url)

    def rows_needing_refinement(self) -> list[tuple[int, ExtractedRow]]:
        """Return (index, row) pairs that still have gaps."""
        return [
            (i, r) for i, r in enumerate(self.rows)
            if r.needs_refinement(self.columns)
        ]

    def column_names(self) -> list[str]:
        """Flat list of column names — convenience for DataFrame creation."""
        return [c.name for c in self.columns]

    def summary(self) -> str:
        """One-line summary for logging/display."""
        filled = [r.fill_rate(self.columns) for r in self.rows]
        avg_fill = sum(filled) / len(filled) if filled else 0.0
        return (
            f"Session {self.session_id} | topic='{self.topic[:40]}' | "
            f"{len(self.rows)} rows | avg fill={avg_fill:.0%} | "
            f"{len(self.processed_sources)} sources processed"
        )

    def ensure_dataset_dirs(self) -> None:
        """
        Create a per-run dataset folder and subfolders for extracted artifacts.
        Safe to call repeatedly.
        """
        if self.dataset_dir:
            return

        topic_slug = re.sub(r"[^a-zA-Z0-9]+", "_", (self.topic or "dataset")).strip("_").lower()
        topic_slug = topic_slug[:80] or "dataset"
        base_name = f"{self.dataset_id}_{topic_slug}"
        dataset_root = Path(cfg.OUTPUT_DIR) / base_name

        self.dataset_dir = str(dataset_root)
        self.pdf_dir = str(dataset_root / "pdfs")
        self.tables_dir = str(dataset_root / "tables")
        self.images_dir = str(dataset_root / "images")
        self.supplementary_dir = str(dataset_root / "supplementary")
        self.live_csv_path = str(dataset_root / "live_rows.csv")

        for d in [self.dataset_dir, self.pdf_dir, self.tables_dir, self.images_dir, self.supplementary_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # Initialize live CSV file once
        live_csv = Path(self.live_csv_path)
        if not live_csv.exists():
            live_csv.write_text("timestamp,source_url,row_json\n", encoding="utf-8")

    # ── Persistence ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "columns": [
                {
                    "name": c.name,
                    "description": c.description,
                    "example": c.example,
                    "required": c.required,
                }
                for c in self.columns
            ],
            "source_limit": self.source_limit,
            "dataset_id": self.dataset_id,
            "dataset_dir": self.dataset_dir,
            "pdf_dir": self.pdf_dir,
            "tables_dir": self.tables_dir,
            "images_dir": self.images_dir,
            "supplementary_dir": self.supplementary_dir,
            "live_csv_path": self.live_csv_path,
            "pending_sources": self.pending_sources,
            "processed_sources": list(self.processed_sources),
            "dead_sources": list(self.dead_sources),
            "rows": [
                {
                    "data": r.data,
                    "source_url": r.source_url,
                    "confidence": r.confidence,
                    "refinement_done": r.refinement_done,
                }
                for r in self.rows
            ],
            "history": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in self.history
            ],
            "refinement_rounds": self.refinement_rounds,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        """Deserialize from a saved dict."""
        s = cls(session_id=d["session_id"])
        s.topic = d.get("topic", "")
        s.columns = [
            ColumnDef(**col) for col in d.get("columns", [])
        ]
        s.source_limit = d.get("source_limit", cfg.DEFAULT_SOURCE_LIMIT)
        s.dataset_id = d.get("dataset_id", s.dataset_id)
        s.dataset_dir = d.get("dataset_dir", "")
        s.pdf_dir = d.get("pdf_dir", "")
        s.tables_dir = d.get("tables_dir", "")
        s.images_dir = d.get("images_dir", "")
        s.supplementary_dir = d.get("supplementary_dir", "")
        s.live_csv_path = d.get("live_csv_path", "")
        s.pending_sources = d.get("pending_sources", [])
        s.processed_sources = set(d.get("processed_sources", []))
        s.dead_sources = set(d.get("dead_sources", []))
        s.rows = [
            ExtractedRow(
                data=r["data"],
                source_url=r["source_url"],
                confidence=r.get("confidence", {}),
                refinement_done=r.get("refinement_done", False),
            )
            for r in d.get("rows", [])
        ]
        s.history = [
            ChatMessage(role=m["role"], content=m["content"],
                        timestamp=m.get("timestamp", 0.0))
            for m in d.get("history", [])
        ]
        s.refinement_rounds = d.get("refinement_rounds", 0)
        s.created_at = d.get("created_at", time.time())
        return s

    def save(self, path: str | None = None) -> str:
        """
        Save the session to disk as JSON.
        Returns the path written to.
        """
        if path is None:
            path = str(
                Path(cfg.SESSION_DIR) / f"session_{self.session_id}.json"
            )
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        self.last_saved = time.time()
        return path

    @classmethod
    def load(cls, path: str) -> "SessionState":
        """Load a previously saved session from disk."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
