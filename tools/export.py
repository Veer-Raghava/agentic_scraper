"""Export and preview helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from state import SessionState

console = Console()


def _rows_to_df(state: SessionState) -> pd.DataFrame:
    cols = state.column_names() or sorted({k for r in state.rows for k in r.data.keys()})
    records = []
    for r in state.rows:
        rec = {c: r.data.get(c, "N/A") for c in cols}
        rec["Source URL"] = r.source_url or rec.get("Source URL", "")
        records.append(rec)
    return pd.DataFrame(records)


def save_csv(state: SessionState) -> str:
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if state.dataset_dir:
        out = Path(state.dataset_dir) / "output.csv"
    else:
        out = Path(cfg.OUTPUT_DIR) / f"dataset_{state.session_id}.csv"
    df = _rows_to_df(state)
    df.to_csv(out, index=False)
    return str(out)


def save_xlsx(state: SessionState) -> str:
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if state.dataset_dir:
        out = Path(state.dataset_dir) / "output.xlsx"
    else:
        out = Path(cfg.OUTPUT_DIR) / f"dataset_{state.session_id}.xlsx"
    df = _rows_to_df(state)
    df.to_excel(out, index=False)
    return str(out)


def print_preview(state: SessionState, max_rows: int = 8) -> None:
    if not state.rows:
        console.print("[yellow]No rows to preview.[/yellow]")
        return
    cols = state.column_names()
    table = Table(title=f"Preview ({min(max_rows, len(state.rows))}/{len(state.rows)})")
    for c in cols[:8]:
        table.add_column(c)
    for r in state.rows[:max_rows]:
        table.add_row(*[str(r.data.get(c, "N/A"))[:50] for c in cols[:8]])
    console.print(table)


def print_fill_report(state: SessionState) -> None:
    if not state.rows:
        return
    cols = state.column_names()
    table = Table(title="Fill Report")
    table.add_column("Row")
    table.add_column("Fill %")
    table.add_column("Missing")
    for i, r in enumerate(state.rows, start=1):
        missing = r.missing_fields(state.columns)
        table.add_row(str(i), f"{r.fill_rate(state.columns)*100:.0f}%", ", ".join(missing[:6]))
    console.print(table)
