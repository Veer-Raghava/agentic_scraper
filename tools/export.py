"""
tools/export.py — Data export utilities.

Handles:
  • CSV export (UTF-8 with BOM for Excel compatibility)
  • XLSX export with column auto-sizing
  • Rich console table preview
  • Incremental/partial save (called after every successful source)

Usage:
    from tools.export import save_csv, save_xlsx, print_preview
    path = save_csv(state)
    save_xlsx(state, path.replace(".csv", ".xlsx"))
    print_preview(state, max_rows=8)
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

import config as cfg
from state import SessionState

console = Console()


# ── DataFrame builder ─────────────────────────────────────────────────────────

def state_to_dataframe(state: SessionState) -> pd.DataFrame:
    """
    Convert session rows to a pandas DataFrame.

    Column order matches state.columns.  Missing columns are filled with N/A.
    Source_URL and Dataset_Links are always appended at the end if present.
    """
    if not state.rows:
        return pd.DataFrame(columns=state.column_names())

    col_names = state.column_names()
    records   = []

    for row in state.rows:
        record = {}
        for col in col_names:
            record[col] = row.data.get(col, "N/A")

        # Always append source URL
        record["Source_URL"] = row.source_url

        # Dataset links if present
        if "Dataset_Links" in row.data:
            record["Dataset_Links"] = row.data["Dataset_Links"]

        records.append(record)

    df = pd.DataFrame(records)

    # Ensure column order: user columns → Source_URL → Dataset_Links
    ordered = col_names.copy()
    if "Source_URL" not in ordered:
        ordered.append("Source_URL")
    if "Dataset_Links" in df.columns and "Dataset_Links" not in ordered:
        ordered.append("Dataset_Links")

    extra = [c for c in df.columns if c not in ordered]
    final_cols = ordered + extra

    for c in final_cols:
        if c not in df.columns:
            df[c] = "N/A"

    return df[final_cols]


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_csv(state: SessionState, path: str | None = None) -> str:
    """
    Save session data to a CSV file.

    Args:
        state — the session state to export
        path  — output path (auto-generated if None)

    Returns:
        The path that was written.
    """
    if path is None:
        safe_topic = "".join(c if c.isalnum() or c in " _-" else "_"
                             for c in state.topic[:40]).strip()
        filename = f"{safe_topic}_{state.session_id}.csv"
        path = str(Path(cfg.OUTPUT_DIR) / filename)

    df = state_to_dataframe(state)

    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        # File is open in Excel — add timestamp to avoid collision
        stem, ext = os.path.splitext(path)
        path = f"{stem}_{datetime.now().strftime('%H%M%S')}{ext}"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        console.print(f"[yellow]⚠ Original locked — saved as {os.path.basename(path)}[/yellow]")

    console.print(f"[green]✓ CSV saved → {path}[/green]")
    return path


# ── XLSX ──────────────────────────────────────────────────────────────────────

def save_xlsx(state: SessionState, path: str | None = None) -> str:
    """
    Save session data to an XLSX file with auto-sized columns.

    Args:
        state — the session state to export
        path  — output path (auto-generated if None)

    Returns:
        The path that was written.
    """
    if path is None:
        safe_topic = "".join(c if c.isalnum() or c in " _-" else "_"
                             for c in state.topic[:40]).strip()
        filename = f"{safe_topic}_{state.session_id}.xlsx"
        path = str(Path(cfg.OUTPUT_DIR) / filename)

    df = state_to_dataframe(state)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
        ws = writer.sheets["Results"]

        # Auto-size columns (max 60 chars wide)
        for column_cells in ws.columns:
            max_len = max(
                (len(str(cell.value)) if cell.value else 0)
                for cell in column_cells
            )
            ws.column_dimensions[column_cells[0].column_letter].width = min(
                max_len + 4, 60
            )

    console.print(f"[green]✓ XLSX saved → {path}[/green]")
    return path


# ── Console preview ───────────────────────────────────────────────────────────

def print_preview(state: SessionState, max_rows: int = 8) -> None:
    """
    Print a rich table preview of the current results to the console.

    Shows up to max_rows rows and up to 7 columns (to fit the terminal).
    Fill rate per row is shown as a colour-coded indicator.
    """
    if not state.rows:
        console.print("[dim]No rows to preview yet.[/dim]")
        return

    df        = state_to_dataframe(state)
    col_names = state.column_names()
    preview_cols = col_names[:7]  # cap at 7 columns for readability

    tbl = Table(
        title=f"Results preview — {len(df)} row(s) | topic: {state.topic[:50]}",
        show_lines=True,
        expand=False,
    )

    # Header columns
    tbl.add_column("Fill", style="dim", width=5)
    for col in preview_cols:
        tbl.add_column(str(col), max_width=28, overflow="ellipsis")

    # Data rows
    for i, (_, dfrow) in enumerate(df.head(max_rows).iterrows()):
        if i >= len(state.rows):
            break
        row   = state.rows[i]
        fill  = row.fill_rate(state.columns)

        if fill >= 0.8:
            fill_str = "[green]●●●[/green]"
        elif fill >= 0.5:
            fill_str = "[yellow]●●○[/yellow]"
        else:
            fill_str = "[red]●○○[/red]"

        values = [str(dfrow.get(c, "N/A"))[:28] for c in preview_cols]
        tbl.add_row(fill_str, *values)

    console.print(tbl)

    if len(state.columns) > 7:
        console.print(
            f"[dim]…and {len(state.columns) - 7} more columns not shown. "
            "Export to CSV/XLSX to see all.[/dim]"
        )


# ── Fill-rate summary ─────────────────────────────────────────────────────────

def print_fill_report(state: SessionState) -> None:
    """
    Print a per-column fill rate report.
    Useful after refinement to see which fields still have gaps.
    """
    if not state.rows:
        return

    col_names = state.column_names()
    console.print("\n[bold]Field fill rates:[/bold]")

    for col_def in state.columns:
        empty = {"n/a", "none", "not specified", "not mentioned",
                 "not provided", "unspecified", "unknown", "na", "",
                 "not available", "not found"}
        filled = sum(
            1 for r in state.rows
            if str(r.data.get(col_def.name, "N/A")).strip().lower() not in empty
        )
        rate = filled / len(state.rows)
        bar  = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))

        color = "green" if rate >= 0.8 else ("yellow" if rate >= 0.5 else "red")
        console.print(
            f"  [{color}]{bar}[/{color}] {rate:.0%}  {col_def.name}"
        )