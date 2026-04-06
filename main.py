"""
main.py — ARIA: Agentic Research & Intelligence Assistant

Entry point for the interactive chat-based data extraction system.

Usage:
    python main.py                      # interactive chat mode
    python main.py --resume session.json # resume a previous session
    python main.py --debug              # enable verbose logging

The system is fully conversational — there are no command-line flags for
topic, columns, or sources. Everything is specified through natural language
in the chat interface.

Architecture overview:
    User ↔ Chat Loop (main.py)
              ↕
         Orchestrator (agents/orchestrator.py)
              ↕
    ┌─────────────────────────────────────────┐
    │  Search → Scrape → Extract → Critic → Refine  │
    └─────────────────────────────────────────┘
              ↕
         SessionState (state.py) — shared truth
"""

import argparse
import sys
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

import config as cfg
from agents.orchestrator import Orchestrator
from state import SessionState
from tools.browser import BrowserPool
from tools.llm_tools import LLMClient

console = Console()

# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = """[bold cyan]
    ███████╗██████╗ ██╗ █████╗
    ██╔══██║██╔══██╗██║██╔══██╗
    ███████║██████╔╝██║███████║
    ██╔══██║██╔══██╗██║██╔══██║
    ███████║██║  ██║██║██║  ██║
    ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
[/bold cyan]
[bold]Agentic Research & Intelligence Assistant[/bold]
[dim]Search · Scrape · Extract · Critique · Refine[/dim]"""

# ── Prompt styling ────────────────────────────────────────────────────────────
_PROMPT_STYLE = Style.from_dict({
    "prompt": "bold cyan",
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ARIA — Agentic Research & Intelligence Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--resume", "-r",
        metavar="SESSION_JSON",
        help="Path to a session JSON file to resume a previous session",
    )
    p.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return p.parse_args()


def setup(args: argparse.Namespace) -> tuple[Orchestrator, SessionState]:
    """
    Validate config, initialise all singletons, and return the orchestrator
    + session state.
    """
    # Debug mode
    if args.debug:
        cfg.DEBUG = True

    # Validate API key early — fail fast with a clear message
    try:
        cfg.validate()
    except ValueError as exc:
        console.print(f"[red]✗ Configuration error:[/red] {exc}")
        sys.exit(1)

    # Initialise shared singletons
    console.print("[dim]Initialising LLM client…[/dim]")
    llm = LLMClient()

    console.print("[dim]Initialising browser…[/dim]")
    browser = BrowserPool()
    browser.start()   # launches Playwright — non-fatal if it fails

    # Session state — resume or fresh
    if args.resume:
        path = args.resume
        if not Path(path).exists():
            console.print(f"[red]Session file not found: {path}[/red]")
            sys.exit(1)
        state = SessionState.load(path)
        console.print(f"[green]✓ Resumed session {state.session_id}[/green]")
        console.print(f"  Topic: {state.topic}")
        console.print(f"  Rows:  {len(state.rows)}")
    else:
        state = SessionState()
        console.print(f"[dim]New session: {state.session_id}[/dim]")

    orchestrator = Orchestrator(llm=llm, browser=browser, state=state)
    return orchestrator, state


def run_chat_loop(orchestrator: Orchestrator, state: SessionState) -> None:
    """
    The main interactive chat loop.

    Uses prompt_toolkit for:
      - Persistent input history (up-arrow to recall previous messages)
      - Auto-suggestion from history (faint ghost text)
      - Multi-line input (Escape+Enter)
    """
    # Store input history in the session directory
    history_file = str(Path(cfg.SESSION_DIR) / ".aria_history")
    session = PromptSession(
        history       = FileHistory(history_file),
        auto_suggest  = AutoSuggestFromHistory(),
        style         = _PROMPT_STYLE,
        mouse_support = False,
    )

    # Print greeting
    greeting = orchestrator.greet()
    console.print(Panel(greeting, border_style="cyan", title="ARIA"))

    while True:
        try:
            # Get user input
            user_input = session.prompt(
                [("class:prompt", "\nYou ❯ ")],
            ).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Ctrl+C — type 'stop' to exit cleanly.[/dim]")
            continue

        if not user_input:
            continue

        # Handle built-in shortcuts that bypass the orchestrator
        lower = user_input.lower()
        if lower in ("exit", "quit", "q", ":q"):
            user_input = "stop"

        # ── Route to orchestrator ────────────────────────────────────────────
        console.print()  # breathing room before response
        try:
            response = orchestrator.handle(user_input)
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Interrupted. Type 'stop' to exit.[/yellow]")
            continue
        except Exception as exc:
            if cfg.DEBUG:
                import traceback
                console.print_exception()
            else:
                console.print(f"[red]✗ Error: {exc}[/red]")
            continue

        # ── Print response ────────────────────────────────────────────────────
        console.print(Rule(style="dim"))
        console.print(f"[bold]ARIA[/bold] ❯  {response}")

        # ── Auto-save on every turn ───────────────────────────────────────────
        if state.rows:
            state.save()

        # ── Stop condition ────────────────────────────────────────────────────
        if state.stop_requested:
            break

    console.print(Rule(style="dim"))
    console.print("[bold cyan]Session ended. Goodbye![/bold cyan]")


def main() -> None:
    """Entry point."""
    args = parse_args()

    # Print banner
    console.print(BANNER)
    console.print()

    orchestrator, state = setup(args)

    try:
        run_chat_loop(orchestrator, state)
    finally:
        # Always clean up the browser, even if we crash
        try:
            orchestrator.browser.stop()
        except Exception:
            pass

        # Final save
        if state.rows:
            from tools.export import save_csv
            path = save_csv(state)
            console.print(f"[dim]Final save → {path}[/dim]")


if __name__ == "__main__":
    main()