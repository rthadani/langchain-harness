"""Streaming one-shot and REPL runners for the harness agent."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

import aiosqlite
from rich.console import Console
from rich.markup import escape
from rich.text import Text

from langchain_harness import sessions
from langchain_harness.events import COMPACT_PROMPT, stream_turn_events
from langchain_harness.memory import capture_session_note


def _render_tool_call(console: Console, name: str, input_data: Any) -> None:
    """Print a concise note that a tool is being invoked."""
    if isinstance(input_data, dict):
        brief = ", ".join(
            f"{escape(str(k))}={escape(str(v))}"
            for k, v in list(input_data.items())[:3]
        )
    else:
        brief = "..."
    safe_name = escape(str(name))
    console.print(f"\n[dim]→ running tool [bold]{safe_name}[/bold]({brief})[/dim]")


async def _stream_turn(
    agent: Any,
    prompt: str,
    thread_id: str,
    console: Console,
    *,
    tool_end_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Stream a single turn from *agent* to the console."""
    async for event in stream_turn_events(agent, prompt, thread_id):
        etype = event["type"]
        if etype == "token":
            console.print(Text(event["text"]), end="")
        elif etype == "tool_start":
            _render_tool_call(console, event["name"], event.get("input"))
        elif etype == "tool_end" and tool_end_callback is not None:
            tool_end_callback(event)
        elif etype == "done":
            pass

    console.print()


async def _upsert_session(
    db_conn: aiosqlite.Connection | None,
    thread_id: str,
    title: str | None,
) -> None:
    if db_conn is not None:
        await sessions.upsert(db_conn, thread_id, title=title)


async def run_once(
    agent: Any,
    prompt: str,
    thread_id: str,
    *,
    console: Console | None = None,
    db_conn: aiosqlite.Connection | None = None,
    model: Any | None = None,
    memory_root: Path | None = None,
) -> None:
    """Run a single user prompt and stream the result.

    When *model* and *memory_root* are provided, the finished session is
    summarized into the memory review inbox — a one-shot run is headless, so
    there is no chance for the user to ask for it later.
    """
    if console is None:
        console = Console()
    try:
        try:
            await _stream_turn(agent, prompt, thread_id, console)
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[dim]Turn interrupted.[/dim]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {escape(str(exc))}")
    finally:
        await _upsert_session(db_conn, thread_id, title=prompt)
        # Capture memory before anything else can clear the session state.
        if model is not None and memory_root is not None:
            await capture_session_note(model, agent, thread_id, memory_root, title=prompt)


async def _list_sessions_command(
    console: Console,
    db_conn: aiosqlite.Connection | None,
) -> None:
    if db_conn is None:
        console.print("[red]Error:[/red] session storage is not available")
        return
    rows = await sessions.list_sessions(db_conn)
    if not rows:
        console.print("No saved sessions.")
        return
    for row in rows:
        tid = row["thread_id"]
        updated = row["updated_at"]
        title = row["title"] or "(no title)"
        console.print(f"{tid}  {updated}  {title}")


async def _resume_command(
    console: Console,
    db_conn: aiosqlite.Connection | None,
    prefix: str,
) -> str | None:
    if db_conn is None:
        console.print("[red]Error:[/red] session storage is not available")
        return None
    try:
        return await sessions.resolve_prefix(db_conn, prefix)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        return None


async def _compact_command(
    agent: Any,
    thread_id: str,
    console: Console,
    db_conn: aiosqlite.Connection | None,
) -> None:
    """Run a turn that asks the model to compact the conversation.

    The compact_conversation tool output is shown verbatim so the user sees a
    refusal when the trigger threshold has not been reached.
    """
    compact_outputs: list[str] = []

    def _on_tool_end(event: dict[str, Any]) -> None:
        if event.get("name") == "compact_conversation":
            text = event.get("output")
            if text is not None:
                compact_outputs.append(text)

    try:
        await _stream_turn(
            agent,
            COMPACT_PROMPT,
            thread_id,
            console,
            tool_end_callback=_on_tool_end,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[dim]Turn interrupted.[/dim]")
        return
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        return

    if compact_outputs:
        for text in compact_outputs:
            console.print(text, markup=False)
    else:
        console.print(
            "[dim]No compaction response received. "
            "Compaction may not be configured for this session.[/dim]"
        )

    await _upsert_session(db_conn, thread_id, title="/compact")


async def _summarize_command(
    agent: Any,
    thread_id: str,
    console: Console,
    model: Any | None,
    memory_root: Path | None,
    title: str,
) -> None:
    """Queue a durable memory note for the current session."""
    if model is None or memory_root is None:
        console.print("[red]Error:[/red] summarization is not available")
        return
    note = await capture_session_note(
        model, agent, thread_id, memory_root, title=title
    )
    if note is None:
        console.print("Nothing to summarize.")
    else:
        console.print(
            f"[green]Queued memory note[/green] {escape(note.stem)} "
            "— review it with [bold]harness memory --review[/bold]."
        )


async def repl(
    agent: Any,
    thread_id: str,
    *,
    console: Console | None = None,
    db_conn: aiosqlite.Connection | None = None,
    model: Any | None = None,
    memory_root: Path | None = None,
) -> None:
    """Run an interactive multi-turn REPL."""
    if console is None:
        console = Console()
    console.print(
        "[bold green]harness[/bold green] interactive mode. "
        "Type /exit or press Ctrl-D to quit. "
        "Commands: /sessions, /resume <id>, /compact, /summarize."
    )

    first_prompt: str | None = None

    while True:
        try:
            prompt = console.input("harness> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break

        if not prompt:
            continue
        lower = prompt.lower()
        if lower in {"/exit", "/quit"}:
            console.print("Goodbye.")
            break
        if lower == "/sessions":
            await _list_sessions_command(console, db_conn)
            continue
        if lower.startswith("/resume "):
            _, _, rest = prompt.partition(" ")
            new_thread_id = await _resume_command(console, db_conn, rest.strip())
            if new_thread_id is not None:
                thread_id = new_thread_id
                console.print(f"[dim]Resumed session {thread_id}.[/dim]")
            continue
        if lower == "/compact":
            await _compact_command(agent, thread_id, console, db_conn)
            continue
        if lower == "/summarize":
            await _summarize_command(
                agent,
                thread_id,
                console,
                model,
                memory_root,
                first_prompt or "REPL session",
            )
            continue

        if first_prompt is None:
            first_prompt = prompt

        try:
            await _stream_turn(agent, prompt, thread_id, console)
            await _upsert_session(db_conn, thread_id, title=prompt)
        except (KeyboardInterrupt, asyncio.CancelledError):
            console.print("\n[dim]Turn interrupted.[/dim]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {escape(str(exc))}")
