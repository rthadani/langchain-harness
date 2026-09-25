"""Streaming one-shot and REPL runners for the harness agent."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from rich.console import Console

REPL_THREAD_ID = "harness-repl"
ONCE_THREAD_ID = "harness-run"


def _render_tool_call(console: Console, event: dict[str, Any]) -> None:
    """Print a concise note that a tool is being invoked."""
    name = event.get("name", "unknown")
    data = event.get("data", {})
    input_data = data.get("input") if isinstance(data, dict) else None
    if isinstance(input_data, dict):
        brief = ", ".join(f"{k}={v!r}" for k, v in list(input_data.items())[:3])
    else:
        brief = "..."
    console.print(f"\n[dim]→ running tool [bold]{name}[/bold]({brief})[/dim]")


async def _stream_turn(
    agent: Any,
    prompt: str,
    thread_id: str,
    console: Console,
) -> None:
    """Stream a single turn from *agent* to the console."""
    config = {"configurable": {"thread_id": thread_id}}
    messages = {"messages": [HumanMessage(content=prompt)]}

    async for event in agent.astream_events(
        {"root": messages},
        config,
        version="v2",
    ):
        kind = event.get("event")
        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            content = getattr(chunk, "content", "")
            if content:
                console.print(content, end="")
        elif kind == "on_tool_start":
            _render_tool_call(console, event)

    console.print()


async def run_once(agent: Any, prompt: str) -> None:
    """Run a single user prompt and stream the result."""
    console = Console()
    await _stream_turn(agent, prompt, ONCE_THREAD_ID, console)


async def repl(agent: Any) -> None:
    """Run an interactive multi-turn REPL."""
    console = Console()
    console.print("[bold green]harness[/bold green] interactive mode. Type /exit or press Ctrl-D to quit.")

    while True:
        try:
            prompt = console.input("harness> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"/exit", "/quit"}:
            console.print("Goodbye.")
            break

        await _stream_turn(agent, prompt, REPL_THREAD_ID, console)
