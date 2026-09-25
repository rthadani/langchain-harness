"""Typer CLI entry point for ``harness``."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer

from langchain_harness.agent_builder import build_agent
from langchain_harness.config import resolve_config
from langchain_harness.runner import repl, run_once

app = typer.Typer(help="LangChain deepagents harness (Claude Code style).")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )


@app.command()
def run(
    prompt: Optional[str] = typer.Argument(None, help="Prompt to run in one-shot mode."),
    provider: Optional[str] = typer.Option(None, "--provider", help="Model provider (e.g. anthropic, openai)."),
    model: Optional[str] = typer.Option(None, "--model", help="Model name (e.g. claude-3-5-sonnet-latest)."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Custom base URL for OpenAI-compatible endpoints."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for the provider."),
) -> None:
    """Run a prompt or start an interactive REPL."""
    _setup_logging()

    cwd = Path.cwd()
    config = resolve_config(
        cwd,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )

    async def _run() -> None:
        agent = await build_agent(config, cwd)
        if prompt is not None:
            await run_once(agent, prompt)
        else:
            await repl(agent)

    asyncio.run(_run())


def main() -> None:
    """Console entry point used by the ``harness`` script."""
    app()


if __name__ == "__main__":
    app()
