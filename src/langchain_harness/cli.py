"""Typer CLI entry point for ``harness``."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from uuid import uuid4

import typer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.console import Console

from langchain_harness.agent_builder import build_agent
from langchain_harness.config import resolve_config
from langchain_harness.runner import repl, run_once
from langchain_harness import sessions
from langchain_harness.sessions import connect, get_latest, resolve_prefix

app = typer.Typer(help="LangChain deepagents harness (Claude Code style).")
users_app = typer.Typer(help="Manage API users for server mode.")
app.add_typer(users_app, name="users")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )


def _db_path(cwd: Path) -> Path:
    db_dir = cwd / ".harness"
    db_dir.mkdir(exist_ok=True)
    return db_dir / "sessions.db"


# ---------------------------------------------------------------------------
# run command (unchanged CLI behavior)
# ---------------------------------------------------------------------------


@app.command()
def run(
    prompt_arg: str | None = typer.Argument(None, metavar="PROMPT", help="Prompt to run in one-shot mode."),
    prompt: str | None = typer.Option(None, "-p", "--prompt", help="Prompt to run in one-shot mode."),
    provider: str | None = typer.Option(None, "--provider", help="Model provider (e.g. anthropic, openai)."),
    model: str | None = typer.Option(None, "--model", help="Model name (e.g. claude-3-5-sonnet-latest)."),
    base_url: str | None = typer.Option(None, "--base-url", help="Custom base URL for OpenAI-compatible endpoints."),
    api_key: str | None = typer.Option(None, "--api-key", help="API key for the provider."),
    continue_: bool = typer.Option(False, "--continue", help="Resume the latest session."),
    resume: str | None = typer.Option(None, "--resume", help="Resume a session by full id or unambiguous prefix."),
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

    if prompt == "" or prompt_arg == "":
        print("Error: prompt cannot be empty.", file=sys.stderr)
        raise typer.Exit(1)

    db = _db_path(cwd)

    async def _run() -> None:
        async with connect(db) as db_conn:
            if continue_:
                latest = await get_latest(db_conn)
                if latest is None:
                    print("Error: no sessions to continue.", file=sys.stderr)
                    raise typer.Exit(1)
                thread_id = latest["thread_id"]
            elif resume is not None:
                try:
                    thread_id = await resolve_prefix(db_conn, resume)
                except ValueError as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    raise typer.Exit(1)
            else:
                thread_id = uuid4().hex

            async with AsyncSqliteSaver.from_conn_string(str(db)) as checkpointer:
                agent = await build_agent(config, cwd, checkpointer=checkpointer)
                user_prompt = prompt_arg if prompt is None else prompt
                if user_prompt is not None:
                    await run_once(agent, user_prompt, thread_id, db_conn=db_conn)
                else:
                    await repl(agent, thread_id, db_conn=db_conn)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# sessions command
# ---------------------------------------------------------------------------


@app.command(name="sessions")
def list_sessions() -> None:
    """List recent persisted sessions."""
    _setup_logging()
    db = _db_path(Path.cwd())

    async def _list() -> None:
        async with connect(db) as db_conn:
            rows = await sessions.list_sessions(db_conn)
            console = Console()
            if not rows:
                console.print("No saved sessions.")
                return
            for row in rows:
                console.print(
                    f"{row['thread_id']}  {row['updated_at']}  "
                    f"{row['title'] or '(no title)'}"
                )

    asyncio.run(_list())


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host", help="Bind host (default 127.0.0.1)."),
    port: int | None = typer.Option(None, "--port", help="Bind port (default 8000)."),
    provider: str | None = typer.Option(None, "--provider", help="Model provider."),
    model: str | None = typer.Option(None, "--model", help="Model name."),
    base_url: str | None = typer.Option(None, "--base-url", help="Custom base URL."),
    api_key: str | None = typer.Option(None, "--api-key", help="API key for the provider."),
) -> None:
    """Start the multi-user web API server."""
    _setup_logging()
    cwd = Path.cwd()
    config = resolve_config(
        cwd,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )

    bind_host = host or config.server.host
    bind_port = port or config.server.port

    if config.server.enable_shell:
        print(
            "WARNING: [server] enable_shell is true — per-user privacy is "
            "best-effort only. Shell commands can access any path.",
            file=sys.stderr,
        )

    from langchain_harness.api.app import create_app

    fastapi_app = create_app(cwd, config)

    import uvicorn

    uvicorn.run(fastapi_app, host=bind_host, port=bind_port)


# ---------------------------------------------------------------------------
# users subcommands
# ---------------------------------------------------------------------------


@users_app.command("add")
def users_add(
    name: str = typer.Argument(..., help="Username to add."),
) -> None:
    """Add a new API user. Prints the token ONCE."""
    _setup_logging()
    from langchain_harness.serverdb import add_user, init_server_db, server_db_path

    db_path = server_db_path(Path.cwd())
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async def _add() -> None:
        import aiosqlite

        async with aiosqlite.connect(str(db_path)) as conn:
            await init_server_db(conn)
            try:
                token = await add_user(conn, name)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                raise typer.Exit(1)
            print(token)

    asyncio.run(_add())


@users_app.command("list")
def users_list() -> None:
    """List all API users."""
    _setup_logging()
    from langchain_harness.serverdb import init_server_db, list_users, server_db_path

    db_path = server_db_path(Path.cwd())

    async def _list() -> None:
        import aiosqlite

        async with aiosqlite.connect(str(db_path)) as conn:
            await init_server_db(conn)
            rows = await list_users(conn)
        console = Console()
        if not rows:
            console.print("No users.")
            return
        for row in rows:
            status = "active" if row["is_active"] else "revoked"
            console.print(
                f"{row['username']}  {row['user_id']}  {row['created_at']}  {status}"
            )

    asyncio.run(_list())


@users_app.command("revoke")
def users_revoke(
    name: str = typer.Argument(..., help="Username to revoke."),
) -> None:
    """Revoke an API user (deactivates their token)."""
    _setup_logging()
    from langchain_harness.serverdb import init_server_db, revoke_user, server_db_path

    db_path = server_db_path(Path.cwd())

    async def _revoke() -> None:
        import aiosqlite

        async with aiosqlite.connect(str(db_path)) as conn:
            await init_server_db(conn)
            found = await revoke_user(conn, name)
        if not found:
            print(f"Error: user {name!r} not found.", file=sys.stderr)
            raise typer.Exit(1)
        print(f"Revoked user {name!r}.")

    asyncio.run(_revoke())


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

_KNOWN_SUBCOMMANDS = {"run", "sessions", "serve", "users", "--help", "-h"}


def main() -> None:
    """Console entry point used by the ``harness`` script.

    ``harness`` and ``harness <prompt>`` continue to dispatch to the ``run``
    command so existing one-shot and REPL usage keeps working after the
    ``sessions``, ``serve``, and ``users`` subcommands were added.
    """
    if len(sys.argv) > 1 and sys.argv[1] in {"sessions", "serve", "users"}:
        app()
        return

    # For all other invocations, default to the run command.
    if len(sys.argv) <= 1 or sys.argv[1] not in _KNOWN_SUBCOMMANDS:
        sys.argv.insert(1, "run")
    app()


if __name__ == "__main__":
    main()
