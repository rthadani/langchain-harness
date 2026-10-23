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
from rich.markup import escape

from langchain_harness import memory
from langchain_harness.agent_builder import build_agent, build_model
from langchain_harness.config import resolve_config
from langchain_harness.discovery import discover_memory
from langchain_harness.memory import eval_session
from langchain_harness.runner import _print_eval_result, repl, run_once
from langchain_harness import sessions
from langchain_harness.sessions import connect, get_latest, resolve_prefix

app = typer.Typer(
    help="LangChain deepagents harness (Claude Code style).",
    context_settings={"help_option_names": ["--help", "-h"]},
)
db_app = typer.Typer(help="Manage the shared database schema.")
app.add_typer(db_app, name="db")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )


def _db_path(root: Path) -> Path:
    db_dir = root / ".harness"
    db_dir.mkdir(exist_ok=True)
    return db_dir / "sessions.db"


def _resolve_root(workspace: str | None) -> Path:
    """``--workspace X`` is equivalent to ``cd X && harness``."""
    if not workspace or not workspace.strip():
        return Path.cwd()
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        print(
            f"Error: --workspace {workspace!r} is not an existing directory.",
            file=sys.stderr,
        )
        raise typer.Exit(1)
    return root


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
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        help="Working folder to run in; discovery, harness.toml, sessions, and the agent backend follow it. A relative path resolves against the current directory; ~ expands.",
    ),
) -> None:
    """Run a prompt or start an interactive REPL."""
    _setup_logging()

    root = _resolve_root(workspace)
    config = resolve_config(
        root,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )

    if prompt == "" or prompt_arg == "":
        print("Error: prompt cannot be empty.", file=sys.stderr)
        raise typer.Exit(1)

    db = _db_path(root)

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
                await checkpointer.setup()
                model = build_model(config)
                agent = await build_agent(
                    config, root, checkpointer=checkpointer, model=model
                )
                user_prompt = prompt_arg if prompt is None else prompt
                if user_prompt is not None:
                    await run_once(
                        agent,
                        user_prompt,
                        thread_id,
                        db_conn=db_conn,
                        model=model,
                        memory_root=root,
                        prompts=config.prompts,
                    )
                else:
                    await repl(
                        agent,
                        thread_id,
                        db_conn=db_conn,
                        model=model,
                        memory_root=root,
                        prompts=config.prompts,
                    )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# sessions command
# ---------------------------------------------------------------------------


@app.command(name="sessions")
def list_sessions(
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        help="Working folder to list sessions from. A relative path resolves against the current directory; ~ expands.",
    ),
) -> None:
    """List recent persisted sessions."""
    _setup_logging()
    db = _db_path(_resolve_root(workspace))

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


@app.command(name="memory")
def memory_command(
    review: bool = typer.Option(False, "--review", help="Interactively review pending notes."),
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        help="Working folder for the memory store. A relative path resolves against the current directory; ~ expands.",
    ),
) -> None:
    """List or review queued session-memory notes."""
    _setup_logging()
    root = _resolve_root(workspace)
    console = Console()
    notes = memory.list_inbox(root)

    if not review:
        if not notes:
            console.print("No pending notes.")
            return
        for note in notes:
            preview = note.summary.splitlines()[0] if note.summary else ""
            console.print(
                f"{escape(note.note_id)}  {escape(note.created_at)}  "
                f"{escape(note.title)}  {escape(preview)}"
            )
        return

    if not notes:
        console.print("No pending notes.")
        return

    for note in notes:
        console.print(
            f"\n[bold]{escape(note.note_id)}[/bold]  {escape(note.title)}  "
            f"[dim]({escape(note.created_at)})[/dim]"
        )
        console.print(escape(note.summary))
        try:
            choice = console.input(
                escape("[a]pprove / [s]kip / [d]elete / [q]uit: ")
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\nStopped.")
            return

        if choice.startswith("q"):
            break
        if choice.startswith("a"):
            if memory.approve_note(root, note.note_id):
                console.print("[green]Approved.[/green]")
            continue
        if choice.startswith("d"):
            if memory.reject_note(root, note.note_id):
                console.print("[yellow]Deleted.[/yellow]")
            continue
        console.print("[dim]Skipped.[/dim]")


# ---------------------------------------------------------------------------
# eval command
# ---------------------------------------------------------------------------


@app.command(name="eval")
def eval_command(
    session_id: str = typer.Argument(..., metavar="SESSION", help="Session id or unambiguous prefix to evaluate."),
    provider: str | None = typer.Option(None, "--provider", help="Model provider."),
    model: str | None = typer.Option(None, "--model", help="Model name."),
    base_url: str | None = typer.Option(None, "--base-url", help="Custom base URL."),
    api_key: str | None = typer.Option(None, "--api-key", help="API key for the provider."),
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        help="Working folder. A relative path resolves against the current directory; ~ expands.",
    ),
) -> None:
    """Score a completed session on scope adherence, task completion, and tool use quality."""
    _setup_logging()
    root = _resolve_root(workspace)
    config = resolve_config(root, provider=provider, model=model, base_url=base_url, api_key=api_key)
    db = _db_path(root)
    console = Console()

    async def _eval() -> None:
        async with connect(db) as db_conn:
            try:
                thread_id = await resolve_prefix(db_conn, session_id)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                raise typer.Exit(1)

        async with AsyncSqliteSaver.from_conn_string(str(db)) as checkpointer:
            await checkpointer.setup()
            model_obj = build_model(config)
            agent = await build_agent(config, root, checkpointer=checkpointer, model=model_obj)

            memory_texts: list[str] = []
            for path in discover_memory(root):
                try:
                    memory_texts.append(path.read_text(encoding="utf-8"))
                except OSError:
                    pass

            result = await eval_session(
                model_obj, agent, thread_id,
                memory_texts=memory_texts,
                system_prompt=config.prompts.eval,
            )
            if result is None:
                console.print("Nothing to evaluate.")
                return
            _print_eval_result(console, result)

    asyncio.run(_eval())


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


@db_app.command("init")
def db_init() -> None:
    """Create the sessions and checkpoint schema (idempotent).

    Uses Postgres when HARNESS_DATABASE_URL is set, otherwise SQLite under
    <cwd>/.harness.
    """
    _setup_logging()
    from langchain_harness.store import open_store

    async def _init() -> None:
        async with open_store(Path.cwd()) as store:
            Console().print(
                f"Initialized {store.backend} store at {store.describe()}."
            )

    asyncio.run(_init())


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

_KNOWN_SUBCOMMANDS = {"run", "sessions", "serve", "memory", "eval", "db"}

# keep in sync with value-taking typer.Option declarations above
_OPTIONS_WITH_VALUE = {
    "--workspace",
    "--provider",
    "--model",
    "--base-url",
    "--api-key",
    "--resume",
    "--host",
    "--port",
    "-p",
    "--prompt",
}


def _first_positional_index(argv: list[str]) -> int | None:
    """Return the index of the first non-option token after the program name.

    Options that take a value are skipped together with that value;
    ``--opt=value`` and boolean flags advance one token each.
    """
    i = 1
    while i < len(argv):
        token = argv[i]
        if not token.startswith("-"):
            return i
        if token in _OPTIONS_WITH_VALUE:
            i += 2  # skip the option and its value
        else:
            i += 1  # boolean flag or self-contained --opt=value
    return None


def main() -> None:
    """Console entry point used by the ``harness`` script.

    ``harness`` and ``harness <prompt>`` continue to dispatch to the ``run``
    command so existing one-shot and REPL usage keeps working after the
    ``sessions``, ``serve``, and ``users`` subcommands were added.
    """
    if len(sys.argv) <= 1:
        sys.argv.insert(1, "run")
        app()
        return

    idx = _first_positional_index(sys.argv)

    if idx is not None and sys.argv[idx] in _KNOWN_SUBCOMMANDS and "--" not in sys.argv[1:idx]:
        # Move the subcommand ahead of any leading options so typer parses it.
        if idx != 1:
            sys.argv.insert(1, sys.argv.pop(idx))
        app()
        return

    # ``harness --help``/``-h`` shows the top-level help; everything else
    # (a prompt, or options only) defaults to the run command.
    if idx is None and set(sys.argv[1:]) <= {"--help", "-h"}:
        app()
        return

    sys.argv.insert(1, "run")
    app()


if __name__ == "__main__":
    main()
