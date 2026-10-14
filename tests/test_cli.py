import asyncio
import sys
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from langchain_harness import cli
from langchain_harness import sessions as sessions_module
from langchain_harness.config import Config


class _FakeAgent:
    async def astream_events(self, *args, **kwargs):
        if False:
            yield


@pytest.fixture
def isolate_cwd(monkeypatch, tmp_path: Path) -> Path:
    """Run CLI tests in a temp directory so they do not touch the real .harness."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_empty_prompt_option_errors(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    monkeypatch.setattr("langchain_harness.cli.build_agent", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["harness", "-p", ""])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1 or str(exc_info.value.code) == "1"


def test_top_level_help_flags(isolate_cwd, monkeypatch, capsys) -> None:
    for flag in ("--help", "-h"):
        monkeypatch.setattr("sys.argv", ["harness", flag])

        with pytest.raises(SystemExit) as exc_info:
            cli.main()

        assert exc_info.value.code == 0
        assert "Usage: harness [OPTIONS]" in capsys.readouterr().out


def test_sessions_command_lists_rows(isolate_cwd, monkeypatch, capsys) -> None:
    db_dir = isolate_cwd / ".harness"
    db_dir.mkdir()
    db_path = db_dir / "sessions.db"

    async def _seed() -> None:
        async with aiosqlite.connect(str(db_path)) as db_conn:
            await sessions_module.init_db(db_conn)
            await sessions_module.upsert(db_conn, "abc-123", title="seed session")

    asyncio.run(_seed())
    monkeypatch.setattr("sys.argv", ["harness", "sessions"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert "seed session" in capsys.readouterr().out


def test_run_continue_resumes_latest(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    db_dir = isolate_cwd / ".harness"
    db_dir.mkdir()
    db_path = db_dir / "sessions.db"

    async def _seed() -> None:
        async with aiosqlite.connect(str(db_path)) as db_conn:
            await sessions_module.init_db(db_conn)
            await sessions_module.upsert(db_conn, "latest-thread", title="latest")

    asyncio.run(_seed())

    captured: dict[str, Any] = {}

    async def fake_build_agent(config: Config, cwd: Path, checkpointer: Any, **kwargs: Any) -> Any:
        captured["checkpointer"] = checkpointer
        return _FakeAgent()

    async def fake_repl(agent: Any, thread_id: str, **kwargs: Any) -> None:
        captured["agent"] = agent
        captured["thread_id"] = thread_id

    monkeypatch.setattr("langchain_harness.cli.build_agent", fake_build_agent)
    monkeypatch.setattr("langchain_harness.cli.repl", fake_repl)
    monkeypatch.setattr("sys.argv", ["harness", "--continue"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["thread_id"] == "latest-thread"


def test_run_continue_errors_when_no_sessions(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    monkeypatch.setattr("langchain_harness.cli.build_agent", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["harness", "--continue"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code != 0


def test_run_resume_resolves_prefix(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    db_dir = isolate_cwd / ".harness"
    db_dir.mkdir()
    db_path = db_dir / "sessions.db"

    async def _seed() -> None:
        async with aiosqlite.connect(str(db_path)) as db_conn:
            await sessions_module.init_db(db_conn)
            await sessions_module.upsert(db_conn, "target-thread", title="target")

    asyncio.run(_seed())

    captured: dict[str, Any] = {}

    async def fake_build_agent(config: Config, cwd: Path, checkpointer: Any, **kwargs: Any) -> Any:
        return _FakeAgent()

    async def fake_repl(agent: Any, thread_id: str, **kwargs: Any) -> None:
        captured["thread_id"] = thread_id

    monkeypatch.setattr("langchain_harness.cli.build_agent", fake_build_agent)
    monkeypatch.setattr("langchain_harness.cli.repl", fake_repl)
    monkeypatch.setattr("sys.argv", ["harness", "--resume", "target"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["thread_id"] == "target-thread"


def test_run_resume_errors_on_ambiguous_prefix(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    db_dir = isolate_cwd / ".harness"
    db_dir.mkdir()
    db_path = db_dir / "sessions.db"

    async def _seed() -> None:
        async with aiosqlite.connect(str(db_path)) as db_conn:
            await sessions_module.init_db(db_conn)
            await sessions_module.upsert(db_conn, "abc-123", title="x")
            await sessions_module.upsert(db_conn, "abc-456", title="y")

    asyncio.run(_seed())
    monkeypatch.setattr("langchain_harness.cli.build_agent", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["harness", "--resume", "abc"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code != 0


def test_bare_harness_starts_repl(isolate_cwd, monkeypatch) -> None:
    """``harness`` with no args still dispatches to the run command / REPL."""
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    captured: dict[str, Any] = {}

    async def fake_build_agent(*args, **kwargs) -> Any:
        return _FakeAgent()

    async def fake_repl(agent: Any, thread_id: str, **kwargs: Any) -> None:
        captured["thread_id"] = thread_id

    monkeypatch.setattr("langchain_harness.cli.build_agent", fake_build_agent)
    monkeypatch.setattr("langchain_harness.cli.repl", fake_repl)
    monkeypatch.setattr("sys.argv", ["harness"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert "thread_id" in captured


def test_run_workspace_flag_points_agent_and_db_at_workspace(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    ws = isolate_cwd / "ws"
    ws.mkdir()
    db_dir = ws / ".harness"
    db_dir.mkdir()
    db_path = db_dir / "sessions.db"

    async def _seed() -> None:
        async with aiosqlite.connect(str(db_path)) as db_conn:
            await sessions_module.init_db(db_conn)
            await sessions_module.upsert(db_conn, "ws-thread", title="workspace session")

    asyncio.run(_seed())

    captured: dict[str, Any] = {}

    async def fake_build_agent(config: Config, cwd: Path, checkpointer: Any, **kwargs: Any) -> Any:
        captured["cwd"] = cwd
        return _FakeAgent()

    async def fake_repl(agent: Any, thread_id: str, **kwargs: Any) -> None:
        captured["thread_id"] = thread_id

    monkeypatch.setattr("langchain_harness.cli.build_agent", fake_build_agent)
    monkeypatch.setattr("langchain_harness.cli.repl", fake_repl)
    monkeypatch.setattr("sys.argv", ["harness", "--workspace", str(ws), "--continue"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["cwd"] == ws.resolve()
    assert captured["thread_id"] == "ws-thread"
    assert not (isolate_cwd / ".harness").exists()


def test_run_workspace_missing_dir_errors(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    missing = isolate_cwd / "does-not-exist"
    monkeypatch.setattr("sys.argv", ["harness", "--workspace", str(missing)])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1 or str(exc_info.value.code) == "1"
    assert not (isolate_cwd / ".harness").exists()


def test_run_workspace_relative_path_resolves_against_cwd(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    ws = isolate_cwd / "ws"
    ws.mkdir()

    captured: dict[str, Any] = {}

    async def fake_build_agent(config: Config, cwd: Path, checkpointer: Any, **kwargs: Any) -> Any:
        captured["cwd"] = cwd
        return _FakeAgent()

    async def fake_repl(agent: Any, thread_id: str, **kwargs: Any) -> None:
        captured["thread_id"] = thread_id

    monkeypatch.setattr("langchain_harness.cli.build_agent", fake_build_agent)
    monkeypatch.setattr("langchain_harness.cli.repl", fake_repl)
    monkeypatch.setattr("sys.argv", ["harness", "--workspace", "ws"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["cwd"] == ws.resolve()


def test_sessions_command_with_workspace_lists_from_workspace_db(isolate_cwd, monkeypatch, capsys) -> None:
    ws = isolate_cwd / "ws"
    ws.mkdir()
    db_dir = ws / ".harness"
    db_dir.mkdir()
    db_path = db_dir / "sessions.db"

    async def _seed() -> None:
        async with aiosqlite.connect(str(db_path)) as db_conn:
            await sessions_module.init_db(db_conn)
            await sessions_module.upsert(db_conn, "ws-thread", title="workspace session")

    asyncio.run(_seed())
    monkeypatch.setattr("sys.argv", ["harness", "sessions", "--workspace", str(ws)])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert not (isolate_cwd / ".harness").exists()
    assert "workspace session" in capsys.readouterr().out


def test_run_flag_before_prompt_routes_to_run(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    ws = isolate_cwd / "ws"
    ws.mkdir()

    captured: dict[str, Any] = {}

    async def fake_build_agent(config: Config, cwd: Path, checkpointer: Any, **kwargs: Any) -> Any:
        captured["cwd"] = cwd
        return _FakeAgent()

    async def fake_run_once(agent: Any, prompt: str, thread_id: str, **kwargs: Any) -> None:
        captured["prompt"] = prompt
        captured["thread_id"] = thread_id

    monkeypatch.setattr("langchain_harness.cli.build_agent", fake_build_agent)
    monkeypatch.setattr("langchain_harness.cli.run_once", fake_run_once)
    monkeypatch.setattr("sys.argv", ["harness", "--workspace", str(ws), "do thing"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["cwd"] == ws.resolve()
    assert captured["prompt"] == "do thing"


def test_run_flag_before_subcommand_routes_to_sessions(isolate_cwd, monkeypatch, capsys) -> None:
    ws = isolate_cwd / "ws"
    ws.mkdir()
    db_dir = ws / ".harness"
    db_dir.mkdir()
    db_path = db_dir / "sessions.db"

    async def _seed() -> None:
        async with aiosqlite.connect(str(db_path)) as db_conn:
            await sessions_module.init_db(db_conn)
            await sessions_module.upsert(db_conn, "ws-thread", title="workspace session")

    asyncio.run(_seed())
    monkeypatch.setattr("sys.argv", ["harness", "--workspace", str(ws), "sessions"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert "workspace session" in capsys.readouterr().out


def test_run_workspace_whitespace_only_uses_cwd(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")

    captured: dict[str, Any] = {}

    async def fake_build_agent(config: Config, cwd: Path, checkpointer: Any, **kwargs: Any) -> Any:
        captured["cwd"] = cwd
        return _FakeAgent()

    async def fake_repl(agent: Any, thread_id: str, **kwargs: Any) -> None:
        captured["thread_id"] = thread_id

    monkeypatch.setattr("langchain_harness.cli.build_agent", fake_build_agent)
    monkeypatch.setattr("langchain_harness.cli.repl", fake_repl)
    monkeypatch.setattr("sys.argv", ["harness", "--workspace", "   "])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["cwd"] == isolate_cwd.resolve()


def test_run_prompt_option_value_routes_to_run(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    captured: dict[str, Any] = {}

    async def fake_build_agent(config: Config, cwd: Path, checkpointer: Any, **kwargs: Any) -> Any:
        captured["cwd"] = cwd
        return _FakeAgent()

    async def fake_run_once(agent: Any, prompt: str, thread_id: str, **kwargs: Any) -> None:
        captured["prompt"] = prompt
        captured["thread_id"] = thread_id

    monkeypatch.setattr("langchain_harness.cli.build_agent", fake_build_agent)
    monkeypatch.setattr("langchain_harness.cli.run_once", fake_run_once)
    monkeypatch.setattr("sys.argv", ["harness", "-p", "sessions"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["prompt"] == "sessions"
    assert captured["cwd"] == isolate_cwd.resolve()


def test_double_dash_before_prompt_routes_to_run(isolate_cwd, monkeypatch) -> None:
    """``harness -- sessions`` treats ``sessions`` as a prompt, not a subcommand."""
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    captured: dict[str, Any] = {}

    async def fake_build_agent(config: Config, cwd: Path, checkpointer: Any, **kwargs: Any) -> Any:
        captured["cwd"] = cwd
        return _FakeAgent()

    async def fake_run_once(agent: Any, prompt: str, thread_id: str, **kwargs: Any) -> None:
        captured["prompt"] = prompt
        captured["thread_id"] = thread_id

    monkeypatch.setattr("langchain_harness.cli.build_agent", fake_build_agent)
    monkeypatch.setattr("langchain_harness.cli.run_once", fake_run_once)
    monkeypatch.setattr("sys.argv", ["harness", "--", "sessions"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["prompt"] == "sessions"


def test_workspace_with_help_shows_run_help(isolate_cwd, monkeypatch, capsys) -> None:
    """``harness --workspace X --help`` shows run help, not a parse error."""
    ws = isolate_cwd / "ws"
    ws.mkdir()
    monkeypatch.setattr("sys.argv", ["harness", "--workspace", str(ws), "--help"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert "Usage: harness run" in capsys.readouterr().out


def test_run_workspace_self_contained_form(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    ws = isolate_cwd / "ws"
    ws.mkdir()

    captured: dict[str, Any] = {}

    async def fake_build_agent(config: Config, cwd: Path, checkpointer: Any, **kwargs: Any) -> Any:
        captured["cwd"] = cwd
        return _FakeAgent()

    async def fake_run_once(agent: Any, prompt: str, thread_id: str, **kwargs: Any) -> None:
        captured["prompt"] = prompt

    monkeypatch.setattr("langchain_harness.cli.build_agent", fake_build_agent)
    monkeypatch.setattr("langchain_harness.cli.run_once", fake_run_once)
    monkeypatch.setattr("sys.argv", ["harness", f"--workspace={ws}", "do thing"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["cwd"] == ws.resolve()


def test_first_positional_index_skips_host_and_port() -> None:
    assert cli._first_positional_index(["harness", "--host", "127.0.0.1", "serve"]) == 3
    assert cli._first_positional_index(["harness", "--port", "8000", "serve"]) == 3


# ---------------------------------------------------------------------------
# model wiring for session memory
# ---------------------------------------------------------------------------


def test_run_forwards_model_and_memory_root_to_run_once(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    captured: dict[str, Any] = {}

    async def fake_build_agent(
        config: Config, cwd: Path, checkpointer: Any, **kwargs: Any
    ) -> Any:
        captured["agent_model"] = kwargs.get("model")
        return _FakeAgent()

    async def fake_run_once(agent: Any, prompt: str, thread_id: str, **kwargs: Any) -> None:
        captured["run_once"] = kwargs

    monkeypatch.setattr("langchain_harness.cli.build_agent", fake_build_agent)
    monkeypatch.setattr("langchain_harness.cli.run_once", fake_run_once)
    monkeypatch.setattr("sys.argv", ["harness", "do thing"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["agent_model"] is not None
    assert captured["run_once"]["model"] is captured["agent_model"]
    assert captured["run_once"]["memory_root"] == isolate_cwd.resolve()


def test_repl_receives_model_and_memory_root(isolate_cwd, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    captured: dict[str, Any] = {}

    async def fake_build_agent(
        config: Config, cwd: Path, checkpointer: Any, **kwargs: Any
    ) -> Any:
        return _FakeAgent()

    async def fake_repl(agent: Any, thread_id: str, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("langchain_harness.cli.build_agent", fake_build_agent)
    monkeypatch.setattr("langchain_harness.cli.repl", fake_repl)
    monkeypatch.setattr("sys.argv", ["harness"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert captured["model"] is not None
    assert captured["memory_root"] == isolate_cwd.resolve()


# ---------------------------------------------------------------------------
# memory command
# ---------------------------------------------------------------------------


def test_memory_command_lists_pending_notes(isolate_cwd, monkeypatch, capsys) -> None:
    from langchain_harness import memory as memory_module

    memory_module.write_inbox_note(
        isolate_cwd,
        thread_id="t1",
        title="First note",
        summary="- bullet one\n- bullet two",
    )
    monkeypatch.setattr("sys.argv", ["harness", "memory"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "t1" in out
    assert "First note" in out
    assert "- bullet one" in out


def test_memory_command_reports_no_pending_notes(isolate_cwd, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["harness", "memory"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert "No pending notes." in capsys.readouterr().out


def test_memory_review_approves_and_deletes_notes(isolate_cwd, monkeypatch, capsys) -> None:
    from rich.console import Console as RichConsole

    from langchain_harness import memory as memory_module

    memory_module.write_inbox_note(
        isolate_cwd, thread_id="t1", title="First", summary="- one"
    )
    memory_module.write_inbox_note(
        isolate_cwd, thread_id="t2", title="Second", summary="- two"
    )

    answers = iter(["a", "d"])
    monkeypatch.setattr(RichConsole, "input", lambda self, prompt="": next(answers))
    monkeypatch.setattr("sys.argv", ["harness", "memory", "--review"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "- one" in out and "- two" in out

    assert memory_module.list_inbox(isolate_cwd) == []
    notes = memory_module.active_memory_path(isolate_cwd).read_text(encoding="utf-8")
    assert "## First" in notes
    assert "## Second" not in notes


def test_memory_review_skip_leaves_note_pending(isolate_cwd, monkeypatch) -> None:
    from rich.console import Console as RichConsole

    from langchain_harness import memory as memory_module

    memory_module.write_inbox_note(
        isolate_cwd, thread_id="t1", title="First", summary="- one"
    )

    answers = iter(["s"])
    monkeypatch.setattr(RichConsole, "input", lambda self, prompt="": next(answers))
    monkeypatch.setattr("sys.argv", ["harness", "memory", "--review"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert len(memory_module.list_inbox(isolate_cwd)) == 1
    assert not memory_module.active_memory_path(isolate_cwd).exists()


def test_memory_command_lists_notes_from_workspace(isolate_cwd, monkeypatch, capsys) -> None:
    from langchain_harness import memory as memory_module

    workspace = isolate_cwd / "other-workspace"
    workspace.mkdir()
    memory_module.write_inbox_note(
        workspace, thread_id="ws-1", title="Workspace note", summary="- ws bullet"
    )
    monkeypatch.setattr(
        "sys.argv", ["harness", "memory", "--workspace", str(workspace)]
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "ws-1" in out
    assert "Workspace note" in out
    assert "- ws bullet" in out


def test_memory_command_rejects_missing_workspace(isolate_cwd, monkeypatch, capsys) -> None:
    missing = isolate_cwd / "does-not-exist"
    monkeypatch.setattr(
        "sys.argv", ["harness", "memory", "--workspace", str(missing)]
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1
    assert "not an existing directory" in capsys.readouterr().err


def test_memory_review_quit_stops_loop(isolate_cwd, monkeypatch) -> None:
    from rich.console import Console as RichConsole

    from langchain_harness import memory as memory_module

    memory_module.write_inbox_note(
        isolate_cwd, thread_id="t1", title="First", summary="- one"
    )

    monkeypatch.setattr(RichConsole, "input", lambda self, prompt="": "q")
    monkeypatch.setattr("sys.argv", ["harness", "memory", "--review"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert len(memory_module.list_inbox(isolate_cwd)) == 1
    assert not memory_module.active_memory_path(isolate_cwd).exists()


def test_memory_review_stops_on_eof(isolate_cwd, monkeypatch, capsys) -> None:
    from rich.console import Console as RichConsole

    from langchain_harness import memory as memory_module

    memory_module.write_inbox_note(
        isolate_cwd, thread_id="t1", title="First", summary="- one"
    )

    def _raise_eof(self, prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr(RichConsole, "input", _raise_eof)
    monkeypatch.setattr("sys.argv", ["harness", "memory", "--review"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert "Stopped." in capsys.readouterr().out
    assert len(memory_module.list_inbox(isolate_cwd)) == 1
