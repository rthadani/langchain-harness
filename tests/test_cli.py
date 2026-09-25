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


def test_sessions_command_lists_rows(isolate_cwd, monkeypatch) -> None:
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

    async def fake_build_agent(config: Config, cwd: Path, checkpointer: Any) -> Any:
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

    async def fake_build_agent(config: Config, cwd: Path, checkpointer: Any) -> Any:
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
