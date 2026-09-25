"""Tests for Part A polish: empty positional, run_once exception, LIKE escaping, upsert-on-error."""

from __future__ import annotations

import asyncio
import io
from typing import Any
from unittest.mock import AsyncMock

import aiosqlite
import pytest
from rich.console import Console

from langchain_harness import runner as runner_module
from langchain_harness import sessions
from langchain_harness.runner import run_once


# --- empty positional prompt ---


def test_empty_positional_prompt_errors(monkeypatch, tmp_path) -> None:
    """harness '' should error just like harness -p ''."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HARNESS_MODEL", "claude-3")
    monkeypatch.setattr("langchain_harness.cli.build_agent", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["harness", ""])

    from langchain_harness import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1 or str(exc_info.value.code) == "1"


# --- run_once catches generic exceptions ---


class _RaisingAgent:
    async def astream_events(self, *args, **kwargs):
        raise RuntimeError("API connection failed")
        yield  # make this an async generator


@pytest.mark.asyncio
async def test_run_once_catches_generic_exception(monkeypatch, tmp_path) -> None:
    """run_once should print a clean error, not a raw traceback, on API errors."""
    output = io.StringIO()
    console = Console(file=output)

    agent = _RaisingAgent()
    # Should not raise
    await run_once(agent, "hello", "thread-1", console=console)

    captured = output.getvalue()
    assert "Error:" in captured
    assert "API connection failed" in captured


@pytest.mark.asyncio
async def test_run_once_upserts_session_on_error(tmp_path) -> None:
    """Even when the turn errors, the session row should be recorded."""
    db_path = tmp_path / "sessions.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(db_path)) as db_conn:
        await sessions.init_db(db_conn)

        agent = _RaisingAgent()
        console = Console(file=io.StringIO())
        await run_once(agent, "hello", "thread-err", console=console, db_conn=db_conn)

        cursor = await db_conn.execute(
            "SELECT thread_id, title FROM sessions WHERE thread_id = ?", ("thread-err",)
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "thread-err"
        assert row[1] == "hello"


@pytest.mark.asyncio
async def test_run_once_upserts_session_on_success(tmp_path) -> None:
    """Normal turns still upsert the session."""
    db_path = tmp_path / "sessions.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    class _OkAgent:
        async def astream_events(self, *args, **kwargs):
            if False:
                yield

    async with aiosqlite.connect(str(db_path)) as db_conn:
        await sessions.init_db(db_conn)
        agent = _OkAgent()
        console = Console(file=io.StringIO())
        await run_once(agent, "hello", "thread-ok", console=console, db_conn=db_conn)

        cursor = await db_conn.execute(
            "SELECT thread_id FROM sessions WHERE thread_id = ?", ("thread-ok",)
        )
        assert await cursor.fetchone() is not None


# --- LIKE/GLOB escaping ---


@pytest.mark.asyncio
async def test_resolve_prefix_with_percent_does_not_act_as_wildcard(tmp_path) -> None:
    """A % in the prefix should not match arbitrary characters."""
    db_path = tmp_path / "sessions.db"
    async with aiosqlite.connect(str(db_path)) as conn:
        await sessions.init_db(conn)
        await sessions.upsert(conn, "abcdef", title="x")
        await sessions.upsert(conn, "abxcdf", title="y")

        # With LIKE, "ab%def" would match both. With GLOB, "ab%def*" should
        # only match if a thread_id literally starts with "ab%def".
        # Neither thread_id starts with "ab%def", so it should raise ValueError.
        with pytest.raises(ValueError, match="No session found"):
            await sessions.resolve_prefix(conn, "ab%def")


@pytest.mark.asyncio
async def test_resolve_prefix_with_underscore_does_not_act_as_wildcard(tmp_path) -> None:
    """A _ in the prefix should not match any single character."""
    db_path = tmp_path / "sessions.db"
    async with aiosqlite.connect(str(db_path)) as conn:
        await sessions.init_db(conn)
        await sessions.upsert(conn, "abcdef", title="x")

        # With LIKE, "ab_def" would match "abcdef". With GLOB it should not.
        with pytest.raises(ValueError, match="No session found"):
            await sessions.resolve_prefix(conn, "ab_def")


@pytest.mark.asyncio
async def test_resolve_prefix_still_works_for_normal_prefixes(tmp_path) -> None:
    """Normal prefixes still resolve correctly after the GLOB change."""
    db_path = tmp_path / "sessions.db"
    async with aiosqlite.connect(str(db_path)) as conn:
        await sessions.init_db(conn)
        await sessions.upsert(conn, "abc-123", title="x")
        assert await sessions.resolve_prefix(conn, "abc") == "abc-123"
