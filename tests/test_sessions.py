from pathlib import Path

import pytest
import aiosqlite

from langchain_harness import sessions


@pytest.fixture
async def conn(tmp_path: Path) -> aiosqlite.Connection:
    db_path = tmp_path / "sessions.db"
    async with aiosqlite.connect(str(db_path)) as db_conn:
        await sessions.init_db(db_conn)
        yield db_conn


@pytest.mark.asyncio
async def test_init_creates_table(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    async with aiosqlite.connect(str(db_path)) as db_conn:
        await sessions.init_db(db_conn)
        cursor = await db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        assert await cursor.fetchone() is not None


@pytest.mark.asyncio
async def test_upsert_creates_and_updates(conn: aiosqlite.Connection) -> None:
    await sessions.upsert(conn, "thread-1", title="first turn")
    row = (await sessions.list_sessions(conn))[0]
    assert row["thread_id"] == "thread-1"
    assert row["title"] == "first turn"

    await sessions.upsert(conn, "thread-1", title="second turn")
    rows = await sessions.list_sessions(conn)
    assert len(rows) == 1
    assert rows[0]["title"] == "second turn"


@pytest.mark.asyncio
async def test_upsert_keeps_existing_title_when_none_given(conn: aiosqlite.Connection) -> None:
    await sessions.upsert(conn, "thread-1", title="keep me")
    first = (await sessions.list_sessions(conn))[0]["updated_at"]
    await sessions.upsert(conn, "thread-1", title=None)
    rows = await sessions.list_sessions(conn)
    assert rows[0]["title"] == "keep me"
    assert rows[0]["updated_at"] != first


@pytest.mark.asyncio
async def test_title_truncated_to_sixty_chars(conn: aiosqlite.Connection) -> None:
    long_title = "x" * 100
    await sessions.upsert(conn, "thread-1", title=long_title)
    row = (await sessions.list_sessions(conn))[0]
    assert len(row["title"]) <= 60
    assert row["title"].endswith("...")


@pytest.mark.asyncio
async def test_list_orders_by_updated_at(conn: aiosqlite.Connection) -> None:
    await sessions.upsert(conn, "a", title="a")
    await sessions.upsert(conn, "b", title="b")
    await sessions.upsert(conn, "a", title="a again")
    rows = await sessions.list_sessions(conn)
    assert [r["thread_id"] for r in rows] == ["a", "b"]


@pytest.mark.asyncio
async def test_list_respects_limit(conn: aiosqlite.Connection) -> None:
    for i in range(5):
        await sessions.upsert(conn, f"t{i}", title=str(i))
    rows = await sessions.list_sessions(conn, limit=2)
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_get_latest_returns_most_recent(conn: aiosqlite.Connection) -> None:
    await sessions.upsert(conn, "old", title="old")
    await sessions.upsert(conn, "new", title="new")
    latest = await sessions.get_latest(conn)
    assert latest is not None
    assert latest["thread_id"] == "new"


@pytest.mark.asyncio
async def test_get_latest_returns_none_when_empty(conn: aiosqlite.Connection) -> None:
    assert await sessions.get_latest(conn) is None


@pytest.mark.asyncio
async def test_resolve_prefix_unique(conn: aiosqlite.Connection) -> None:
    await sessions.upsert(conn, "abc-123", title="x")
    await sessions.upsert(conn, "def-456", title="y")
    assert await sessions.resolve_prefix(conn, "abc") == "abc-123"


@pytest.mark.asyncio
async def test_resolve_prefix_ambiguous(conn: aiosqlite.Connection) -> None:
    await sessions.upsert(conn, "abc-123", title="x")
    await sessions.upsert(conn, "abc-456", title="y")
    with pytest.raises(ValueError, match="ambiguous"):
        await sessions.resolve_prefix(conn, "abc")


@pytest.mark.asyncio
async def test_resolve_prefix_missing(conn: aiosqlite.Connection) -> None:
    with pytest.raises(ValueError, match="No session found"):
        await sessions.resolve_prefix(conn, "zzz")


@pytest.mark.asyncio
async def test_connect_initializes_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    async with sessions.connect(db_path) as db_conn:
        cursor = await db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        assert await cursor.fetchone() is not None
