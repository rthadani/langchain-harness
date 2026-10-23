"""Persistence for harness session metadata.

Sessions live in the same SQLite file used by the LangGraph checkpointer
(``<cwd>/.harness/sessions.db``) but in their own table.  They are managed
through a separate ``aiosqlite`` connection so the runner can upsert metadata
without touching the checkpointer's internals.

``user_id`` is stored on every row. The local CLI leaves it empty; server mode
sets it so a shared Postgres database can filter sessions per user.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

_SESSIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    thread_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    title TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);
"""

# Added separately so databases created before user_id existed can migrate
# before the index that depends on the column is created.
_USER_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_sessions_user_updated "
    "ON sessions(user_id, updated_at DESC)"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def truncate_title(title: str | None) -> str:
    if title is None:
        return ""
    # Truncate on a word boundary when possible.
    title = title.strip().replace("\n", " ")
    if len(title) <= 60:
        return title
    return title[:57].rsplit(" ", 1)[0] + "..."


async def init_db(conn: aiosqlite.Connection) -> None:
    """Create the sessions table and indexes, migrating older databases."""
    await conn.executescript(_SESSIONS_SCHEMA)
    cursor = await conn.execute("PRAGMA table_info(sessions)")
    columns = {row[1] for row in await cursor.fetchall()}
    if "user_id" not in columns:
        await conn.execute(
            "ALTER TABLE sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT ''"
        )
    await conn.execute(_USER_INDEX)
    await conn.commit()


@asynccontextmanager
async def connect(db_path: Path | str) -> AsyncIterator[aiosqlite.Connection]:
    """Open an aiosqlite connection to *db_path* and initialize the schema."""
    async with aiosqlite.connect(str(db_path)) as conn:
        await init_db(conn)
        yield conn


async def upsert(
    conn: aiosqlite.Connection,
    thread_id: str,
    title: str | None = None,
    *,
    user_id: str = "",
) -> None:
    """Insert or update a session row, timestamped in UTC."""
    now = now_iso()
    truncated = truncate_title(title)
    await conn.execute(
        """
        INSERT INTO sessions (thread_id, user_id, created_at, updated_at, title)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            updated_at = excluded.updated_at,
            title = COALESCE(NULLIF(excluded.title, ''), sessions.title)
        """,
        (thread_id, user_id, now, now, truncated),
    )
    await conn.commit()


async def list_sessions(
    conn: aiosqlite.Connection,
    *,
    limit: int = 20,
    user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return the most recently updated sessions first."""
    if user_id is None:
        cursor = await conn.execute(
            """
            SELECT thread_id, created_at, updated_at, title
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
    else:
        cursor = await conn.execute(
            """
            SELECT thread_id, created_at, updated_at, title
            FROM sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
    rows = await cursor.fetchall()
    return [
        {
            "thread_id": row[0],
            "created_at": row[1],
            "updated_at": row[2],
            "title": row[3],
        }
        for row in rows
    ]


async def get_latest(conn: aiosqlite.Connection) -> dict[str, Any] | None:
    """Return the single most recently updated session, or ``None``."""
    rows = await list_sessions(conn, limit=1)
    return rows[0] if rows else None


async def get_session(
    conn: aiosqlite.Connection,
    thread_id: str,
    *,
    user_id: str | None = None,
) -> dict[str, Any] | None:
    """Return a single session by thread_id, or ``None`` if not found."""
    if user_id is None:
        cursor = await conn.execute(
            "SELECT thread_id, created_at, updated_at, title "
            "FROM sessions WHERE thread_id = ?",
            (thread_id,),
        )
    else:
        cursor = await conn.execute(
            "SELECT thread_id, created_at, updated_at, title "
            "FROM sessions WHERE thread_id = ? AND user_id = ?",
            (thread_id, user_id),
        )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "thread_id": row[0],
        "created_at": row[1],
        "updated_at": row[2],
        "title": row[3],
    }


async def delete_session(
    conn: aiosqlite.Connection,
    thread_id: str,
    *,
    user_id: str | None = None,
) -> None:
    """Delete a session row, optionally scoped to its owner."""
    if user_id is None:
        await conn.execute(
            "DELETE FROM sessions WHERE thread_id = ?", (thread_id,)
        )
    else:
        await conn.execute(
            "DELETE FROM sessions WHERE thread_id = ? AND user_id = ?",
            (thread_id, user_id),
        )
    await conn.commit()


async def resolve_prefix(conn: aiosqlite.Connection, prefix: str) -> str:
    """Resolve a thread id prefix to a full thread id.

    Raises:
        ValueError: if the prefix matches zero or more than one session.
    """
    cursor = await conn.execute(
        """
        SELECT thread_id
        FROM sessions
        WHERE thread_id GLOB ?
        ORDER BY updated_at DESC
        """,
        (f"{prefix}*",),
    )
    rows = await cursor.fetchall()
    if not rows:
        raise ValueError(f"No session found with id/prefix {prefix!r}")
    if len(rows) > 1:
        raise ValueError(f"Session prefix {prefix!r} is ambiguous")
    return rows[0][0]
