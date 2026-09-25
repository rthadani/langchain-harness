"""Persistence for harness session metadata.

Sessions live in the same SQLite file used by the LangGraph checkpointer
(``<cwd>/.harness/sessions.db``) but in their own table.  They are managed
through a separate ``aiosqlite`` connection so the runner can upsert metadata
without touching the checkpointer's internals.
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
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    title TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate_title(title: str | None) -> str:
    if title is None:
        return ""
    # Truncate on a word boundary when possible.
    title = title.strip().replace("\n", " ")
    if len(title) <= 60:
        return title
    return title[:57].rsplit(" ", 1)[0] + "..."


async def init_db(conn: aiosqlite.Connection) -> None:
    """Create the sessions table and indexes."""
    await conn.executescript(_SESSIONS_SCHEMA)
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
) -> None:
    """Insert or update a session row, timestamped in UTC."""
    now = _now_iso()
    truncated = _truncate_title(title)
    await conn.execute(
        """
        INSERT INTO sessions (thread_id, created_at, updated_at, title)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            updated_at = excluded.updated_at,
            title = COALESCE(NULLIF(excluded.title, ''), sessions.title)
        """,
        (thread_id, now, now, truncated),
    )
    await conn.commit()


async def list_sessions(
    conn: aiosqlite.Connection,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return the most recently updated sessions first."""
    cursor = await conn.execute(
        """
        SELECT thread_id, created_at, updated_at, title
        FROM sessions
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,),
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


async def resolve_prefix(conn: aiosqlite.Connection, prefix: str) -> str:
    """Resolve a thread id prefix to a full thread id.

    Raises:
        ValueError: if the prefix matches zero or more than one session.
    """
    cursor = await conn.execute(
        """
        SELECT thread_id
        FROM sessions
        WHERE thread_id LIKE ?
        ORDER BY updated_at DESC
        """,
        (f"{prefix}%",),
    )
    rows = await cursor.fetchall()
    if not rows:
        raise ValueError(f"No session found with id/prefix {prefix!r}")
    if len(rows) > 1:
        raise ValueError(f"Session prefix {prefix!r} is ambiguous")
    return rows[0][0]
