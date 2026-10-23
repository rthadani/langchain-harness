"""Global server database for user auth (separate from per-user session DBs).

Manages the ``users`` table in ``<server-cwd>/.harness/server.db``.
Tokens are stored as SHA-256 hex hashes; the plaintext is shown only once at
creation time.
"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite


class UserIdConflict(Exception):
    """Raised when a requested user id already belongs to another auth source."""

    def __init__(self, user_id: str) -> None:
        super().__init__(
            f"User id {user_id!r} belongs to a different authentication source"
        )
        self.user_id = user_id


_SERVER_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    token_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    auth_source TEXT NOT NULL DEFAULT ''
);
"""

_TOKEN_PREFIX = "harn_"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def generate_token() -> str:
    """Generate a new random API token (plaintext, shown to user once)."""
    return _TOKEN_PREFIX + secrets.token_urlsafe(32)


def verify_token(token: str, expected_hash: str) -> bool:
    """Check a plaintext token against a stored hash using constant-time compare."""
    actual = _hash_token(token)
    return secrets.compare_digest(actual, expected_hash)


def hash_token(token: str) -> str:
    """Hash a plaintext token for storage."""
    return _hash_token(token)


def server_db_path(cwd: Path) -> Path:
    """Return the path to the global server.db."""
    return cwd / ".harness" / "server.db"


async def init_server_db(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_SERVER_SCHEMA)
    # Migrate databases created before auth_source existed.
    cursor = await conn.execute("PRAGMA table_info(users)")
    columns = {row[1] for row in await cursor.fetchall()}
    if "auth_source" not in columns:
        await conn.execute(
            "ALTER TABLE users ADD COLUMN auth_source TEXT NOT NULL DEFAULT ''"
        )
    await conn.commit()


async def _select_user(
    conn: aiosqlite.Connection, where: str, params: tuple[Any, ...]
) -> dict[str, Any] | None:
    cursor = await conn.execute(
        f"SELECT user_id, username, is_active, auth_source FROM users WHERE {where}",
        params,
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {
        "user_id": row[0],
        "username": row[1],
        "is_active": bool(row[2]),
        "auth_source": row[3],
    }


async def get_or_create_user(
    conn: aiosqlite.Connection, user_id: str, *, source: str = ""
) -> dict[str, Any]:
    """Return the user row for *user_id*, inserting it on first sight.

    Header/JWT modes call this instead of requiring ``harness users add``.
    A username is reused only when the existing row's effective auth source
    matches *source* (legacy rows with an empty auth_source count as ``token``).
    Cross-source adoption is forbidden to prevent a JWT or trusted-header
    subject from inheriting a pre-provisioned token user's identity.
    """
    existing = await _select_user(conn, "username = ?", (user_id,))
    if existing is None:
        # OR IGNORE: a concurrent replica may have inserted the same id first.
        await conn.execute(
            "INSERT OR IGNORE INTO users "
            "(user_id, username, token_hash, created_at, is_active, auth_source) "
            "VALUES (?, ?, '', ?, 1, ?)",
            (user_id, user_id, _now_iso(), source),
        )
        await conn.commit()
        existing = await _select_user(conn, "username = ?", (user_id,))

    if existing is None:  # pragma: no cover - insert just wrote it
        raise RuntimeError(f"user {user_id!r} could not be created")

    effective_source = existing["auth_source"] or "token"
    if effective_source != source:
        raise UserIdConflict(user_id)
    return existing


async def add_user(conn: aiosqlite.Connection, username: str) -> str:
    """Add a new user, returning the plaintext token (shown once).

    Raises ``ValueError`` if the username already exists.
    """
    cursor = await conn.execute(
        "SELECT 1 FROM users WHERE username = ?", (username,)
    )
    if await cursor.fetchone() is not None:
        raise ValueError(f"User {username!r} already exists")

    user_id = uuid.uuid4().hex
    token = generate_token()
    token_hash = _hash_token(token)
    await conn.execute(
        "INSERT INTO users "
        "(user_id, username, token_hash, created_at, is_active, auth_source) "
        "VALUES (?, ?, ?, ?, 1, 'token')",
        (user_id, username, token_hash, _now_iso()),
    )
    await conn.commit()
    return token


async def list_users(conn: aiosqlite.Connection) -> list[dict[str, Any]]:
    """Return all users ordered by creation time."""
    cursor = await conn.execute(
        "SELECT user_id, username, created_at, is_active, auth_source "
        "FROM users ORDER BY created_at"
    )
    rows = await cursor.fetchall()
    return [
        {
            "user_id": row[0],
            "username": row[1],
            "created_at": row[2],
            "is_active": bool(row[3]),
            "auth_source": row[4],
        }
        for row in rows
    ]


async def revoke_user(conn: aiosqlite.Connection, username: str) -> bool:
    """Deactivate a user by username. Returns True if a user was found."""
    cursor = await conn.execute(
        "UPDATE users SET is_active = 0 WHERE username = ?", (username,)
    )
    await conn.commit()
    return cursor.rowcount > 0


async def lookup_by_token_hash(
    conn: aiosqlite.Connection, token_hash: str
) -> dict[str, Any] | None:
    """Look up a user by their token hash. Returns None if not found or inactive."""
    cursor = await conn.execute(
        "SELECT user_id, username, is_active FROM users WHERE token_hash = ?",
        (token_hash,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {"user_id": row[0], "username": row[1], "is_active": bool(row[2])}


async def get_user_by_username(
    conn: aiosqlite.Connection, username: str
) -> dict[str, Any] | None:
    """Return a user row by username, or None."""
    return await _select_user(conn, "username = ?", (username,))


async def count_provisioned_users(conn: aiosqlite.Connection) -> int:
    """Return the number of auto-provisioned (non-token) users."""
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM users "
        "WHERE auth_source IS NOT NULL "
        "AND auth_source != '' "
        "AND auth_source != 'token'"
    )
    row = await cursor.fetchone()
    return row[0]


async def get_user_id(conn: aiosqlite.Connection, username: str) -> str | None:
    """Return the user_id for a username, or None."""
    cursor = await conn.execute(
        "SELECT user_id FROM users WHERE username = ?", (username,)
    )
    row = await cursor.fetchone()
    return row[0] if row else None
