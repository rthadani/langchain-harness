"""Server persistence: API users, session metadata, and agent checkpoints.

Two interchangeable backends:

- ``SqliteStore`` (default) keeps the existing on-disk layout:
  ``<cwd>/.harness/server.db`` for users and
  ``<cwd>/.harness/users/<user_id>/sessions.db`` for sessions + checkpoints.
- ``PostgresStore`` stores all three in one Postgres database. Select it with
  ``HARNESS_DATABASE_URL``. It is the only backend that works across multiple
  replicas, because every pod sees the same users and session state.

Workspace files are still per-pod; sharing them needs a ReadWriteMany volume.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

import aiosqlite

from langchain_harness import serverdb, sessions
from langchain_harness.workspaces import user_sessions_db

_DEFAULT_POOL_SIZE = 10

# Serializes schema and checkpointer migrations when several replicas start at
# the same time. Arbitrary constant, stable across releases.
_PG_INIT_LOCK = 0x6861726E

_PG_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        token_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        auth_source TEXT NOT NULL DEFAULT ''
    )
    """,
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_source TEXT NOT NULL DEFAULT ''",
    """
    CREATE TABLE IF NOT EXISTS sessions (
        thread_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        title TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sessions_user_updated "
    "ON sessions (user_id, updated_at DESC)",
)


class Store(Protocol):
    """Backend-agnostic interface used by the API, CLI, and agent cache."""

    backend: str

    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def ping(self) -> None: ...

    def describe(self) -> str: ...

    async def add_user(self, username: str) -> str: ...

    async def get_or_create_user(
        self, user_id: str, *, source: str = ""
    ) -> dict[str, Any]: ...

    async def get_user_by_username(
        self, username: str
    ) -> dict[str, Any] | None: ...

    async def count_provisioned_users(self) -> int: ...

    async def list_users(self) -> list[dict[str, Any]]: ...

    async def revoke_user(self, username: str) -> bool: ...

    async def get_user_by_token_hash(
        self, token_hash: str
    ) -> dict[str, Any] | None: ...

    async def create_session(
        self, user_id: str, thread_id: str, title: str | None
    ) -> dict[str, Any]: ...

    async def upsert_session(
        self, user_id: str, thread_id: str, title: str | None
    ) -> None: ...

    async def list_sessions(
        self, user_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]: ...

    async def get_session(
        self, user_id: str, thread_id: str
    ) -> dict[str, Any] | None: ...

    async def delete_session(self, user_id: str, thread_id: str) -> None: ...

    def checkpointer(self, user_id: str) -> Any: ...


class SqliteStore:
    """Single-host backend, preserving the pre-Postgres behavior."""

    backend = "sqlite"

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def describe(self) -> str:
        return str(serverdb.server_db_path(self.cwd))

    async def initialize(self) -> None:
        path = serverdb.server_db_path(self.cwd)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(path)) as conn:
            await serverdb.init_server_db(conn)

    async def close(self) -> None:
        return None

    async def ping(self) -> None:
        path = serverdb.server_db_path(self.cwd)
        async with aiosqlite.connect(str(path)) as conn:
            await conn.execute("SELECT 1")

    async def add_user(self, username: str) -> str:
        path = serverdb.server_db_path(self.cwd)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(path)) as conn:
            await serverdb.init_server_db(conn)
            return await serverdb.add_user(conn, username)

    async def list_users(self) -> list[dict[str, Any]]:
        path = serverdb.server_db_path(self.cwd)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(path)) as conn:
            await serverdb.init_server_db(conn)
            return await serverdb.list_users(conn)

    async def revoke_user(self, username: str) -> bool:
        path = serverdb.server_db_path(self.cwd)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(path)) as conn:
            await serverdb.init_server_db(conn)
            return await serverdb.revoke_user(conn, username)

    async def get_user_by_token_hash(
        self, token_hash: str
    ) -> dict[str, Any] | None:
        path = serverdb.server_db_path(self.cwd)
        async with aiosqlite.connect(str(path)) as conn:
            await serverdb.init_server_db(conn)
            return await serverdb.lookup_by_token_hash(conn, token_hash)

    async def get_or_create_user(
        self, user_id: str, *, source: str = ""
    ) -> dict[str, Any]:
        path = serverdb.server_db_path(self.cwd)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(path)) as conn:
            await serverdb.init_server_db(conn)
            return await serverdb.get_or_create_user(conn, user_id, source=source)

    async def get_user_by_username(
        self, username: str
    ) -> dict[str, Any] | None:
        path = serverdb.server_db_path(self.cwd)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(path)) as conn:
            await serverdb.init_server_db(conn)
            return await serverdb.get_user_by_username(conn, username)

    async def count_provisioned_users(self) -> int:
        path = serverdb.server_db_path(self.cwd)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(path)) as conn:
            await serverdb.init_server_db(conn)
            return await serverdb.count_provisioned_users(conn)

    def _session_db(self, user_id: str) -> Path:
        path = user_sessions_db(self.cwd, user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    async def create_session(
        self, user_id: str, thread_id: str, title: str | None
    ) -> dict[str, Any]:
        path = self._session_db(user_id)
        async with sessions.connect(path) as conn:
            await sessions.upsert(conn, thread_id, title=title, user_id=user_id)
            row = await sessions.get_session(conn, thread_id, user_id=user_id)
        if row is None:  # pragma: no cover - upsert just wrote it
            raise RuntimeError("session insert did not persist")
        return row

    async def upsert_session(
        self, user_id: str, thread_id: str, title: str | None
    ) -> None:
        async with sessions.connect(self._session_db(user_id)) as conn:
            await sessions.upsert(conn, thread_id, title=title, user_id=user_id)

    async def list_sessions(
        self, user_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        path = user_sessions_db(self.cwd, user_id)
        if not path.exists():
            return []
        async with sessions.connect(path) as conn:
            return await sessions.list_sessions(conn, limit=limit, user_id=user_id)

    async def get_session(
        self, user_id: str, thread_id: str
    ) -> dict[str, Any] | None:
        path = user_sessions_db(self.cwd, user_id)
        if not path.exists():
            return None
        async with sessions.connect(path) as conn:
            return await sessions.get_session(conn, thread_id, user_id=user_id)

    async def delete_session(self, user_id: str, thread_id: str) -> None:
        path = user_sessions_db(self.cwd, user_id)
        if not path.exists():
            return
        async with sessions.connect(path) as conn:
            await sessions.delete_session(conn, thread_id, user_id=user_id)

    @asynccontextmanager
    async def checkpointer(self, user_id: str) -> AsyncIterator[Any]:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        path = self._session_db(user_id)
        async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
            # from_conn_string does not create the checkpoint tables.
            await saver.setup()
            yield saver


class PostgresStore:
    """Shared backend for multi-replica deployments.

    Requires the ``postgres`` extra:
    ``pip install 'langchain-harness[postgres]'``.
    """

    backend = "postgres"

    def __init__(self, dsn: str) -> None:
        if not dsn.startswith(("postgresql://", "postgres://")):
            raise ValueError(
                "HARNESS_DATABASE_URL must be a postgresql:// DSN "
                "(SQLite URLs are not supported; unset it to use SQLite)"
            )
        self.dsn = dsn
        self._pool: Any = None

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PostgresStore is not initialized")
        return self._pool

    def describe(self) -> str:
        from urllib.parse import urlsplit

        parts = urlsplit(self.dsn)
        host = parts.hostname or "localhost"
        if parts.port:
            host = f"{host}:{parts.port}"
        user = f"{parts.username}@" if parts.username else ""
        database = parts.path.lstrip("/") or "postgres"
        return f"postgresql://{user}{host}/{database}"

    async def initialize(self) -> None:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg_pool import AsyncConnectionPool
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "HARNESS_DATABASE_URL is set but the Postgres dependencies are "
                "missing. Install 'langchain-harness[postgres]'."
            ) from exc

        if self._pool is None:
            size = int(os.environ.get(
                "HARNESS_DATABASE_POOL_SIZE", str(_DEFAULT_POOL_SIZE)
            ))
            # autocommit is required: the checkpointer's setup() and writes rely
            # on statements committing when the pooled connection returns.
            self._pool = AsyncConnectionPool(
                self.dsn,
                min_size=1,
                max_size=max(1, size),
                open=False,
                kwargs={"autocommit": True},
            )
            await self._pool.open()
            await self._pool.wait(timeout=15)

        # CREATE TABLE IF NOT EXISTS is not race-free in Postgres: two replicas
        # starting together can both try to create the same type. We serialize
        # them with a pg advisory lock. LangGraph's setup() runs on the same
        # connection for the same reason (and to avoid needing a second pool
        # slot).
        #
        # Use pg_try_advisory_lock + retry instead of pg_advisory_lock: a
        # blocking advisory-lock call holds an active statement transaction,
        # and CREATE INDEX CONCURRENTLY (inside LangGraph's migrations) waits
        # for every active virtual transaction to finish. With several replicas
        # starting together, that produces a circular wait and a deadlock.
        # pg_try_advisory_lock returns immediately, so waiters never keep a
        # transaction open.
        async with self._pool.connection() as conn:
            await conn.execute("SET lock_timeout = '30s'")
            acquired = False
            loop = asyncio.get_running_loop()
            deadline = loop.time() + 30
            while not acquired:
                cur = await conn.execute(
                    "SELECT pg_try_advisory_lock(%s)", (_PG_INIT_LOCK,)
                )
                acquired = (await cur.fetchone())[0]
                if not acquired:
                    if loop.time() >= deadline:
                        raise RuntimeError(
                            "Timed out waiting for the Postgres init lock"
                        )
                    await asyncio.sleep(0.1)
            try:
                for statement in _PG_SCHEMA:
                    await conn.execute(statement)
                # Idempotent LangGraph migrations for checkpoints/blobs/writes.
                await AsyncPostgresSaver(conn).setup()
            finally:
                await conn.execute(
                    "SELECT pg_advisory_unlock(%s)", (_PG_INIT_LOCK,)
                )
                await conn.execute("RESET lock_timeout")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def ping(self) -> None:
        async with self._require_pool().connection() as conn:
            await conn.execute("SELECT 1")

    async def add_user(self, username: str) -> str:
        from psycopg.errors import UniqueViolation

        user_id = uuid.uuid4().hex
        token = serverdb.generate_token()
        try:
            async with self._require_pool().connection() as conn:
                await conn.execute(
                    "INSERT INTO users "
                    "(user_id, username, token_hash, created_at, is_active, "
                    "auth_source) "
                    "VALUES (%s, %s, %s, %s, TRUE, 'token')",
                    (
                        user_id,
                        username,
                        serverdb.hash_token(token),
                        sessions.now_iso(),
                    ),
                )
        except UniqueViolation as exc:
            raise ValueError(f"User {username!r} already exists") from exc
        return token

    async def get_or_create_user(
        self, user_id: str, *, source: str = ""
    ) -> dict[str, Any]:
        """Return the user row for *user_id*, inserting it on first sight.

        ``ON CONFLICT DO NOTHING`` covers both the user_id primary key and the
        username unique constraint, so racing replicas converge on one row.
        A username is reused only when the existing row's effective source
        matches *source*; otherwise ``UserIdConflict`` is raised.
        """
        columns = (
            "SELECT user_id, username, is_active, auth_source "
            "FROM users WHERE "
        )
        async with self._require_pool().connection() as conn:
            cur = await conn.execute(columns + "username = %s", (user_id,))
            row = await cur.fetchone()
            if row is None:
                await conn.execute(
                    "INSERT INTO users "
                    "(user_id, username, token_hash, created_at, is_active, "
                    "auth_source) "
                    "VALUES (%s, %s, '', %s, TRUE, %s) "
                    "ON CONFLICT DO NOTHING",
                    (user_id, user_id, sessions.now_iso(), source),
                )
                cur = await conn.execute(columns + "username = %s", (user_id,))
                row = await cur.fetchone()
        if row is None:  # pragma: no cover - insert just wrote it
            raise RuntimeError(f"user {user_id!r} could not be created")
        effective_source = row[3] or "token"
        if effective_source != source:
            raise serverdb.UserIdConflict(user_id)
        return {
            "user_id": row[0],
            "username": row[1],
            "is_active": bool(row[2]),
            "auth_source": row[3],
        }

    async def get_user_by_username(
        self, username: str
    ) -> dict[str, Any] | None:
        async with self._require_pool().connection() as conn:
            cur = await conn.execute(
                "SELECT user_id, username, is_active, auth_source "
                "FROM users WHERE username = %s",
                (username,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return {
            "user_id": row[0],
            "username": row[1],
            "is_active": bool(row[2]),
            "auth_source": row[3],
        }

    async def count_provisioned_users(self) -> int:
        async with self._require_pool().connection() as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) FROM users "
                "WHERE auth_source IS NOT NULL "
                "AND auth_source != '' "
                "AND auth_source != 'token'"
            )
            row = await cur.fetchone()
        return row[0]

    async def list_users(self) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row

        async with self._require_pool().connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT user_id, username, created_at, is_active, auth_source "
                    "FROM users ORDER BY created_at"
                )
                rows = await cur.fetchall()
        return [
            {
                "user_id": row["user_id"],
                "username": row["username"],
                "created_at": row["created_at"],
                "is_active": bool(row["is_active"]),
                "auth_source": row["auth_source"],
            }
            for row in rows
        ]

    async def revoke_user(self, username: str) -> bool:
        async with self._require_pool().connection() as conn:
            cur = await conn.execute(
                "UPDATE users SET is_active = FALSE WHERE username = %s",
                (username,),
            )
            return cur.rowcount > 0

    async def get_user_by_token_hash(
        self, token_hash: str
    ) -> dict[str, Any] | None:
        from psycopg.rows import dict_row

        async with self._require_pool().connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT user_id, username, is_active FROM users "
                    "WHERE token_hash = %s",
                    (token_hash,),
                )
                row = await cur.fetchone()
        if row is None:
            return None
        return {
            "user_id": row["user_id"],
            "username": row["username"],
            "is_active": bool(row["is_active"]),
        }

    async def create_session(
        self, user_id: str, thread_id: str, title: str | None
    ) -> dict[str, Any]:
        await self.upsert_session(user_id, thread_id, title)
        row = await self.get_session(user_id, thread_id)
        if row is None:  # pragma: no cover - upsert just wrote it
            raise RuntimeError("session insert did not persist")
        return row

    async def upsert_session(
        self, user_id: str, thread_id: str, title: str | None
    ) -> None:
        now = sessions.now_iso()
        truncated = sessions.truncate_title(title)
        async with self._require_pool().connection() as conn:
            await conn.execute(
                """
                INSERT INTO sessions
                    (thread_id, user_id, created_at, updated_at, title)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (thread_id) DO UPDATE SET
                    updated_at = EXCLUDED.updated_at,
                    title = COALESCE(NULLIF(EXCLUDED.title, ''), sessions.title)
                WHERE sessions.user_id = EXCLUDED.user_id
                """,
                (thread_id, user_id, now, now, truncated),
            )

    async def list_sessions(
        self, user_id: str, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row

        async with self._require_pool().connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT thread_id, created_at, updated_at, title "
                    "FROM sessions WHERE user_id = %s "
                    "ORDER BY updated_at DESC LIMIT %s",
                    (user_id, limit),
                )
                rows = await cur.fetchall()
        return [
            {
                "thread_id": row["thread_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "title": row["title"],
            }
            for row in rows
        ]

    async def get_session(
        self, user_id: str, thread_id: str
    ) -> dict[str, Any] | None:
        from psycopg.rows import dict_row

        async with self._require_pool().connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT thread_id, created_at, updated_at, title "
                    "FROM sessions WHERE thread_id = %s AND user_id = %s",
                    (thread_id, user_id),
                )
                row = await cur.fetchone()
        if row is None:
            return None
        return dict(row)

    async def delete_session(self, user_id: str, thread_id: str) -> None:
        async with self._require_pool().connection() as conn:
            await conn.execute(
                "DELETE FROM sessions WHERE thread_id = %s AND user_id = %s",
                (thread_id, user_id),
            )

    @asynccontextmanager
    async def checkpointer(self, user_id: str) -> AsyncIterator[Any]:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        # One saver per user gives each its own lock; the shared pool provides
        # connections, so users are not serialized behind a single lock.
        yield AsyncPostgresSaver(self._require_pool())


def database_url() -> str | None:
    """Return the configured Postgres DSN, or ``None`` for SQLite."""
    url = os.environ.get("HARNESS_DATABASE_URL", "").strip()
    return url or None


def make_store(cwd: Path) -> Store:
    """Build the store selected by ``HARNESS_DATABASE_URL``."""
    url = database_url()
    if url is not None:
        return PostgresStore(url)
    return SqliteStore(cwd)


@asynccontextmanager
async def open_store(cwd: Path) -> AsyncIterator[Store]:
    """Open and initialize a store, closing it when the block exits."""
    store = make_store(cwd)
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()
