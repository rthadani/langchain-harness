"""Store contract tests: SQLite always, Postgres when a DSN is provided.

Run the Postgres half with:
    HARNESS_TEST_POSTGRES_URL=postgresql://... pytest tests/test_store.py
"""

from __future__ import annotations

import os
import uuid

import pytest

from langchain_harness.serverdb import UserIdConflict, hash_token
from langchain_harness.store import PostgresStore, SqliteStore, make_store

PG_URL = os.environ.get("HARNESS_TEST_POSTGRES_URL")


@pytest.fixture
async def sqlite_store(tmp_path):
    store = SqliteStore(tmp_path)
    await store.initialize()
    yield store
    await store.close()


async def _assert_user_lifecycle(store) -> None:
    token = await store.add_user("alice")
    assert token.startswith("harn_")

    with pytest.raises(ValueError, match="already exists"):
        await store.add_user("alice")

    users = await store.list_users()
    assert len(users) == 1
    assert users[0]["username"] == "alice"
    assert users[0]["is_active"] is True

    user = await store.get_user_by_token_hash(hash_token(token))
    assert user is not None
    assert user["user_id"] == users[0]["user_id"]
    assert user["is_active"] is True

    assert await store.revoke_user("alice") is True
    revoked = await store.get_user_by_token_hash(hash_token(token))
    assert revoked is not None
    assert revoked["is_active"] is False
    assert await store.revoke_user("nobody") is False


async def _assert_get_or_create_user(store, suffix: str) -> None:
    name = f"get-or-create-{suffix}"
    token_name = f"token-{suffix}"
    token2_name = f"token2-{suffix}"
    provisioned_name = f"provisioned-{suffix}"

    # Create with source.
    user = await store.get_or_create_user(name, source="trust")
    assert user["username"] == name
    assert user["is_active"] is True
    assert user["auth_source"] == "trust"

    # Idempotent re-call returns the same row.
    again = await store.get_or_create_user(name, source="trust")
    assert again["user_id"] == user["user_id"]

    # Same-source reuse by username: a token-backed row reused with token source.
    await store.add_user(token_name)
    token_user = await store.get_or_create_user(token_name, source="token")
    token_lookup = await store.get_user_by_username(token_name)
    assert token_user["user_id"] == token_lookup["user_id"]

    # Cross-source conflict raises UserIdConflict.
    with pytest.raises(UserIdConflict):
        await store.get_or_create_user(token_name, source="trust")

    # count_provisioned_users counts non-token rows only.
    before = await store.count_provisioned_users()
    await store.add_user(token2_name)
    after_token = await store.count_provisioned_users()
    assert after_token == before

    await store.get_or_create_user(provisioned_name, source="jwt")
    after_provisioned = await store.count_provisioned_users()
    assert after_provisioned == before + 1


async def _assert_session_isolation(store) -> None:
    first = await store.create_session("u1", "t1", "hello")
    assert first["thread_id"] == "t1"
    assert first["title"] == "hello"

    await store.upsert_session("u1", "t1", "second turn")
    rows = await store.list_sessions("u1")
    assert [r["thread_id"] for r in rows] == ["t1"]
    assert rows[0]["title"] == "second turn"

    await store.create_session("u2", "t2", "other user")
    assert await store.get_session("u2", "t1") is None
    assert await store.get_session("u1", "t2") is None
    assert [r["thread_id"] for r in await store.list_sessions("u2")] == ["t2"]

    # Deleting another user's thread must not remove it.
    await store.delete_session("u2", "t1")
    assert await store.get_session("u1", "t1") is not None

    await store.delete_session("u1", "t1")
    assert await store.get_session("u1", "t1") is None


async def _assert_checkpointer_usable(store) -> None:
    async with store.checkpointer("u1") as saver:
        # Exercising a query proves the saver is wired to the database.
        await saver.adelete_thread("no-such-thread")


@pytest.mark.asyncio
async def test_sqlite_user_lifecycle(sqlite_store) -> None:
    await _assert_user_lifecycle(sqlite_store)


@pytest.mark.asyncio
async def test_sqlite_session_isolation(sqlite_store) -> None:
    await _assert_session_isolation(sqlite_store)


@pytest.mark.asyncio
async def test_sqlite_checkpointer_usable(sqlite_store) -> None:
    await _assert_checkpointer_usable(sqlite_store)


@pytest.mark.asyncio
async def test_make_store_defaults_to_sqlite(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HARNESS_DATABASE_URL", raising=False)
    assert isinstance(make_store(tmp_path), SqliteStore)


@pytest.mark.asyncio
async def test_make_store_rejects_non_postgres_dsn(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_DATABASE_URL", "sqlite:///tmp/x.db")
    with pytest.raises(ValueError, match="postgresql://"):
        make_store(tmp_path)


@pytest.mark.asyncio
async def test_postgres_store_rejects_bad_scheme() -> None:
    with pytest.raises(ValueError, match="postgresql://"):
        PostgresStore("mysql://localhost/db")


@pytest.mark.skipif(not PG_URL, reason="HARNESS_TEST_POSTGRES_URL not set")
@pytest.mark.asyncio
async def test_postgres_concurrent_initialize() -> None:
    """Several replicas booting together must not race on CREATE TABLE."""
    import asyncio

    assert PG_URL is not None
    first = PostgresStore(PG_URL)
    await first.initialize()
    async with first._require_pool().connection() as conn:
        await conn.execute(
            "DROP TABLE IF EXISTS checkpoint_writes, checkpoint_blobs, "
            "checkpoints, checkpoint_migrations, sessions, users CASCADE"
        )
    await first.close()

    stores = [PostgresStore(PG_URL) for _ in range(4)]
    try:
        await asyncio.gather(*(store.initialize() for store in stores))
        for store in stores:
            await store.ping()
    finally:
        await asyncio.gather(*(store.close() for store in stores))


@pytest.mark.skipif(not PG_URL, reason="HARNESS_TEST_POSTGRES_URL not set")
@pytest.mark.asyncio
async def test_postgres_user_lifecycle() -> None:
    assert PG_URL is not None
    store = PostgresStore(PG_URL)
    await store.initialize()
    suffix = uuid.uuid4().hex[:8]
    username = f"pytest-{suffix}"
    try:
        token = await store.add_user(username)
        assert token.startswith("harn_")
        users = await store.list_users()
        assert any(u["username"] == username for u in users)
        user = await store.get_user_by_token_hash(hash_token(token))
        assert user is not None and user["is_active"] is True
        assert await store.revoke_user(username) is True
        revoked = await store.get_user_by_token_hash(hash_token(token))
        assert revoked is not None and revoked["is_active"] is False
    finally:
        async with store._require_pool().connection() as conn:
            await conn.execute("DELETE FROM users WHERE username = %s", (username,))
        await store.close()


@pytest.mark.skipif(not PG_URL, reason="HARNESS_TEST_POSTGRES_URL not set")
@pytest.mark.asyncio
async def test_postgres_session_isolation_and_checkpointer() -> None:
    assert PG_URL is not None
    store = PostgresStore(PG_URL)
    # Idempotent: the server calls this on every startup.
    await store.initialize()
    await store.initialize()

    suffix = uuid.uuid4().hex[:8]
    u1, u2 = f"u1-{suffix}", f"u2-{suffix}"
    t1, t2 = f"t1-{suffix}", f"t2-{suffix}"
    try:
        await store.create_session(u1, t1, "hello")
        await store.upsert_session(u1, t1, "second turn")
        await store.create_session(u2, t2, "other user")

        assert (await store.list_sessions(u1))[0]["title"] == "second turn"
        assert await store.get_session(u2, t1) is None
        await store.delete_session(u2, t1)
        assert await store.get_session(u1, t1) is not None

        await _assert_checkpointer_usable(store)

        await store.delete_session(u1, t1)
        assert await store.get_session(u1, t1) is None
    finally:
        async with store._require_pool().connection() as conn:
            await conn.execute(
                "DELETE FROM sessions WHERE thread_id IN (%s, %s)", (t1, t2)
            )
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_get_or_create_user(sqlite_store) -> None:
    await _assert_get_or_create_user(sqlite_store, uuid.uuid4().hex[:8])


@pytest.mark.skipif(not PG_URL, reason="HARNESS_TEST_POSTGRES_URL not set")
@pytest.mark.asyncio
async def test_postgres_get_or_create_user() -> None:
    assert PG_URL is not None
    store = PostgresStore(PG_URL)
    await store.initialize()
    suffix = uuid.uuid4().hex[:8]
    try:
        await _assert_get_or_create_user(store, suffix)
    finally:
        async with store._require_pool().connection() as conn:
            await conn.execute(
                "DELETE FROM users WHERE username LIKE %s",
                (f"%-{suffix}",),
            )
        await store.close()
