"""Tests for the global server.db users table."""

from __future__ import annotations

import hashlib
from pathlib import Path

import aiosqlite
import pytest

from langchain_harness.serverdb import (
    add_user,
    generate_token,
    get_user_id,
    hash_token,
    init_server_db,
    list_users,
    lookup_by_token_hash,
    revoke_user,
    server_db_path,
    verify_token,
)


@pytest.fixture
async def conn(tmp_path: Path) -> aiosqlite.Connection:
    db_path = tmp_path / "server.db"
    async with aiosqlite.connect(str(db_path)) as conn:
        await init_server_db(conn)
        yield conn


def test_generate_token_has_prefix() -> None:
    token = generate_token()
    assert token.startswith("harn_")
    assert len(token) > 10


def test_hash_and_verify_token() -> None:
    token = generate_token()
    h = hash_token(token)
    assert h == hashlib.sha256(token.encode()).hexdigest()
    assert verify_token(token, h) is True
    assert verify_token("wrong", h) is False


def test_verify_token_rejects_wrong_token() -> None:
    token = generate_token()
    h = hash_token(token)
    assert verify_token(token, h) is True
    assert verify_token(token[:-1] + "x", h) is False


@pytest.mark.asyncio
async def test_add_user_returns_token(conn: aiosqlite.Connection) -> None:
    token = await add_user(conn, "alice")
    assert token.startswith("harn_")

    # Verify the user can be looked up by token hash
    h = hash_token(token)
    user = await lookup_by_token_hash(conn, h)
    assert user is not None
    assert user["username"] == "alice"
    assert user["is_active"] is True


@pytest.mark.asyncio
async def test_add_duplicate_user_raises(conn: aiosqlite.Connection) -> None:
    await add_user(conn, "alice")
    with pytest.raises(ValueError, match="already exists"):
        await add_user(conn, "alice")


@pytest.mark.asyncio
async def test_list_users(conn: aiosqlite.Connection) -> None:
    await add_user(conn, "alice")
    await add_user(conn, "bob")
    users = await list_users(conn)
    assert len(users) == 2
    assert {u["username"] for u in users} == {"alice", "bob"}


@pytest.mark.asyncio
async def test_revoke_user(conn: aiosqlite.Connection) -> None:
    token = await add_user(conn, "alice")
    h = hash_token(token)

    found = await revoke_user(conn, "alice")
    assert found is True

    user = await lookup_by_token_hash(conn, h)
    assert user is not None
    assert user["is_active"] is False


@pytest.mark.asyncio
async def test_revoke_nonexistent_returns_false(conn: aiosqlite.Connection) -> None:
    found = await revoke_user(conn, "nobody")
    assert found is False


@pytest.mark.asyncio
async def test_lookup_nonexistent_token_returns_none(conn: aiosqlite.Connection) -> None:
    h = hash_token("harn_nonexistent")
    assert await lookup_by_token_hash(conn, h) is None


@pytest.mark.asyncio
async def test_lookup_revoked_user_returns_inactive(conn: aiosqlite.Connection) -> None:
    """Revoked users are returned with is_active=False (auth layer filters)."""
    token = await add_user(conn, "alice")
    h = hash_token(token)
    await revoke_user(conn, "alice")
    user = await lookup_by_token_hash(conn, h)
    assert user is not None
    assert user["is_active"] is False


@pytest.mark.asyncio
async def test_get_user_id(conn: aiosqlite.Connection) -> None:
    await add_user(conn, "alice")
    uid = await get_user_id(conn, "alice")
    assert uid is not None
    assert len(uid) > 0
    assert await get_user_id(conn, "nobody") is None


def test_server_db_path(tmp_path: Path) -> None:
    path = server_db_path(tmp_path)
    assert path == tmp_path / ".harness" / "server.db"
