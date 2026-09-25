"""Integration tests for the multi-user web API using httpx ASGI transport."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from langchain_harness.api.app import create_app
from langchain_harness.config import Config
from langchain_harness.serverdb import add_user, hash_token, init_server_db, server_db_path


class _FakeState:
    def __init__(self, messages: list[Any] | None = None) -> None:
        self.values = {"messages": messages or []} if messages else {}


class _FakeMsg:
    """Fake LangChain message with .type and .content for history tests."""

    def __init__(self, msg_type: str, content: str) -> None:
        self.type = msg_type
        self.content = content


class _FakeAgent:
    """Fake agent for testing SSE streaming and history."""

    def __init__(
        self,
        events: list[dict[str, Any]] | None = None,
        state_messages: list[Any] | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self._events = events or []
        self._state_messages = state_messages or []
        self.stream_calls: list[tuple[Any, Any]] = []
        self.checkpointer = checkpointer

    async def astream_events(self, input_data: Any, config: Any, *, version: str):
        self.stream_calls.append((input_data, config))
        for event in self._events:
            yield event

    async def aget_state(self, config: Any) -> _FakeState:
        return _FakeState(messages=self._state_messages)


class _FakeAgentCache:
    """Minimal cache that returns a fake agent without building anything."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    async def get_agent(self, user_id: str) -> Any:
        return self._agent

    async def close_all(self) -> None:
        pass


class _FakeCheckpointer:
    """Records adelete_thread calls for checkpoint deletion tests."""

    def __init__(self) -> None:
        self.deleted_threads: list[str] = []

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)


@pytest.fixture
async def test_env(tmp_path, monkeypatch):
    """Set up a full test environment with two users and their tokens."""
    monkeypatch.chdir(tmp_path)

    # Create server.db and add two users
    db_path = server_db_path(tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    import aiosqlite

    token_a: str = ""
    token_b: str = ""

    async with aiosqlite.connect(str(db_path)) as conn:
        await init_server_db(conn)
        token_a = await add_user(conn, "alice")
        token_b = await add_user(conn, "bob")

    config = Config(provider="anthropic", model="test-model")

    app = create_app(tmp_path, config)

    # Create a fake agent that yields some token events
    fake_agent = _FakeAgent(events=[
        {"event": "on_chat_model_stream", "data": {"chunk": _Chunk("Hello ")}},
        {"event": "on_chat_model_stream", "data": {"chunk": _Chunk("world!")}},
    ])
    fake_cache = _FakeAgentCache(fake_agent)

    # We need to manage the lifespan manually
    async with app.router.lifespan_context(app):
        app.state.get_agent = fake_cache.get_agent

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield {
                "client": client,
                "token_a": token_a,
                "token_b": token_b,
                "agent": fake_agent,
                "tmp_path": tmp_path,
            }


class _Chunk:
    def __init__(self, content: Any) -> None:
        self.content = content


# --- auth tests ---


@pytest.mark.asyncio
async def test_auth_valid_token_returns_200(test_env) -> None:
    resp = await test_env["client"].get(
        "/v1/sessions",
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_invalid_token_returns_401(test_env) -> None:
    resp = await test_env["client"].get(
        "/v1/sessions",
        headers={"Authorization": "Bearer harn_invalid_token"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_missing_header_returns_401(test_env) -> None:
    resp = await test_env["client"].get("/v1/sessions")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_revoked_user_returns_401(test_env, tmp_path) -> None:
    # Revoke alice
    from langchain_harness.serverdb import revoke_user
    import aiosqlite

    async with aiosqlite.connect(str(server_db_path(tmp_path))) as conn:
        await revoke_user(conn, "alice")

    resp = await test_env["client"].get(
        "/v1/sessions",
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    assert resp.status_code == 401


# --- privacy tests ---


@pytest.mark.asyncio
async def test_privacy_user_b_cannot_see_user_a_session(test_env) -> None:
    # Alice creates a session
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "alice's session"},
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    # Bob tries to access it → 404
    resp = await test_env["client"].get(
        f"/v1/sessions/{session_id}/history",
        headers={"Authorization": f"Bearer {test_env['token_b']}"},
    )
    assert resp.status_code == 404

    # Bob tries to delete it → 404
    resp = await test_env["client"].delete(
        f"/v1/sessions/{session_id}",
        headers={"Authorization": f"Bearer {test_env['token_b']}"},
    )
    assert resp.status_code == 404


# --- sessions CRUD ---


@pytest.mark.asyncio
async def test_create_session(test_env) -> None:
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "test session"},
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["title"] == "test session"
    assert "created_at" in data


@pytest.mark.asyncio
async def test_list_sessions_only_own(test_env) -> None:
    # Alice creates 2 sessions
    for i in range(2):
        await test_env["client"].post(
            "/v1/sessions",
            json={"title": f"alice-{i}"},
            headers={"Authorization": f"Bearer {test_env['token_a']}"},
        )
    # Bob creates 1 session
    await test_env["client"].post(
        "/v1/sessions",
        json={"title": "bob-0"},
        headers={"Authorization": f"Bearer {test_env['token_b']}"},
    )

    # Alice sees only her 2
    resp = await test_env["client"].get(
        "/v1/sessions",
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    assert resp.status_code == 200
    sessions = resp.json()
    assert len(sessions) == 2
    assert all(s["title"].startswith("alice") for s in sessions)

    # Bob sees only his 1
    resp = await test_env["client"].get(
        "/v1/sessions",
        headers={"Authorization": f"Bearer {test_env['token_b']}"},
    )
    assert resp.status_code == 200
    sessions = resp.json()
    assert len(sessions) == 1
    assert sessions[0]["title"] == "bob-0"


@pytest.mark.asyncio
async def test_delete_session(test_env) -> None:
    # Create
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "to delete"},
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    session_id = resp.json()["session_id"]

    # Delete
    resp = await test_env["client"].delete(
        f"/v1/sessions/{session_id}",
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    assert resp.status_code == 200

    # List should be empty
    resp = await test_env["client"].get(
        "/v1/sessions",
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    assert len(resp.json()) == 0


@pytest.mark.asyncio
async def test_delete_session_calls_checkpointer_adelete_thread(tmp_path, monkeypatch) -> None:
    """When the agent exposes a checkpointer, delete_session awaits
    adelete_thread with the correct thread_id and still deletes the row."""
    monkeypatch.chdir(tmp_path)

    import aiosqlite
    from langchain_harness.api.app import create_app
    from langchain_harness.serverdb import add_user, init_server_db, server_db_path

    db_path = server_db_path(tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(db_path)) as conn:
        await init_server_db(conn)
        token = await add_user(conn, "alice")

    checkpointer = _FakeCheckpointer()
    fake_agent = _FakeAgent(checkpointer=checkpointer)
    fake_cache = _FakeAgentCache(fake_agent)

    config = Config(provider="anthropic", model="test-model")
    app = create_app(tmp_path, config)

    async with app.router.lifespan_context(app):
        app.state.get_agent = fake_cache.get_agent
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Create a session
            resp = await client.post(
                "/v1/sessions",
                json={"title": "checkpoint test"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200
            session_id = resp.json()["session_id"]

            # Delete it
            resp = await client.delete(
                f"/v1/sessions/{session_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200

            # Checkpointer was awaited with the right thread_id
            assert checkpointer.deleted_threads == [session_id]

            # Session row was also deleted
            resp = await client.get(
                "/v1/sessions",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.json() == []


async def test_delete_foreign_session_404(test_env) -> None:
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "alice's"},
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    session_id = resp.json()["session_id"]

    resp = await test_env["client"].delete(
        f"/v1/sessions/{session_id}",
        headers={"Authorization": f"Bearer {test_env['token_b']}"},
    )
    assert resp.status_code == 404


# --- SSE streaming ---


@pytest.mark.asyncio
async def test_sse_streaming_shows_tokens_and_done(test_env) -> None:
    # Create a session first
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "sse test"},
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    session_id = resp.json()["session_id"]

    # Send a message and stream the response
    resp = await test_env["client"].post(
        f"/v1/sessions/{session_id}/messages",
        json={"content": "hello"},
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    body = resp.text
    assert "event: token" in body
    assert "Hello " in body
    assert "world!" in body
    assert "event: done" in body


@pytest.mark.asyncio
async def test_sse_404_on_foreign_session(test_env) -> None:
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "alice's"},
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    session_id = resp.json()["session_id"]

    resp = await test_env["client"].post(
        f"/v1/sessions/{session_id}/messages",
        json={"content": "hello"},
        headers={"Authorization": f"Bearer {test_env['token_b']}"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_compact_endpoint_streams(test_env) -> None:
    # Create a session
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "compact test"},
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    session_id = resp.json()["session_id"]

    resp = await test_env["client"].post(
        f"/v1/sessions/{session_id}/compact",
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "event: done" in resp.text


# --- history ---


@pytest.mark.asyncio
async def test_history_empty_session(test_env) -> None:
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "history test"},
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    session_id = resp.json()["session_id"]

    resp = await test_env["client"].get(
        f"/v1/sessions/{session_id}/history",
        headers={"Authorization": f"Bearer {test_env['token_a']}"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_history_returns_messages_after_send(tmp_path, monkeypatch) -> None:
    """Send a message, then fetch history — real message shapes must be present."""
    monkeypatch.chdir(tmp_path)

    import aiosqlite

    db_path = server_db_path(tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(db_path)) as conn:
        await init_server_db(conn)
        token = await add_user(conn, "alice")

    config = Config(provider="anthropic", model="test-model")
    app = create_app(tmp_path, config)

    fake_agent = _FakeAgent(
        events=[
            {"event": "on_chat_model_stream", "data": {"chunk": _Chunk("Hello")}},
        ],
        state_messages=[
            _FakeMsg("human", "hello"),
            _FakeMsg("ai", "Hello"),
        ],
    )
    fake_cache = _FakeAgentCache(fake_agent)

    async with app.router.lifespan_context(app):
        app.state.get_agent = fake_cache.get_agent
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/sessions",
                json={"title": "history test"},
                headers={"Authorization": f"Bearer {token}"},
            )
            session_id = resp.json()["session_id"]

            # Send a message (streams the fake response)
            resp = await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"content": "hello"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200

            # Fetch history — should contain real message shapes
            resp = await client.get(
                f"/v1/sessions/{session_id}/history",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200
            messages = resp.json()
            assert len(messages) == 2
            assert messages[0]["role"] == "human"
            assert messages[0]["content"] == "hello"
            assert messages[1]["role"] == "ai"
            assert messages[1]["content"] == "Hello"


# --- smoke isolation ---


@pytest.mark.asyncio
async def test_smoke_isolation_two_users(tmp_path, monkeypatch):
    """Smoke test: add two users, list sessions for each, assert isolation."""
    monkeypatch.chdir(tmp_path)

    import aiosqlite
    from langchain_harness.serverdb import add_user, init_server_db

    db_path = server_db_path(tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(db_path)) as conn:
        await init_server_db(conn)
        token_a = await add_user(conn, "alice")
        token_b = await add_user(conn, "bob")

    config = Config(provider="anthropic", model="test-model")
    app = create_app(tmp_path, config)

    fake_agent = _FakeAgent(events=[])
    fake_cache = _FakeAgentCache(fake_agent)

    async with app.router.lifespan_context(app):
        app.state.get_agent = fake_cache.get_agent
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Alice creates 3 sessions
            for i in range(3):
                await client.post(
                    "/v1/sessions",
                    json={"title": f"a{i}"},
                    headers={"Authorization": f"Bearer {token_a}"},
                )
            # Bob creates 1 session
            await client.post(
                "/v1/sessions",
                json={"title": "b0"},
                headers={"Authorization": f"Bearer {token_b}"},
            )

            # Alice sees 3
            resp = await client.get(
                "/v1/sessions",
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert resp.status_code == 200
            assert len(resp.json()) == 3

            # Bob sees 1
            resp = await client.get(
                "/v1/sessions",
                headers={"Authorization": f"Bearer {token_b}"},
            )
            assert resp.status_code == 200
            assert len(resp.json()) == 1
