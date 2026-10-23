"""Integration tests for the multi-user web API using httpx ASGI transport."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from langchain_harness.api.app import create_app
from langchain_harness.config import Config
from langchain_harness.memory import inbox_dir, list_inbox


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

    def __init__(self, agent: Any, shared_model: Any | None = None) -> None:
        self._agent = agent
        self.shared_model = shared_model

    async def get_agent(self, user_id: str) -> Any:
        return self._agent

    async def close_all(self) -> None:
        pass


class _FakeSummarizingModel:
    """Fake shared model used by the summarize endpoint."""

    def __init__(self, content: str = "- summary bullet") -> None:
        self.content = content
        self.calls = 0

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.calls += 1
        return type("Result", (), {"content": self.content})()


class _FailingSummarizingModel:
    """Shared model whose summarization call always raises."""

    async def ainvoke(self, messages: list[Any]) -> Any:
        raise RuntimeError("model down")


class _FakeCheckpointer:
    """Records adelete_thread calls for checkpoint deletion tests."""

    def __init__(self) -> None:
        self.deleted_threads: list[str] = []

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)


@pytest.fixture
async def test_env(tmp_path, monkeypatch):
    """Set up a full test environment with two trust-mode identities."""
    monkeypatch.chdir(tmp_path)

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
        app.state.agent_cache = fake_cache

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield {
                "client": client,
                "agent": fake_agent,
                "tmp_path": tmp_path,
            }


class _Chunk:
    def __init__(self, content: Any) -> None:
        self.content = content


# --- health tests ---


@pytest.mark.asyncio
async def test_healthz_needs_no_auth(test_env) -> None:
    resp = await test_env["client"].get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readyz_needs_no_auth(test_env) -> None:
    resp = await test_env["client"].get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


@pytest.mark.asyncio
async def test_readyz_503_before_startup(tmp_path) -> None:
    app = create_app(tmp_path, Config(provider="anthropic", model="test-model"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json() == {"status": "starting"}


# --- auth tests ---


@pytest.mark.asyncio
async def test_auth_valid_header_returns_200(test_env) -> None:
    resp = await test_env["client"].get(
        "/v1/sessions",
        headers={"X-User-Id": "alice"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_missing_header_returns_400(test_env) -> None:
    resp = await test_env["client"].get("/v1/sessions")
    assert resp.status_code == 400
    assert "X-User-Id" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_auth_invalid_header_returns_400(test_env) -> None:
    resp = await test_env["client"].get(
        "/v1/sessions",
        headers={"X-User-Id": "../etc"},
    )
    assert resp.status_code == 400


# --- privacy tests ---


@pytest.mark.asyncio
async def test_privacy_user_b_cannot_see_user_a_session(test_env) -> None:
    # Alice creates a session
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "alice's session"},
        headers={"X-User-Id": "alice"},
    )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    # Bob tries to access it → 404
    resp = await test_env["client"].get(
        f"/v1/sessions/{session_id}/history",
        headers={"X-User-Id": "bob"},
    )
    assert resp.status_code == 404

    # Bob tries to delete it → 404
    resp = await test_env["client"].delete(
        f"/v1/sessions/{session_id}",
        headers={"X-User-Id": "bob"},
    )
    assert resp.status_code == 404


# --- sessions CRUD ---


@pytest.mark.asyncio
async def test_create_session(test_env) -> None:
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "test session"},
        headers={"X-User-Id": "alice"},
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
            headers={"X-User-Id": "alice"},
        )
    # Bob creates 1 session
    await test_env["client"].post(
        "/v1/sessions",
        json={"title": "bob-0"},
        headers={"X-User-Id": "bob"},
    )

    # Alice sees only her 2
    resp = await test_env["client"].get(
        "/v1/sessions",
        headers={"X-User-Id": "alice"},
    )
    assert resp.status_code == 200
    sessions = resp.json()
    assert len(sessions) == 2
    assert all(s["title"].startswith("alice") for s in sessions)

    # Bob sees only his 1
    resp = await test_env["client"].get(
        "/v1/sessions",
        headers={"X-User-Id": "bob"},
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
        headers={"X-User-Id": "alice"},
    )
    session_id = resp.json()["session_id"]

    # Delete
    resp = await test_env["client"].delete(
        f"/v1/sessions/{session_id}",
        headers={"X-User-Id": "alice"},
    )
    assert resp.status_code == 200

    # List should be empty
    resp = await test_env["client"].get(
        "/v1/sessions",
        headers={"X-User-Id": "alice"},
    )
    assert len(resp.json()) == 0


@pytest.mark.asyncio
async def test_delete_session_calls_checkpointer_adelete_thread(tmp_path, monkeypatch) -> None:
    """When the agent exposes a checkpointer, delete_session awaits
    adelete_thread with the correct thread_id and still deletes the row."""
    monkeypatch.chdir(tmp_path)

    checkpointer = _FakeCheckpointer()
    fake_agent = _FakeAgent(checkpointer=checkpointer)
    fake_cache = _FakeAgentCache(fake_agent)

    config = Config(provider="anthropic", model="test-model")
    app = create_app(tmp_path, config)

    async with app.router.lifespan_context(app):
        app.state.get_agent = fake_cache.get_agent
        app.state.agent_cache = fake_cache
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Create a session
            resp = await client.post(
                "/v1/sessions",
                json={"title": "checkpoint test"},
                headers={"X-User-Id": "alice"},
            )
            assert resp.status_code == 200
            session_id = resp.json()["session_id"]

            # Delete it
            resp = await client.delete(
                f"/v1/sessions/{session_id}",
                headers={"X-User-Id": "alice"},
            )
            assert resp.status_code == 200

            # Checkpointer was awaited with the right thread_id
            assert checkpointer.deleted_threads == [session_id]

            # Session row was also deleted
            resp = await client.get(
                "/v1/sessions",
                headers={"X-User-Id": "alice"},
            )
            assert resp.json() == []


async def test_delete_foreign_session_404(test_env) -> None:
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "alice's"},
        headers={"X-User-Id": "alice"},
    )
    session_id = resp.json()["session_id"]

    resp = await test_env["client"].delete(
        f"/v1/sessions/{session_id}",
        headers={"X-User-Id": "bob"},
    )
    assert resp.status_code == 404


# --- SSE streaming ---


@pytest.mark.asyncio
async def test_sse_streaming_shows_tokens_and_done(test_env) -> None:
    # Create a session first
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "sse test"},
        headers={"X-User-Id": "alice"},
    )
    session_id = resp.json()["session_id"]

    # Send a message and stream the response
    resp = await test_env["client"].post(
        f"/v1/sessions/{session_id}/messages",
        json={"content": "hello"},
        headers={"X-User-Id": "alice"},
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
        headers={"X-User-Id": "alice"},
    )
    session_id = resp.json()["session_id"]

    resp = await test_env["client"].post(
        f"/v1/sessions/{session_id}/messages",
        json={"content": "hello"},
        headers={"X-User-Id": "bob"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_compact_endpoint_streams(test_env) -> None:
    # Create a session
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "compact test"},
        headers={"X-User-Id": "alice"},
    )
    session_id = resp.json()["session_id"]

    resp = await test_env["client"].post(
        f"/v1/sessions/{session_id}/compact",
        headers={"X-User-Id": "alice"},
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
        headers={"X-User-Id": "alice"},
    )
    session_id = resp.json()["session_id"]

    resp = await test_env["client"].get(
        f"/v1/sessions/{session_id}/history",
        headers={"X-User-Id": "alice"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_history_returns_messages_after_send(tmp_path, monkeypatch) -> None:
    """Send a message, then fetch history — real message shapes must be present."""
    monkeypatch.chdir(tmp_path)

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
        app.state.agent_cache = fake_cache
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/sessions",
                json={"title": "history test"},
                headers={"X-User-Id": "alice"},
            )
            session_id = resp.json()["session_id"]

            # Send a message (streams the fake response)
            resp = await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"content": "hello"},
                headers={"X-User-Id": "alice"},
            )
            assert resp.status_code == 200

            # Fetch history — should contain real message shapes
            resp = await client.get(
                f"/v1/sessions/{session_id}/history",
                headers={"X-User-Id": "alice"},
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
    """Smoke test: two identities via trust mode, list sessions for each, assert isolation."""
    monkeypatch.chdir(tmp_path)

    config = Config(provider="anthropic", model="test-model")
    app = create_app(tmp_path, config)

    fake_agent = _FakeAgent(events=[])
    fake_cache = _FakeAgentCache(fake_agent)

    async with app.router.lifespan_context(app):
        app.state.get_agent = fake_cache.get_agent
        app.state.agent_cache = fake_cache
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Alice creates 3 sessions
            for i in range(3):
                await client.post(
                    "/v1/sessions",
                    json={"title": f"a{i}"},
                    headers={"X-User-Id": "alice"},
                )
            # Bob creates 1 session
            await client.post(
                "/v1/sessions",
                json={"title": "b0"},
                headers={"X-User-Id": "bob"},
            )

            # Alice sees 3
            resp = await client.get(
                "/v1/sessions",
                headers={"X-User-Id": "alice"},
            )
            assert resp.status_code == 200
            assert len(resp.json()) == 3

            # Bob sees 1
            resp = await client.get(
                "/v1/sessions",
                headers={"X-User-Id": "bob"},
            )
            assert resp.status_code == 200
            assert len(resp.json()) == 1


# --- session memory ---


async def _make_app_env(tmp_path, monkeypatch, *, agent, shared_model):
    """Create an app whose per-user agent and shared model are fakes."""
    monkeypatch.chdir(tmp_path)

    config = Config(provider="anthropic", model="test-model")
    app = create_app(tmp_path, config)
    cache = _FakeAgentCache(agent, shared_model=shared_model)
    return app, cache


@pytest.mark.asyncio
async def test_summarize_endpoint_queues_note(tmp_path, monkeypatch) -> None:
    agent = _FakeAgent(
        state_messages=[_FakeMsg("human", "hi"), _FakeMsg("ai", "hello")]
    )
    model = _FakeSummarizingModel("- summary bullet")
    app, cache = await _make_app_env(
        tmp_path, monkeypatch, agent=agent, shared_model=model
    )

    async with app.router.lifespan_context(app):
        app.state.get_agent = cache.get_agent
        app.state.agent_cache = cache
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/sessions",
                json={"title": "memory session"},
                headers={"X-User-Id": "alice"},
            )
            session_id = resp.json()["session_id"]

            resp = await client.post(
                f"/v1/sessions/{session_id}/summarize",
                headers={"X-User-Id": "alice"},
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "queued"

    notes = list_inbox(tmp_path)
    assert len(notes) == 1
    assert notes[0].thread_id == session_id
    assert notes[0].note_id == body["note_id"]
    assert notes[0].note_id.endswith(f"--{session_id}")
    assert notes[0].title == "memory session"
    assert notes[0].summary == "- summary bullet"
    assert notes[0].user_id is not None
    # Pending until a human approves it.
    assert not (tmp_path / ".harness" / "memory" / "notes.md").exists()


@pytest.mark.asyncio
async def test_summarize_endpoint_skips_without_shared_model(test_env) -> None:
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "no model"},
        headers={"X-User-Id": "alice"},
    )
    session_id = resp.json()["session_id"]

    resp = await test_env["client"].post(
        f"/v1/sessions/{session_id}/summarize",
        headers={"X-User-Id": "alice"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"status": "skipped", "reason": "no model available"}
    assert list_inbox(test_env["tmp_path"]) == []


@pytest.mark.asyncio
async def test_summarize_endpoint_skips_without_substantive_content(
    tmp_path, monkeypatch
) -> None:
    agent = _FakeAgent()
    model = _FakeSummarizingModel()
    app, cache = await _make_app_env(
        tmp_path, monkeypatch, agent=agent, shared_model=model
    )

    async with app.router.lifespan_context(app):
        app.state.get_agent = cache.get_agent
        app.state.agent_cache = cache
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/sessions",
                json={"title": "empty"},
                headers={"X-User-Id": "alice"},
            )
            session_id = resp.json()["session_id"]

            resp = await client.post(
                f"/v1/sessions/{session_id}/summarize",
                headers={"X-User-Id": "alice"},
            )

    assert resp.status_code == 200
    assert resp.json() == {"status": "skipped", "reason": "not captured"}
    assert model.calls == 0
    assert list_inbox(tmp_path) == []


@pytest.mark.asyncio
async def test_summarize_endpoint_skips_when_model_raises(tmp_path, monkeypatch) -> None:
    agent = _FakeAgent(
        state_messages=[_FakeMsg("human", "hi"), _FakeMsg("ai", "hello")]
    )
    app, cache = await _make_app_env(
        tmp_path, monkeypatch, agent=agent, shared_model=_FailingSummarizingModel()
    )

    async with app.router.lifespan_context(app):
        app.state.get_agent = cache.get_agent
        app.state.agent_cache = cache
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/sessions",
                json={"title": "boom"},
                headers={"X-User-Id": "alice"},
            )
            session_id = resp.json()["session_id"]

            resp = await client.post(
                f"/v1/sessions/{session_id}/summarize",
                headers={"X-User-Id": "alice"},
            )

    assert resp.status_code == 200
    assert resp.json() == {"status": "skipped", "reason": "not captured"}
    assert list_inbox(tmp_path) == []


@pytest.mark.asyncio
async def test_summarize_endpoint_404_on_foreign_session(test_env) -> None:
    resp = await test_env["client"].post(
        "/v1/sessions",
        json={"title": "alice's"},
        headers={"X-User-Id": "alice"},
    )
    session_id = resp.json()["session_id"]

    resp = await test_env["client"].post(
        f"/v1/sessions/{session_id}/summarize",
        headers={"X-User-Id": "bob"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_session_does_not_write_memory_note(tmp_path, monkeypatch) -> None:
    agent = _FakeAgent(
        state_messages=[_FakeMsg("human", "hi"), _FakeMsg("ai", "hello")]
    )
    model = _FakeSummarizingModel()
    app, cache = await _make_app_env(
        tmp_path, monkeypatch, agent=agent, shared_model=model
    )

    async with app.router.lifespan_context(app):
        app.state.get_agent = cache.get_agent
        app.state.agent_cache = cache
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/sessions",
                json={"title": "to delete"},
                headers={"X-User-Id": "alice"},
            )
            session_id = resp.json()["session_id"]

            resp = await client.delete(
                f"/v1/sessions/{session_id}",
                headers={"X-User-Id": "alice"},
            )

    assert resp.status_code == 200
    assert model.calls == 0
    assert not inbox_dir(tmp_path).exists()
    assert not (tmp_path / ".harness" / "memory" / "notes.md").exists()
