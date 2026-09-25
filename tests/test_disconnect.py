"""Test SSE disconnect cancellation with a fake agent.

The httpx ASGI transport does not model real TCP disconnects — it collects
the entire response body before returning, so ``request.is_disconnected()``
never fires and the response generator is never cancelled through the
transport.  Instead we test the cancellation path directly: cancel the
``stream_sse_response`` consumer and verify the cancellation propagates to
the underlying agent.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import httpx
import pytest

from langchain_harness.api.app import create_app
from langchain_harness.api.sse import stream_sse_response
from langchain_harness.config import Config
from langchain_harness.serverdb import add_user, init_server_db, server_db_path


class _Chunk:
    def __init__(self, content: Any) -> None:
        self.content = content


class _SlowAgent:
    """Agent that yields tokens slowly, observing cancellation."""

    def __init__(self) -> None:
        self.was_cancelled = False

    async def astream_events(self, input_data: Any, config: Any, *, version: str):
        try:
            for i in range(10000):
                await asyncio.sleep(0.001)
                yield {
                    "event": "on_chat_model_stream",
                    "data": {"chunk": _Chunk(f"chunk-{i}")},
                }
        except asyncio.CancelledError:
            self.was_cancelled = True
            raise

    async def aget_state(self, config: Any) -> Any:
        class _State:
            values = {}
        return _State()


class _FakeRequest:
    """Minimal Request stub whose is_disconnected always returns False."""

    async def is_disconnected(self) -> bool:
        return False


class _FakeCache:
    def __init__(self, agent: Any) -> None:
        self._agent = agent

    async def get_agent(self, user_id: str) -> Any:
        return self._agent

    async def close_all(self) -> None:
        pass


@pytest.mark.asyncio
async def test_disconnect_cancels_streaming(tmp_path, monkeypatch) -> None:
    """When stream_sse_response's consumer is cancelled, the agent is cancelled.

    We verify the SSE handler's task-cancellation path works by directly
    cancelling the stream_sse_response generator and checking that the
    CancelledError propagates to the underlying agent.
    """
    monkeypatch.chdir(tmp_path)

    slow_agent = _SlowAgent()
    request = _FakeRequest()

    async def _consume():
        async for _chunk in stream_sse_response(slow_agent, "hello", "thread-1", request):
            pass

    task = asyncio.create_task(_consume())
    # Let the agent enter its streaming loop
    await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    # The cancellation must have propagated to the agent
    assert slow_agent.was_cancelled, "CancelledError did not propagate to the agent"


@pytest.mark.asyncio
async def test_streaming_completes_normally(tmp_path, monkeypatch) -> None:
    """Verify that a normal (non-cancelled) stream completes with done event."""
    monkeypatch.chdir(tmp_path)

    import aiosqlite

    db_path = server_db_path(tmp_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(db_path)) as conn:
        await init_server_db(conn)
        token = await add_user(conn, "alice")

    config = Config(provider="anthropic", model="test-model")
    app = create_app(tmp_path, config)

    class _QuickAgent:
        async def astream_events(self, input_data: Any, config: Any, *, version: str):
            yield {"event": "on_chat_model_stream", "data": {"chunk": _Chunk("hi")}}
            yield {"event": "on_chat_model_stream", "data": {"chunk": _Chunk(" there")}}

        async def aget_state(self, config: Any) -> Any:
            class _State:
                values = {}
            return _State()

    fake_cache = _FakeCache(_QuickAgent())

    async with app.router.lifespan_context(app):
        app.state.get_agent = fake_cache.get_agent
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/sessions",
                json={"title": "test"},
                headers={"Authorization": f"Bearer {token}"},
            )
            session_id = resp.json()["session_id"]

            resp = await client.post(
                f"/v1/sessions/{session_id}/messages",
                json={"content": "hello"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200
            body = resp.text
            assert "event: token" in body
            assert "event: done" in body
