"""Tests for the shared stream_turn_events generator."""

from __future__ import annotations

from typing import Any

import pytest

from langchain_harness.events import stream_turn_events


class _FakeAgent:
    """Yields pre-configured events from astream_events."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self.calls: list[tuple[Any, Any, str]] = []

    async def astream_events(self, input_data: Any, config: Any, *, version: str):
        self.calls.append((input_data, config, version))
        for event in self._events:
            yield event


class _Chunk:
    def __init__(self, content: Any) -> None:
        self.content = content


@pytest.mark.asyncio
async def test_stream_turn_events_yields_token_events() -> None:
    agent = _FakeAgent([
        {"event": "on_chat_model_stream", "data": {"chunk": _Chunk("hello ")}},
        {"event": "on_chat_model_stream", "data": {"chunk": _Chunk("world")}},
    ])

    events = [e async for e in stream_turn_events(agent, "hi", "thread-1")]

    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 2
    assert token_events[0]["text"] == "hello "
    assert token_events[1]["text"] == "world"
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_stream_turn_events_yields_tool_events() -> None:
    agent = _FakeAgent([
        {"event": "on_tool_start", "name": "search", "data": {"input": {"q": "test"}}},
        {
            "event": "on_tool_end",
            "name": "search",
            "data": {"output": type("Out", (), {"content": "result"})},
        },
    ])

    events = [e async for e in stream_turn_events(agent, "search", "thread-1")]

    starts = [e for e in events if e["type"] == "tool_start"]
    assert len(starts) == 1
    assert starts[0]["name"] == "search"
    assert starts[0]["input"] == {"q": "test"}

    ends = [e for e in events if e["type"] == "tool_end"]
    assert len(ends) == 1
    assert ends[0]["name"] == "search"
    assert ends[0]["output"] == "result"


@pytest.mark.asyncio
async def test_stream_turn_events_handles_list_content_blocks() -> None:
    agent = _FakeAgent([
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": _Chunk([
                    {"type": "text", "text": "part1 "},
                    {"type": "image", "source": "ignored"},
                    {"type": "text", "text": "part2"},
                ])
            },
        },
    ])

    events = [e async for e in stream_turn_events(agent, "hi", "thread-1")]

    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 2
    assert token_events[0]["text"] == "part1 "
    assert token_events[1]["text"] == "part2"


@pytest.mark.asyncio
async def test_stream_turn_events_passes_correct_config() -> None:
    agent = _FakeAgent([])

    async for _ in stream_turn_events(agent, "hi", "my-thread"):
        pass

    assert len(agent.calls) == 1
    input_data, config, version = agent.calls[0]
    assert version == "v2"
    assert config == {"configurable": {"thread_id": "my-thread"}}
    assert "messages" in input_data
    assert input_data["messages"][0].content == "hi"


@pytest.mark.asyncio
async def test_stream_turn_events_empty_stream_still_yields_done() -> None:
    agent = _FakeAgent([])

    events = [e async for e in stream_turn_events(agent, "hi", "thread-1")]

    assert len(events) == 1
    assert events[0]["type"] == "done"
