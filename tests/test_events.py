"""Tests for the shared stream_turn_events generator."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from uuid import UUID, uuid4

import pytest

from langchain_harness import __version__
from langchain_harness.events import stream_turn_events
from langchain_harness.tracing import TurnMetricsHandler

METRICS_LOGGER = "langchain_harness.metrics"


class _FakeAgent:
    """Yields pre-configured events from astream_events."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self.calls: list[tuple[Any, Any, str]] = []

    async def astream_events(self, input_data: Any, config: Any, *, version: str):
        self.calls.append((input_data, config, version))
        for event in self._events:
            yield event


class _UsageMessage:
    def __init__(self, usage: dict[str, int] | None) -> None:
        self.usage_metadata = usage


class _UsageResponse:
    """Fake LLMResult carrying usage, so no live provider is needed."""

    def __init__(self, usage: dict[str, int] | None) -> None:
        self.generations = [[type("Gen", (), {"message": _UsageMessage(usage)})()]]


class _ToolOutput:
    def __init__(self, status: str = "success") -> None:
        self.status = status
        self.content = "result"


class _CallbackAgent:
    """Fake agent that also drives the metrics handler on the config.

    ``astream_events`` in the real graph invokes callbacks itself; this fake
    must do it explicitly so the handler's accounting is exercised.
    """

    def __init__(self, *, error: BaseException | None = None) -> None:
        self._error = error
        self.config: dict[str, Any] | None = None

    async def astream_events(self, input_data: Any, config: Any, *, version: str):
        self.config = config
        handler = config["callbacks"][0]
        llm_run = uuid4()
        tool_run = uuid4()
        handler.on_chat_model_start(
            {}, [[]], run_id=llm_run, metadata={"langgraph_node": "model"}
        )
        handler.on_llm_new_token("hi", run_id=llm_run)
        handler.on_llm_end(
            _UsageResponse({"input_tokens": 5, "output_tokens": 2}), run_id=llm_run
        )
        handler.on_tool_start({"name": "search"}, "{}", run_id=tool_run, name="search")
        handler.on_tool_end(_ToolOutput(), run_id=tool_run)
        yield {"event": "on_chat_model_stream", "data": {"chunk": _Chunk("hi")}}
        if self._error is not None:
            raise self._error


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
    assert config["configurable"] == {"thread_id": "my-thread"}
    assert isinstance(config["run_id"], UUID)
    assert config["run_name"] == "harness_turn"
    assert config["tags"] == ["harness"]
    assert config["metadata"]["thread_id"] == "my-thread"
    assert config["metadata"]["user_id"] is None
    assert config["metadata"]["app_version"] == __version__
    assert isinstance(config["callbacks"][0], TurnMetricsHandler)
    assert "messages" in input_data
    assert input_data["messages"][0].content == "hi"


@pytest.mark.asyncio
async def test_stream_turn_events_empty_stream_still_yields_done() -> None:
    agent = _FakeAgent([])

    events = [e async for e in stream_turn_events(agent, "hi", "thread-1")]

    assert len(events) == 1
    assert events[0]["type"] == "done"
    assert isinstance(events[0]["run_id"], str)
    assert len(events[0]["run_id"]) == 36


@pytest.mark.asyncio
async def test_stream_turn_events_wires_metrics_handler_into_config() -> None:
    agent = _FakeAgent([])

    async for _ in stream_turn_events(agent, "hi", "thread-1", user_id="alice"):
        pass

    _input_data, config, _version = agent.calls[0]
    handler = config["callbacks"][0]
    assert isinstance(handler, TurnMetricsHandler)
    assert handler.metrics.run_id == str(config["run_id"])
    assert handler.metrics.thread_id == "thread-1"
    assert handler.metrics.user_id == "alice"


@pytest.mark.asyncio
async def test_stream_turn_events_forwards_tags_and_metadata() -> None:
    agent = _FakeAgent([])

    async for _ in stream_turn_events(
        agent,
        "hi",
        "thread-1",
        user_id="alice",
        tags=["eval_run"],
        metadata={"eval_run": "run-7", "thread_id": "spoofed"},
    ):
        pass

    _input_data, config, _version = agent.calls[0]
    assert config["tags"] == ["harness", "eval_run"]
    assert config["metadata"]["eval_run"] == "run-7"
    # Fixed trace identifiers win over caller-supplied metadata.
    assert config["metadata"]["thread_id"] == "thread-1"
    assert config["metadata"]["user_id"] == "alice"
    assert config["metadata"]["app_version"] == __version__


@pytest.mark.asyncio
async def test_stream_turn_events_logs_metrics_on_success(caplog) -> None:
    caplog.set_level(logging.INFO, logger=METRICS_LOGGER)
    agent = _CallbackAgent()

    events = [e async for e in stream_turn_events(agent, "hi", "thread-1", user_id="alice")]

    assert events[-1]["type"] == "done"
    records = [r for r in caplog.records if r.name == METRICS_LOGGER]
    assert len(records) == 1
    payload = json.loads(records[0].getMessage())
    assert payload["outcome"] == "ok"
    assert payload["model_calls"] == 1
    assert payload["input_tokens"] == 5
    assert payload["output_tokens"] == 2
    assert payload["tool_calls"] == 1
    assert payload["tool_errors"] == 0
    assert payload["thread_id"] == "thread-1"
    assert payload["user_id"] == "alice"
    assert payload["run_id"] == events[-1]["run_id"]


@pytest.mark.asyncio
async def test_stream_turn_events_logs_error_outcome_and_reraises(caplog) -> None:
    caplog.set_level(logging.INFO, logger=METRICS_LOGGER)
    agent = _CallbackAgent(error=RuntimeError("provider down"))

    with pytest.raises(RuntimeError, match="provider down"):
        async for _ in stream_turn_events(agent, "hi", "thread-1"):
            pass

    payload = json.loads(
        next(r for r in caplog.records if r.name == METRICS_LOGGER).getMessage()
    )
    assert payload["outcome"] == "error"


@pytest.mark.asyncio
async def test_stream_turn_events_logs_cancel_outcome(caplog) -> None:
    caplog.set_level(logging.INFO, logger=METRICS_LOGGER)
    agent = _CallbackAgent(error=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        async for _ in stream_turn_events(agent, "hi", "thread-1"):
            pass

    payload = json.loads(
        next(r for r in caplog.records if r.name == METRICS_LOGGER).getMessage()
    )
    assert payload["outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_stream_turn_events_labels_client_disconnect_outcome(caplog) -> None:
    caplog.set_level(logging.INFO, logger=METRICS_LOGGER)
    agent = _FakeAgent([
        {"event": "on_chat_model_stream", "data": {"chunk": _Chunk("hello")}},
    ])

    stream = stream_turn_events(
        agent, "hi", "thread-1", cancel_outcome="client_disconnect"
    )
    await stream.__anext__()
    # Closing the generator early is GeneratorExit, not CancelledError.
    await stream.aclose()

    payload = json.loads(
        next(r for r in caplog.records if r.name == METRICS_LOGGER).getMessage()
    )
    assert payload["outcome"] == "client_disconnect"


@pytest.mark.asyncio
async def test_stream_turn_events_survives_metrics_failure(caplog, monkeypatch) -> None:
    def _boom(self: TurnMetricsHandler, outcome: str | None = None) -> None:
        raise RuntimeError("metrics broken")

    monkeypatch.setattr(TurnMetricsHandler, "finish", _boom)

    events = [e async for e in stream_turn_events(_FakeAgent([]), "hi", "thread-1")]
    assert events[-1]["type"] == "done"

    # A failing finish() must not replace the turn's own exception.
    failing = _CallbackAgent(error=ValueError("real failure"))
    with pytest.raises(ValueError, match="real failure"):
        async for _ in stream_turn_events(failing, "hi", "thread-1"):
            pass

    assert any("Failed to record turn metrics" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_done_run_id_matches_config_run_id() -> None:
    agent = _FakeAgent([])

    events = [e async for e in stream_turn_events(agent, "hi", "thread-1")]

    _input_data, config, _version = agent.calls[0]
    assert events[-1] == {"type": "done", "run_id": str(config["run_id"])}


@pytest.mark.asyncio
async def test_stream_turn_events_filters_summarization_tokens() -> None:
    agent = _FakeAgent([
        {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "SummarizationMiddleware.before_model"},
            "data": {"chunk": _Chunk("internal summary text")},
        },
        {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "model"},
            "data": {"chunk": _Chunk("visible reply")},
        },
    ])

    events = [e async for e in stream_turn_events(agent, "hi", "thread-1")]

    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 1
    assert token_events[0]["text"] == "visible reply"


@pytest.mark.asyncio
async def test_stream_turn_events_passes_tokens_without_node_metadata() -> None:
    agent = _FakeAgent([
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": _Chunk("hello")},
        },
    ])

    events = [e async for e in stream_turn_events(agent, "hi", "thread-1")]

    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 1
    assert token_events[0]["text"] == "hello"
