"""Tests for the per-turn metrics callback handler."""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from uuid import uuid4

from langchain_harness.tracing import ToolStat, TurnMetrics, TurnMetricsHandler

METRICS_LOGGER = "langchain_harness.metrics"


class _Message:
    def __init__(self, usage: dict[str, Any] | None) -> None:
        self.usage_metadata = usage


class _Generation:
    def __init__(self, usage: dict[str, Any] | None) -> None:
        self.message = _Message(usage)


class _Response:
    """Minimal LLMResult: generations is a list of lists of generations."""

    def __init__(self, usage: dict[str, Any] | None) -> None:
        self.generations = [[_Generation(usage)]]


class _ToolOutput:
    def __init__(self, status: str) -> None:
        self.status = status


def _handler() -> TurnMetricsHandler:
    return TurnMetricsHandler(uuid4(), "thread-1")


def test_metrics_to_json_counts_tools_and_errors() -> None:
    metrics = TurnMetrics(
        run_id="run-1",
        thread_id="thread-1",
        tools=[
            ToolStat("search", 12.0),
            ToolStat("read_file", 3.0, "tool_status_error"),
        ],
    )

    payload = json.loads(metrics.to_json())

    assert payload["run_id"] == "run-1"
    assert payload["thread_id"] == "thread-1"
    assert payload["tool_calls"] == 2
    assert payload["tool_errors"] == 1
    assert payload["outcome"] == "ok"


def test_handler_accumulates_tokens_and_model_time() -> None:
    handler = _handler()
    run_id = uuid4()

    handler.on_chat_model_start({}, [[]], run_id=run_id, metadata={"langgraph_node": "model"})
    time.sleep(0.002)
    handler.on_llm_end(
        _Response({"input_tokens": 11, "output_tokens": 7}), run_id=run_id
    )

    m = handler.metrics
    assert m.model_calls == 1
    assert m.input_tokens == 11
    assert m.output_tokens == 7
    assert m.overhead_calls == 0
    assert m.model_ms > 0


def test_handler_separates_compaction_overhead() -> None:
    handler = _handler()
    run_id = uuid4()

    handler.on_chat_model_start(
        {},
        [[]],
        run_id=run_id,
        metadata={"langgraph_node": "SummarizationMiddleware.before_model"},
    )
    handler.on_llm_end(_Response({"input_tokens": 900, "output_tokens": 100}), run_id=run_id)

    m = handler.metrics
    assert m.overhead_calls == 1
    assert m.overhead_tokens == 1000
    assert m.model_calls == 0
    assert m.input_tokens == 0
    assert m.output_tokens == 0


def test_handler_records_time_to_first_token() -> None:
    handler = _handler()
    run_id = uuid4()

    handler.on_chat_model_start({}, [[]], run_id=run_id, metadata={"langgraph_node": "model"})
    time.sleep(0.002)
    handler.on_llm_new_token("hi", run_id=run_id)

    assert handler.metrics.ttft_ms is not None
    assert handler.metrics.ttft_ms > 0


def test_handler_records_tool_success_and_status_error() -> None:
    handler = _handler()
    ok_run, error_run = uuid4(), uuid4()

    handler.on_tool_start({"name": "search"}, "{}", run_id=ok_run)
    handler.on_tool_end(_ToolOutput("success"), run_id=ok_run)
    handler.on_tool_start({"name": "read_file"}, "{}", run_id=error_run)
    handler.on_tool_end(_ToolOutput("error"), run_id=error_run)

    tools = handler.metrics.tools
    assert [(t.name, t.error) for t in tools] == [
        ("search", None),
        ("read_file", "tool_status_error"),
    ]
    assert all(t.ms >= 0 for t in tools)


def test_handler_records_tool_exception() -> None:
    handler = _handler()
    run_id = uuid4()

    handler.on_tool_start({"name": "search"}, "{}", run_id=run_id)
    handler.on_tool_error(ValueError("boom"), run_id=run_id)

    (tool,) = handler.metrics.tools
    assert tool.name == "search"
    assert tool.error is not None
    assert "boom" in tool.error


def test_handler_marks_llm_error_outcome() -> None:
    handler = _handler()
    run_id = uuid4()

    handler.on_chat_model_start({}, [[]], run_id=run_id, metadata={"langgraph_node": "model"})
    handler.on_llm_error(RuntimeError("provider down"), run_id=run_id)

    assert handler.metrics.outcome == "error"
    assert handler.metrics.error is not None
    assert "provider down" in handler.metrics.error


def test_finish_is_idempotent_and_logs_one_json_line(caplog) -> None:
    caplog.set_level(logging.INFO, logger=METRICS_LOGGER)
    handler = _handler()

    first = handler.finish("ok")
    total_ms = first.total_ms
    second = handler.finish("error")

    records = [r for r in caplog.records if r.name == METRICS_LOGGER]
    assert len(records) == 1
    payload = json.loads(records[0].getMessage())
    assert payload["outcome"] == "ok"
    assert payload["run_id"] == first.run_id
    # A stray second call must not re-time or re-label the turn.
    assert second.total_ms == total_ms
    assert second.outcome == "ok"


def test_handler_tolerates_missing_usage_and_serialized_fields() -> None:
    handler = _handler()
    run_id = uuid4()

    handler.on_chat_model_start({}, [[]], run_id=run_id, metadata=None)
    handler.on_llm_end(_Response(None), run_id=run_id)
    handler.on_tool_start({}, "", run_id=uuid4())

    m = handler.metrics
    assert m.model_calls == 1
    assert m.input_tokens == 0
    assert m.output_tokens == 0
    assert handler.metrics.tools == []

    # Unknown serialized/name data still produces a tool stat once it ends.
    tool_run = uuid4()
    handler.on_tool_start({}, "", run_id=tool_run)
    handler.on_tool_end(None, run_id=tool_run)
    assert handler.metrics.tools[-1].name == "unknown"
