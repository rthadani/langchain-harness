"""Tests for the per-turn metrics callback handler."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from langchain_harness import __version__
from langchain_harness.tracing import (
    _MAX_THREADS,
    _MAX_TURNS_PER_THREAD,
    ToolStat,
    TurnMetrics,
    TurnMetricsHandler,
    app_version,
    clear_metrics,
    default_metrics_path,
    recent_turn_metrics,
    record_metrics,
)

METRICS_LOGGER = "langchain_harness.metrics"


@pytest.fixture(autouse=True)
def _empty_registry():
    """The registry is module state; do not let one test's turns leak."""
    clear_metrics()
    yield
    clear_metrics()


def _turn(run_id: str, thread_id: str, **kwargs: Any) -> TurnMetrics:
    fields: dict[str, Any] = {"total_ms": 1000.0}
    fields.update(kwargs)
    return TurnMetrics(run_id=run_id, thread_id=thread_id, **fields)


def _logged_metrics(metrics: TurnMetrics, path: Path) -> None:
    """Record *metrics* only to *path*, as a previous process would have."""
    record_metrics(metrics, path)
    clear_metrics()


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


# --- registry and durable sink ---


def test_record_metrics_keeps_turns_per_thread() -> None:
    record_metrics(_turn("run-1", "thread-1"))
    record_metrics(_turn("run-2", "thread-1"))
    record_metrics(_turn("run-3", "thread-2"))

    assert [m.run_id for m in recent_turn_metrics("thread-1")] == ["run-1", "run-2"]
    assert [m.run_id for m in recent_turn_metrics("thread-2")] == ["run-3"]
    assert recent_turn_metrics("thread-missing") == []


def test_recent_metrics_is_bounded_per_thread() -> None:
    for i in range(_MAX_TURNS_PER_THREAD + 5):
        record_metrics(_turn(f"run-{i}", "thread-1"))

    turns = recent_turn_metrics("thread-1", limit=_MAX_TURNS_PER_THREAD)

    assert len(turns) == _MAX_TURNS_PER_THREAD
    assert turns[0].run_id == "run-5"
    assert turns[-1].run_id == f"run-{_MAX_TURNS_PER_THREAD + 4}"


def test_recent_metrics_evicts_oldest_thread() -> None:
    last = _MAX_THREADS + 4
    for i in range(last + 1):
        record_metrics(_turn(f"run-{i}", f"thread-{i}"))

    assert recent_turn_metrics("thread-0") == []
    assert [m.run_id for m in recent_turn_metrics(f"thread-{last}")] == [f"run-{last}"]


def test_record_metrics_appends_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "metrics" / "turns.jsonl"

    record_metrics(_turn("run-1", "thread-1"), path)
    record_metrics(_turn("run-2", "thread-1"), path)

    payloads = [json.loads(line) for line in path.read_text().splitlines()]
    assert [p["run_id"] for p in payloads] == ["run-1", "run-2"]
    assert all(p["thread_id"] == "thread-1" for p in payloads)


def test_recent_metrics_reads_jsonl_when_memory_empty(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    _logged_metrics(
        _turn(
            "run-1",
            "thread-1",
            tools=[ToolStat("read_file", 12.0), ToolStat("grep", 3.0, "tool_status_error")],
        ),
        path,
    )
    _logged_metrics(_turn("run-2", "thread-2"), path)

    turns = recent_turn_metrics("thread-1", path=path)

    assert [m.run_id for m in turns] == ["run-1"]
    assert turns[0].total_ms == 1000.0
    assert [(t.name, t.error) for t in turns[0].tools] == [
        ("read_file", None),
        ("grep", "tool_status_error"),
    ]


def test_recent_metrics_ignores_malformed_jsonl_lines(tmp_path: Path) -> None:
    path = tmp_path / "turns.jsonl"
    path.write_text(
        "not json\n"
        + json.dumps({"run_id": "run-ok", "thread_id": "thread-1", "total_ms": 5.0})
        + "\n[]\n",
        encoding="utf-8",
    )

    turns = recent_turn_metrics("thread-1", path=path)

    assert [m.run_id for m in turns] == ["run-ok"]


def test_record_metrics_swallows_write_errors(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    record_metrics(_turn("run-1", "thread-1"), blocker / "turns.jsonl")

    # The turn itself is unaffected, and the in-memory copy still exists.
    assert [m.run_id for m in recent_turn_metrics("thread-1")] == ["run-1"]


def test_default_metrics_path(tmp_path: Path) -> None:
    assert default_metrics_path(tmp_path) == tmp_path / ".harness" / "metrics" / "turns.jsonl"


def test_app_version_prefers_env(monkeypatch) -> None:
    monkeypatch.delenv("HARNESS_APP_VERSION", raising=False)
    assert app_version() == __version__

    monkeypatch.setenv("HARNESS_APP_VERSION", "build-42")
    assert app_version() == "build-42"
