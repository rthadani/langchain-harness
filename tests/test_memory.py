"""Tests for the session-memory layer: inbox, approval, and active notes."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from langchain_harness.frontmatter import parse_frontmatter
from langchain_harness.memory import (
    _CONVERSATION_CHAR_LIMIT,
    _METRICS_BLOCK_CHAR_LIMIT,
    _TRUNCATION_MARKER,
    EvalResult,
    active_memory_path,
    approve_note,
    capture_session_note,
    eval_session,
    inbox_dir,
    list_inbox,
    reject_note,
    summarize_session,
    write_inbox_note,
)
from langchain_harness.tracing import (
    ToolStat,
    TurnMetrics,
    app_version,
    clear_metrics,
    record_metrics,
)

EVAL_LOGGER = "langchain_harness.eval"


@pytest.fixture(autouse=True)
def _empty_metrics_registry():
    """Eval reads the shared registry; keep tests from seeing each other."""
    clear_metrics()
    yield
    clear_metrics()


def _turn(run_id: str, *, thread_id: str = "t1", **kwargs: Any) -> TurnMetrics:
    fields: dict[str, Any] = {"total_ms": 3400.0, "ttft_ms": 480.0, "model_calls": 4}
    fields.update(kwargs)
    return TurnMetrics(run_id=run_id, thread_id=thread_id, **fields)


def _eval_prompt(model: _FakeModel) -> str:
    return model.calls[0][1].content


class _FakeMsg:
    def __init__(self, msg_type: str, content: Any) -> None:
        self.type = msg_type
        self.content = content


class _FakeState:
    def __init__(self, messages: list[Any]) -> None:
        self.values = {"messages": messages}


class _FakeAgent:
    def __init__(self, messages: list[Any] | None = None) -> None:
        self._messages = messages or []
        self.configs: list[Any] = []

    async def aget_state(self, config: Any) -> _FakeState:
        self.configs.append(config)
        return _FakeState(self._messages)


class _FakeModel:
    """Fake chat model returning preset content and recording invocations."""

    def __init__(self, content: Any = "  summarized  ") -> None:
        self.content = content
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.calls.append(messages)
        return type("Result", (), {"content": self.content})()


# --- summarize_session ---


@pytest.mark.asyncio
async def test_summarize_session_skips_state_without_messages() -> None:
    model = _FakeModel()
    assert await summarize_session(model, _FakeAgent([]), "t1") is None
    assert model.calls == []


@pytest.mark.asyncio
async def test_summarize_session_skips_human_only_state() -> None:
    model = _FakeModel()
    agent = _FakeAgent([_FakeMsg("human", "hello")])
    assert await summarize_session(model, agent, "t1") is None
    assert model.calls == []


@pytest.mark.asyncio
async def test_summarize_session_skips_empty_ai_text() -> None:
    model = _FakeModel()
    agent = _FakeAgent([_FakeMsg("human", "hello"), _FakeMsg("ai", "   ")])
    assert await summarize_session(model, agent, "t1") is None
    assert model.calls == []


@pytest.mark.asyncio
async def test_summarize_session_returns_trimmed_text() -> None:
    model = _FakeModel("  - did a thing\n")
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])

    text = await summarize_session(model, agent, "t1", title="A session")

    assert text == "- did a thing"
    assert len(model.calls) == 1
    system, human = model.calls[0]
    assert "long-term memory" in system.content
    assert "A session" in human.content
    assert "ai: done" in human.content


@pytest.mark.asyncio
async def test_summarize_session_extracts_text_blocks() -> None:
    model = _FakeModel([{"type": "text", "text": "block summary"}])
    agent = _FakeAgent([_FakeMsg("ai", [{"type": "text", "text": "reply"}])])

    assert await summarize_session(model, agent, "t1") == "block summary"


@pytest.mark.asyncio
async def test_summarize_session_returns_none_when_state_fetch_raises() -> None:
    class _BrokenAgent:
        async def aget_state(self, config: Any) -> Any:
            raise RuntimeError("checkpointer down")

    assert await summarize_session(_FakeModel(), _BrokenAgent(), "t1") is None


@pytest.mark.asyncio
async def test_summarize_session_returns_none_when_state_is_none() -> None:
    class _NoneStateAgent:
        async def aget_state(self, config: Any) -> Any:
            return None

    model = _FakeModel()
    assert await summarize_session(model, _NoneStateAgent(), "t1") is None
    assert model.calls == []


@pytest.mark.asyncio
async def test_summarize_session_returns_none_when_state_values_is_none() -> None:
    class _NoValuesAgent:
        async def aget_state(self, config: Any) -> Any:
            return type("State", (), {"values": None})()

    model = _FakeModel()
    assert await summarize_session(model, _NoValuesAgent(), "t1") is None
    assert model.calls == []


@pytest.mark.asyncio
async def test_summarize_session_returns_none_when_model_content_is_empty() -> None:
    model = _FakeModel("   ")
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])

    assert await summarize_session(model, agent, "t1") is None
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_summarize_session_passes_thread_id_in_config() -> None:
    model = _FakeModel()
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])

    await summarize_session(model, agent, "thread-abc")

    assert agent.configs == [{"configurable": {"thread_id": "thread-abc"}}]


@pytest.mark.asyncio
async def test_summarize_session_caps_total_rendered_conversation() -> None:
    model = _FakeModel()
    agent = _FakeAgent([_FakeMsg("ai", "x" * 3000) for _ in range(20)])

    await summarize_session(model, agent, "t1")

    human = model.calls[0][1]
    rendered = human.content.split("Conversation:\n", 1)[1]
    assert rendered.endswith(_TRUNCATION_MARKER)
    assert len(rendered) == _CONVERSATION_CHAR_LIMIT + len(_TRUNCATION_MARKER)


# --- write_inbox_note / list_inbox ---


def test_write_inbox_note_names_and_frontmatter(tmp_path) -> None:
    path = write_inbox_note(
        tmp_path,
        thread_id="t1",
        title="Refactor: parser",
        summary="line one\nline two",
    )

    assert path == inbox_dir(tmp_path) / "t1.md"
    metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert metadata["title"] == "Refactor: parser"
    assert metadata["thread_id"] == "t1"
    assert metadata["created_at"]
    assert "user_id" not in metadata
    assert body.strip() == "line one\nline two"


def test_write_inbox_note_scopes_note_id_by_user(tmp_path) -> None:
    path = write_inbox_note(
        tmp_path, thread_id="t1", title="x", summary="s", user_id="alice"
    )

    assert path.name == "alice--t1.md"
    metadata, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert metadata["user_id"] == "alice"


def test_write_inbox_note_falls_back_to_default_title(tmp_path) -> None:
    path = write_inbox_note(tmp_path, thread_id="t1", title="   ", summary="s")

    metadata, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert metadata["title"] == "Untitled session"


def test_list_inbox_round_trips_notes(tmp_path) -> None:
    write_inbox_note(tmp_path, thread_id="t2", title="Second", summary="b")
    write_inbox_note(tmp_path, thread_id="t1", title="First", summary="a")

    notes = list_inbox(tmp_path)

    assert [n.note_id for n in notes] == ["t1", "t2"]
    assert notes[0].title == "First"
    assert notes[0].thread_id == "t1"
    assert notes[0].summary == "a"
    assert notes[0].created_at
    assert notes[0].user_id is None


def test_list_inbox_without_directory(tmp_path) -> None:
    assert list_inbox(tmp_path) == []


# --- approve_note / reject_note ---


def test_approve_note_appends_to_active_memory(tmp_path) -> None:
    write_inbox_note(tmp_path, thread_id="t1", title="First", summary="- one")

    assert approve_note(tmp_path, "t1") is True

    assert not (inbox_dir(tmp_path) / "t1.md").exists()
    notes = active_memory_path(tmp_path).read_text(encoding="utf-8")
    assert notes.startswith("## First\n")
    assert "_session `t1` · approved `" in notes
    assert notes.rstrip().endswith("- one")


def test_approve_note_separates_appended_notes(tmp_path) -> None:
    write_inbox_note(tmp_path, thread_id="t1", title="First", summary="- one")
    write_inbox_note(tmp_path, thread_id="t2", title="Second", summary="- two")

    approve_note(tmp_path, "t1")
    approve_note(tmp_path, "t2")

    notes = active_memory_path(tmp_path).read_text(encoding="utf-8")
    assert notes.count("## ") == 2
    assert "\n\n## Second" in notes


def test_approve_note_missing_returns_false(tmp_path) -> None:
    assert approve_note(tmp_path, "nope") is False
    assert not active_memory_path(tmp_path).exists()


def test_reject_note_deletes_only_inbox_file(tmp_path) -> None:
    write_inbox_note(tmp_path, thread_id="t1", title="First", summary="- one")
    write_inbox_note(tmp_path, thread_id="t2", title="Second", summary="- two")

    assert reject_note(tmp_path, "t1") is True

    assert not (inbox_dir(tmp_path) / "t1.md").exists()
    assert (inbox_dir(tmp_path) / "t2.md").exists()
    assert not active_memory_path(tmp_path).exists()


def test_reject_note_missing_returns_false(tmp_path) -> None:
    assert reject_note(tmp_path, "nope") is False


# --- eval_session ---

_VALID_EVAL_JSON = """{
    "scope_adherence": 8,
    "scope_reason": "stayed on topic",
    "task_completion": 9,
    "task_reason": "fully done",
    "tool_use_quality": 7,
    "tool_use_reason": "one redundant read",
    "verdict": "good session"
}"""


@pytest.mark.asyncio
async def test_eval_session_returns_result_on_valid_json() -> None:
    model = _FakeModel(_VALID_EVAL_JSON)
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])

    result = await eval_session(model, agent, "t1")

    assert isinstance(result, EvalResult)
    assert result.scope_adherence == 8
    assert result.scope_reason == "stayed on topic"
    assert result.task_completion == 9
    assert result.task_reason == "fully done"
    assert result.tool_use_quality == 7
    assert result.tool_use_reason == "one redundant read"
    assert result.verdict == "good session"


@pytest.mark.asyncio
async def test_eval_session_strips_markdown_fences() -> None:
    fenced = f"```json\n{_VALID_EVAL_JSON}\n```"
    model = _FakeModel(fenced)
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])

    result = await eval_session(model, agent, "t1")

    assert result is not None
    assert result.scope_adherence == 8


@pytest.mark.asyncio
async def test_eval_session_skips_empty_session() -> None:
    model = _FakeModel(_VALID_EVAL_JSON)
    assert await eval_session(model, _FakeAgent([]), "t1") is None
    assert model.calls == []


@pytest.mark.asyncio
async def test_eval_session_skips_human_only_session() -> None:
    model = _FakeModel(_VALID_EVAL_JSON)
    agent = _FakeAgent([_FakeMsg("human", "hello")])
    assert await eval_session(model, agent, "t1") is None
    assert model.calls == []


@pytest.mark.asyncio
async def test_eval_session_returns_none_on_invalid_json() -> None:
    model = _FakeModel("not json at all")
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])
    assert await eval_session(model, agent, "t1") is None


@pytest.mark.asyncio
async def test_eval_session_returns_none_on_missing_keys() -> None:
    model = _FakeModel('{"scope_adherence": 8}')
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])
    assert await eval_session(model, agent, "t1") is None


@pytest.mark.asyncio
async def test_eval_session_returns_none_when_model_raises() -> None:
    class _BrokenModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            raise RuntimeError("model down")

    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])
    assert await eval_session(_BrokenModel(), agent, "t1") is None


@pytest.mark.asyncio
async def test_eval_session_includes_memory_texts_in_prompt() -> None:
    model = _FakeModel(_VALID_EVAL_JSON)
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])

    await eval_session(model, agent, "t1", memory_texts=["scope: coding only"])

    human_msg = model.calls[0][1]
    assert "scope: coding only" in human_msg.content


@pytest.mark.asyncio
async def test_eval_session_passes_thread_id_in_config() -> None:
    model = _FakeModel(_VALID_EVAL_JSON)
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])

    await eval_session(model, agent, "thread-xyz")

    assert agent.configs == [{"configurable": {"thread_id": "thread-xyz"}}]


# --- eval_session with turn metrics ---


@pytest.mark.asyncio
async def test_eval_session_includes_metrics_block() -> None:
    model = _FakeModel(_VALID_EVAL_JSON)
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])
    metrics = [
        _turn(
            "5dd0e629-aaaa",
            tools=[
                ToolStat("read_file", 1.0),
                ToolStat("read_file", 2.0),
                ToolStat("read_file", 3.0),
                ToolStat("grep", 4.0, "tool_status_error"),
            ],
        )
    ]

    await eval_session(
        model, agent, "t1", memory_texts=["scope: coding only"], metrics=metrics
    )

    prompt = _eval_prompt(model)
    assert "Turn metrics (most recent last; showing last 1 of 1 recorded turns):" in prompt
    assert (
        "- run=5dd0e629 outcome=ok model_calls=4 tools=[read_file x3, grep x1 (1 err)] "
        "tool_errors=1 ttft_ms=480 total_ms=3400 overhead_tokens=0"
    ) in prompt
    assert prompt.index("Project instructions:") < prompt.index("Turn metrics")
    assert prompt.index("Turn metrics") < prompt.index("Conversation:")


@pytest.mark.asyncio
async def test_eval_session_omits_metrics_block_without_metrics() -> None:
    model = _FakeModel(_VALID_EVAL_JSON)
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])

    result = await eval_session(model, agent, "t1")

    assert result is not None
    assert "Turn metrics" not in _eval_prompt(model)


@pytest.mark.asyncio
async def test_eval_session_reads_metrics_from_path(tmp_path) -> None:
    model = _FakeModel(_VALID_EVAL_JSON)
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])
    path = tmp_path / "turns.jsonl"
    record_metrics(_turn("run-path-1", tools=[ToolStat("grep", 1.0)]), path)
    # A fresh process (``harness eval``) has no in-memory turns at all.
    clear_metrics()

    result = await eval_session(model, agent, "t1", metrics_path=path)

    assert result is not None
    assert "grep x1" in _eval_prompt(model)


@pytest.mark.asyncio
async def test_eval_session_uses_latest_metric_run_id() -> None:
    model = _FakeModel(_VALID_EVAL_JSON)
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])

    result = await eval_session(
        model, agent, "t1", metrics=[_turn("run-1"), _turn("run-2")]
    )

    assert result is not None
    assert result.run_id == "run-2"


@pytest.mark.asyncio
async def test_eval_session_explicit_run_id_wins() -> None:
    model = _FakeModel(_VALID_EVAL_JSON)
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])

    result = await eval_session(
        model, agent, "t1", metrics=[_turn("run-1")], run_id="run-explicit"
    )

    assert result is not None
    assert result.run_id == "run-explicit"


@pytest.mark.asyncio
async def test_eval_session_logs_joined_record(caplog) -> None:
    caplog.set_level(logging.INFO, logger=EVAL_LOGGER)
    model = _FakeModel(_VALID_EVAL_JSON)
    agent = _FakeAgent([_FakeMsg("human", "do the secret thing"), _FakeMsg("ai", "done")])
    metrics = [
        _turn("run-1", tools=[ToolStat("read_file", 1.0)]),
        _turn("run-2", tools=[ToolStat("read_file", 1.0), ToolStat("read_file", 2.0)]),
    ]

    await eval_session(
        model, agent, "t1", memory_texts=["scope: coding only"], metrics=metrics
    )

    records = [r for r in caplog.records if r.name == EVAL_LOGGER]
    assert len(records) == 1
    message = records[0].getMessage()
    payload = json.loads(message)
    assert set(payload) == {
        "event",
        "scope",
        "run_id",
        "run_ids",
        "thread_id",
        "app_version",
        "scores",
        "verdict",
        "metrics",
    }
    assert payload["event"] == "eval"
    assert payload["scope"] == "session"
    assert payload["run_id"] == "run-2"
    assert payload["run_ids"] == ["run-1", "run-2"]
    assert payload["thread_id"] == "t1"
    assert payload["app_version"] == app_version()
    assert payload["scores"] == {
        "scope_adherence": 8,
        "task_completion": 9,
        "tool_use_quality": 7,
    }
    assert payload["verdict"] == "good session"
    assert payload["metrics"]["turn_count"] == 2
    assert payload["metrics"]["totals"]["tool_calls"] == 3
    # Compact JSON, and no prompt or conversation text anywhere in the record.
    assert '": ' not in message
    assert "do the secret thing" not in message
    assert "Project instructions" not in message


@pytest.mark.asyncio
async def test_joined_record_has_null_metrics_without_turns(caplog) -> None:
    caplog.set_level(logging.INFO, logger=EVAL_LOGGER)
    model = _FakeModel(_VALID_EVAL_JSON)
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])

    result = await eval_session(model, agent, "t1")

    assert result is not None
    payload = json.loads(
        next(r for r in caplog.records if r.name == EVAL_LOGGER).getMessage()
    )
    assert payload["metrics"] is None
    assert payload["run_id"] is None
    assert payload["run_ids"] == []


@pytest.mark.asyncio
async def test_metrics_block_truncates_to_limit() -> None:
    model = _FakeModel(_VALID_EVAL_JSON)
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])
    metrics = [
        _turn(
            f"run-{i:02d}-abcdef",
            tools=[
                ToolStat("read_file", 1.0),
                ToolStat("read_file", 1.0),
                ToolStat("read_file", 1.0, "tool_status_error"),
                ToolStat("grep", 1.0),
                ToolStat("write_file", 1.0),
                ToolStat("bash", 1.0),
            ],
        )
        for i in range(30)
    ]

    await eval_session(model, agent, "t1", metrics=metrics)

    block = _eval_prompt(model).split("\n\nConversation:", 1)[0]
    assert "of 30 recorded turns" in block
    assert "showing last 10 of 30 recorded turns" in block
    assert block.endswith(_TRUNCATION_MARKER)
    assert len(block) == _METRICS_BLOCK_CHAR_LIMIT + len(_TRUNCATION_MARKER)


@pytest.mark.asyncio
async def test_metrics_block_skips_unfinished_turns() -> None:
    model = _FakeModel(_VALID_EVAL_JSON)
    agent = _FakeAgent([_FakeMsg("human", "do it"), _FakeMsg("ai", "done")])
    metrics = [
        _turn("run-aaaa1111"),
        _turn("run-bbbb2222", total_ms=None),
    ]

    result = await eval_session(model, agent, "t1", metrics=metrics)

    prompt = _eval_prompt(model)
    assert "showing last 1 of 1 recorded turns" in prompt
    assert "run-aaaa" in prompt
    assert "run-bbbb" not in prompt
    assert result is not None
    assert result.run_id == "run-aaaa1111"


# --- capture_session_note ---


@pytest.mark.asyncio
async def test_capture_session_note_writes_note(tmp_path) -> None:
    agent = _FakeAgent([_FakeMsg("human", "hi"), _FakeMsg("ai", "hello")])

    note = await capture_session_note(
        _FakeModel("- bullet"), agent, "t1", tmp_path, title="T", user_id="alice"
    )

    assert note is not None
    assert note.name == "alice--t1.md"
    notes = list_inbox(tmp_path)
    assert len(notes) == 1
    assert notes[0].summary == "- bullet"
    assert notes[0].user_id == "alice"


@pytest.mark.asyncio
async def test_capture_session_note_returns_none_without_content(tmp_path) -> None:
    agent = _FakeAgent([_FakeMsg("human", "hi")])

    assert await capture_session_note(_FakeModel(), agent, "t1", tmp_path) is None
    assert list_inbox(tmp_path) == []


@pytest.mark.asyncio
async def test_capture_session_note_never_raises_when_model_fails(tmp_path) -> None:
    class _BrokenModel:
        async def ainvoke(self, messages: list[Any]) -> Any:
            raise RuntimeError("model down")

    agent = _FakeAgent([_FakeMsg("human", "hi"), _FakeMsg("ai", "hello")])

    assert await capture_session_note(_BrokenModel(), agent, "t1", tmp_path) is None
    assert list_inbox(tmp_path) == []
