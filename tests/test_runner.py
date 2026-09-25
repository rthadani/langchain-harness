import asyncio
import io
from typing import Any

import aiosqlite
import pytest
from rich.console import Console

from langchain_harness import runner as runner_module
from langchain_harness.runner import (
    _render_tool_call,
    _stream_turn,
    repl,
)


class _FakeAgent:
    """Records the input passed to ``astream_events`` and yields nothing."""

    def __init__(self) -> None:
        self.inputs: list[Any] = []

    async def astream_events(self, input_data: Any, config: Any, *, version: str):
        self.inputs.append((input_data, config, version))
        if False:
            yield


def _patch_console(monkeypatch, inputs: list[str]) -> tuple[Console, list[Any]]:
    """Return a console whose ``input`` reads from *inputs* and record print calls."""
    printed: list[Any] = []
    real_console = Console()
    iterator = iter(inputs)

    def _input(_prompt: str = "") -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError

    def _print(*args, **kwargs):
        printed.append((args, kwargs))

    monkeypatch.setattr(real_console, "input", _input, raising=False)
    monkeypatch.setattr(real_console, "print", _print, raising=False)
    monkeypatch.setattr(runner_module, "Console", lambda: real_console)
    return real_console, printed


@pytest.mark.asyncio
async def test_stream_turn_passes_messages_dict_directly(monkeypatch) -> None:
    agent = _FakeAgent()
    console, _ = _patch_console(monkeypatch, [])

    await _stream_turn(agent, "hello", "thread-1", console)

    assert len(agent.inputs) == 1
    input_data, config, version = agent.inputs[0]
    assert version == "v2"
    assert config == {"configurable": {"thread_id": "thread-1"}}
    assert isinstance(input_data, dict)
    assert "messages" in input_data
    assert len(input_data["messages"]) == 1
    assert input_data["messages"][0].content == "hello"


@pytest.mark.asyncio
async def test_repl_exit_command(monkeypatch) -> None:
    agent = _FakeAgent()
    _, printed = _patch_console(monkeypatch, ["/exit"])

    await repl(agent, "thread-1")

    assert not agent.inputs
    assert any("Goodbye" in str(p[0]) for p in printed)


@pytest.mark.asyncio
async def test_repl_quit_command(monkeypatch) -> None:
    agent = _FakeAgent()
    _, printed = _patch_console(monkeypatch, ["/quit"])

    await repl(agent, "thread-1")

    assert not agent.inputs
    assert any("Goodbye" in str(p[0]) for p in printed)


@pytest.mark.asyncio
async def test_repl_empty_prompt_skipped(monkeypatch) -> None:
    agent = _FakeAgent()
    _, _ = _patch_console(monkeypatch, ["", "", "/exit"])

    await repl(agent, "thread-1")

    assert not agent.inputs


@pytest.mark.asyncio
async def test_repl_eof_quits(monkeypatch) -> None:
    agent = _FakeAgent()
    _, printed = _patch_console(monkeypatch, [])

    await repl(agent, "thread-1")

    assert not agent.inputs
    assert any("Goodbye" in str(p[0]) for p in printed)


@pytest.mark.asyncio
async def test_repl_turn_error_does_not_kill_loop(monkeypatch) -> None:
    class _BrokenAgent:
        async def astream_events(self, *args, **kwargs):
            raise RuntimeError("boom")
            yield  # make this an async generator

    agent = _BrokenAgent()
    _, printed = _patch_console(monkeypatch, ["hello", "/exit"])

    await repl(agent, "thread-1")

    assert any("Error:" in str(p[0]) and "boom" in str(p[0]) for p in printed)


@pytest.mark.asyncio
async def test_repl_turn_error_with_brackets_does_not_raise_markup_error() -> None:
    class _BracketErrorAgent:
        async def astream_events(self, *args, **kwargs):
            raise RuntimeError("[tool] failed [/path]")
            yield

    agent = _BracketErrorAgent()
    output = io.StringIO()
    console = Console(file=output)
    inputs = iter(["hello", "/exit"])
    console.input = lambda _prompt="": next(inputs)  # type: ignore[method-assign]

    await repl(agent, "thread-1", console=console)

    captured = output.getvalue()
    assert "Error:" in captured
    assert "[tool] failed [/path]" in captured


@pytest.mark.asyncio
async def test_repl_keyboard_interrupt_continues_loop(monkeypatch) -> None:
    class _InterruptAgent:
        async def astream_events(self, *args, **kwargs):
            raise KeyboardInterrupt()
            yield

    agent = _InterruptAgent()
    _, printed = _patch_console(monkeypatch, ["hello", "/exit"])

    await repl(agent, "thread-1")

    assert any("interrupted" in str(p[0]).lower() for p in printed)


@pytest.mark.asyncio
async def test_repl_cancelled_error_continues_loop(monkeypatch) -> None:
    class _CancelledAgent:
        async def astream_events(self, *args, **kwargs):
            raise asyncio.CancelledError()
            yield

    agent = _CancelledAgent()
    _, printed = _patch_console(monkeypatch, ["hello", "/exit"])

    await repl(agent, "thread-1")

    assert any("interrupted" in str(p[0]).lower() for p in printed)


class _Chunk:
    def __init__(self, content: Any) -> None:
        self.content = content


@pytest.mark.asyncio
async def test_stream_turn_renders_text_content_blocks_only(monkeypatch) -> None:
    events = [
        {
            "event": "on_chat_model_stream",
            "data": {
                "chunk": _Chunk(
                    [
                        {"type": "text", "text": "hello "},
                        {"type": "image", "source": "ignored"},
                        {"type": "text", "text": "world"},
                    ]
                )
            },
        }
    ]

    class _BlockAgent:
        async def astream_events(self, *args, **kwargs):
            for event in events:
                yield event

    agent = _BlockAgent()
    console, printed = _patch_console(monkeypatch, [])

    await _stream_turn(agent, "hi", "thread-blocks", console)

    output = "".join(str(p[0][0]) for p in printed if p[0])
    assert "hello world" in output
    assert "ignored" not in output
    assert "image" not in output.lower()


def test_render_tool_call_non_dict_input() -> None:
    console = Console()
    printed: list[Any] = []
    console.print = lambda *args, **kwargs: printed.append((args, kwargs))  # type: ignore[method-assign]

    _render_tool_call(console, {"name": "tool", "data": {"input": "not-a-dict"}})

    assert any("..." in str(p[0]) for p in printed)


def test_render_tool_call_truncates_more_than_three_keys() -> None:
    console = Console()
    printed: list[Any] = []
    console.print = lambda *args, **kwargs: printed.append((args, kwargs))  # type: ignore[method-assign]

    _render_tool_call(
        console,
        {"name": "tool", "data": {"input": {"a": 1, "b": 2, "c": 3, "d": 4}}},
    )

    output = str(printed[0][0])
    assert "a=1" in output
    assert "b=2" in output
    assert "c=3" in output
    assert "d=4" not in output


def test_render_tool_call_escapes_markup_in_args() -> None:
    output = io.StringIO()
    console = Console(file=output)

    _render_tool_call(
        console,
        {"name": "tool", "data": {"input": {"[key]": "[value]"}}},
    )

    captured = output.getvalue()
    assert "[key]" in captured
    assert "[value]" in captured


@pytest.mark.asyncio
async def test_repl_sessions_command_lists_rows(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    async with aiosqlite.connect(str(db_path)) as db_conn:
        await runner_module.sessions.init_db(db_conn)
        await runner_module.sessions.upsert(db_conn, "abc-123", title="first turn")

    agent = _FakeAgent()
    _, printed = _patch_console(monkeypatch, ["/sessions", "/exit"])

    async with aiosqlite.connect(str(db_path)) as db_conn:
        await runner_module.sessions.init_db(db_conn)
        await repl(agent, "thread-1", db_conn=db_conn)

    output = "\n".join(str(p[0][0]) for p in printed if p[0])
    assert "abc-123" in output
    assert "first turn" in output


@pytest.mark.asyncio
async def test_repl_resume_command_swaps_thread_id(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    async with aiosqlite.connect(str(db_path)) as db_conn:
        await runner_module.sessions.init_db(db_conn)
        await runner_module.sessions.upsert(db_conn, "target-thread", title="old")

    class _RecordingAgent:
        def __init__(self) -> None:
            self.thread_ids: list[str] = []

        async def astream_events(self, input_data: Any, config: Any, *, version: str):
            self.thread_ids.append(config["configurable"]["thread_id"])
            if False:
                yield

    agent = _RecordingAgent()
    _, _ = _patch_console(monkeypatch, ["/resume target", "hi", "/exit"])

    async with aiosqlite.connect(str(db_path)) as db_conn:
        await runner_module.sessions.init_db(db_conn)
        await repl(agent, "thread-1", db_conn=db_conn)

    assert agent.thread_ids == ["target-thread"]


@pytest.mark.asyncio
async def test_repl_compact_command_streams_compact_prompt(monkeypatch) -> None:
    class _CompactAgent:
        async def astream_events(self, input_data: Any, config: Any, *, version: str):
            msg = input_data["messages"][0]
            assert "compact_conversation" in msg.content
            # Yield a tool-end event so the command surfaces output.
            yield {
                "event": "on_tool_end",
                "name": "compact_conversation",
                "data": {"output": type("ToolOutput", (), {"content": "refused"})},
            }

    agent = _CompactAgent()
    _, printed = _patch_console(monkeypatch, ["/compact", "/exit"])

    await repl(agent, "thread-1")

    assert any("refused" in str(p[0][0]) for p in printed if p[0])
