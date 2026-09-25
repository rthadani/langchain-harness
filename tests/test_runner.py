from typing import Any

import pytest
from rich.console import Console

from langchain_harness import runner as runner_module
from langchain_harness.runner import _stream_turn, repl


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

    await repl(agent)

    assert not agent.inputs
    assert any("Goodbye" in str(p[0]) for p in printed)


@pytest.mark.asyncio
async def test_repl_quit_command(monkeypatch) -> None:
    agent = _FakeAgent()
    _, printed = _patch_console(monkeypatch, ["/quit"])

    await repl(agent)

    assert not agent.inputs
    assert any("Goodbye" in str(p[0]) for p in printed)


@pytest.mark.asyncio
async def test_repl_empty_prompt_skipped(monkeypatch) -> None:
    agent = _FakeAgent()
    _, _ = _patch_console(monkeypatch, ["", "", "/exit"])

    await repl(agent)

    assert not agent.inputs


@pytest.mark.asyncio
async def test_repl_eof_quits(monkeypatch) -> None:
    agent = _FakeAgent()
    _, printed = _patch_console(monkeypatch, [])

    await repl(agent)

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

    await repl(agent)

    assert any("Error:" in str(p[0]) and "boom" in str(p[0]) for p in printed)
