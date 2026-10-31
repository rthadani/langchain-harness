"""Tests for host-side global guardrails."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import SystemMessage

from langchain_harness.agent_builder import _load_subagents, build_agent
from langchain_harness.config import Config
from langchain_harness.guardrails import (
    GUARDRAILS_FILENAME,
    MAX_GUARDRAILS_CHARS,
    GuardrailMiddleware,
    _append_to_system,
    discover_guardrails,
    guardrail_permissions,
    load_guardrails,
)


# --- discover / load ---


def test_discover_guardrails_missing(tmp_path: Path) -> None:
    assert discover_guardrails(tmp_path) is None


def test_discover_guardrails_present(tmp_path: Path) -> None:
    path = tmp_path / GUARDRAILS_FILENAME
    path.write_text("rules", encoding="utf-8")
    assert discover_guardrails(tmp_path) == path


def test_load_guardrails_none_path() -> None:
    assert load_guardrails(None) is None


def test_load_guardrails_whitespace_only(tmp_path: Path) -> None:
    path = tmp_path / GUARDRAILS_FILENAME
    path.write_text("   \n\n\t", encoding="utf-8")
    assert load_guardrails(path) is None


def test_load_guardrails_returns_none_on_os_error(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    path = tmp_path / GUARDRAILS_FILENAME
    path.write_text("rules", encoding="utf-8")

    def boom(self: Path, *args: Any, **kwargs: Any) -> str:
        raise OSError("unreadable")

    monkeypatch.setattr(Path, "read_text", boom)
    with caplog.at_level("WARNING"):
        assert load_guardrails(path) is None
    assert "Could not read guardrails file" in caplog.text


def test_load_guardrails_truncates_oversize_file(tmp_path: Path, caplog) -> None:
    path = tmp_path / GUARDRAILS_FILENAME
    path.write_text("x" * (MAX_GUARDRAILS_CHARS + 100), encoding="utf-8")

    with caplog.at_level("WARNING"):
        text = load_guardrails(path)

    assert text is not None
    assert len(text) == MAX_GUARDRAILS_CHARS
    assert "truncating" in caplog.text


def test_load_guardrails_strips_text(tmp_path: Path) -> None:
    path = tmp_path / GUARDRAILS_FILENAME
    path.write_text("\n  RULE  \n", encoding="utf-8")
    assert load_guardrails(path) == "RULE"


# --- system message composition ---


def test_append_to_system_without_existing_message() -> None:
    message = _append_to_system(None, "RULE")
    assert isinstance(message, SystemMessage)
    assert "RULE" in message.text
    assert "immutable" in message.text
    assert "outranks" in message.text


def test_append_to_system_preserves_existing_blocks() -> None:
    original = SystemMessage(
        content_blocks=[
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
    )
    message = _append_to_system(original, "RULE")

    blocks = message.content_blocks
    assert [block["text"] for block in blocks[:2]] == ["first", "second"]
    assert "RULE" in blocks[-1]["text"]
    assert blocks[-1]["text"].startswith("\n\n")


# --- middleware ---


class _StubRequest:
    def __init__(self, system_message: SystemMessage | None = None) -> None:
        self.system_message = system_message

    def override(self, **kwargs: Any) -> "_StubRequest":
        return _StubRequest(kwargs.get("system_message"))


def test_guardrail_middleware_rewrites_request_system_message() -> None:
    captured: list[_StubRequest] = []

    def handler(request: _StubRequest) -> object:
        captured.append(request)
        return object()

    middleware = GuardrailMiddleware("HOST RULE")
    middleware.wrap_model_call(_StubRequest(), handler)

    assert len(captured) == 1
    system_message = captured[0].system_message
    assert system_message is not None
    assert "HOST RULE" in system_message.text


@pytest.mark.asyncio
async def test_guardrail_middleware_async_rewrites_request_system_message() -> None:
    captured: list[_StubRequest] = []

    async def handler(request: _StubRequest) -> object:
        captured.append(request)
        return object()

    middleware = GuardrailMiddleware("HOST RULE")
    await middleware.awrap_model_call(_StubRequest(), handler)

    assert len(captured) == 1
    system_message = captured[0].system_message
    assert system_message is not None
    assert "HOST RULE" in system_message.text


# --- permissions ---


def test_guardrail_permissions_shell_mode_returns_empty() -> None:
    assert guardrail_permissions(enable_shell=True) == []


def test_guardrail_permissions_server_mode_denies_write() -> None:
    permissions = guardrail_permissions(enable_shell=False)
    assert len(permissions) == 1
    rule = permissions[0]
    assert rule.mode == "deny"
    assert rule.operations == ["write"]
    assert rule.paths == ["/GUARDRAILS.md"]


# --- subagent loading ---


def _write_agent(root: Path, name: str) -> Path:
    agents_dir = root / ".claude" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / f"{name}.md"
    path.write_text(
        f"---\nname: {name}\ndescription: helper\n---\nbody", encoding="utf-8"
    )
    return path


def test_load_subagents_adds_guardrail_middleware(tmp_path: Path) -> None:
    path = _write_agent(tmp_path, "helper")
    subagents, _ = _load_subagents([path], "anthropic", guardrails_text="HOST RULE")

    assert len(subagents) == 1
    middleware = subagents[0]["middleware"]
    assert len(middleware) == 1
    assert isinstance(middleware[0], GuardrailMiddleware)
    assert middleware[0].text == "HOST RULE"


def test_load_subagents_without_guardrails_has_no_middleware(tmp_path: Path) -> None:
    path = _write_agent(tmp_path, "helper")
    subagents, _ = _load_subagents([path], "anthropic")

    assert "middleware" not in subagents[0]


# --- build_agent integration ---


def _patch_agent_build(monkeypatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(
        "langchain_harness.agent_builder.create_deep_agent", fake_create_deep_agent
    )
    monkeypatch.setattr(
        "langchain_harness.agent_builder.load_mcp_connections", lambda _path: {}
    )
    return captured


def _guardrail_texts(middleware: list[Any]) -> list[str]:
    return [m.text for m in middleware if isinstance(m, GuardrailMiddleware)]


@pytest.mark.asyncio
async def test_build_agent_reads_guardrails_host_side(
    tmp_path: Path, monkeypatch
) -> None:
    """A GUARDRAILS.md in the user workspace cannot replace the host copy."""
    captured = _patch_agent_build(monkeypatch)

    server_cwd = tmp_path / "server"
    server_cwd.mkdir()
    (server_cwd / GUARDRAILS_FILENAME).write_text("HOST RULE", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / GUARDRAILS_FILENAME).write_text("TAMPERED", encoding="utf-8")

    config = Config(provider="anthropic", model="claude-3")
    await build_agent(
        config,
        server_cwd,
        checkpointer=object(),
        enable_shell=False,
        root_dir=workspace,
        skills_override=[],
        memory_override=[],
        backend=object(),
    )

    assert _guardrail_texts(captured["middleware"]) == ["HOST RULE"]


@pytest.mark.asyncio
async def test_build_agent_guardrails_reach_every_subagent(
    tmp_path: Path, monkeypatch
) -> None:
    captured = _patch_agent_build(monkeypatch)

    (tmp_path / GUARDRAILS_FILENAME).write_text("HOST RULE", encoding="utf-8")
    # Defined in the cwd, the only place subagents are discovered from.
    _write_agent(tmp_path, "helper")

    config = Config(provider="anthropic", model="claude-3")
    await build_agent(
        config,
        tmp_path,
        checkpointer=object(),
        enable_shell=False,
        skills_override=[],
        memory_override=[],
        backend=object(),
    )

    subagents = captured["subagents"]
    names = [s["name"] for s in subagents]
    assert "helper" in names
    for spec in subagents:
        middleware = spec["middleware"]
        assert len(middleware) == 1
        assert isinstance(middleware[0], GuardrailMiddleware)
        assert middleware[0].text == "HOST RULE"


@pytest.mark.asyncio
async def test_build_agent_adds_general_purpose_subagent(
    tmp_path: Path, monkeypatch
) -> None:
    captured = _patch_agent_build(monkeypatch)
    (tmp_path / GUARDRAILS_FILENAME).write_text("HOST RULE", encoding="utf-8")

    config = Config(provider="anthropic", model="claude-3")
    await build_agent(
        config,
        tmp_path,
        checkpointer=object(),
        enable_shell=False,
        skills_override=[],
        memory_override=[],
        backend=object(),
    )

    by_name = {s["name"]: s for s in captured["subagents"]}
    general = by_name["general-purpose"]
    assert len(general["middleware"]) == 1
    assert isinstance(general["middleware"][0], GuardrailMiddleware)


@pytest.mark.asyncio
async def test_build_agent_without_guardrails_adds_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    captured = _patch_agent_build(monkeypatch)

    config = Config(provider="anthropic", model="claude-3")
    await build_agent(
        config,
        tmp_path,
        checkpointer=object(),
        enable_shell=False,
        skills_override=[],
        memory_override=[],
        backend=object(),
    )

    assert _guardrail_texts(captured.get("middleware") or []) == []
    assert captured["subagents"] is None
    # Server mode still protects the guardrail path even with no file to load.
    assert captured["permissions"][0].paths == ["/GUARDRAILS.md"]


@pytest.mark.asyncio
async def test_build_agent_denies_guardrail_writes_in_server_mode(
    tmp_path: Path, monkeypatch
) -> None:
    captured = _patch_agent_build(monkeypatch)
    (tmp_path / GUARDRAILS_FILENAME).write_text("HOST RULE", encoding="utf-8")

    config = Config(provider="anthropic", model="claude-3")
    await build_agent(
        config,
        tmp_path,
        checkpointer=object(),
        enable_shell=False,
        skills_override=[],
        memory_override=[],
        backend=object(),
    )

    permissions = captured["permissions"]
    assert len(permissions) == 1
    assert permissions[0].mode == "deny"
    assert permissions[0].paths == ["/GUARDRAILS.md"]


@pytest.mark.asyncio
async def test_build_agent_warns_and_omits_permissions_under_shell(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    captured = _patch_agent_build(monkeypatch)
    (tmp_path / GUARDRAILS_FILENAME).write_text("HOST RULE", encoding="utf-8")

    config = Config(provider="anthropic", model="claude-3")
    with caplog.at_level("WARNING"):
        await build_agent(
            config,
            tmp_path,
            checkpointer=object(),
            enable_shell=True,
            skills_override=[],
            memory_override=[],
            backend=object(),
        )

    assert "permissions" not in captured
    assert "advisory only under shell" in caplog.text
