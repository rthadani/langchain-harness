from pathlib import Path
from typing import Any

import pytest

from langchain_harness.agent_builder import (
    _MCP_TOOLS_TIMEOUT,
    _build_model,
    _build_summarization_middleware,
    _load_subagents,
    _resolve_subagent_model,
    build_agent,
)
from langchain_harness.config import CompactionConfig, Config


def test_load_subagents_skips_missing_name_or_description(tmp_path: Path, caplog) -> None:
    agents_dir = tmp_path / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "bad.md").write_text("---\ndescription: desc\n---\nbody", encoding="utf-8")
    (agents_dir / "good.md").write_text("---\nname: foo\ndescription: desc\n---\nbody", encoding="utf-8")

    with caplog.at_level("WARNING"):
        subagents, sources = _load_subagents([agents_dir / "bad.md", agents_dir / "good.md"], "anthropic")

    assert len(subagents) == 1
    assert subagents[0]["name"] == "foo"
    assert "must include 'name' and 'description'" in caplog.text
    assert sources == {"foo": agents_dir / "good.md"}


def test_load_subagents_translates_model_aliases(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "a.md").write_text(
        "---\nname: sonnet-agent\ndescription: d\nmodel: sonnet\n---\nbody",
        encoding="utf-8",
    )
    (agents_dir / "b.md").write_text(
        "---\nname: opus-agent\ndescription: d\nmodel: opus\n---\nbody",
        encoding="utf-8",
    )
    (agents_dir / "c.md").write_text(
        "---\nname: inherit-agent\ndescription: d\nmodel: inherit\n---\nbody",
        encoding="utf-8",
    )

    subagents, _ = _load_subagents([
        agents_dir / "a.md",
        agents_dir / "b.md",
        agents_dir / "c.md",
    ], "anthropic")

    by_name = {s["name"]: s for s in subagents}
    assert by_name["sonnet-agent"]["model"] == "anthropic:claude-sonnet-4-5"
    assert by_name["opus-agent"]["model"] == "anthropic:claude-opus-4-5"
    assert "model" not in by_name["inherit-agent"]


def test_load_subagents_warns_and_skips_bad_model(tmp_path: Path, caplog) -> None:
    agents_dir = tmp_path / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "a.md").write_text(
        "---\nname: bad-model\ndescription: d\nmodel: 'anthropic:'\n---\nbody",
        encoding="utf-8",
    )

    with caplog.at_level("WARNING"):
        subagents, _ = _load_subagents([agents_dir / "a.md"], "anthropic")

    assert len(subagents) == 1
    assert "model" not in subagents[0]
    assert "Unrecognised model" in caplog.text


def test_load_subagents_accepts_provider_prefixed_model(tmp_path: Path) -> None:
    agents_dir = tmp_path / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "a.md").write_text(
        "---\nname: openai-agent\ndescription: d\nmodel: openai:gpt-4o\n---\nbody",
        encoding="utf-8",
    )

    subagents, _ = _load_subagents([agents_dir / "a.md"], "anthropic")
    assert subagents[0]["model"] == "openai:gpt-4o"


def test_resolve_subagent_model_case_insensitive() -> None:
    assert _resolve_subagent_model("SONNET", "anthropic", Path("x.md")) == "anthropic:claude-sonnet-4-5"
    assert _resolve_subagent_model(" Inherit ", "anthropic", Path("x.md")) is None


def test_resolve_subagent_model_prefixes_bare_model() -> None:
    assert _resolve_subagent_model("claude-sonnet-4-5", "anthropic", Path("x.md")) == "anthropic:claude-sonnet-4-5"


def test_load_subagents_prefixes_bare_model_with_provider(tmp_path: Path, caplog) -> None:
    agents_dir = tmp_path / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "a.md").write_text(
        "---\nname: bare-model\ndescription: d\nmodel: claude-sonnet-4-5\n---\nbody",
        encoding="utf-8",
    )

    with caplog.at_level("INFO"):
        subagents, _ = _load_subagents([agents_dir / "a.md"], "anthropic")

    assert subagents[0]["model"] == "anthropic:claude-sonnet-4-5"
    assert "Prefixed bare model" in caplog.text


def test_build_model_kwargs_matrix(monkeypatch) -> None:
    calls = []

    def fake_init(model: str, **kwargs):
        calls.append((model, kwargs))
        return object()

    monkeypatch.setattr("langchain_harness.agent_builder.init_chat_model", fake_init)

    _build_model(Config(provider="anthropic", model="claude-3"))
    assert calls[-1] == ("claude-3", {"model_provider": "anthropic"})

    _build_model(Config(provider="openai", model="gpt-4", base_url="http://x"))
    assert calls[-1] == ("gpt-4", {"model_provider": "openai", "base_url": "http://x"})

    _build_model(Config(provider="openai", model="gpt-4", api_key="k"))
    assert calls[-1] == ("gpt-4", {"model_provider": "openai", "api_key": "k"})

    _build_model(Config(provider="openai", model="gpt-4", base_url="http://x", api_key="k"))
    assert calls[-1] == (
        "gpt-4",
        {"model_provider": "openai", "base_url": "http://x", "api_key": "k"},
    )


@pytest.mark.asyncio
async def test_build_agent_passes_inherit_env_to_shell_backend(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_shell_backend(*, inherit_env: bool = False, **kwargs: Any) -> object:
        captured["inherit_env"] = inherit_env
        captured["kwargs"] = kwargs
        return object()

    def fake_create_deep_agent(*args: Any, **kwargs: Any) -> object:
        return object()

    monkeypatch.setattr("langchain_harness.agent_builder.LocalShellBackend", fake_shell_backend)
    monkeypatch.setattr("langchain_harness.agent_builder.create_deep_agent", fake_create_deep_agent)

    config = Config(provider="anthropic", model="claude-3")
    await build_agent(config, tmp_path, checkpointer=object())

    assert captured.get("inherit_env") is True


def test_mcp_tools_timeout_is_module_constant() -> None:
    assert isinstance(_MCP_TOOLS_TIMEOUT, int)
    assert _MCP_TOOLS_TIMEOUT == 30


@pytest.mark.asyncio
async def test_build_agent_gracefully_degrades_when_one_mcp_server_fails(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """If one MCP server raises, the healthy server's tools still load."""
    calls: list[tuple[str, dict[str, Any]]] = []

    class _FailingClient:
        def __init__(self, servers: dict[str, Any], *, tool_name_prefix: bool = False) -> None:
            calls.append(("client", servers))

        async def get_tools(self) -> list[Any]:
            raise RuntimeError("server is down")

    class _HealthyClient:
        def __init__(self, servers: dict[str, Any], *, tool_name_prefix: bool = False) -> None:
            calls.append(("client", servers))

        async def get_tools(self) -> list[Any]:
            return [object()]

    def fake_client(servers: dict[str, Any], *, tool_name_prefix: bool = False) -> Any:
        name = next(iter(servers))
        if name == "bad":
            return _FailingClient(servers, tool_name_prefix=tool_name_prefix)
        return _HealthyClient(servers, tool_name_prefix=tool_name_prefix)

    def fake_create_deep_agent(**kwargs: Any) -> object:
        return kwargs

    monkeypatch.setattr(
        "langchain_harness.agent_builder.MultiServerMCPClient", fake_client
    )
    monkeypatch.setattr(
        "langchain_harness.agent_builder.create_deep_agent", fake_create_deep_agent
    )
    monkeypatch.setattr(
        "langchain_harness.agent_builder.load_mcp_connections",
        lambda _path: {"bad": {"command": "echo"}, "good": {"command": "echo"}},
    )

    config = Config(provider="anthropic", model="claude-3")
    with caplog.at_level("WARNING"):
        result = await build_agent(config, tmp_path, checkpointer=object())

    assert "Could not load MCP server" in caplog.text
    tools = result.get("tools") or []
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_build_agent_passes_middleware_when_compaction_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain_harness.agent_builder.create_deep_agent", fake_create_deep_agent
    )

    config = Config(
        provider="anthropic",
        model="claude-3",
        compaction=CompactionConfig(trigger_tokens=1000, keep_messages=10),
    )
    await build_agent(config, tmp_path, checkpointer=object())

    assert "middleware" in captured
    middleware = captured["middleware"]
    assert len(middleware) == 2
    assert middleware[0].__class__.__name__ == "_DeepAgentsSummarizationMiddleware"
    assert middleware[1].__class__.__name__ == "SummarizationToolMiddleware"
    inner = middleware[0]._lc_helper
    assert inner.trigger == ("tokens", 1000)
    assert inner.keep == ("messages", 10)


@pytest.mark.asyncio
async def test_build_agent_omits_middleware_when_compaction_not_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain_harness.agent_builder.create_deep_agent", fake_create_deep_agent
    )

    config = Config(provider="anthropic", model="claude-3")
    await build_agent(config, tmp_path, checkpointer=object())

    assert "middleware" not in captured


@pytest.mark.asyncio
async def test_build_agent_uses_message_trigger_when_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain_harness.agent_builder.create_deep_agent", fake_create_deep_agent
    )

    config = Config(
        provider="anthropic",
        model="claude-3",
        compaction=CompactionConfig(trigger_messages=50, keep_messages=5),
    )
    await build_agent(config, tmp_path, checkpointer=object())

    middleware = captured["middleware"]
    inner = middleware[0]._lc_helper
    assert inner.trigger == ("messages", 50)
    assert inner.keep == ("messages", 5)


def test_build_summarization_middleware_returns_none_when_no_trigger() -> None:
    model = object()
    backend = object()
    compaction = CompactionConfig()
    assert _build_summarization_middleware(model, backend, compaction) is None


# --- build_agent override path ---


@pytest.mark.asyncio
async def test_build_agent_override_skips_model_backend_tools_discovery(
    tmp_path: Path, monkeypatch
) -> None:
    """When all overrides are provided, _build_model, _build_backend, MCP
    loading, and discover_all are all skipped."""
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain_harness.agent_builder.create_deep_agent", fake_create_deep_agent
    )

    # init_chat_model must NOT be called when model is overridden
    def fail_init_chat_model(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("init_chat_model should not be called when model is overridden")

    monkeypatch.setattr(
        "langchain_harness.agent_builder.init_chat_model", fail_init_chat_model
    )

    # MultiServerMCPClient must NOT be called when tools_override is set
    class _ExplodingMCPClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("MCP client should not be created when tools_override is set")

    monkeypatch.setattr(
        "langchain_harness.agent_builder.MultiServerMCPClient", _ExplodingMCPClient
    )

    # discover_all must NOT be called when skills_override / memory_override are set
    discover_calls: list[Any] = []

    def fake_discover_all(cwd: Path) -> Any:
        discover_calls.append(cwd)
        return (None, [], [])

    monkeypatch.setattr(
        "langchain_harness.agent_builder.discover_all", fake_discover_all
    )

    fake_model = object()
    fake_backend = object()

    config = Config(provider="anthropic", model="claude-3")
    await build_agent(
        config,
        tmp_path,
        checkpointer=object(),
        model=fake_model,
        tools_override=[],
        backend=fake_backend,
        skills_override=[],
        memory_override=[],
    )

    assert captured["model"] is fake_model
    assert captured["backend"] is fake_backend
    # tools_override=[] -> mcp_tools=[] -> `mcp_tools or None` -> None
    assert captured["tools"] is None
    # skills_override=[] (falsy) -> skills = None
    assert captured["skills"] is None
    assert discover_calls == []


@pytest.mark.asyncio
async def test_build_agent_truthy_skills_override_passed_through(
    tmp_path: Path, monkeypatch
) -> None:
    """A truthy skills_override list is passed through as the skills source."""
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain_harness.agent_builder.create_deep_agent", fake_create_deep_agent
    )

    config = Config(provider="anthropic", model="claude-3")
    await build_agent(
        config,
        tmp_path,
        checkpointer=object(),
        skills_override=["/some/path"],
        memory_override=[],
    )

    assert captured["skills"] == ["/some/path"]


@pytest.mark.asyncio
async def test_build_agent_model_override_used_while_defaults_load(
    tmp_path: Path, monkeypatch
) -> None:
    """When only model is overridden, the override is used while other defaults
    still load (discovery, MCP, backend)."""
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "langchain_harness.agent_builder.create_deep_agent", fake_create_deep_agent
    )

    # init_chat_model must NOT be called when model is overridden
    def fail_init_chat_model(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("init_chat_model should not be called when model is overridden")

    monkeypatch.setattr(
        "langchain_harness.agent_builder.init_chat_model", fail_init_chat_model
    )

    fake_model = object()

    config = Config(provider="anthropic", model="claude-3")
    await build_agent(
        config,
        tmp_path,
        checkpointer=object(),
        model=fake_model,
    )

    assert captured["model"] is fake_model
    # Defaults still loaded: no skills dir in tmp_path -> skills=None
    assert captured["skills"] is None
    # No mcp.json -> mcp_tools=[] -> tools=None
    assert captured["tools"] is None
    # backend was built via _build_backend (not None)
    assert captured["backend"] is not None
