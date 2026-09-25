from pathlib import Path
from typing import Any

import pytest

from langchain_harness.agent_builder import (
    _build_model,
    _load_subagents,
    _resolve_subagent_model,
    build_agent,
)
from langchain_harness.config import Config


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
    await build_agent(config, tmp_path)

    assert captured.get("inherit_env") is True
