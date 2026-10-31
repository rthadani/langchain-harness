"""HTTP tests for the private per-user published skill/agent endpoints."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from langchain_harness.agent_builder import build_agent
from langchain_harness.api.app import UserAgentCache, create_app
from langchain_harness.config import Config
from langchain_harness.guardrails import GuardrailMiddleware
from langchain_harness.published import (
    PublishError,
    create_draft,
    draft_dir,
    make_draft_tools,
    publish,
    published_agent_files,
    published_dir,
)
from langchain_harness.workspaces import provision_user_workspace, user_workspace


class _FakeAgentCache:
    def __init__(self) -> None:
        self.invalidated: list[str] = []
        self.shared_mcp_tools: list[Any] = []

    async def get_agent(self, user_id: str) -> Any:
        return object()

    async def invalidate(self, user_id: str) -> None:
        self.invalidated.append(user_id)

    async def close_all(self) -> None:
        pass


@pytest.fixture
async def api(tmp_path: Path, monkeypatch):
    """App with trust auth and a fake agent cache (no model/MCP work)."""
    monkeypatch.chdir(tmp_path)
    config = Config(provider="anthropic", model="test-model")
    app = create_app(tmp_path, config)
    cache = _FakeAgentCache()

    async with app.router.lifespan_context(app):
        app.state.agent_cache = cache
        app.state.get_agent = cache.get_agent
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield {"client": client, "cache": cache, "cwd": tmp_path}


def _headers(user_id: str) -> dict[str, str]:
    return {"X-User-Id": user_id}


def _draft_id(
    cwd: Path,
    user_id: str = "alice",
    *,
    kind: str = "skills",
    name: str = "notes",
    body: str = "# Notes",
) -> str:
    workspace = provision_user_workspace(cwd, user_id)
    draft = create_draft(
        user_id, workspace, kind=kind, name=name, description=f"{name} item", body=body
    )
    return draft["draft_id"]


async def _publish_via_http(
    client: httpx.AsyncClient, cwd: Path, user_id: str, name: str
) -> str:
    draft_id = _draft_id(cwd, user_id, name=name)
    response = await client.post(f"/v1/published/{draft_id}", headers=_headers(user_id))
    assert response.status_code == 200, response.text
    return draft_id


# --- end to end ---


@pytest.mark.asyncio
async def test_publish_flow_end_to_end(api) -> None:
    cwd = api["cwd"]
    client = api["client"]

    workspace = provision_user_workspace(cwd, "alice")
    tools = {tool.name: tool for tool in make_draft_tools("alice", workspace)}
    drafted = json.loads(
        tools["draft_skill"].invoke(
            {"name": "notes", "description": "my notes", "body": "# Notes\nuse me"}
        )
    )
    draft_id = drafted["draft_id"]

    # The draft is visible as a preview but nothing is published yet.
    response = await client.get(f"/v1/published/drafts/{draft_id}", headers=_headers("alice"))
    assert response.status_code == 200
    preview = response.json()
    assert preview["name"] == "notes"
    assert preview["body"] == "# Notes\nuse me"
    response = await client.get("/v1/published", headers=_headers("alice"))
    assert response.json() == []

    # The POST is the confirmation.
    response = await client.post(f"/v1/published/{draft_id}", headers=_headers("alice"))
    assert response.status_code == 200
    published = response.json()
    assert published["kind"] == "skills"
    assert published["name"] == "notes"
    assert published["body"].strip() == "# Notes\nuse me"

    response = await client.get("/v1/published", headers=_headers("alice"))
    assert [item["name"] for item in response.json()] == ["notes"]
    response = await client.get("/v1/published?kind=agents", headers=_headers("alice"))
    assert response.json() == []
    response = await client.get(
        "/v1/published/items/skills/notes", headers=_headers("alice")
    )
    assert response.status_code == 200
    assert response.json()["description"] == "my notes"

    assert (published_dir(cwd, "alice", "skills") / "notes" / "SKILL.md").is_file()
    # The draft was consumed by the publish.
    response = await client.get(f"/v1/published/drafts/{draft_id}", headers=_headers("alice"))
    assert response.status_code == 404
    assert api["cache"].invalidated == ["alice"]


@pytest.mark.asyncio
async def test_publish_and_delete_invalidate_agent_cache(api) -> None:
    client = api["client"]
    cwd = api["cwd"]

    await _publish_via_http(client, cwd, "alice", "notes")
    assert api["cache"].invalidated == ["alice"]

    response = await client.delete(
        "/v1/published/items/skills/notes", headers=_headers("alice")
    )
    assert response.status_code == 200
    assert api["cache"].invalidated == ["alice", "alice"]
    assert (await client.get("/v1/published", headers=_headers("alice"))).json() == []

    response = await client.delete(
        "/v1/published/items/skills/notes", headers=_headers("alice")
    )
    assert response.status_code == 404


# --- privacy ---


@pytest.mark.asyncio
async def test_user_b_cannot_see_or_delete_user_a_items(api) -> None:
    client = api["client"]
    cwd = api["cwd"]

    await _publish_via_http(client, cwd, "alice", "notes")

    assert (await client.get("/v1/published", headers=_headers("bob"))).json() == []
    assert (
        await client.get("/v1/published/items/skills/notes", headers=_headers("bob"))
    ).status_code == 404
    assert (
        await client.delete("/v1/published/items/skills/notes", headers=_headers("bob"))
    ).status_code == 404
    # The item is still there for its owner.
    assert (
        await client.get("/v1/published/items/skills/notes", headers=_headers("alice"))
    ).status_code == 200

    # A draft lives in its creator's workspace; only the creator can confirm it.
    alice_draft = _draft_id(cwd, "alice", name="second")
    assert (
        await client.get(f"/v1/published/drafts/{alice_draft}", headers=_headers("bob"))
    ).status_code == 404
    assert (
        await client.post(f"/v1/published/{alice_draft}", headers=_headers("bob"))
    ).status_code == 404
    response = await client.post(f"/v1/published/{alice_draft}", headers=_headers("alice"))
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_publish_requires_auth(api) -> None:
    response = await api["client"].post(
        "/v1/published/" + "0" * 32,
    )
    assert response.status_code == 400
    assert "X-User-Id" in response.json()["detail"]


# --- build wiring ---


class _FakeStore:
    @asynccontextmanager
    async def checkpointer(self, user_id: str):
        yield object()


@pytest.mark.asyncio
async def test_build_user_agent_never_mutates_shared_tools(tmp_path: Path, monkeypatch) -> None:
    """Draft tools are per-user closures, not additions to the shared list."""
    cache = UserAgentCache(
        tmp_path, Config(provider="anthropic", model="claude-3"), _FakeStore()
    )
    cache.shared_mcp_tools = ["mcp-tool"]
    captured: dict[str, Any] = {}

    async def fake_build_agent(*args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("langchain_harness.api.app.build_agent", fake_build_agent)

    await cache._build_user_agent("alice")

    tools = captured["tools_override"]
    assert tools[0] == "mcp-tool"
    assert {tool.name for tool in tools[1:]} == {"draft_skill", "draft_agent"}
    assert cache.shared_mcp_tools == ["mcp-tool"]

    alice_tool = tools[1]
    result = json.loads(
        alice_tool.invoke({"name": "a-skill", "description": "a", "body": "a"})
    )
    alice_drafts = draft_dir(user_workspace(tmp_path, "alice"))
    bob_drafts = draft_dir(user_workspace(tmp_path, "bob"))
    assert (alice_drafts / f"{result['draft_id']}.json").is_file()
    assert not (bob_drafts / f"{result['draft_id']}.json").exists()

    await cache._build_user_agent("bob")
    assert cache.shared_mcp_tools == ["mcp-tool"]


@pytest.mark.asyncio
async def test_invalidate_retires_stack_without_closing(tmp_path: Path) -> None:
    cache = UserAgentCache(
        tmp_path, Config(provider="anthropic", model="claude-3"), _FakeStore()
    )
    closed: list[str] = []

    class _Stack:
        async def aclose(self) -> None:
            closed.append("closed")

    cache._cache["alice"] = object()
    cache._stacks["alice"] = _Stack()  # type: ignore[assignment]

    await cache.invalidate("alice")

    assert "alice" not in cache._cache
    assert cache._stacks == {}
    # Retired, not closed: an in-flight SSE stream may still be using it.
    assert len(cache._retired_stacks) == 1
    assert closed == []

    await cache.close_all()
    assert closed == ["closed"]
    assert cache._retired_stacks == []


@pytest.mark.asyncio
async def test_published_agent_keeps_guardrails(tmp_path: Path, monkeypatch) -> None:
    cwd = tmp_path
    (cwd / "GUARDRAILS.md").write_text("HOST RULE", encoding="utf-8")
    workspace = provision_user_workspace(cwd, "alice")
    draft = create_draft(
        "alice",
        workspace,
        kind="agents",
        name="helper",
        description="helper agent",
        body="do things",
    )
    publish(cwd, "alice", workspace, draft["draft_id"])

    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(
        "langchain_harness.agent_builder.create_deep_agent", fake_create_deep_agent
    )

    await build_agent(
        Config(provider="anthropic", model="claude-3"),
        cwd,
        checkpointer=object(),
        enable_shell=False,
        root_dir=workspace,
        skills_override=[],
        memory_override=[],
        tools_override=[],
        backend=object(),
        agents_override=published_agent_files(cwd, "alice"),
    )

    specs = [spec for spec in captured["subagents"] if spec["name"] == "helper"]
    assert len(specs) == 1
    middleware = specs[0]["middleware"]
    assert len(middleware) == 1
    assert isinstance(middleware[0], GuardrailMiddleware)
    assert middleware[0].text == "HOST RULE"


@pytest.mark.asyncio
async def test_published_agent_cannot_shadow_global(tmp_path: Path, monkeypatch) -> None:
    cwd = tmp_path
    agents_dir = cwd / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    global_path = agents_dir / "helper.md"
    global_path.write_text(
        "---\nname: helper\ndescription: global helper\n---\nglobal body",
        encoding="utf-8",
    )

    workspace = provision_user_workspace(cwd, "alice")
    draft = create_draft(
        "alice",
        workspace,
        kind="agents",
        name="helper",
        description="published helper",
        body="published body",
    )
    # The publish itself rejects shadowing a global name.
    with pytest.raises(PublishError) as excinfo:
        publish(
            cwd,
            "alice",
            workspace,
            draft["draft_id"],
            global_agent_names={"helper"},
        )
    assert excinfo.value.status == 409

    # Even a store file placed out of band is shadowed at build time: global
    # files are listed first and the first definition of a name wins.
    published_path = published_dir(cwd, "alice", "agents") / "helper.md"
    published_path.parent.mkdir(parents=True, exist_ok=True)
    published_path.write_text(
        "---\nname: helper\ndescription: published helper\n---\npublished body",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> object:
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(
        "langchain_harness.agent_builder.create_deep_agent", fake_create_deep_agent
    )

    await build_agent(
        Config(provider="anthropic", model="claude-3"),
        cwd,
        checkpointer=object(),
        enable_shell=False,
        root_dir=workspace,
        skills_override=[],
        memory_override=[],
        tools_override=[],
        backend=object(),
        agents_override=[global_path, *published_agent_files(cwd, "alice")],
    )

    specs = [spec for spec in captured["subagents"] if spec["name"] == "helper"]
    assert len(specs) == 1
    assert specs[0]["system_prompt"].strip() == "global body"


@pytest.mark.asyncio
async def test_shell_mode_skill_sources_are_relative_to_the_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    """Shell mode uses a non-virtual backend, so sources must be relative.

    A host absolute path or a leading-slash virtual path would resolve outside
    the workspace there; ``_build_user_agent`` strips the prefix instead.
    """
    from deepagents.backends import LocalShellBackend
    from deepagents.middleware.skills import _list_skills

    from langchain_harness.config import ServerConfig

    config = Config(
        provider="anthropic",
        model="claude-3",
        server=ServerConfig(enable_shell=True),
    )
    cache = UserAgentCache(tmp_path, config, _FakeStore())
    captured: dict[str, Any] = {}

    async def fake_build_agent(*args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("langchain_harness.api.app.build_agent", fake_build_agent)

    workspace = provision_user_workspace(tmp_path, "alice")
    skill_dir = workspace / ".claude" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo skill\n---\nbody", encoding="utf-8"
    )

    await cache._build_user_agent("alice")

    sources = captured["skills_override"]
    assert sources == [".claude/skills"]
    assert not any(source.startswith("/") for source in sources)
    backend = LocalShellBackend(root_dir=workspace, virtual_mode=False, inherit_env=True)
    assert [skill["name"] for skill in _list_skills(backend, sources[0])] == ["demo"]


@pytest.mark.asyncio
async def test_server_mode_memory_sources_are_virtual_paths(
    tmp_path: Path, monkeypatch
) -> None:
    """Server mode uses a virtual-mode backend, so memory sources carry the
    leading slash and must reach ``build_agent`` unmodified (host absolute paths
    would resolve under the workspace root and never load)."""
    config = Config(provider="anthropic", model="claude-3")
    cache = UserAgentCache(tmp_path, config, _FakeStore())
    captured: dict[str, Any] = {}

    async def fake_build_agent(*args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("langchain_harness.api.app.build_agent", fake_build_agent)

    workspace = provision_user_workspace(tmp_path, "alice")
    (workspace / "CLAUDE.md").write_text("# Claude", encoding="utf-8")
    (workspace / "AGENTS.md").write_text("# Agents", encoding="utf-8")

    await cache._build_user_agent("alice")

    assert captured["memory_override"] == ["/CLAUDE.md", "/AGENTS.md"]


@pytest.mark.asyncio
async def test_shell_mode_memory_sources_are_relative_to_the_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    """Shell mode uses a non-virtual backend, which treats a leading slash as a
    host path; ``_build_user_agent`` strips the prefix instead."""
    from deepagents.backends import LocalShellBackend

    from langchain_harness.config import ServerConfig

    config = Config(
        provider="anthropic",
        model="claude-3",
        server=ServerConfig(enable_shell=True),
    )
    cache = UserAgentCache(tmp_path, config, _FakeStore())
    captured: dict[str, Any] = {}

    async def fake_build_agent(*args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("langchain_harness.api.app.build_agent", fake_build_agent)

    workspace = provision_user_workspace(tmp_path, "alice")
    (workspace / "CLAUDE.md").write_text("# Claude", encoding="utf-8")

    await cache._build_user_agent("alice")

    sources = captured["memory_override"]
    assert sources == ["CLAUDE.md"]
    assert not any(source.startswith("/") for source in sources)
    backend = LocalShellBackend(root_dir=workspace, virtual_mode=False, inherit_env=True)
    assert [response.error for response in backend.download_files(sources)] == [None]


@pytest.mark.asyncio
async def test_get_agent_rebuilds_after_invalidate(tmp_path: Path, monkeypatch) -> None:
    cache = UserAgentCache(
        tmp_path, Config(provider="anthropic", model="claude-3"), _FakeStore()
    )
    built: list[str] = []

    async def fake_build(user_id: str) -> object:
        built.append(user_id)
        return object()

    monkeypatch.setattr(cache, "_build_user_agent", fake_build)

    first = await cache.get_agent("alice")
    assert await cache.get_agent("alice") is first
    assert built == ["alice"]

    await cache.invalidate("alice")
    second = await cache.get_agent("alice")

    assert second is not first
    assert built == ["alice", "alice"]
