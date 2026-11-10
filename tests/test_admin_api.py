"""Integration tests for the admin API (dream + memory) over ASGI transport."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from langchain_harness.api.app import create_app
from langchain_harness.config import Config, ServerConfig
from langchain_harness import dream, sessions
from langchain_harness.workspaces import user_sessions_db

AUTH = {"Authorization": "Bearer sekrit"}


class _FakeState:
    def __init__(self, messages: list[Any] | None = None) -> None:
        self.values = {"messages": messages or []} if messages else {}


class _FakeMsg:
    def __init__(self, msg_type: str, content: str) -> None:
        self.type = msg_type
        self.content = content


class _FakeAgent:
    """Fake agent exposing state for the summarizer."""

    def __init__(self, messages: list[Any] | None = None) -> None:
        self._messages = messages or []

    async def aget_state(self, config: Any) -> _FakeState:
        return _FakeState(messages=self._messages)


class _FakeModel:
    """Fake shared model returning preset content and recording calls."""

    def __init__(self, content: str = "- captured fact") -> None:
        self.content = content
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.calls.append(messages)
        return type("Result", (), {"content": self.content})()


class _FakeAdminCache:
    """Minimal cache recording invalidations for admin endpoint tests."""

    def __init__(
        self,
        agent: Any,
        shared_model: Any | None = None,
        known_users: list[str] | None = None,
    ) -> None:
        self._agent = agent
        self.shared_model = shared_model
        self.known_users = known_users or []
        self.invalidated: list[str] = []
        self.invalidated_all_calls = 0

    async def get_agent(self, user_id: str) -> Any:
        return self._agent

    async def invalidate(self, user_id: str) -> None:
        self.invalidated.append(user_id)

    async def invalidate_all(self) -> list[str]:
        self.invalidated_all_calls += 1
        return list(self.known_users)

    async def close_all(self) -> None:
        pass


async def _seed_user(
    root: Path, uid: str, thread_id: str = "t1", title: str = "A session"
) -> None:
    db = user_sessions_db(root, uid)
    db.parent.mkdir(parents=True, exist_ok=True)
    async with sessions.connect(db) as conn:
        await sessions.upsert(conn, thread_id, title=title, user_id=uid)


def _write_user_memory(root: Path, uid: str, text: str) -> None:
    path = dream.user_memory_path(root, uid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_draft(root: Path, draft_id: str, body: str) -> str:
    """Write a minimal valid draft; returns the draft id."""
    directory = dream.drafts_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.safe_dump(
        {
            "kind": "notes-draft",
            "created_at": "2025-02-01T00:00:00+00:00",
            "items": 1,
            "users": ["u1"],
        },
        sort_keys=False,
        default_style="'",
        allow_unicode=True,
    ).strip()
    (directory / f"{draft_id}.md").write_text(
        f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8"
    )
    return draft_id


@pytest.fixture
async def admin_env(tmp_path, monkeypatch):
    """App with an admin token, fake cache, and two seeded users."""
    monkeypatch.chdir(tmp_path)
    config = Config(
        provider="anthropic",
        model="test-model",
        server=ServerConfig(admin_token="sekrit"),
    )
    app = create_app(tmp_path, config)

    fake_agent = _FakeAgent([_FakeMsg("human", "hi"), _FakeMsg("ai", "hello")])
    fake_cache = _FakeAdminCache(
        fake_agent,
        shared_model=_FakeModel("- captured fact"),
        known_users=["u1", "u2"],
    )

    async with app.router.lifespan_context(app):
        app.state.get_agent = fake_cache.get_agent
        app.state.agent_cache = fake_cache

        await _seed_user(tmp_path, "u1")
        await _seed_user(tmp_path, "u2")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield {
                "client": client,
                "cache": fake_cache,
                "tmp_path": tmp_path,
            }


# --- auth ---


@pytest.mark.asyncio
async def test_admin_disabled_without_token(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    app = create_app(tmp_path, Config(provider="anthropic", model="test-model"))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/v1/admin/memory/status")
    assert resp.status_code == 403
    assert "Admin API disabled" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_admin_wrong_token_returns_403(admin_env) -> None:
    resp = await admin_env["client"].get(
        "/v1/admin/memory/status",
        headers={"Authorization": "Bearer nope"},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Invalid admin token"


@pytest.mark.asyncio
async def test_admin_correct_token_returns_200(admin_env) -> None:
    resp = await admin_env["client"].get("/v1/admin/memory/status", headers=AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert [u["user_id"] for u in data["users"]] == ["u1", "u2"]


# --- dream capture ---


@pytest.mark.asyncio
async def test_capture_endpoint_writes_user_memory(admin_env) -> None:
    client = admin_env["client"]
    root = admin_env["tmp_path"]

    resp = await client.post("/v1/admin/dream/capture", headers=AUTH)

    assert resp.status_code == 200
    data = resp.json()
    assert data["captured"] == 2
    assert [entry["user_id"] for entry in data["users"]] == ["u1", "u2"]
    memory_path = dream.user_memory_path(root, "u1")
    other_path = dream.user_memory_path(root, "u2")
    assert memory_path.is_file()
    assert "### " in memory_path.read_text(encoding="utf-8")
    assert other_path.is_file()
    # Users with captured blocks get their agent invalidated.
    assert sorted(admin_env["cache"].invalidated) == ["u1", "u2"]


@pytest.mark.asyncio
async def test_capture_rejects_unsafe_user_id(admin_env) -> None:
    resp = await admin_env["client"].post(
        "/v1/admin/dream/capture",
        params={"user_id": "../etc"},
        headers=AUTH,
    )
    assert resp.status_code == 400


# --- consolidate + drafts ---


@pytest.mark.asyncio
async def test_consolidate_creates_draft_and_lists_it(admin_env) -> None:
    client = admin_env["client"]
    root = admin_env["tmp_path"]
    _write_user_memory(
        root,
        "u1",
        "### 2025-06-01T10:00:00+00:00 — Prefs (session t1)\n\n- likes dark mode\n",
    )

    resp = await client.post("/v1/admin/dream/consolidate", headers=AUTH)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "draft"
    assert data["draft_id"].startswith("notes-")
    assert data["items"] == 1

    listing = await client.get("/v1/admin/memory/drafts", headers=AUTH)
    assert listing.status_code == 200
    assert [d["draft_id"] for d in listing.json()] == [data["draft_id"]]

    detail = await client.get(f"/v1/admin/memory/drafts/{data['draft_id']}", headers=AUTH)
    assert detail.status_code == 200
    assert detail.json()["draft_id"] == data["draft_id"]
    # The fake model's canned output is what lands in the draft body.
    assert "- captured fact" in detail.json()["content"]


@pytest.mark.asyncio
async def test_approve_replaces_notes_and_invalidates_all(admin_env) -> None:
    client = admin_env["client"]
    root = admin_env["tmp_path"]
    draft_id = _write_draft(root, "notes-a", "# Notes\n\n- merged fact")

    resp = await client.post(
        f"/v1/admin/memory/drafts/{draft_id}/approve", headers=AUTH
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["approved"] is True
    assert data["invalidated_users"] == ["u1", "u2"]
    assert admin_env["cache"].invalidated_all_calls == 1

    notes = (root / ".harness" / "memory" / "notes.md").read_text(encoding="utf-8")
    assert "# Notes" in notes
    assert "- merged fact" in notes
    assert not (root / ".harness" / "memory" / "drafts" / f"{draft_id}.md").exists()


@pytest.mark.asyncio
async def test_approve_missing_draft_returns_404(admin_env) -> None:
    resp = await admin_env["client"].post(
        "/v1/admin/memory/drafts/notes-missing/approve", headers=AUTH
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_draft(admin_env) -> None:
    client = admin_env["client"]
    root = admin_env["tmp_path"]
    draft_id = _write_draft(root, "notes-a", "body")

    resp = await client.delete(f"/v1/admin/memory/drafts/{draft_id}", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True}

    again = await client.delete(f"/v1/admin/memory/drafts/{draft_id}", headers=AUTH)
    assert again.status_code == 404

    missing = await client.get("/v1/admin/memory/drafts/notes-a", headers=AUTH)
    assert missing.status_code == 404


# --- agents invalidate ---


@pytest.mark.asyncio
async def test_invalidate_single_agent(admin_env) -> None:
    client = admin_env["client"]

    resp = await client.post(
        "/v1/admin/agents/invalidate",
        params={"user_id": "u1"},
        headers=AUTH,
    )

    assert resp.status_code == 200
    assert resp.json() == {"invalidated": ["u1"]}
    assert admin_env["cache"].invalidated == ["u1"]


@pytest.mark.asyncio
async def test_invalidate_all_agents(admin_env) -> None:
    client = admin_env["client"]

    resp = await client.post("/v1/admin/agents/invalidate", headers=AUTH)

    assert resp.status_code == 200
    assert resp.json() == {"invalidated": ["u1", "u2"]}
    assert admin_env["cache"].invalidated_all_calls == 1
