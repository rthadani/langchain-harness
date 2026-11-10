"""Tests for the dream worker: capture, consolidate, and draft approval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from langchain_harness import dream, memory, sessions
from langchain_harness.frontmatter import parse_frontmatter
from langchain_harness.workspaces import user_sessions_db


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


def _get_agent(agent: Any) -> Callable[[str], Any]:
    async def _get(uid: str) -> Any:
        return agent

    return _get


async def _seed_session(
    root: Path, uid: str, thread_id: str, title: str = "A session"
) -> None:
    db = user_sessions_db(root, uid)
    db.parent.mkdir(parents=True, exist_ok=True)
    async with sessions.connect(db) as conn:
        await sessions.upsert(conn, thread_id, title=title, user_id=uid)


def _write_draft(
    root: Path,
    draft_id: str,
    created_at: str,
    body: str,
    users: list[str] | None = None,
) -> None:
    directory = dream.drafts_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.safe_dump(
        {
            "kind": "notes-draft",
            "created_at": created_at,
            "items": 1,
            "users": users or ["u1"],
        },
        sort_keys=False,
        default_style="'",
        allow_unicode=True,
    ).strip()
    (directory / f"{draft_id}.md").write_text(
        f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8"
    )


# --- capture ---


async def test_capture_appends_block_and_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path
    await _seed_session(root, "u1", "t1", title="Prefs")
    model = _FakeModel("- prefers dark mode")
    agent = _FakeAgent([_FakeMsg("human", "set theme"), _FakeMsg("ai", "done")])

    result = await dream.capture_user(
        cwd=root, user_id="u1", get_agent=_get_agent(agent), model=model
    )

    assert result == {"user_id": "u1", "captured": 1, "skipped": 0}
    text = dream.user_memory_path(root, "u1").read_text(encoding="utf-8")
    assert "- prefers dark mode" in text
    assert "(session t1)" in text

    state = json.loads(
        dream.dream_state_path(root, "u1").read_text(encoding="utf-8")
    )
    assert "t1" in state["processed"]

    second = await dream.capture_user(
        cwd=root, user_id="u1", get_agent=_get_agent(agent), model=model
    )
    assert second == {"user_id": "u1", "captured": 0, "skipped": 0}
    assert len(model.calls) == 1


async def test_capture_skips_empty_session_but_marks_processed(tmp_path: Path) -> None:
    root = tmp_path
    await _seed_session(root, "u1", "t1")
    model = _FakeModel("- should never be called")
    agent = _FakeAgent([_FakeMsg("human", "just looking")])

    result = await dream.capture_user(
        cwd=root, user_id="u1", get_agent=_get_agent(agent), model=model
    )

    assert result == {"user_id": "u1", "captured": 0, "skipped": 1}
    assert model.calls == []
    assert not dream.user_memory_path(root, "u1").exists()
    state = json.loads(
        dream.dream_state_path(root, "u1").read_text(encoding="utf-8")
    )
    assert "t1" in state["processed"]


async def test_capture_all_skips_missing_db_without_error(tmp_path: Path) -> None:
    root = tmp_path
    (root / ".harness" / "users" / "u1").mkdir(parents=True)

    result = await dream.capture_all(
        cwd=root,
        get_agent=_get_agent(_FakeAgent()),
        model=_FakeModel(),
    )

    assert result["captured"] == 0
    assert result["skipped"] == 0
    assert result["users"] == [{"user_id": "u1", "captured": 0, "skipped": 0}]


async def test_capture_all_respects_user_id_param(tmp_path: Path) -> None:
    root = tmp_path
    await _seed_session(root, "u1", "t1")
    await _seed_session(root, "u2", "t2")
    agent = _FakeAgent([_FakeMsg("human", "hi"), _FakeMsg("ai", "hello")])

    result = await dream.capture_all(
        cwd=root,
        get_agent=_get_agent(agent),
        model=_FakeModel("- fact"),
        user_id="u2",
    )

    assert [entry["user_id"] for entry in result["users"]] == ["u2"]
    assert result["captured"] == 1
    assert dream.user_memory_path(root, "u2").exists()
    assert not dream.user_memory_path(root, "u1").exists()


async def test_unsafe_user_id_raises_value_error(tmp_path: Path) -> None:
    root = tmp_path
    with pytest.raises(ValueError):
        await dream.capture_user(
            cwd=root, user_id="../x", get_agent=_get_agent(_FakeAgent()), model=_FakeModel()
        )
    with pytest.raises(ValueError):
        await dream.capture_all(
            cwd=root, get_agent=_get_agent(_FakeAgent()), model=_FakeModel(), user_id="../x"
        )


# --- consolidate ---


async def test_consolidate_writes_draft_but_not_notes(tmp_path: Path) -> None:
    root = tmp_path
    memory.write_inbox_note(
        root, thread_id="t9", title="Session title", summary="- inbox fact", user_id="u1"
    )
    model = _FakeModel("# Notes\n\n- inbox fact")

    result = await dream.consolidate(cwd=root, model=model)

    assert result["status"] == "draft"
    assert result["draft_id"].startswith("notes-")
    assert result["items"] == 1
    assert result["users"] == ["u1"]
    assert not memory.active_memory_path(root).exists()

    draft_text = dream.read_draft(root, result["draft_id"])
    assert draft_text is not None
    assert "# Notes" in draft_text
    metadata, _body = parse_frontmatter(draft_text)
    assert metadata["kind"] == "notes-draft"
    assert metadata["items"] == 1
    assert metadata["users"] == ["u1"]

    # The candidate line rendered into the prompt uses host-style fields.
    human = model.calls[0][1].content
    assert "- [user=u1 session=t9 captured=" in human
    assert "Current notes.md:" in human


async def test_second_consolidate_after_approve_is_nothing_new(tmp_path: Path) -> None:
    root = tmp_path
    memory_path = dream.user_memory_path(root, "u1")
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        "### 2025-01-01T00:00:00+00:00 — Prefs (session t1)\n\n- likes dark mode\n",
        encoding="utf-8",
    )
    _write_draft(root, "notes-old", "2025-03-01T00:00:00+00:00", "old body")

    assert dream.approve_draft(root, "notes-old") is True

    result = await dream.consolidate(cwd=root, model=_FakeModel("# Notes"))
    assert result["status"] == "nothing-new"
    assert result["draft_id"] == ""
    assert dream.list_drafts(root) == []


async def test_consolidate_error_when_model_returns_empty(tmp_path: Path) -> None:
    root = tmp_path
    memory.write_inbox_note(root, thread_id="t1", title="T", summary="- fact")

    result = await dream.consolidate(cwd=root, model=_FakeModel("   "))

    assert result["status"] == "error"
    assert dream.list_drafts(root) == []


# --- drafts ---


def test_render_parse_roundtrip_with_tricky_title() -> None:
    text = dream.render_memory_block(
        captured_at="2025-06-01T10:00:00+00:00",
        title="Debug — quest (session 2)",
        thread_id="abc-123",
        summary="Fixed the thing",
    )
    blocks = dream.parse_memory_blocks(text, user_id="u9")

    assert len(blocks) == 1
    block = blocks[0]
    assert block.captured_at == "2025-06-01T10:00:00+00:00"
    assert block.title == "Debug — quest (session 2)"
    assert block.thread_id == "abc-123"
    assert block.body == "Fixed the thing"
    assert block.user_id == "u9"

    # Default title falls back, and multiple blocks parse independently.
    two = text + dream.render_memory_block(
        captured_at="2025-06-02T10:00:00+00:00",
        title="",
        thread_id="def-456",
        summary="Second",
    )
    blocks = dream.parse_memory_blocks(two)
    assert [b.thread_id for b in blocks] == ["abc-123", "def-456"]
    assert blocks[1].title == memory._DEFAULT_TITLE
    assert blocks[1].user_id == ""


def test_approve_draft_replaces_notes_and_sets_cursor(tmp_path: Path) -> None:
    root = tmp_path
    _write_draft(root, "notes-a", "2025-02-01T00:00:00+00:00", "# Notes\n\n- merged fact")

    assert dream.approve_draft(root, "notes-a") is True

    notes = memory.active_memory_path(root).read_text(encoding="utf-8")
    assert "# Notes" in notes
    assert "- merged fact" in notes
    assert not (dream.drafts_dir(root) / "notes-a.md").exists()
    state = json.loads(
        dream.consolidation_state_path(root).read_text(encoding="utf-8")
    )
    assert state["consolidated_through"] == "2025-02-01T00:00:00+00:00"
    assert dream.approve_draft(root, "notes-a") is False


def test_approve_never_regresses_cursor(tmp_path: Path) -> None:
    root = tmp_path
    _write_draft(root, "notes-older", "2025-01-01T00:00:00+00:00", "older body")
    _write_draft(root, "notes-newer", "2025-02-01T00:00:00+00:00", "newer body")

    assert dream.approve_draft(root, "notes-newer") is True
    state = json.loads(
        dream.consolidation_state_path(root).read_text(encoding="utf-8")
    )
    assert state["consolidated_through"] == "2025-02-01T00:00:00+00:00"

    assert dream.approve_draft(root, "notes-older") is True
    state = json.loads(
        dream.consolidation_state_path(root).read_text(encoding="utf-8")
    )
    assert state["consolidated_through"] == "2025-02-01T00:00:00+00:00"


def test_delete_draft(tmp_path: Path) -> None:
    root = tmp_path
    _write_draft(root, "notes-a", "2025-02-01T00:00:00+00:00", "body")

    assert dream.delete_draft(root, "notes-a") is True
    assert not (dream.drafts_dir(root) / "notes-a.md").exists()
    assert dream.delete_draft(root, "notes-a") is False
    assert dream.read_draft(root, "notes-a") is None
    # Non-filename ids never resolve to a path outside the drafts dir.
    assert dream.delete_draft(root, "../evil") is False


def test_memory_status_counts_blocks_and_drafts(tmp_path: Path) -> None:
    root = tmp_path
    memory_path = dream.user_memory_path(root, "u1")
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(
        "### 2025-01-01T00:00:00+00:00 — Prefs (session t1)\n\n- likes dark mode\n",
        encoding="utf-8",
    )
    _write_draft(root, "notes-a", "2025-02-01T00:00:00+00:00", "body")

    status = dream.memory_status(root)

    assert status["users"] == [
        {
            "user_id": "u1",
            "memory_blocks": 1,
            "last_captured_at": "2025-01-01T00:00:00+00:00",
            "processed_sessions": 0,
        }
    ]
    assert status["notes_md"] == {"exists": False, "bytes": 0}
    assert status["pending_inbox_notes"] == 0
    assert status["pending_drafts"] == ["notes-a"]
    assert status["consolidated_through"] == ""
