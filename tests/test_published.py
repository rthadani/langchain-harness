"""Tests for the private per-user published skill/agent store."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from langchain_harness.frontmatter import parse_file
from langchain_harness.published import (
    DRAFT_TTL_SECONDS,
    MAX_BODY,
    MAX_DESCRIPTION,
    MAX_DRAFTS,
    PublishError,
    create_draft,
    delete_published,
    draft_dir,
    global_agent_names,
    global_skill_names,
    list_published,
    load_draft,
    make_draft_tools,
    publish,
    published_agent_files,
    published_dir,
    published_root,
    read_published,
    validate_name,
)
from langchain_harness.workspaces import provision_user_workspace, user_workspace


def _workspace(cwd: Path, user_id: str = "alice") -> Path:
    return provision_user_workspace(cwd, user_id)


def _draft(
    cwd: Path,
    *,
    user_id: str = "alice",
    kind: str = "skills",
    name: str = "notes",
    description: str = "notes skill",
    body: str = "# Notes",
) -> dict[str, str]:
    return create_draft(
        user_id,
        _workspace(cwd, user_id),
        kind=kind,
        name=name,
        description=description,
        body=body,
    )


def _published_path(cwd: Path, kind: str, name: str, user_id: str = "alice") -> Path:
    store = published_dir(cwd, user_id, kind)
    return store / name / "SKILL.md" if kind == "skills" else store / f"{name}.md"


# --- layout ---


def test_store_layout(tmp_path: Path) -> None:
    assert published_root(tmp_path, "alice") == (
        tmp_path / ".harness" / "users" / "alice" / "published"
    )
    assert published_dir(tmp_path, "alice", "agents") == (
        tmp_path / ".harness" / "users" / "alice" / "published" / "agents"
    )
    assert draft_dir(tmp_path / "workspace") == tmp_path / "workspace" / ".publish" / "drafts"


# --- validation ---


@pytest.mark.parametrize(
    "bad",
    ["../x", "a/b", ".hidden", "/etc/passwd", "", "UPPER", "a b", "x" * 65, "a" * 64 + "b"],
)
def test_create_draft_rejects_invalid_name(tmp_path: Path, bad: str) -> None:
    with pytest.raises(PublishError) as excinfo:
        _draft(tmp_path, name=bad)
    assert excinfo.value.status == 422


def test_create_draft_rejects_empty_and_oversized_description(tmp_path: Path) -> None:
    with pytest.raises(PublishError) as excinfo:
        _draft(tmp_path, description="   ")
    assert excinfo.value.status == 422

    with pytest.raises(PublishError) as excinfo:
        _draft(tmp_path, description="x" * (MAX_DESCRIPTION + 1))
    assert excinfo.value.status == 422


def test_create_draft_rejects_empty_and_oversized_body(tmp_path: Path) -> None:
    with pytest.raises(PublishError) as excinfo:
        _draft(tmp_path, body=" \n\t ")
    assert excinfo.value.status == 422

    with pytest.raises(PublishError) as excinfo:
        _draft(tmp_path, body="x" * (MAX_BODY + 1))
    assert excinfo.value.status == 422


def test_validate_name_rejects_reserved_agent_name() -> None:
    with pytest.raises(PublishError) as excinfo:
        validate_name("general-purpose", "agents")
    assert excinfo.value.status == 422
    assert validate_name("general-purpose", "skills") == "general-purpose"


# --- publishing ---


@pytest.mark.parametrize("kind", ["skills", "agents"])
def test_publish_writes_artifact_with_server_rendered_frontmatter(
    tmp_path: Path, kind: str
) -> None:
    draft = _draft(tmp_path, kind=kind, name="helper", description="does things", body="# Body")
    item = publish(tmp_path, "alice", _workspace(tmp_path), draft["draft_id"])

    path = _published_path(tmp_path, kind, "helper")
    assert path.is_file()
    assert item["kind"] == kind
    assert item["name"] == "helper"
    assert item["description"] == "does things"

    metadata, body = parse_file(path)
    assert metadata == {"name": "helper", "description": "does things"}
    assert body.strip() == "# Body"


def test_publish_strips_capability_keys(tmp_path: Path) -> None:
    """User text never becomes frontmatter, so capability keys cannot be set."""
    body = (
        "---\n"
        "name: evil\n"
        "model: gpt-4\n"
        "allowed-tools: [shell]\n"
        "tools: [run]\n"
        "middleware: [x]\n"
        "permissions: [y]\n"
        "---\n"
        "# Real body"
    )
    draft = _draft(tmp_path, kind="agents", name="helper", body=body)
    publish(tmp_path, "alice", _workspace(tmp_path), draft["draft_id"])

    metadata, parsed_body = parse_file(_published_path(tmp_path, "agents", "helper"))
    assert metadata == {"name": "helper", "description": "notes skill"}
    assert parsed_body.startswith("---\nname: evil")


@pytest.mark.parametrize("kind", ["skills", "agents"])
def test_publish_rejects_global_name_collision(tmp_path: Path, kind: str) -> None:
    draft = _draft(tmp_path, kind=kind, name="shared")

    with pytest.raises(PublishError) as excinfo:
        publish(
            tmp_path,
            "alice",
            _workspace(tmp_path),
            draft["draft_id"],
            global_skill_names={"shared"} if kind == "skills" else set(),
            global_agent_names={"shared"} if kind == "agents" else set(),
        )

    assert excinfo.value.status == 409
    assert "global" in excinfo.value.detail
    # The draft survives a collision so the user can choose another name.
    assert load_draft(_workspace(tmp_path), draft["draft_id"])["name"] == "shared"


def test_publish_rejects_reserved_agent_name(tmp_path: Path) -> None:
    with pytest.raises(PublishError) as excinfo:
        _draft(tmp_path, kind="agents", name="general-purpose")
    assert excinfo.value.status == 422


def test_publish_replaces_atomically_on_republish(tmp_path: Path) -> None:
    first = _draft(tmp_path, body="# one")
    publish(tmp_path, "alice", _workspace(tmp_path), first["draft_id"])
    path = _published_path(tmp_path, "skills", "notes")
    assert "one" in path.read_text(encoding="utf-8")

    second = _draft(tmp_path, body="# two")
    publish(tmp_path, "alice", _workspace(tmp_path), second["draft_id"])

    text = path.read_text(encoding="utf-8")
    assert "two" in text
    assert "one" not in text
    assert list(published_dir(tmp_path, "alice", "skills").rglob("*.tmp")) == []


def test_publish_rejects_foreign_draft(tmp_path: Path) -> None:
    draft = _draft(tmp_path, user_id="alice")

    with pytest.raises(PublishError) as excinfo:
        publish(tmp_path, "bob", user_workspace(tmp_path, "alice"), draft["draft_id"])

    assert excinfo.value.status == 404


def test_delete_published_missing_is_404(tmp_path: Path) -> None:
    with pytest.raises(PublishError) as excinfo:
        delete_published(tmp_path, "alice", "agents", "ghost")
    assert excinfo.value.status == 404


# --- privacy ---


def test_store_is_scoped_to_the_owner(tmp_path: Path) -> None:
    draft = _draft(tmp_path)
    publish(tmp_path, "alice", _workspace(tmp_path), draft["draft_id"])

    assert [item["name"] for item in list_published(tmp_path, "alice")] == ["notes"]
    assert list_published(tmp_path, "bob") == []

    with pytest.raises(PublishError) as excinfo:
        read_published(tmp_path, "bob", "skills", "notes")
    assert excinfo.value.status == 404

    with pytest.raises(PublishError) as excinfo:
        delete_published(tmp_path, "bob", "skills", "notes")
    assert excinfo.value.status == 404

    # Alice's item survived Bob's attempts.
    assert read_published(tmp_path, "alice", "skills", "notes")[0]["name"] == "notes"


def test_symlink_in_store_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text(
        "---\nname: leak\ndescription: secret\n---\nsecret", encoding="utf-8"
    )

    store = published_dir(tmp_path, "alice", "skills")
    store.mkdir(parents=True)
    (store / "leak").symlink_to(outside, target_is_directory=True)

    assert list_published(tmp_path, "alice") == []

    with pytest.raises(PublishError) as excinfo:
        read_published(tmp_path, "alice", "skills", "leak")
    assert excinfo.value.status == 404

    with pytest.raises(PublishError) as excinfo:
        delete_published(tmp_path, "alice", "skills", "leak")
    assert excinfo.value.status == 404
    assert (outside / "SKILL.md").is_file()

    draft = _draft(tmp_path, name="leak")
    with pytest.raises(PublishError) as excinfo:
        publish(tmp_path, "alice", _workspace(tmp_path), draft["draft_id"])
    assert excinfo.value.status == 404
    assert (outside / "SKILL.md").read_text(encoding="utf-8").endswith("secret")

    # The disposable workspace copy skips symlinked entries too.
    workspace = _workspace(tmp_path)
    assert not (workspace / ".claude" / "user-skills" / "leak").exists()


# --- drafts ---


def test_draft_ttl_expiry(tmp_path: Path) -> None:
    draft = _draft(tmp_path)
    path = draft_dir(_workspace(tmp_path)) / f"{draft['draft_id']}.json"
    expired = datetime.now(timezone.utc) - timedelta(seconds=DRAFT_TTL_SECONDS + 60)

    data = json.loads(path.read_text(encoding="utf-8"))
    data["created_at"] = expired.isoformat()
    path.write_text(json.dumps(data), encoding="utf-8")
    os.utime(path, (expired.timestamp(), expired.timestamp()))

    with pytest.raises(PublishError) as excinfo:
        load_draft(_workspace(tmp_path), draft["draft_id"])
    assert excinfo.value.status == 404
    assert not path.exists()


def test_draft_limit(tmp_path: Path) -> None:
    for index in range(MAX_DRAFTS):
        _draft(tmp_path, name=f"skill-{index}")

    with pytest.raises(PublishError) as excinfo:
        _draft(tmp_path, name="one-too-many")
    assert excinfo.value.status == 422
    assert "Too many pending drafts" in excinfo.value.detail


def test_draft_tool_returns_confirm_path(tmp_path: Path) -> None:
    tools = {tool.name: tool for tool in make_draft_tools("alice", _workspace(tmp_path))}
    result = json.loads(
        tools["draft_skill"].invoke(
            {"name": "notes", "description": "notes skill", "body": "# Notes"}
        )
    )

    assert result["status"] == "draft"
    assert result["name"] == "notes"
    assert result["confirm"] == f"POST /v1/published/{result['draft_id']}"
    assert "confirm" in result["message"]


def test_draft_tools_are_constructed_per_user(tmp_path: Path) -> None:
    alice_workspace = _workspace(tmp_path, "alice")
    bob_workspace = _workspace(tmp_path, "bob")

    alice_tools = make_draft_tools("alice", alice_workspace)
    bob_tools = make_draft_tools("bob", bob_workspace)
    assert alice_tools[0] is not bob_tools[0]

    alice_draft = json.loads(
        alice_tools[0].invoke({"name": "a-skill", "description": "a", "body": "a"})
    )
    bob_draft = json.loads(
        bob_tools[1].invoke({"name": "b-agent", "description": "b", "body": "b"})
    )

    assert (draft_dir(alice_workspace) / f"{alice_draft['draft_id']}.json").is_file()
    assert not (draft_dir(alice_workspace) / f"{bob_draft['draft_id']}.json").exists()
    assert (draft_dir(bob_workspace) / f"{bob_draft['draft_id']}.json").is_file()


# --- global name lookups ---


def test_global_skill_and_agent_names(tmp_path: Path) -> None:
    skill = tmp_path / ".claude" / "skills" / "dir-name" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: real-skill\ndescription: d\n---\nbody", encoding="utf-8")

    agent = tmp_path / ".claude" / "agents" / "file.md"
    agent.parent.mkdir(parents=True)
    agent.write_text("---\nname: real-agent\ndescription: d\n---\nbody", encoding="utf-8")

    assert global_skill_names(tmp_path) == {"real-skill"}
    assert global_agent_names(tmp_path) == {"real-agent"}


def test_publish_survives_a_provisioned_rebuild(tmp_path: Path) -> None:
    """Provisioning again (an agent rebuild) neither loses nor duplicates items."""
    skill = _draft(tmp_path, name="notes")
    publish(tmp_path, "alice", _workspace(tmp_path), skill["draft_id"])
    agent = _draft(tmp_path, kind="agents", name="helper", description="helper agent")
    publish(tmp_path, "alice", _workspace(tmp_path), agent["draft_id"])

    provision_user_workspace(tmp_path, "alice")
    provision_user_workspace(tmp_path, "alice")

    by_name = {item["name"]: item["kind"] for item in list_published(tmp_path, "alice")}
    assert by_name == {"notes": "skills", "helper": "agents"}
    assert published_agent_files(tmp_path, "alice") == [
        published_dir(tmp_path, "alice", "agents") / "helper.md"
    ]
    assert _published_path(tmp_path, "skills", "notes").is_file()
    assert read_published(tmp_path, "alice", "agents", "helper")[1].strip() == "# Notes"
