"""Tests for per-user workspace provisioning and config sync."""

from __future__ import annotations

from pathlib import Path

import pytest

from langchain_harness.workspaces import (
    get_workspace_memory_paths,
    get_workspace_skills_dir,
    provision_user_workspace,
    user_dir,
    user_sessions_db,
    user_workspace,
    users_root,
)


def test_users_root(tmp_path: Path) -> None:
    assert users_root(tmp_path) == tmp_path / ".harness" / "users"


def test_user_dir(tmp_path: Path) -> None:
    assert user_dir(tmp_path, "u123") == tmp_path / ".harness" / "users" / "u123"


def test_user_workspace(tmp_path: Path) -> None:
    assert user_workspace(tmp_path, "u123") == tmp_path / ".harness" / "users" / "u123" / "workspace"


def test_user_sessions_db(tmp_path: Path) -> None:
    assert user_sessions_db(tmp_path, "u123") == tmp_path / ".harness" / "users" / "u123" / "sessions.db"


def test_provision_creates_directory_tree(tmp_path: Path) -> None:
    workspace = provision_user_workspace(tmp_path, "u123")
    assert workspace.is_dir()
    assert user_sessions_db(tmp_path, "u123").parent.is_dir()


def test_provision_syncs_skills(tmp_path: Path) -> None:
    # Create shared skills
    skills_dir = tmp_path / ".claude" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "test-skill").mkdir()
    (skills_dir / "test-skill" / "SKILL.md").write_text("# Test Skill")

    workspace = provision_user_workspace(tmp_path, "u123")

    dst_skills = workspace / ".claude" / "skills" / "test-skill" / "SKILL.md"
    assert dst_skills.is_file()
    assert dst_skills.read_text() == "# Test Skill"


def test_provision_syncs_memory_files(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Agents")
    (tmp_path / "CLAUDE.md").write_text("# Claude")

    workspace = provision_user_workspace(tmp_path, "u123")

    assert (workspace / "AGENTS.md").is_file()
    assert (workspace / "AGENTS.md").read_text() == "# Agents"
    assert (workspace / "CLAUDE.md").is_file()
    assert (workspace / "CLAUDE.md").read_text() == "# Claude"


def test_provision_resyncs_skills(tmp_path: Path) -> None:
    """Re-provisioning should update skills, not duplicate or error."""
    skills_dir = tmp_path / ".claude" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "v1").mkdir()
    (skills_dir / "v1" / "SKILL.md").write_text("v1")

    workspace = provision_user_workspace(tmp_path, "u123")
    assert (workspace / ".claude" / "skills" / "v1" / "SKILL.md").read_text() == "v1"

    # Update and re-sync
    (skills_dir / "v1" / "SKILL.md").write_text("v2")
    (skills_dir / "v2").mkdir()
    (skills_dir / "v2" / "SKILL.md").write_text("new skill")

    provision_user_workspace(tmp_path, "u123")
    assert (workspace / ".claude" / "skills" / "v1" / "SKILL.md").read_text() == "v2"
    assert (workspace / ".claude" / "skills" / "v2" / "SKILL.md").read_text() == "new skill"


def test_get_workspace_skills_dir(tmp_path: Path) -> None:
    workspace = provision_user_workspace(tmp_path, "u123")
    # No skills dir was created (no source skills)
    assert get_workspace_skills_dir(workspace) is None

    # Create skills and re-provision
    skills_dir = tmp_path / ".claude" / "skills"
    skills_dir.mkdir(parents=True)
    (skills_dir / "x").mkdir()
    (skills_dir / "x" / "SKILL.md").write_text("x")

    workspace = provision_user_workspace(tmp_path, "u123")
    skills = get_workspace_skills_dir(workspace)
    assert skills is not None
    assert skills.is_dir()


def test_get_workspace_memory_paths(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("# Agents")
    workspace = provision_user_workspace(tmp_path, "u123")
    memory = get_workspace_memory_paths(workspace)
    assert len(memory) == 1
    assert memory[0].name == "AGENTS.md"


def test_provision_no_skills_or_memory(tmp_path: Path) -> None:
    """Provisioning works even with no shared config to sync."""
    workspace = provision_user_workspace(tmp_path, "u123")
    assert workspace.is_dir()
    assert get_workspace_skills_dir(workspace) is None
    assert get_workspace_memory_paths(workspace) == []
