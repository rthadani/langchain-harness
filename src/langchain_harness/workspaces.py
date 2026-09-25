"""Per-user workspace provisioning and shared-config sync.

Each user gets ``<server-cwd>/.harness/users/<user_id>/`` containing:
- ``workspace/`` — the agent's filesystem backend root
- ``sessions.db`` — their checkpoints + sessions table
- ``conversation_history/`` — lands inside workspace automatically

On first provision, shared skills/memory are copied INTO the workspace so the
agent's per-user backend can read them.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def users_root(cwd: Path) -> Path:
    """Return the root directory for all user workspaces."""
    return cwd / ".harness" / "users"


def user_dir(cwd: Path, user_id: str) -> Path:
    """Return the directory for a specific user."""
    return users_root(cwd) / user_id


def user_workspace(cwd: Path, user_id: str) -> Path:
    """Return the workspace directory for a user (the agent backend root)."""
    return user_dir(cwd, user_id) / "workspace"


def user_sessions_db(cwd: Path, user_id: str) -> Path:
    """Return the path to a user's sessions.db."""
    return user_dir(cwd, user_id) / "sessions.db"


def provision_user_workspace(cwd: Path, user_id: str) -> Path:
    """Create the directory tree for a user and sync shared config.

    Returns the workspace path (the agent's backend root).
    """
    workspace = user_workspace(cwd, user_id)
    workspace.mkdir(parents=True, exist_ok=True)

    # conversation_history dir lands inside workspace via deepagents
    # — no need to pre-create it.

    _sync_shared_config(cwd, workspace)
    return workspace


def _sync_shared_config(cwd: Path, workspace: Path) -> None:
    """Copy shared skills/memory into the user's workspace.

    - ``<cwd>/.claude/skills`` → ``<workspace>/.claude/skills``
    - ``CLAUDE.md`` / ``AGENTS.md`` → ``<workspace>/``
    """
    # Skills
    src_skills = cwd / ".claude" / "skills"
    dst_skills = workspace / ".claude" / "skills"
    if src_skills.is_dir():
        if dst_skills.exists():
            shutil.rmtree(dst_skills)
        shutil.copytree(src_skills, dst_skills)
        logger.info("Synced skills to %s", dst_skills)
    else:
        # Ensure the .claude dir exists so skills/memory paths are valid
        dst_skills.parent.mkdir(parents=True, exist_ok=True)

    # Memory files (CLAUDE.md, AGENTS.md)
    for name in ("CLAUDE.md", "AGENTS.md"):
        src = cwd / name
        if src.is_file():
            dst = workspace / name
            shutil.copy2(src, dst)
            logger.info("Synced %s to %s", name, dst)


def get_workspace_skills_dir(workspace: Path) -> Path | None:
    """Return the in-workspace skills directory if it exists."""
    skills = workspace / ".claude" / "skills"
    return skills if skills.is_dir() else None


def get_workspace_memory_paths(workspace: Path) -> list[Path]:
    """Return in-workspace memory files that exist."""
    candidates = [workspace / "CLAUDE.md", workspace / "AGENTS.md"]
    return [p for p in candidates if p.is_file()]
