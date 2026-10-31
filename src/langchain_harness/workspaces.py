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

from langchain_harness.published import copy_into_workspace

logger = logging.getLogger(__name__)

# Backend (virtual) skill source paths. Order matters: the skills middleware
# loads sources in order with later sources winning, so the user directory is
# listed FIRST and the global directory LAST (global wins a name collision).
GLOBAL_SKILLS_VPATH = "/.claude/skills"
USER_SKILLS_VPATH = "/.claude/user-skills"


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
    copy_into_workspace(cwd, user_id, workspace)
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

    # Approved session notes only — the pending inbox stays host-side so
    # unapproved summaries never reach an agent.
    src_notes = cwd / ".harness" / "memory" / "notes.md"
    if src_notes.is_file():
        dst_notes = workspace / ".harness" / "memory" / "notes.md"
        dst_notes.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_notes, dst_notes)
        logger.info("Synced memory notes to %s", dst_notes)


def workspace_skill_sources(workspace: Path) -> list[str]:
    """Return backend (virtual) skill source paths for a provisioned workspace.

    The user directory comes first and the global directory last because the
    skills middleware lets later sources win, so a global skill shadows a
    user-published skill with the same name.
    """
    sources: list[str] = []
    if (workspace / ".claude" / "user-skills").is_dir():
        sources.append(USER_SKILLS_VPATH)
    if (workspace / ".claude" / "skills").is_dir():
        sources.append(GLOBAL_SKILLS_VPATH)
    return sources


def workspace_memory_sources(workspace: Path) -> list[str]:
    """Return backend (virtual) memory source paths for a provisioned workspace.

    Sources are workspace-relative virtual paths because the server backend runs
    in virtual mode, where it resolves them under the workspace root. Host
    absolute paths would resolve under the root as well and never be found.
    """
    candidates = (
        ("CLAUDE.md", "/CLAUDE.md"),
        ("AGENTS.md", "/AGENTS.md"),
        (".harness/memory/notes.md", "/.harness/memory/notes.md"),
    )
    return [vpath for rel, vpath in candidates if (workspace / rel).is_file()]


def get_workspace_skills_dir(workspace: Path) -> Path | None:
    """Return the in-workspace skills directory if it exists."""
    skills = workspace / ".claude" / "skills"
    return skills if skills.is_dir() else None


def get_workspace_memory_paths(workspace: Path) -> list[Path]:
    """Return in-workspace memory files that exist."""
    candidates = [
        workspace / "CLAUDE.md",
        workspace / "AGENTS.md",
        workspace / ".harness" / "memory" / "notes.md",
    ]
    return [p for p in candidates if p.is_file()]
