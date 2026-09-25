"""Discovery of skills, agents, memory files, and MCP config in a project directory."""

from __future__ import annotations

from pathlib import Path


def discover_skills(root: Path) -> Path | None:
    """Return the skills directory if it exists, otherwise ``None``."""
    skills_dir = root / ".claude" / "skills"
    return skills_dir if skills_dir.is_dir() else None


def discover_agents(root: Path) -> list[Path]:
    """Return sorted paths to agent definition markdown files."""
    agents_dir = root / ".claude" / "agents"
    if not agents_dir.is_dir():
        return []
    return sorted(p for p in agents_dir.glob("*.md") if p.is_file())


def discover_memory(root: Path) -> list[Path]:
    """Return existing AGENTS.md / CLAUDE.md memory files in precedence order."""
    candidates = [root / "CLAUDE.md", root / "AGENTS.md"]
    return [p for p in candidates if p.is_file()]


def discover_mcp_config(root: Path) -> Path:
    """Return the path to the project's ``mcp.json`` file."""
    return root / "mcp.json"


def discover_all(root: Path) -> tuple[Path | None, list[Path], list[Path]]:
    """Discover skills directory, agent files, and memory files.

    Returns ``(skills_dir, agent_files, memory_files)``.
    """
    return discover_skills(root), discover_agents(root), discover_memory(root)
