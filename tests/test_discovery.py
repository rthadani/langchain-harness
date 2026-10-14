from pathlib import Path

from langchain_harness.discovery import discover_agents, discover_all, discover_memory, discover_skills


def test_discover_skills(tmp_path: Path) -> None:
    skills = tmp_path / ".claude" / "skills"
    skills.mkdir(parents=True)
    assert discover_skills(tmp_path) == skills


def test_discover_skills_missing(tmp_path: Path) -> None:
    assert discover_skills(tmp_path) is None


def test_discover_agents(tmp_path: Path) -> None:
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "b.md").write_text("b", encoding="utf-8")
    (agents / "a.md").write_text("a", encoding="utf-8")
    assert discover_agents(tmp_path) == [agents / "a.md", agents / "b.md"]


def test_discover_agents_missing_dir(tmp_path: Path) -> None:
    assert discover_agents(tmp_path) == []


def test_discover_memory(tmp_path: Path) -> None:
    claude = tmp_path / "CLAUDE.md"
    agents = tmp_path / "AGENTS.md"
    claude.write_text("c", encoding="utf-8")
    agents.write_text("a", encoding="utf-8")
    assert discover_memory(tmp_path) == [claude, agents]


def test_discover_memory_missing(tmp_path: Path) -> None:
    assert discover_memory(tmp_path) == []


def test_discover_memory_includes_harness_notes(tmp_path: Path) -> None:
    notes = tmp_path / ".harness" / "memory" / "notes.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("## Note", encoding="utf-8")
    assert discover_memory(tmp_path) == [notes]


def test_discover_memory_orders_project_files_before_notes(tmp_path: Path) -> None:
    claude = tmp_path / "CLAUDE.md"
    agents = tmp_path / "AGENTS.md"
    notes = tmp_path / ".harness" / "memory" / "notes.md"
    claude.write_text("c", encoding="utf-8")
    agents.write_text("a", encoding="utf-8")
    notes.parent.mkdir(parents=True)
    notes.write_text("n", encoding="utf-8")
    assert discover_memory(tmp_path) == [claude, agents, notes]


def test_discover_all(tmp_path: Path) -> None:
    skills = tmp_path / ".claude" / "skills"
    skills.mkdir(parents=True)
    agents = tmp_path / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "x.md").write_text("x", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("m", encoding="utf-8")
    notes = tmp_path / ".harness" / "memory" / "notes.md"
    notes.parent.mkdir(parents=True)
    notes.write_text("n", encoding="utf-8")

    s, a, m = discover_all(tmp_path)
    assert s == skills
    assert a == [agents / "x.md"]
    assert m == [tmp_path / "AGENTS.md", notes]
