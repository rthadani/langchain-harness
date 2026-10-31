"""Private per-user skills and agents published from inside a session.

Artifacts are durable and host-side, outside the agent-writable workspace::

    <cwd>/.harness/users/<user_id>/published/skills/<name>/SKILL.md
    <cwd>/.harness/users/<user_id>/published/agents/<name>.md

The agent never writes there. Its ``draft_skill`` / ``draft_agent`` tools drop
a workspace-scoped draft under ``<workspace>/.publish/drafts/``, and only
``POST /v1/published/{draft_id}`` promotes a draft into the store. In-workspace
copies (``<workspace>/.claude/user-skills``) are disposable and refreshed on
every build.

Server-rendered frontmatter carries only ``name`` and ``description``. A draft
never supplies frontmatter, so a published file cannot set ``model``, ``tools``,
``allowed-tools``, ``middleware``, or ``permissions``.

Every store path is a function of the authenticated ``user_id``; there is no
cross-user lookup path. Foreign items therefore read as ``404``, not ``403``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from langchain_harness.discovery import discover_agents, discover_skills
from langchain_harness.frontmatter import parse_file

logger = logging.getLogger(__name__)

KINDS = ("skills", "agents")
MAX_NAME = 64
MAX_DESCRIPTION = 1024
MAX_BODY = 100_000
MAX_DRAFTS = 20
DRAFT_TTL_SECONDS = 24 * 3600
RESERVED_AGENT_NAMES = frozenset({"general-purpose"})

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_DRAFT_ID_RE = re.compile(r"^[0-9a-f]{32}$")

_KIND_LABEL = {"skills": "skill", "agents": "agent"}


class PublishedItem(TypedDict):
    """Summary of one published artifact."""

    kind: str
    name: str
    description: str
    updated_at: str


class PublishError(Exception):
    """Validation, collision, or lookup failure carrying an HTTP status."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class DraftInput(BaseModel):
    """Arguments shared by the ``draft_skill`` and ``draft_agent`` tools."""

    name: str = Field(
        description="Lowercase name of 1-64 letters, digits, and hyphens, starting with a letter or digit."
    )
    description: str = Field(description="One-line description, at most 1024 characters.")
    body: str = Field(description="Markdown instructions, at most 100000 characters.")


# --- paths ---


def published_root(cwd: Path, user_id: str) -> Path:
    """Return the durable store root for one user's published artifacts."""
    return cwd / ".harness" / "users" / user_id / "published"


def published_dir(cwd: Path, user_id: str, kind: str) -> Path:
    """Return the store directory for one kind of artifact."""
    if kind not in KINDS:
        raise PublishError(422, f"Unknown kind {kind!r}; expected one of {', '.join(KINDS)}")
    return published_root(cwd, user_id) / kind


def published_agent_files(cwd: Path, user_id: str) -> list[Path]:
    """Return sorted host paths to a user's published agent files."""
    agents_dir = published_dir(cwd, user_id, "agents")
    if not agents_dir.is_dir():
        return []
    return sorted(
        path
        for path in agents_dir.glob("*.md")
        if path.is_file() and not path.is_symlink()
    )


def draft_dir(workspace: Path) -> Path:
    """Return the draft directory inside a user's disposable workspace."""
    return workspace / ".publish" / "drafts"


# --- validation ---


def validate_name(raw: str, kind: str) -> str:
    """Return a clean artifact name or raise ``PublishError(422)``."""
    name = str(raw or "").strip()
    if not _NAME_RE.match(name):
        raise PublishError(
            422,
            "Name must be 1-64 characters of lowercase letters, digits, and hyphens, "
            "starting with a letter or digit.",
        )
    if kind == "agents" and name in RESERVED_AGENT_NAMES:
        raise PublishError(422, f"{name!r} is a reserved agent name; choose another.")
    return name


def validate_description(raw: str) -> str:
    """Return a clean description or raise ``PublishError(422)``."""
    description = str(raw or "").strip()
    if not description:
        raise PublishError(422, "Description must not be empty.")
    if len(description) > MAX_DESCRIPTION:
        raise PublishError(
            422, f"Description must be at most {MAX_DESCRIPTION} characters."
        )
    return description


def validate_body(raw: str) -> str:
    """Return the artifact body or raise ``PublishError(422)``."""
    body = str(raw or "")
    if not body.strip():
        raise PublishError(422, "Body must not be empty.")
    if len(body) > MAX_BODY:
        raise PublishError(422, f"Body must be at most {MAX_BODY} characters.")
    return body


# --- drafts ---


def create_draft(
    user_id: str,
    workspace: Path,
    *,
    kind: str,
    name: str,
    description: str,
    body: str,
) -> dict[str, str]:
    """Validate and store a pending draft for *user_id*; nothing is active yet."""
    if kind not in KINDS:
        raise PublishError(422, f"Unknown kind {kind!r}; expected one of {', '.join(KINDS)}")

    draft = {
        "draft_id": uuid4().hex,
        "user_id": user_id,
        "kind": kind,
        "name": validate_name(name, kind),
        "description": validate_description(description),
        "body": validate_body(body),
        "created_at": _now(),
    }

    drafts = draft_dir(workspace)
    drafts.mkdir(parents=True, exist_ok=True)
    _collect_expired_drafts(drafts)
    if len(list(drafts.glob("*.json"))) >= MAX_DRAFTS:
        raise PublishError(
            422,
            f"Too many pending drafts (limit {MAX_DRAFTS}); publish or wait for them to expire.",
        )

    _write_text_atomic(drafts / f"{draft['draft_id']}.json", json.dumps(draft))
    return draft


def load_draft(workspace: Path, draft_id: str) -> dict[str, str]:
    """Return a live draft, or raise ``PublishError(404)`` when missing/expired."""
    path = _draft_path(workspace, draft_id)
    data = _read_draft(path)
    if data is None:
        raise PublishError(404, "Draft not found")
    if _draft_expired(path, data):
        path.unlink(missing_ok=True)
        raise PublishError(404, "Draft not found")
    return data


def drop_draft(workspace: Path, draft_id: str) -> None:
    """Remove a draft file if it exists."""
    _draft_path(workspace, draft_id).unlink(missing_ok=True)


def _draft_path(workspace: Path, draft_id: str) -> Path:
    if not _DRAFT_ID_RE.match(str(draft_id or "")):
        raise PublishError(404, "Draft not found")
    return draft_dir(workspace) / f"{draft_id}.json"


def _read_draft(path: Path) -> dict[str, str] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        path.unlink(missing_ok=True)
        return None
    if not isinstance(data, dict):
        path.unlink(missing_ok=True)
        return None
    required = ("draft_id", "user_id", "kind", "name", "description", "body", "created_at")
    if not all(isinstance(data.get(key), str) for key in required):
        path.unlink(missing_ok=True)
        return None
    if data["draft_id"] != path.stem or data["kind"] not in KINDS:
        path.unlink(missing_ok=True)
        return None
    return data  # type: ignore[return-value]


def _draft_expired(path: Path, data: dict[str, str]) -> bool:
    try:
        created = datetime.fromisoformat(data["created_at"])
    except (KeyError, TypeError, ValueError):
        created = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - created).total_seconds()
    return age > DRAFT_TTL_SECONDS


def _collect_expired_drafts(drafts: Path) -> None:
    for path in drafts.glob("*.json"):
        data = _read_draft(path)
        if data is not None and _draft_expired(path, data):
            path.unlink(missing_ok=True)


# --- publishing ---


def publish(
    cwd: Path,
    user_id: str,
    workspace: Path,
    draft_id: str,
    *,
    global_skill_names: frozenset[str] | set[str] = frozenset(),
    global_agent_names: frozenset[str] | set[str] = frozenset(),
) -> PublishedItem:
    """Promote a draft into the user's durable store and return its summary.

    The file is rendered from validated fields only, so user-supplied YAML is
    never interpreted as frontmatter. A name that collides with a global
    artifact is a ``409``: it could never take effect, and silently losing it
    would be worse.
    """
    draft = load_draft(workspace, draft_id)
    if draft["user_id"] != user_id:
        raise PublishError(404, "Draft not found")

    kind = draft["kind"]
    name = validate_name(draft["name"], kind)
    description = validate_description(draft["description"])
    body = validate_body(draft["body"])

    globals_for_kind = global_skill_names if kind == "skills" else global_agent_names
    if name in globals_for_kind:
        raise PublishError(
            409,
            f"{name!r} is a global {_KIND_LABEL[kind]} name; choose another name.",
        )

    target = _item_path(cwd, user_id, kind, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "skills":
        target.mkdir(parents=True, exist_ok=True)
        entry = target
        path = target / "SKILL.md"
    else:
        entry = target
        path = target

    _write_text_atomic(path, _render(name, description, body))
    drop_draft(workspace, draft_id)

    item = _describe(kind, entry)
    if item is None:
        raise PublishError(500, "Published file could not be read back")
    return item


def list_published(cwd: Path, user_id: str, kind: str | None = None) -> list[PublishedItem]:
    """Return the user's published artifacts, optionally filtered by kind."""
    if kind is not None and kind not in KINDS:
        raise PublishError(422, f"Unknown kind {kind!r}; expected one of {', '.join(KINDS)}")

    items: list[PublishedItem] = []
    for current in (kind,) if kind is not None else KINDS:
        store = published_dir(cwd, user_id, current)
        if not store.is_dir():
            continue
        for entry in sorted(store.iterdir()):
            item = _describe(current, entry)
            if item is not None:
                items.append(item)
    return items


def read_published(
    cwd: Path, user_id: str, kind: str, name: str
) -> tuple[PublishedItem, str]:
    """Return one published artifact and its body; ``404`` when absent."""
    clean = validate_name(name, kind)
    target = _item_path(cwd, user_id, kind, clean)
    path = target / "SKILL.md" if kind == "skills" else target
    if kind == "skills" and not target.is_dir():
        raise PublishError(404, "Published item not found")
    if not path.is_file():
        raise PublishError(404, "Published item not found")
    _ensure_inside(path, published_dir(cwd, user_id, kind))

    metadata, body = parse_file(path)
    item = PublishedItem(
        kind=kind,
        name=str(metadata.get("name") or clean),
        description=str(metadata.get("description") or ""),
        updated_at=_iso(path.stat().st_mtime),
    )
    return item, body


def delete_published(cwd: Path, user_id: str, kind: str, name: str) -> None:
    """Delete one published artifact; ``404`` when absent."""
    clean = validate_name(name, kind)
    target = _item_path(cwd, user_id, kind, clean)
    if kind == "skills":
        if not target.is_dir():
            raise PublishError(404, "Published item not found")
        shutil.rmtree(target)
        return
    if not target.is_file():
        raise PublishError(404, "Published item not found")
    target.unlink()


def copy_into_workspace(cwd: Path, user_id: str, workspace: Path) -> None:
    """Replace the workspace's disposable copy of the user's published skills.

    Runs after the global config sync. The directories are disjoint, so the
    global sync's ``rmtree`` of ``.claude/skills`` never touches these copies.
    """
    source = published_dir(cwd, user_id, "skills")
    destination = workspace / ".claude" / "user-skills"

    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        shutil.rmtree(destination)

    if not source.is_dir():
        return

    copied = 0
    for entry in sorted(source.iterdir()):
        if entry.is_symlink() or not entry.is_dir():
            continue
        if not (entry / "SKILL.md").is_file():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(entry, destination / entry.name)
        copied += 1

    if copied:
        logger.info("Copied %d published skill(s) to %s", copied, destination)


# --- global name lookups (for collision checks) ---


def global_skill_names(cwd: Path) -> set[str]:
    """Return the names of global skills, as the skills middleware sees them."""
    skills_dir = discover_skills(cwd)
    if skills_dir is None:
        return set()

    names: set[str] = set()
    for entry in skills_dir.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.is_file():
            continue
        try:
            metadata, _ = parse_file(skill_file)
        except OSError:
            continue
        names.add(str(metadata.get("name") or entry.name))
    return names


def global_agent_names(cwd: Path) -> set[str]:
    """Return the names of global agent definitions."""
    names: set[str] = set()
    for path in discover_agents(cwd):
        try:
            metadata, _ = parse_file(path)
        except OSError:
            continue
        names.add(str(metadata.get("name") or path.stem))
    return names


# --- session-side draft tools ---


def make_draft_tools(user_id: str, workspace: Path) -> list[BaseTool]:
    """Build the draft tools for one user.

    Constructed per agent build, never shared: a tool closes over *user_id* and
    *workspace*, so it must not be appended to a server-wide tool list.
    """

    def _draft(kind: str, name: str, description: str, body: str) -> str:
        try:
            draft = create_draft(
                user_id,
                workspace,
                kind=kind,
                name=name,
                description=description,
                body=body,
            )
        except PublishError as exc:
            return f"Rejected ({exc.status}): {exc.detail}"

        return json.dumps(
            {
                "status": "draft",
                "draft_id": draft["draft_id"],
                "kind": kind,
                "name": draft["name"],
                "confirm": f"POST /v1/published/{draft['draft_id']}",
                "message": (
                    "Nothing is active yet. The user must confirm before this is "
                    "published and usable."
                ),
            }
        )

    def draft_skill(name: str, description: str, body: str) -> str:
        """Save a private skill draft for this user; it is not active yet."""
        return _draft("skills", name, description, body)

    def draft_agent(name: str, description: str, body: str) -> str:
        """Save a private agent draft for this user; it is not active yet."""
        return _draft("agents", name, description, body)

    return [
        StructuredTool.from_function(
            func=draft_skill,
            name="draft_skill",
            description=(
                "Draft a private skill for this user. Writes a draft and returns a "
                "draft id; nothing is active until the user confirms with "
                "POST /v1/published/{draft_id}."
            ),
            args_schema=DraftInput,
        ),
        StructuredTool.from_function(
            func=draft_agent,
            name="draft_agent",
            description=(
                "Draft a private subagent for this user. Writes a draft and returns "
                "a draft id; nothing is active until the user confirms with "
                "POST /v1/published/{draft_id}."
            ),
            args_schema=DraftInput,
        ),
    ]


# --- internals ---


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _render(name: str, description: str, body: str) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {json.dumps(description)}\n"
        "---\n"
        f"{body.strip()}\n"
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _item_path(cwd: Path, user_id: str, kind: str, name: str) -> Path:
    store = published_dir(cwd, user_id, kind)
    target = store / name if kind == "skills" else store / f"{name}.md"
    if target.is_symlink():
        raise PublishError(404, "Published item not found")
    _ensure_inside(target, store)
    return target


def _ensure_inside(target: Path, store: Path) -> None:
    resolved = target.resolve()
    root = store.resolve()
    if resolved != root and root not in resolved.parents:
        raise PublishError(404, "Published item not found")


def _describe(kind: str, entry: Path) -> PublishedItem | None:
    if kind == "skills":
        if entry.is_symlink() or not entry.is_dir():
            return None
        path = entry / "SKILL.md"
    else:
        if entry.is_symlink() or not entry.is_file() or entry.suffix != ".md":
            return None
        path = entry

    try:
        metadata, _ = parse_file(path)
        updated_at = _iso(path.stat().st_mtime)
    except OSError:
        return None

    fallback = entry.name if kind == "skills" else entry.stem
    return PublishedItem(
        kind=kind,
        name=str(metadata.get("name") or fallback),
        description=str(metadata.get("description") or ""),
        updated_at=updated_at,
    )
