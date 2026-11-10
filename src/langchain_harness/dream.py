"""Dream worker: intraday capture, nightly consolidation of agent memory.

Two phases:

- **capture** distills each user's new server sessions into that user's own
  ``user_memory.md`` (under
  ``<cwd>/.harness/users/<uid>/workspace/.harness/memory/``). This layer is
  gate-free: blocks land immediately because only the owning user ever reads
  the file.
- **consolidate** folds new ``user_memory.md`` blocks plus pending review-inbox
  notes into a *draft* of the operator-approved ``notes.md``. Nothing touches
  ``notes.md`` until a human approves the draft, and the consolidation cursor
  advances on approval — never on the draft write, because a draft may still
  be rejected.

Two layers, on purpose: ``user_memory.md`` is per-user and gate-free;
``notes.md`` is operator-approved and rebuilt wholesale from reviewed drafts.
Consolidation crosses user boundaries, so it is host-side, operator-invoked,
and its output is human-reviewed before it ships.

This module targets server workspaces (``.harness/users/``). The local CLI
REPL discovery path is untouched.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_harness import memory, sessions
from langchain_harness.frontmatter import parse_frontmatter
from langchain_harness.workspaces import (
    user_dir,
    user_sessions_db,
    user_workspace,
    users_root,
)

logger = logging.getLogger(__name__)

# Defense in depth for direct/CLI callers: the HTTP layer already sanitizes
# user ids via auth.sanitize_user_id (→ 400). This guard keeps a hostile id
# from reaching the filesystem even when dream is called without that layer.
_SAFE_USER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9@._+-]{0,127}$")

# Greedy ``.*`` backtracks to the last ``(session ...)`` group, so titles
# containing ``(session x)`` still round-trip through parse_memory_blocks.
_HEADER_RE = re.compile(
    r"^### (?P<captured>\S+) — (?P<title>.*) \(session (?P<thread>[^)]+)\)$"
)

DREAM_CAPTURE_SYSTEM_PROMPT = (
    "You distill durable facts about a user from one of their work sessions.\n"
    "Capture decisions, preferences, project state, and constraints in 1-2 "
    "short lines.\n"
    "Omit transient work and any secrets (credentials, tokens, keys).\n"
    "Output nothing when the session contains nothing durable."
)

CONSOLIDATE_SYSTEM_PROMPT = (
    "You maintain an operator-approved memory file. Output ONLY the full "
    "rebuilt markdown — no commentary, no code fences.\n"
    "Preserve every existing fact, merging duplicates into one line.\n"
    "Fold the candidate items in, deduplicating to the most precise wording.\n"
    "Never invent facts that appear in neither the current file nor the "
    "candidates."
)


@dataclass(frozen=True)
class MemoryBlock:
    """One captured memory block parsed out of a ``user_memory.md`` file."""

    captured_at: str
    title: str
    thread_id: str
    body: str
    user_id: str = ""


@dataclass(frozen=True)
class DraftInfo:
    """Metadata about one unapproved ``notes.md`` draft."""

    draft_id: str
    created_at: str
    items: int
    users: list[str]


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def user_memory_path(cwd: Path, user_id: str) -> Path:
    """Return a user's own gate-free memory file inside their workspace."""
    return user_workspace(cwd, user_id) / ".harness" / "memory" / "user_memory.md"


def dream_state_path(cwd: Path, user_id: str) -> Path:
    """Return the per-user capture watermark file."""
    return user_dir(cwd, user_id) / "dream_state.json"


def consolidation_state_path(cwd: Path) -> Path:
    """Return the host-side consolidation cursor file."""
    return memory.memory_root(cwd) / "dream_state.json"


def drafts_dir(cwd: Path) -> Path:
    """Return the directory holding unapproved ``notes.md`` drafts."""
    return memory.memory_root(cwd) / "drafts"


def list_dream_users(cwd: Path) -> list[str]:
    """Return sorted user ids found under the users root (directories only)."""
    root = users_root(cwd)
    if not root.is_dir():
        return []
    return sorted(entry.name for entry in root.iterdir() if entry.is_dir())


# ---------------------------------------------------------------------------
# rendering / parsing
# ---------------------------------------------------------------------------


def render_memory_block(
    *, captured_at: str, title: str, thread_id: str, summary: str
) -> str:
    """Render one memory block; ``parse_memory_blocks`` inverts this exactly."""
    shown_title = title.strip() or memory._DEFAULT_TITLE
    return (
        f"### {captured_at} — {shown_title} (session {thread_id})\n\n"
        f"{summary.strip()}\n"
    )


def parse_memory_blocks(text: str, user_id: str = "") -> list[MemoryBlock]:
    """Parse ``### <ts> — <title> (session <id>)`` blocks out of *text*.

    Unparseable headers are skipped rather than raising.
    """
    blocks: list[MemoryBlock] = []
    current: MemoryBlock | None = None
    body_lines: list[str] = []
    for line in text.splitlines():
        match = _HEADER_RE.match(line)
        if match:
            if current is not None:
                blocks.append(_with_body(current, body_lines))
            current = MemoryBlock(
                captured_at=match.group("captured"),
                title=match.group("title").strip(),
                thread_id=match.group("thread"),
                body="",
                user_id=user_id,
            )
            body_lines = []
        elif current is not None:
            body_lines.append(line)
    if current is not None:
        blocks.append(_with_body(current, body_lines))
    return blocks


def _with_body(block: MemoryBlock, body_lines: list[str]) -> MemoryBlock:
    return MemoryBlock(
        captured_at=block.captured_at,
        title=block.title,
        thread_id=block.thread_id,
        body="\n".join(body_lines).strip(),
        user_id=block.user_id,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: str) -> datetime | None:
    """Parse an ISO timestamp, returning ``None`` when unparseable."""
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _norm_ts(value: str) -> datetime | None:
    """Parse an ISO timestamp; naive values are treated as UTC."""
    parsed = _parse_ts(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# ---------------------------------------------------------------------------
# state files
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON state file; missing or corrupt files read as empty."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read dream state %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON state file; failures are logged, never raised."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write dream state %s: %s", path, exc)


# ---------------------------------------------------------------------------
# capture (intraday, per user)
# ---------------------------------------------------------------------------


async def capture_user(
    *,
    cwd: Path,
    user_id: str,
    get_agent: Callable[[str], Awaitable[Any]],
    model: Any,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Summarize a user's new sessions into their ``user_memory.md``.

    Returns ``{"user_id", "captured", "skipped"}``. A per-thread watermark in
    ``dream_state.json`` makes repeated runs idempotent: a session counts as
    pending only while its ``updated_at`` differs from the watermark.
    """
    if not _SAFE_USER_ID.match(user_id):
        raise ValueError(f"unsafe user_id: {user_id!r}")

    counts: dict[str, Any] = {"user_id": user_id, "captured": 0, "skipped": 0}
    db_path = user_sessions_db(cwd, user_id)
    if not db_path.is_file():
        return counts

    async with sessions.connect(db_path) as conn:
        rows = await sessions.list_sessions(conn, limit=10_000, user_id=user_id)

    state = _read_json(dream_state_path(cwd, user_id))
    processed: dict[str, Any] = state.get("processed") or {}
    if not isinstance(processed, dict):
        processed = {}
    pending = [
        row for row in rows if processed.get(row["thread_id"]) != row["updated_at"]
    ]
    if not pending:
        return counts

    agent = await get_agent(user_id)
    memory_path = user_memory_path(cwd, user_id)
    for row in pending:
        thread_id = str(row["thread_id"])
        summary = await memory.summarize_session(
            model,
            agent,
            thread_id,
            title=str(row["title"] or ""),
            system_prompt=system_prompt or DREAM_CAPTURE_SYSTEM_PROMPT,
        )
        if summary:
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            block = render_memory_block(
                captured_at=_now_iso(),
                title=str(row["title"] or ""),
                thread_id=thread_id,
                summary=summary,
            )
            with memory_path.open("a", encoding="utf-8") as handle:
                handle.write("\n" + block)
            counts["captured"] += 1
        else:
            counts["skipped"] += 1
        # Mark processed either way so an empty session is not retried forever.
        processed[thread_id] = row["updated_at"]

    _write_json(dream_state_path(cwd, user_id), {"processed": processed})
    logger.info(
        "Dream capture for %s: %d captured, %d skipped",
        user_id,
        counts["captured"],
        counts["skipped"],
    )
    return counts


async def capture_all(
    *,
    cwd: Path,
    get_agent: Callable[[str], Awaitable[Any]],
    model: Any,
    user_id: str | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Run capture for one user or for every user under the users root.

    A failing user is recorded as ``{"user_id", "error", "captured": 0}`` and
    never aborts the sweep.
    """
    if user_id is not None and not _SAFE_USER_ID.match(user_id):
        raise ValueError(f"unsafe user_id: {user_id!r}")

    targets = [user_id] if user_id is not None else list_dream_users(cwd)
    users: list[dict[str, Any]] = []
    for uid in targets:
        try:
            users.append(
                await capture_user(
                    cwd=cwd,
                    user_id=uid,
                    get_agent=get_agent,
                    model=model,
                    system_prompt=system_prompt,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dream capture failed for user %s: %s", uid, exc)
            users.append(
                {"user_id": uid, "error": str(exc), "captured": 0, "skipped": 0}
            )
    return {
        "captured": sum(int(entry.get("captured", 0)) for entry in users),
        "skipped": sum(int(entry.get("skipped", 0)) for entry in users),
        "users": users,
    }


# ---------------------------------------------------------------------------
# consolidate (nightly, operator-reviewed)
# ---------------------------------------------------------------------------


async def consolidate(
    *,
    cwd: Path,
    model: Any,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Fold new memory into a *draft* of ``notes.md``; never writes notes.md.

    Returns ``{"status", "draft_id", "items", "users"}``. Candidates are the
    ``user_memory.md`` blocks captured after the consolidation cursor plus the
    pending review-inbox notes.
    """
    state = _read_json(consolidation_state_path(cwd))
    cursor = str(state.get("consolidated_through") or "")
    cursor_ts = _norm_ts(cursor) if cursor else None

    candidates: list[MemoryBlock] = []
    involved: set[str] = set()
    for uid in list_dream_users(cwd):
        path = user_memory_path(cwd, uid)
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read user memory %s: %s", path, exc)
            continue
        for block in parse_memory_blocks(text, user_id=uid):
            captured = _norm_ts(block.captured_at)
            # Unparseable timestamps never become candidates.
            if captured is None:
                continue
            if cursor_ts is None or captured > cursor_ts:
                candidates.append(block)
                involved.add(uid)

    for note in memory.list_inbox(cwd):
        candidates.append(
            MemoryBlock(
                captured_at=note.created_at,
                title=note.title,
                thread_id=note.thread_id,
                body=note.summary,
                user_id=note.user_id or "",
            )
        )
        involved.add(note.user_id or "host")

    if not candidates:
        return {"status": "nothing-new", "draft_id": "", "items": 0, "users": []}

    notes_path = memory.active_memory_path(cwd)
    try:
        current = notes_path.read_text(encoding="utf-8") if notes_path.is_file() else ""
    except OSError as exc:
        logger.warning("Could not read active notes %s: %s", notes_path, exc)
        current = ""

    lines = [
        f"- [user={block.user_id or 'host'} session={block.thread_id} "
        f"captured={block.captured_at}] "
        f"{block.title or memory._DEFAULT_TITLE}: {block.body}"
        for block in candidates
    ]
    human = (
        f"Current notes.md:\n\n{current.strip() or '(empty)'}\n\n"
        "New candidate items:\n\n" + "\n".join(lines)
    )
    result = await model.ainvoke(
        [
            SystemMessage(content=system_prompt or CONSOLIDATE_SYSTEM_PROMPT),
            HumanMessage(content=human),
        ]
    )
    merged = memory._content_text(getattr(result, "content", result))
    if not merged:
        return {"status": "error", "draft_id": "", "items": 0, "users": []}

    users = sorted(involved)
    directory = drafts_dir(cwd)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    draft_path = directory / f"notes-{stamp}.md"
    suffix = 2
    while draft_path.exists():
        draft_path = directory / f"notes-{stamp}-{suffix}.md"
        suffix += 1

    frontmatter = yaml.safe_dump(
        {
            "kind": "notes-draft",
            "created_at": _now_iso(),
            "items": len(candidates),
            "users": users,
        },
        sort_keys=False,
        default_style="'",
        allow_unicode=True,
    ).strip()
    draft_path.write_text(f"---\n{frontmatter}\n---\n\n{merged}\n", encoding="utf-8")
    return {"status": "draft", "draft_id": draft_path.stem, "items": len(candidates), "users": users}


# ---------------------------------------------------------------------------
# drafts
# ---------------------------------------------------------------------------


def _draft_path(cwd: Path, draft_id: str) -> Path | None:
    """Resolve a draft id to its file, or ``None`` for a non-filename id."""
    if (
        not draft_id
        or "/" in draft_id
        or "\\" in draft_id
        or draft_id.startswith(".")
    ):
        return None
    return drafts_dir(cwd) / f"{draft_id}.md"


def list_drafts(cwd: Path) -> list[DraftInfo]:
    """Return metadata for every draft file, oldest filename first."""
    directory = drafts_dir(cwd)
    if not directory.is_dir():
        return []
    drafts: list[DraftInfo] = []
    for path in sorted(directory.glob("*.md")):
        try:
            metadata, _body = parse_frontmatter(path.read_text(encoding="utf-8"))
            raw_users = metadata.get("users") or []
            drafts.append(
                DraftInfo(
                    draft_id=path.stem,
                    created_at=str(metadata.get("created_at") or ""),
                    items=int(metadata.get("items") or 0),
                    users=[str(u) for u in raw_users] if isinstance(raw_users, list) else [],
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read draft %s: %s", path, exc)
    return drafts


def read_draft(cwd: Path, draft_id: str) -> str | None:
    """Return a draft's full text, or ``None`` when it does not exist."""
    path = _draft_path(cwd, draft_id)
    if path is None or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read draft %s: %s", path, exc)
        return None


def delete_draft(cwd: Path, draft_id: str) -> bool:
    """Discard a draft without touching ``notes.md`` or the cursor."""
    path = _draft_path(cwd, draft_id)
    if path is None or not path.is_file():
        return False
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("Could not delete draft %s: %s", path, exc)
        return False
    return True


def _advance_cursor(existing: str, draft_created_at: str) -> str | None:
    """Return the later of two ISO cursors; ``None`` when neither parses."""
    existing_ts = _norm_ts(existing) if existing else None
    draft_ts = _norm_ts(draft_created_at) if draft_created_at else None
    if existing_ts is not None and draft_ts is not None:
        winner = max(existing_ts, draft_ts)
    else:
        winner = existing_ts or draft_ts
    return winner.isoformat() if winner is not None else None


def approve_draft(cwd: Path, draft_id: str) -> bool:
    """Replace ``notes.md`` with a draft's body and advance the cursor.

    The notes file is written to a temp path and ``os.replace``d into place so
    a crash mid-write cannot truncate the operator-approved memory. The
    consolidation cursor moves to the later of the existing cursor and the
    draft's ``created_at``; approving an older draft never regresses it.
    """
    path = _draft_path(cwd, draft_id)
    if path is None or not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read draft %s: %s", path, exc)
        return False
    metadata, body = parse_frontmatter(text)

    notes_path = memory.active_memory_path(cwd)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = notes_path.with_suffix(".md.tmp")
    tmp_path.write_text(body, encoding="utf-8")
    os.replace(tmp_path, notes_path)

    existing = _read_json(consolidation_state_path(cwd))
    new_cursor = _advance_cursor(
        str(existing.get("consolidated_through") or ""),
        str(metadata.get("created_at") or ""),
    )
    if new_cursor is not None:
        _write_json(
            consolidation_state_path(cwd), {"consolidated_through": new_cursor}
        )

    try:
        path.unlink()
    except OSError as exc:
        logger.warning("Could not delete draft %s: %s", path, exc)
    return True


# ---------------------------------------------------------------------------
# operator status
# ---------------------------------------------------------------------------


def memory_status(cwd: Path) -> dict[str, Any]:
    """Summarize dream state for operators: per-user blocks, notes, drafts."""
    users: list[dict[str, Any]] = []
    for uid in list_dream_users(cwd):
        blocks: list[MemoryBlock] = []
        path = user_memory_path(cwd, uid)
        if path.is_file():
            try:
                blocks = parse_memory_blocks(
                    path.read_text(encoding="utf-8"), user_id=uid
                )
            except OSError as exc:
                logger.warning("Could not read user memory %s: %s", path, exc)
        captured = [
            ts for ts in (_norm_ts(b.captured_at) for b in blocks) if ts is not None
        ]
        last_captured = max(captured).isoformat() if captured else ""
        state = _read_json(dream_state_path(cwd, uid))
        processed = state.get("processed") or {}
        users.append(
            {
                "user_id": uid,
                "memory_blocks": len(blocks),
                "last_captured_at": last_captured,
                "processed_sessions": len(processed) if isinstance(processed, dict) else 0,
            }
        )

    notes_path = memory.active_memory_path(cwd)
    host_state = _read_json(consolidation_state_path(cwd))
    return {
        "users": users,
        "notes_md": {
            "exists": notes_path.is_file(),
            "bytes": notes_path.stat().st_size if notes_path.is_file() else 0,
        },
        "pending_inbox_notes": len(memory.list_inbox(cwd)),
        "pending_drafts": [d.draft_id for d in list_drafts(cwd)],
        "consolidated_through": str(host_state.get("consolidated_through") or ""),
    }
