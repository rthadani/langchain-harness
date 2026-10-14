"""Durable session memory: summarization, review inbox, and active notes.

A session can be summarized into a note, queued under
``<root>/.harness/memory/inbox/``, approved by a human through
``harness memory --review``, and then appended to the single active memory
file ``<root>/.harness/memory/notes.md`` that future sessions auto-load.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_harness.frontmatter import parse_frontmatter

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM_PROMPT = (
    "You are maintaining the long-term memory of a coding assistant. "
    "Summarize the conversation below into a short, durable note for future "
    "sessions. Focus on: the user's goal, key decisions and constraints, "
    "important gotchas, and any unresolved follow-ups. Write 3-8 concise "
    "bullets in third person, past tense. Do not invent facts not present in "
    "the conversation."
)

# Messages are truncated to this many characters before rendering so a long
# session cannot blow past the summarization model's context window.
_MESSAGE_CHAR_LIMIT = 2000

# The whole rendered conversation is capped as well, so a session with many
# messages cannot produce an unbounded summarization prompt either.
_CONVERSATION_CHAR_LIMIT = 16000
_TRUNCATION_MARKER = "\n...[truncated]"

_DEFAULT_TITLE = "Untitled session"


@dataclass(frozen=True)
class MemoryNote:
    """A queued or approved session summary."""

    note_id: str
    title: str
    thread_id: str
    created_at: str
    summary: str
    user_id: str | None = None


def memory_root(root: Path) -> Path:
    """Return the memory store directory for *root*."""
    return root / ".harness" / "memory"


def inbox_dir(root: Path) -> Path:
    """Return the directory holding pending (unapproved) notes."""
    return memory_root(root) / "inbox"


def active_memory_path(root: Path) -> Path:
    """Return the file of approved notes that future sessions auto-load."""
    return memory_root(root) / "notes.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _note_id(thread_id: str, user_id: str | None) -> str:
    return f"{user_id}--{thread_id}" if user_id else thread_id


def _render_note(
    *,
    title: str,
    thread_id: str,
    created_at: str,
    summary: str,
    user_id: str | None,
) -> str:
    """Render an inbox note with YAML frontmatter plus a markdown body."""
    metadata: dict[str, Any] = {
        "title": title,
        "thread_id": thread_id,
        "created_at": created_at,
    }
    if user_id is not None:
        metadata["user_id"] = user_id
    # Quoted scalars keep ISO timestamps and titles containing ':' or quotes
    # round-tripping as strings through ``safe_load``.
    frontmatter = yaml.safe_dump(
        metadata, sort_keys=False, default_style="'", allow_unicode=True
    ).strip()
    return f"---\n{frontmatter}\n---\n\n{summary.strip()}\n"


def _note_from_path(path: Path) -> MemoryNote:
    metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    user_id = metadata.get("user_id")
    return MemoryNote(
        note_id=path.stem,
        title=str(metadata.get("title") or _DEFAULT_TITLE),
        thread_id=str(metadata.get("thread_id") or ""),
        created_at=str(metadata.get("created_at") or ""),
        summary=body.strip(),
        user_id=None if user_id is None else str(user_id),
    )


def write_inbox_note(
    root: Path,
    *,
    thread_id: str,
    title: str,
    summary: str,
    user_id: str | None = None,
) -> Path:
    """Write a pending note to the review inbox and return its path."""
    directory = inbox_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_note_id(thread_id, user_id)}.md"
    path.write_text(
        _render_note(
            title=title.strip() or _DEFAULT_TITLE,
            thread_id=thread_id,
            created_at=_now_iso(),
            summary=summary,
            user_id=user_id,
        ),
        encoding="utf-8",
    )
    return path


def list_inbox(root: Path) -> list[MemoryNote]:
    """Return all pending notes, oldest filename first."""
    directory = inbox_dir(root)
    if not directory.is_dir():
        return []
    notes: list[MemoryNote] = []
    for path in sorted(directory.glob("*.md")):
        try:
            notes.append(_note_from_path(path))
        except OSError as exc:
            logger.warning("Could not read memory note %s: %s", path, exc)
    return notes


def approve_note(root: Path, note_id: str) -> bool:
    """Append a pending note to the active memory file and remove it.

    Returns ``False`` when no pending note with *note_id* exists.
    """
    path = inbox_dir(root) / f"{note_id}.md"
    if not path.is_file():
        return False
    note = _note_from_path(path)

    notes_path = active_memory_path(root)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    separator = "\n" if notes_path.exists() and notes_path.stat().st_size > 0 else ""
    block = (
        f"## {note.title}\n\n"
        f"_session `{note.thread_id}` · approved `{_now_iso()}`\n\n"
        f"{note.summary.strip()}\n"
    )
    with notes_path.open("a", encoding="utf-8") as handle:
        handle.write(separator + block)

    path.unlink()
    return True


def reject_note(root: Path, note_id: str) -> bool:
    """Discard a pending note without touching the active memory file."""
    path = inbox_dir(root) / f"{note_id}.md"
    if not path.is_file():
        return False
    path.unlink()
    return True


def _content_text(content: Any) -> str:
    """Extract stripped plain text from a string or a list of content blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def _has_substantive_content(messages: list[Any]) -> bool:
    """True when at least one AI message carries non-empty text."""
    return any(
        getattr(message, "type", "") == "ai"
        and _content_text(getattr(message, "content", "")) != ""
        for message in messages
    )


def _render_conversation(messages: list[Any]) -> str:
    lines: list[str] = []
    for message in messages:
        role = getattr(message, "type", "unknown")
        text = _content_text(getattr(message, "content", ""))[:_MESSAGE_CHAR_LIMIT]
        lines.append(f"{role}: {text}")
    rendered = "\n".join(lines)
    if len(rendered) > _CONVERSATION_CHAR_LIMIT:
        return rendered[:_CONVERSATION_CHAR_LIMIT] + _TRUNCATION_MARKER
    return rendered


async def summarize_session(
    model: Any,
    agent: Any,
    thread_id: str,
    *,
    title: str = "",
) -> str | None:
    """Summarize a session's messages into a durable note.

    Returns ``None`` when the session has no substantive content or when
    summarization fails for any reason — callers never see an exception.
    """
    try:
        state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
        values = getattr(state, "values", None)
        messages: list[Any] = []
        if isinstance(values, dict):
            messages = values.get("messages") or []
        if not _has_substantive_content(messages):
            return None

        result = await model.ainvoke(
            [
                SystemMessage(content=_SUMMARIZE_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Session title: {title}\n\n"
                        f"Conversation:\n{_render_conversation(messages)}"
                    )
                ),
            ]
        )
        text = _content_text(getattr(result, "content", result))
        return text or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not summarize session %s: %s", thread_id, exc)
        return None


async def capture_session_note(
    model: Any,
    agent: Any,
    thread_id: str,
    root: Path,
    *,
    title: str = "",
    user_id: str | None = None,
) -> Path | None:
    """Summarize a session and queue the result in the review inbox.

    Returns the inbox note path, or ``None`` when there is nothing worth
    summarizing or summarization fails. Never raises.
    """
    try:
        summary = await summarize_session(model, agent, thread_id, title=title)
        if not summary:
            return None
        return write_inbox_note(
            root,
            thread_id=thread_id,
            title=title,
            summary=summary,
            user_id=user_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not capture memory note for session %s: %s", thread_id, exc)
        return None
