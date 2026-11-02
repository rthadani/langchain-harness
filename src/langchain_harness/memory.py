"""Durable session memory: summarization, review inbox, and active notes.

A session can be summarized into a note, queued under
``<root>/.harness/memory/inbox/``, approved by a human through
``harness memory --review``, and then appended to the single active memory
file ``<root>/.harness/memory/notes.md`` that future sessions auto-load.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_harness import prompts as _prompts
from langchain_harness.frontmatter import parse_frontmatter
from langchain_harness.tracing import (
    ToolStat,
    TurnMetrics,
    app_version,
    recent_turn_metrics,
)

logger = logging.getLogger(__name__)
eval_logger = logging.getLogger("langchain_harness.eval")

_SUMMARIZE_SYSTEM_PROMPT = _prompts.load("summarize")

# Messages are truncated to this many characters before rendering so a long
# session cannot blow past the summarization model's context window.
_MESSAGE_CHAR_LIMIT = 2000

# The whole rendered conversation is capped as well, so a session with many
# messages cannot produce an unbounded summarization prompt either.
_CONVERSATION_CHAR_LIMIT = 16000
_TRUNCATION_MARKER = "\n...[truncated]"

_DEFAULT_TITLE = "Untitled session"

# The judge only needs enough recent turns to judge tool use; more than this
# competes with the conversation for the model's attention.
_METRICS_TURN_LIMIT = 10
_METRICS_BLOCK_CHAR_LIMIT = 1500

_EVAL_SYSTEM_PROMPT = _prompts.load("eval")


@dataclass(frozen=True)
class EvalResult:
    scope_adherence: int
    scope_reason: str
    task_completion: int
    task_reason: str
    tool_use_quality: int
    tool_use_reason: str
    verdict: str
    run_id: str | None = None


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


def _completed_turns(turns: list[TurnMetrics]) -> list[TurnMetrics]:
    """Keep only finished turns; an unfinished turn has no ``total_ms``."""
    return [turn for turn in turns if turn.total_ms is not None]


def _tool_counts(tools: list[ToolStat]) -> dict[str, int]:
    """Count calls per tool name, preserving first-seen order."""
    counts: dict[str, int] = {}
    for tool in tools:
        counts[tool.name] = counts.get(tool.name, 0) + 1
    return counts


def _tool_error_counts(tools: list[ToolStat]) -> dict[str, int]:
    """Count failed calls per tool name, preserving first-seen order."""
    counts: dict[str, int] = {}
    for tool in tools:
        if tool.error:
            counts[tool.name] = counts.get(tool.name, 0) + 1
    return counts


def _render_tool_summary(tools: list[ToolStat]) -> str:
    """Render ``read_file x3, grep x1 (1 err)`` for a turn's tool calls."""
    errors = _tool_error_counts(tools)
    parts: list[str] = []
    for name, count in _tool_counts(tools).items():
        part = f"{name} x{count}"
        if errors.get(name):
            part += f" ({errors[name]} err)"
        parts.append(part)
    return ", ".join(parts)


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else str(int(round(value)))


def _render_metrics(turns: list[TurnMetrics]) -> str:
    """Render the turn-metrics block for the eval prompt; ``""`` when empty."""
    finished = _completed_turns(turns)
    if not finished:
        return ""
    shown = finished[-_METRICS_TURN_LIMIT:]
    lines = [
        f"Turn metrics (most recent last; showing last {len(shown)} of "
        f"{len(finished)} recorded turns):"
    ]
    for turn in shown:
        tool_errors = sum(1 for tool in turn.tools if tool.error)
        lines.append(
            f"- run={turn.run_id[:8]} outcome={turn.outcome} model_calls={turn.model_calls} "
            f"tools=[{_render_tool_summary(turn.tools)}] tool_errors={tool_errors} "
            f"ttft_ms={_format_ms(turn.ttft_ms)} total_ms={_format_ms(turn.total_ms)} "
            f"overhead_tokens={turn.overhead_tokens}"
        )
    rendered = "\n".join(lines)
    if len(rendered) > _METRICS_BLOCK_CHAR_LIMIT:
        return rendered[:_METRICS_BLOCK_CHAR_LIMIT] + _TRUNCATION_MARKER
    return rendered


def _metrics_summary(turns: list[TurnMetrics]) -> dict[str, Any]:
    """Summarize the turns an eval was judged against, for the joined record."""
    return {
        "turn_count": len(turns),
        "totals": {
            "model_calls": sum(turn.model_calls for turn in turns),
            "tool_calls": sum(len(turn.tools) for turn in turns),
            "tool_errors": sum(1 for turn in turns for tool in turn.tools if tool.error),
            "overhead_tokens": sum(turn.overhead_tokens for turn in turns),
        },
        "turns": [
            {
                "run_id": turn.run_id,
                "outcome": turn.outcome,
                "model_calls": turn.model_calls,
                "tools": _tool_counts(turn.tools),
                "tool_errors": sum(1 for tool in turn.tools if tool.error),
                "ttft_ms": turn.ttft_ms,
                "total_ms": turn.total_ms,
                "overhead_tokens": turn.overhead_tokens,
            }
            for turn in turns
        ],
    }


async def summarize_session(
    model: Any,
    agent: Any,
    thread_id: str,
    *,
    title: str = "",
    system_prompt: str | None = None,
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
                SystemMessage(content=system_prompt or _SUMMARIZE_SYSTEM_PROMPT),
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


async def eval_session(
    model: Any,
    agent: Any,
    thread_id: str,
    *,
    memory_texts: list[str] | None = None,
    system_prompt: str | None = None,
    metrics: list[TurnMetrics] | None = None,
    metrics_path: Path | None = None,
    run_id: str | None = None,
) -> EvalResult | None:
    """Evaluate a session on scope adherence, task completion, and tool use quality.

    The last few turn metrics of *thread_id* are added to the prompt as objective
    evidence for tool use: read from *metrics* when given, otherwise from the
    durable sink at *metrics_path*, otherwise from the in-memory registry.

    Returns None when the session has no substantive content or when the model
    call fails for any reason. Never raises.
    """
    try:
        state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
        values = getattr(state, "values", None)
        messages: list[Any] = []
        if isinstance(values, dict):
            messages = values.get("messages") or []
        if not _has_substantive_content(messages):
            return None

        turns = (
            metrics
            if metrics is not None
            else recent_turn_metrics(
                thread_id, limit=_METRICS_TURN_LIMIT, path=metrics_path
            )
        )
        turns = _completed_turns(turns)
        resolved_run_id = run_id or (turns[-1].run_id if turns else None)

        parts: list[str] = []
        if memory_texts:
            for text in memory_texts:
                if text.strip():
                    parts.append(f"Project instructions:\n{text.strip()}")
        metrics_block = _render_metrics(turns)
        if metrics_block:
            parts.append(metrics_block)
        parts.append(f"Conversation:\n{_render_conversation(messages)}")

        result = await model.ainvoke(
            [
                SystemMessage(content=system_prompt or _EVAL_SYSTEM_PROMPT),
                HumanMessage(content="\n\n".join(parts)),
            ]
        )
        raw = _content_text(getattr(result, "content", result))
        # Strip markdown fences if the model wrapped the JSON anyway.
        lines = raw.strip().splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines)

        data = json.loads(raw)
        eval_result = EvalResult(
            scope_adherence=int(data["scope_adherence"]),
            scope_reason=str(data["scope_reason"]),
            task_completion=int(data["task_completion"]),
            task_reason=str(data["task_reason"]),
            tool_use_quality=int(data["tool_use_quality"]),
            tool_use_reason=str(data["tool_use_reason"]),
            verdict=str(data["verdict"]),
            run_id=resolved_run_id,
        )
        # One joined line per eval: the scores, the turn(s) they were judged
        # against, and the build that produced them. No prompt, no transcript.
        eval_logger.info(
            json.dumps(
                {
                    "event": "eval",
                    "scope": "session",
                    "run_id": resolved_run_id,
                    "run_ids": [turn.run_id for turn in turns],
                    "thread_id": thread_id,
                    "app_version": app_version(),
                    "scores": {
                        "scope_adherence": eval_result.scope_adherence,
                        "task_completion": eval_result.task_completion,
                        "tool_use_quality": eval_result.tool_use_quality,
                    },
                    "verdict": eval_result.verdict,
                    "metrics": _metrics_summary(turns) if turns else None,
                },
                separators=(",", ":"),
            )
        )
        return eval_result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not evaluate session %s: %s", thread_id, exc)
        return None


async def capture_session_note(
    model: Any,
    agent: Any,
    thread_id: str,
    root: Path,
    *,
    title: str = "",
    user_id: str | None = None,
    system_prompt: str | None = None,
) -> Path | None:
    """Summarize a session and queue the result in the review inbox.

    Returns the inbox note path, or ``None`` when there is nothing worth
    summarizing or summarization fails. Never raises.
    """
    try:
        summary = await summarize_session(model, agent, thread_id, title=title, system_prompt=system_prompt)
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
