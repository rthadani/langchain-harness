"""Session CRUD and SSE message/compact routes."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from langchain_harness.api.auth import get_current_user
from langchain_harness.api.schemas import (
    HistoryMessage,
    MessageCreate,
    SessionCreate,
    SessionListItem,
    SessionResponse,
)
from langchain_harness.api.sse import stream_sse_response
from langchain_harness.events import COMPACT_PROMPT
from langchain_harness.memory import capture_session_note
from langchain_harness.sessions import connect, upsert
from langchain_harness import sessions as sessions_module

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


def _user_db_path(request: Request, user_id: str):
    from langchain_harness.workspaces import user_sessions_db
    return user_sessions_db(request.app.state.cwd, user_id)


@router.post("", response_model=SessionResponse)
@router.post("/", response_model=SessionResponse, include_in_schema=False)
async def create_session(
    body: SessionCreate,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> SessionResponse:
    """Create a new session for the authenticated user."""
    user_id = user["user_id"]
    thread_id = uuid4().hex
    db_path = _user_db_path(request, user_id)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with connect(db_path) as conn:
        await upsert(conn, thread_id, title=body.title)
        row = await sessions_module.list_sessions(conn)
        # Find our session
        session = next((r for r in row if r["thread_id"] == thread_id), None)

    return SessionResponse(
        session_id=thread_id,
        title=body.title,
        created_at=session["created_at"] if session else "",
    )


@router.get("", response_model=list[SessionListItem])
@router.get("/", response_model=list[SessionListItem], include_in_schema=False)
async def list_sessions(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> list[SessionListItem]:
    """List the authenticated user's sessions only."""
    user_id = user["user_id"]
    db_path = _user_db_path(request, user_id)
    if not db_path.exists():
        return []
    async with connect(db_path) as conn:
        rows = await sessions_module.list_sessions(conn)
    return [
        SessionListItem(
            session_id=r["thread_id"],
            title=r["title"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


async def _resolve_session(
    request: Request, user: dict[str, Any], session_id: str
) -> dict[str, Any]:
    """Return the session row if it exists in the user's db, else 404."""
    user_id = user["user_id"]
    db_path = _user_db_path(request, user_id)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    async with connect(db_path) as conn:
        session = await sessions_module.get_session(conn, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/{session_id}/history")
async def get_history(
    session_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> list[HistoryMessage]:
    """Return message history for a session. 404 on foreign session."""
    row = await _resolve_session(request, user, session_id)
    thread_id = row["thread_id"]
    agent = await request.app.state.get_agent(user["user_id"])

    # Use the agent's checkpointer to retrieve state
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await agent.aget_state(config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to retrieve state for session %s: %s", thread_id, exc)
        return []

    messages = state.values.get("messages", []) if state and state.values else []
    result: list[HistoryMessage] = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)
        result.append(HistoryMessage(role=role, content=str(content)))
    return result


@router.post("/{session_id}/messages")
async def send_message(
    session_id: str,
    body: MessageCreate,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> StreamingResponse:
    """Send a message and stream the agent's response as SSE."""
    row = await _resolve_session(request, user, session_id)
    thread_id = row["thread_id"]
    agent = await request.app.state.get_agent(user["user_id"])

    db_path = _user_db_path(request, user["user_id"])

    async def _stream():
        try:
            async for chunk in stream_sse_response(agent, body.content, thread_id, request):
                yield chunk
        finally:
            async with connect(db_path) as conn:
                await upsert(conn, thread_id, title=body.content)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{session_id}/compact")
async def compact_session(
    session_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> StreamingResponse:
    """Stream the /compact flow as SSE. 404 on foreign session."""
    row = await _resolve_session(request, user, session_id)
    thread_id = row["thread_id"]
    agent = await request.app.state.get_agent(user["user_id"])
    db_path = _user_db_path(request, user["user_id"])

    compact_prompt = getattr(request.app.state.config.prompts, "compact", None) or COMPACT_PROMPT

    async def _stream():
        try:
            async for chunk in stream_sse_response(agent, compact_prompt, thread_id, request):
                yield chunk
        finally:
            async with connect(db_path) as conn:
                await upsert(conn, thread_id, title="/compact")

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{session_id}/summarize")
async def summarize_session_endpoint(
    session_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, str]:
    """Queue a durable memory note for a session. 404 on foreign session."""
    row = await _resolve_session(request, user, session_id)
    thread_id = row["thread_id"]
    agent = await request.app.state.get_agent(user["user_id"])
    title = row.get("title") or "API session"

    model = getattr(request.app.state.agent_cache, "shared_model", None)
    if model is None:
        return {"status": "skipped", "reason": "no model available"}

    summarize_prompt = getattr(request.app.state.config.prompts, "summarize", None)
    note = await capture_session_note(
        model,
        agent,
        thread_id,
        request.app.state.cwd,
        title=title,
        user_id=user["user_id"],
        system_prompt=summarize_prompt,
    )
    if note is None:
        return {"status": "skipped", "reason": "not captured"}
    return {"status": "queued", "note_id": note.stem}


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, str]:
    """Delete a session. 404 on foreign session."""
    row = await _resolve_session(request, user, session_id)
    thread_id = row["thread_id"]
    agent = await request.app.state.get_agent(user["user_id"])

    # Delete checkpoint data via the saver, then the sessions metadata row
    try:
        await agent.checkpointer.adelete_thread(thread_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to delete checkpoint data for session %s: %s", thread_id, exc)

    db_path = _user_db_path(request, user["user_id"])
    async with connect(db_path) as conn:
        await conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
        await conn.commit()

    return {"status": "deleted"}
