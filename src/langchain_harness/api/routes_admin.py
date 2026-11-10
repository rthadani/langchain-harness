"""Operator-only admin API: dream capture/consolidate and memory drafts.

Every route requires a bearer token matching ``server.admin_token``. When no
token is configured the whole router answers 403, so deployments that never
set ``HARNESS_SERVER_ADMIN_TOKEN`` expose no admin surface at all.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from langchain_harness import dream
from langchain_harness.api import auth

logger = logging.getLogger(__name__)


async def require_admin(request: Request) -> None:
    """Reject any request lacking the configured admin bearer token."""
    token = request.app.state.config.server.admin_token
    if not token:
        raise HTTPException(
            status_code=403,
            detail="Admin API disabled: set HARNESS_SERVER_ADMIN_TOKEN",
        )
    supplied = request.headers.get("Authorization", "")
    if supplied.startswith("Bearer "):
        supplied = supplied[len("Bearer "):].strip()
    else:
        supplied = ""
    if not secrets.compare_digest(supplied, str(token)):
        raise HTTPException(status_code=403, detail="Invalid admin token")


def _job_lock(app: Any, name: str) -> asyncio.Lock:
    """Return a named long-job lock stored on ``app.state``.

    The lock object is created synchronously, before any await, so two
    concurrent requests cannot each create their own lock and both run.
    """
    locks: dict[str, asyncio.Lock] | None = getattr(app.state, "_job_locks", None)
    if locks is None:
        locks = {}
        app.state._job_locks = locks
    lock = locks.get(name)
    if lock is None:
        lock = asyncio.Lock()
        locks[name] = lock
    return lock


router = APIRouter(
    prefix="/v1/admin", tags=["admin"], dependencies=[Depends(require_admin)]
)


@router.post("/dream/capture")
async def dream_capture(
    request: Request,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Capture every user's new sessions into their per-user memory files."""
    if user_id:
        # Raises 400 on unsafe ids before they reach the filesystem layer.
        user_id = auth.sanitize_user_id(user_id)
    cache = request.app.state.agent_cache
    if getattr(cache, "shared_model", None) is None:
        raise HTTPException(status_code=503, detail="no model available")
    lock = _job_lock(request.app, "dream_capture")
    if lock.locked():
        raise HTTPException(status_code=409, detail="dream capture already running")
    async with lock:
        result = await dream.capture_all(
            cwd=request.app.state.cwd,
            get_agent=cache.get_agent,
            model=cache.shared_model,
            user_id=user_id,
        )
    for entry in result.get("users", []):
        if entry.get("captured", 0):
            await cache.invalidate(str(entry["user_id"]))
    return result


@router.post("/dream/consolidate")
async def dream_consolidate(request: Request) -> dict[str, Any]:
    """Fold new memory into a reviewed draft of notes.md (never notes.md itself)."""
    cache = request.app.state.agent_cache
    if getattr(cache, "shared_model", None) is None:
        raise HTTPException(status_code=503, detail="no model available")
    lock = _job_lock(request.app, "dream_consolidate")
    if lock.locked():
        raise HTTPException(
            status_code=409, detail="dream consolidation already running"
        )
    async with lock:
        return await dream.consolidate(
            cwd=request.app.state.cwd,
            model=cache.shared_model,
        )


@router.get("/memory/status")
async def memory_status(request: Request) -> dict[str, Any]:
    """Operator view of memory state across users."""
    return dream.memory_status(request.app.state.cwd)


@router.get("/memory/drafts")
async def list_memory_drafts(request: Request) -> list[dict[str, Any]]:
    """List unapproved notes.md drafts."""
    return [
        dataclasses.asdict(draft) for draft in dream.list_drafts(request.app.state.cwd)
    ]


@router.get("/memory/drafts/{draft_id}")
async def read_memory_draft(request: Request, draft_id: str) -> dict[str, Any]:
    """Return one draft's full markdown text."""
    content = dream.read_draft(request.app.state.cwd, draft_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Draft not found")
    return {"draft_id": draft_id, "content": content}


@router.post("/memory/drafts/{draft_id}/approve")
async def approve_memory_draft(request: Request, draft_id: str) -> dict[str, Any]:
    """Replace notes.md with an approved draft and advance the cursor."""
    # Approving mutates what consolidation reads (notes.md + cursor), so it
    # shares the consolidation lock to avoid racing a running consolidate.
    lock = _job_lock(request.app, "dream_consolidate")
    if lock.locked():
        raise HTTPException(
            status_code=409, detail="dream consolidation already running"
        )
    async with lock:
        if not dream.approve_draft(request.app.state.cwd, draft_id):
            raise HTTPException(status_code=404, detail="Draft not found")
    cache = request.app.state.agent_cache
    invalidated = await cache.invalidate_all() if hasattr(cache, "invalidate_all") else []
    return {"approved": True, "invalidated_users": invalidated}


@router.delete("/memory/drafts/{draft_id}")
async def delete_memory_draft(request: Request, draft_id: str) -> dict[str, bool]:
    """Discard a draft without touching notes.md."""
    if not dream.delete_draft(request.app.state.cwd, draft_id):
        raise HTTPException(status_code=404, detail="Draft not found")
    return {"deleted": True}


@router.post("/agents/invalidate")
async def invalidate_agents(
    request: Request,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Drop cached agents so the next request rebuilds them with fresh memory."""
    cache = request.app.state.agent_cache
    if user_id:
        uid = auth.sanitize_user_id(user_id)
        await cache.invalidate(uid)
        return {"invalidated": [uid]}
    invalidated = await cache.invalidate_all() if hasattr(cache, "invalidate_all") else []
    return {"invalidated": invalidated}
