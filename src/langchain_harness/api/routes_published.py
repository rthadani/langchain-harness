"""HTTP routes for private per-user published skills and agents.

Every handler derives the user id from ``get_current_user`` and passes it into
the store layer. There is no ``user_id`` request parameter, so another user's
items are unreachable by construction and read as ``404``, never ``403``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from langchain_harness.api.auth import get_current_user
from langchain_harness.api.schemas import DraftPreview, PublishedDetail, PublishedItem
from langchain_harness.published import (
    KINDS,
    PublishError,
    delete_published,
    global_agent_names,
    global_skill_names,
    list_published,
    load_draft,
    publish,
    read_published,
)
from langchain_harness.workspaces import user_workspace

router = APIRouter(prefix="/v1/published", tags=["published"])


def _http_error(exc: PublishError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=exc.detail)


def _workspace(cwd: Path, user_id: str) -> Path:
    return user_workspace(cwd, user_id)


@router.get("", response_model=list[PublishedItem])
@router.get("/", response_model=list[PublishedItem], include_in_schema=False)
async def list_items(
    request: Request,
    kind: str | None = None,
    user: dict[str, Any] = Depends(get_current_user),
) -> list[PublishedItem]:
    """List the authenticated user's published skills and agents."""
    if kind is not None and kind not in KINDS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown kind {kind!r}; expected one of {', '.join(KINDS)}",
        )
    try:
        items = list_published(request.app.state.cwd, user["user_id"], kind)
    except PublishError as exc:
        raise _http_error(exc) from exc
    return [PublishedItem(**item) for item in items]


@router.get("/items/{kind}/{name}", response_model=PublishedDetail)
async def read_item(
    kind: str,
    name: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> PublishedDetail:
    """Return one of the authenticated user's published items, including its body."""
    if kind not in KINDS:
        raise HTTPException(status_code=404, detail="Published item not found")
    try:
        item, body = read_published(request.app.state.cwd, user["user_id"], kind, name)
    except PublishError as exc:
        raise _http_error(exc) from exc
    return PublishedDetail(**item, body=body)


@router.get("/drafts/{draft_id}", response_model=DraftPreview)
async def preview_draft(
    draft_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> DraftPreview:
    """Preview a pending draft so the user can confirm or discard it."""
    workspace = _workspace(request.app.state.cwd, user["user_id"])
    try:
        draft = load_draft(workspace, draft_id)
    except PublishError as exc:
        raise _http_error(exc) from exc
    if draft["user_id"] != user["user_id"]:
        raise HTTPException(status_code=404, detail="Draft not found")
    return DraftPreview(
        draft_id=draft["draft_id"],
        kind=draft["kind"],  # type: ignore[arg-type]
        name=draft["name"],
        description=draft["description"],
        body=draft["body"],
        created_at=draft["created_at"],
    )


@router.post("/{draft_id}", response_model=PublishedDetail)
async def publish_draft(
    draft_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> PublishedDetail:
    """Publish a draft. The call itself is the confirmation; there is no body."""
    cwd: Path = request.app.state.cwd
    user_id = user["user_id"]
    workspace = _workspace(cwd, user_id)
    try:
        item = publish(
            cwd,
            user_id,
            workspace,
            draft_id,
            global_skill_names=global_skill_names(cwd),
            global_agent_names=global_agent_names(cwd),
        )
        detail, body = read_published(cwd, user_id, item["kind"], item["name"])
    except PublishError as exc:
        raise _http_error(exc) from exc

    await request.app.state.agent_cache.invalidate(user_id)
    return PublishedDetail(**detail, body=body)


@router.delete("/items/{kind}/{name}")
async def delete_item(
    kind: str,
    name: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, str]:
    """Delete one of the authenticated user's published items."""
    if kind not in KINDS:
        raise HTTPException(status_code=404, detail="Published item not found")
    user_id = user["user_id"]
    try:
        delete_published(request.app.state.cwd, user_id, kind, name)
    except PublishError as exc:
        raise _http_error(exc) from exc

    await request.app.state.agent_cache.invalidate(user_id)
    return {"status": "deleted"}
