"""Pydantic schemas for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class SessionCreate(BaseModel):
    title: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    title: str | None = None
    created_at: str


class SessionListItem(BaseModel):
    session_id: str
    title: str | None = None
    created_at: str
    updated_at: str


class MessageCreate(BaseModel):
    content: str


class HistoryMessage(BaseModel):
    role: str
    content: str


class PublishedItem(BaseModel):
    kind: Literal["skills", "agents"]
    name: str
    description: str
    updated_at: str


class PublishedDetail(PublishedItem):
    body: str


class DraftPreview(BaseModel):
    draft_id: str
    kind: Literal["skills", "agents"]
    name: str
    description: str
    body: str
    created_at: str
