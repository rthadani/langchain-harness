"""Pydantic schemas for the API."""

from __future__ import annotations

from datetime import datetime

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
