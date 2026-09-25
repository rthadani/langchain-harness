"""FastAPI auth dependency: Bearer token → user_id."""

from __future__ import annotations

import hashlib
from typing import Any

import aiosqlite
from fastapi import Depends, HTTPException, Request, status

from langchain_harness.serverdb import init_server_db, lookup_by_token_hash, server_db_path


async def get_current_user(request: Request) -> dict[str, Any]:
    """Resolve the Bearer token to a user dict.

    Raises 401 on missing/invalid/inactive tokens.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )
    token = auth[len("Bearer "):]
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    db_path = server_db_path(request.app.state.cwd)
    async with aiosqlite.connect(str(db_path)) as conn:
        await init_server_db(conn)
        user = await lookup_by_token_hash(conn, token_hash)

    if user is None or not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked token",
        )
    return user
