"""SSE formatting and disconnect-aware streaming."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

from fastapi import Request

from langchain_harness.events import stream_turn_events


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format a single SSE event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def stream_sse_response(
    agent: Any,
    prompt: str,
    thread_id: str,
    request: Request,
) -> AsyncIterator[str]:
    """Stream agent events as SSE, cancelling on client disconnect.

    Yields SSE-formatted strings. If the client disconnects, the underlying
    streaming task is cancelled.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _produce() -> None:
        try:
            async for event in stream_turn_events(agent, prompt, thread_id):
                etype = event["type"]
                if etype == "token":
                    await queue.put(_sse_event("token", {"text": event["text"]}))
                elif etype == "tool_start":
                    await queue.put(_sse_event("tool_start", {
                        "name": event["name"],
                        "input": event.get("input"),
                    }))
                elif etype == "tool_end":
                    await queue.put(_sse_event("tool_end", {
                        "name": event["name"],
                        "output": event.get("output"),
                    }))
                elif etype == "done":
                    await queue.put(_sse_event("done", {}))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            await queue.put(_sse_event("error", {"detail": str(exc)}))
        finally:
            await queue.put(None)

    task = asyncio.create_task(_produce())

    try:
        while True:
            if await request.is_disconnected():
                task.cancel()
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if item is None:
                break
            yield item
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
