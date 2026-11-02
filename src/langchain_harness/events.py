"""Shared streaming logic for REPL and SSE handlers.

Provides ``stream_turn_events`` — an async generator that yields normalized
event dicts from a deep agent's ``astream_events`` stream.  Both the REPL
runner and the SSE handler consume these events so they stay in sync.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator
from uuid import uuid4

from langchain_core.messages import HumanMessage

from langchain_harness import __version__
from langchain_harness import prompts as _prompts
from langchain_harness.tracing import TurnMetricsHandler

logger = logging.getLogger(__name__)

COMPACT_PROMPT = _prompts.load("compact")


async def stream_turn_events(
    agent: Any,
    prompt: str,
    thread_id: str,
    *,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    cancel_outcome: str = "cancelled",
) -> AsyncIterator[dict[str, Any]]:
    """Stream normalized events from a single agent turn.

    Yields dicts with ``type`` in:
    - ``token`` — ``{"type": "token", "text": "..."}`` for text deltas
    - ``tool_start`` — ``{"type": "tool_start", "name": "...", "input": {...}}``
    - ``tool_end`` — ``{"type": "tool_end", "name": "...", "output": "..."}``
    - ``done`` — ``{"type": "done", "run_id": "..."}`` after the stream completes

    The turn is measured by a :class:`TurnMetricsHandler` callback, which logs one
    JSON line per turn on the ``langchain_harness.metrics`` logger. ``run_id`` is
    the root trace id for the turn; callers forward it so feedback can be joined
    to the turn later.
    """
    run_id = uuid4()
    handler = TurnMetricsHandler(run_id, thread_id, user_id)
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "run_id": run_id,
        "run_name": "harness_turn",
        "callbacks": [handler],
        "tags": ["harness", *(tags or [])],
        "metadata": {
            **(metadata or {}),
            "thread_id": thread_id,
            "user_id": user_id,
            "app_version": __version__,
        },
    }
    messages = {"messages": [HumanMessage(content=prompt)]}

    outcome = "ok"
    try:
        # Note: langchain pops ``run_id`` out of this dict, so never reuse it.
        async for event in agent.astream_events(
            messages,
            config,
            version="v2",
        ):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                node = event.get("metadata", {}).get("langgraph_node", "")
                if "SummarizationMiddleware" in node:
                    continue
                chunk = event.get("data", {}).get("chunk")
                content = getattr(chunk, "content", "")
                if content:
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                yield {"type": "token", "text": block.get("text", "")}
                    else:
                        yield {"type": "token", "text": content}
            elif kind == "on_tool_start":
                data = event.get("data", {})
                input_data = data.get("input") if isinstance(data, dict) else None
                yield {
                    "type": "tool_start",
                    "name": event.get("name", "unknown"),
                    "input": input_data,
                }
            elif kind == "on_tool_end":
                data = event.get("data", {})
                output = data.get("output") if isinstance(data, dict) else None
                text = _extract_tool_output_text(output)
                yield {
                    "type": "tool_end",
                    "name": event.get("name", "unknown"),
                    "output": text,
                }

        yield {"type": "done", "run_id": str(run_id)}
    except asyncio.CancelledError:
        outcome = cancel_outcome
        raise
    except GeneratorExit:
        # Closing this generator early is a disconnect too, but GeneratorExit is
        # neither an Exception nor a CancelledError.
        outcome = cancel_outcome
        raise
    except Exception:
        outcome = "error"
        raise
    finally:
        # finish() is the only call that runs outside the callback manager's
        # exception guard, so a metrics failure must not mask the real error.
        try:
            handler.finish(outcome)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to record turn metrics", exc_info=True)


def _extract_tool_output_text(output: Any) -> str | None:
    """Extract a string representation of a tool's output, if present."""
    if output is None:
        return None
    content = getattr(output, "content", output)
    if isinstance(content, list):
        return "\n".join(str(block) for block in content)
    return str(content)
