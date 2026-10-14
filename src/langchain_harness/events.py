"""Shared streaming logic for REPL and SSE handlers.

Provides ``stream_turn_events`` — an async generator that yields normalized
event dicts from a deep agent's ``astream_events`` stream.  Both the REPL
runner and the SSE handler consume these events so they stay in sync.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage

from langchain_harness import prompts as _prompts

COMPACT_PROMPT = _prompts.load("compact")


async def stream_turn_events(
    agent: Any,
    prompt: str,
    thread_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """Stream normalized events from a single agent turn.

    Yields dicts with ``type`` in:
    - ``token`` — ``{"type": "token", "text": "..."}`` for text deltas
    - ``tool_start`` — ``{"type": "tool_start", "name": "...", "input": {...}}``
    - ``tool_end`` — ``{"type": "tool_end", "name": "...", "output": "..."}``
    - ``done`` — ``{"type": "done"}`` after the stream completes
    """
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    messages = {"messages": [HumanMessage(content=prompt)]}

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

    yield {"type": "done"}


def _extract_tool_output_text(output: Any) -> str | None:
    """Extract a string representation of a tool's output, if present."""
    if output is None:
        return None
    content = getattr(output, "content", output)
    if isinstance(content, list):
        return "\n".join(str(block) for block in content)
    return str(content)
