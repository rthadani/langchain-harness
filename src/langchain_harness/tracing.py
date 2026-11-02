"""Per-turn metrics collector for the harness.

A LangChain callback handler that is attached to the config in
``events.stream_turn_events`` so every entry point (REPL, one-shot, SSE,
/compact) is measured the same way. It is vendor-neutral: it emits one JSON
line per turn, which can be shipped to BigQuery / Loki / a file, and it also
works alongside LangSmith or Langfuse.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("langchain_harness.metrics")

# Mirrors events.py: calls made by the compaction middleware are overhead, not
# part of the answer, and are tallied separately.
_OVERHEAD_NODE_MARKER = "SummarizationMiddleware"


@dataclass
class ToolStat:
    name: str
    ms: float = 0.0
    error: str | None = None


@dataclass
class TurnMetrics:
    run_id: str
    thread_id: str
    user_id: str | None = None
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    overhead_calls: int = 0
    overhead_tokens: int = 0
    model_ms: float = 0.0
    ttft_ms: float | None = None
    total_ms: float | None = None
    tools: list[ToolStat] = field(default_factory=list)
    outcome: str = "ok"  # ok | error | cancelled | client_disconnect
    error: str | None = None

    def to_json(self) -> str:
        d = asdict(self)
        d["tool_calls"] = len(self.tools)
        d["tool_errors"] = sum(1 for t in self.tools if t.error)
        return json.dumps(d, separators=(",", ":"))


class TurnMetricsHandler(BaseCallbackHandler):
    """Collects latency, token, and tool statistics for one agent turn."""

    # Keep callbacks ordered with the event stream rather than in a thread.
    run_inline = True

    def __init__(self, run_id: UUID | str, thread_id: str, user_id: str | None = None):
        self.metrics = TurnMetrics(str(run_id), thread_id, user_id)
        self._t0 = time.perf_counter()
        self._llm_start: dict[UUID, float] = {}
        self._llm_overhead: dict[UUID, bool] = {}
        self._tool_start: dict[UUID, tuple[float, str]] = {}

    # -- model calls -----------------------------------------------------
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kw: Any,
    ) -> None:
        self._llm_start[run_id] = time.perf_counter()
        node = (metadata or {}).get("langgraph_node", "")
        self._llm_overhead[run_id] = _OVERHEAD_NODE_MARKER in node

    def on_llm_new_token(self, token: Any, *, run_id: UUID, **kw: Any) -> None:
        if self.metrics.ttft_ms is None and not self._llm_overhead.get(run_id):
            self.metrics.ttft_ms = (time.perf_counter() - self._t0) * 1000

    def on_llm_end(self, response: Any, *, run_id: UUID, **kw: Any) -> None:
        start = self._llm_start.pop(run_id, None)
        overhead = self._llm_overhead.pop(run_id, False)
        tokens_in = tokens_out = 0
        for gens in response.generations:
            for gen in gens:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if usage:
                    tokens_in += usage.get("input_tokens", 0)
                    tokens_out += usage.get("output_tokens", 0)
        m = self.metrics
        if overhead:
            m.overhead_calls += 1
            m.overhead_tokens += tokens_in + tokens_out
            return
        m.model_calls += 1
        m.input_tokens += tokens_in
        m.output_tokens += tokens_out
        if start is not None:
            m.model_ms += (time.perf_counter() - start) * 1000

    def on_llm_error(self, error: Any, *, run_id: UUID, **kw: Any) -> None:
        self._llm_start.pop(run_id, None)
        self._llm_overhead.pop(run_id, None)
        self.metrics.outcome, self.metrics.error = "error", repr(error)[:300]

    # -- tools -----------------------------------------------------------
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        name: str | None = None,
        **kw: Any,
    ) -> None:
        self._tool_start[run_id] = (
            time.perf_counter(),
            name or (serialized or {}).get("name", "unknown"),
        )

    def _finish_tool(self, run_id: UUID, error: str | None = None) -> None:
        start, name = self._tool_start.pop(run_id, (None, "unknown"))
        ms = (time.perf_counter() - start) * 1000 if start else 0.0
        self.metrics.tools.append(ToolStat(name, ms, error))

    def on_tool_end(self, output: Any, *, run_id: UUID, **kw: Any) -> None:
        # ToolNode converts most tool exceptions into a ToolMessage with
        # status="error"; surface that as an error too.
        status = getattr(output, "status", None)
        self._finish_tool(run_id, "tool_status_error" if status == "error" else None)

    def on_tool_error(self, error: Any, *, run_id: UUID, **kw: Any) -> None:
        self._finish_tool(run_id, repr(error)[:200])

    # -- finish ----------------------------------------------------------
    def finish(self, outcome: str | None = None) -> TurnMetrics:
        m = self.metrics
        if m.total_ms is not None:
            # Idempotent: a stray second call must not double-log the turn.
            return m
        m.total_ms = (time.perf_counter() - self._t0) * 1000
        if outcome and m.outcome == "ok":
            m.outcome = outcome
        logger.info(m.to_json())
        return m
