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
import os
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from langchain_harness import __version__

logger = logging.getLogger("langchain_harness.metrics")

# Mirrors events.py: calls made by the compaction middleware are overhead, not
# part of the answer, and are tallied separately.
_OVERHEAD_NODE_MARKER = "SummarizationMiddleware"

# The in-memory registry is a fallback for callers that have no metrics file:
# it is bounded on both axes so a long-lived server cannot grow without limit.
_MAX_TURNS_PER_THREAD = 50
_MAX_THREADS = 200
# Only the tail of the JSONL sink is read, so a large file stays cheap to scan.
_JSONL_TAIL_LINES = 200


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


_RECENT: OrderedDict[str, deque[TurnMetrics]] = OrderedDict()


def app_version() -> str:
    """Version stamped on metrics and eval records; ``HARNESS_APP_VERSION`` wins.

    A static package version cannot separate two builds that behave differently,
    so deploys set the env var and both records share the answer.
    """
    return os.environ.get("HARNESS_APP_VERSION") or __version__


def default_metrics_path(root: Path) -> Path:
    """Return the durable per-turn metrics sink for *root*."""
    return root / ".harness" / "metrics" / "turns.jsonl"


def record_metrics(metrics: TurnMetrics, path: Path | None = None) -> None:
    """Remember *metrics* for its thread and append it to *path* when given.

    The file is the source of truth whenever a path is supplied because a later
    ``harness eval`` runs in a fresh process with an empty registry. Metrics
    must never break a turn, so every failure here is only logged.
    """
    try:
        turns = _RECENT.get(metrics.thread_id)
        if turns is None:
            turns = deque(maxlen=_MAX_TURNS_PER_THREAD)
            _RECENT[metrics.thread_id] = turns
        turns.append(metrics)
        _RECENT.move_to_end(metrics.thread_id)
        while len(_RECENT) > _MAX_THREADS:
            _RECENT.popitem(last=False)

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(metrics.to_json() + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not record turn metrics: %s", exc)


def recent_turn_metrics(
    thread_id: str, *, limit: int = 10, path: Path | None = None
) -> list[TurnMetrics]:
    """Return up to *limit* most recent turns for *thread_id*, oldest first."""
    if path is not None and path.is_file():
        return _read_jsonl_tail(thread_id, limit, path)
    return list(_RECENT.get(thread_id, ()))[-limit:]


def clear_metrics() -> None:
    """Drop the in-memory registry; used by tests."""
    _RECENT.clear()


def _read_jsonl_tail(
    thread_id: str, limit: int, path: Path
) -> list[TurnMetrics]:
    """Rebuild the last *limit* turns for *thread_id* from the JSONL sink."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    turns: list[TurnMetrics] = []
    for line in lines[-_JSONL_TAIL_LINES:]:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict) or data.get("thread_id") != thread_id:
            continue
        try:
            # Rebuild field by field: ``to_json`` adds derived keys
            # (``tool_calls``/``tool_errors``) that are not constructor fields.
            turns.append(
                TurnMetrics(
                    run_id=str(data["run_id"]),
                    thread_id=str(data["thread_id"]),
                    user_id=data.get("user_id"),
                    model_calls=int(data.get("model_calls") or 0),
                    input_tokens=int(data.get("input_tokens") or 0),
                    output_tokens=int(data.get("output_tokens") or 0),
                    overhead_calls=int(data.get("overhead_calls") or 0),
                    overhead_tokens=int(data.get("overhead_tokens") or 0),
                    model_ms=float(data.get("model_ms") or 0.0),
                    ttft_ms=data.get("ttft_ms"),
                    total_ms=data.get("total_ms"),
                    tools=[
                        ToolStat(
                            name=str(tool.get("name", "unknown")),
                            ms=float(tool.get("ms") or 0.0),
                            error=tool.get("error"),
                        )
                        for tool in data.get("tools") or []
                        if isinstance(tool, dict)
                    ],
                    outcome=str(data.get("outcome") or "ok"),
                    error=data.get("error"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return turns[-limit:]


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
