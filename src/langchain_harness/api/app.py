"""FastAPI app with lifespan-managed shared model, MCP tools, and per-user agent cache."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite
from fastapi import FastAPI

from langchain_harness.agent_builder import _build_backend, _build_model, _load_subagents
from langchain_harness.config import Config
from langchain_harness.discovery import discover_agents
from langchain_harness.mcp_loader import load_mcp_connections
from langchain_harness.api.routes_sessions import router as sessions_router
from langchain_harness.serverdb import init_server_db, server_db_path
from langchain_harness.workspaces import (
    get_workspace_memory_paths,
    get_workspace_skills_dir,
    provision_user_workspace,
    user_sessions_db,
)

logger = logging.getLogger(__name__)


class UserAgentCache:
    """Manages per-user agent instances, savers, and MCP exit stacks."""

    def __init__(self, cwd: Path, config: Config) -> None:
        self.cwd = cwd
        self.config = config
        self._cache: dict[str, tuple[Any, Any]] = {}  # user_id → (agent, saver_ctx)
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self.shared_model: Any = None
        self.shared_mcp_tools: list[Any] = []
        self._saver_contexts: dict[str, Any] = {}  # user_id → AsyncSqliteSaver context

    async def initialize_shared(self) -> None:
        """Build the shared model and load MCP tools once at startup."""
        self.shared_model = _build_model(self.config)

        from langchain_mcp_adapters.client import MultiServerMCPClient

        connections = load_mcp_connections(self.cwd / "mcp.json")
        mcp_tools: list[Any] = []
        for name, conn in connections.items():
            try:
                client = MultiServerMCPClient({name: conn}, tool_name_prefix=True)
                tools = await asyncio.wait_for(client.get_tools(), timeout=30)
                mcp_tools.extend(tools)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load MCP server %r: %s", name, exc)
        self.shared_mcp_tools = mcp_tools

    async def get_agent(self, user_id: str) -> Any:
        """Get or build the agent for a user. Thread-safe via per-user lock."""
        async with self._global_lock:
            if user_id not in self._locks:
                self._locks[user_id] = asyncio.Lock()

        async with self._locks[user_id]:
            if user_id in self._cache:
                return self._cache[user_id][0]

            agent = await self._build_user_agent(user_id)
            self._cache[user_id] = (agent, None)
            return agent

    async def _build_user_agent(self, user_id: str) -> Any:
        """Build a deep agent for a specific user with their own workspace/saver."""
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from deepagents.graph import create_deep_agent
        from deepagents.middleware.summarization import (
            SummarizationMiddleware,
            SummarizationToolMiddleware,
        )

        workspace = provision_user_workspace(self.cwd, user_id)
        db_path = user_sessions_db(self.cwd, user_id)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open the AsyncSqliteSaver as an async context manager
        saver_ctx = AsyncSqliteSaver.from_conn_string(str(db_path))
        saver = await saver_ctx.__aenter__()
        self._saver_contexts[user_id] = saver_ctx

        # Build backend
        enable_shell = self.config.server.enable_shell
        if enable_shell:
            logger.warning(
                "Shell is ENABLED for user %s — per-user privacy is best-effort only.",
                user_id,
            )

        backend = _build_backend(
            self.cwd,
            enable_shell=enable_shell,
            root_dir=workspace,
        )

        # Skills/memory from workspace
        skills_dir = get_workspace_skills_dir(workspace)
        skills = [str(skills_dir)] if skills_dir is not None else None
        memory_paths = get_workspace_memory_paths(workspace)
        memory = [str(p) for p in memory_paths] or None

        # Subagents from host-side discovery (parsed host-side, not through backend)
        agent_paths = discover_agents(self.cwd)
        subagents, subagent_sources = _load_subagents(agent_paths, self.config.provider)

        # Middleware
        middleware = None
        if self.config.compaction.trigger_tokens is not None or self.config.compaction.trigger_messages is not None:
            if self.config.compaction.trigger_tokens is not None:
                trigger = ("tokens", self.config.compaction.trigger_tokens)
            else:
                trigger = ("messages", self.config.compaction.trigger_messages)
            keep = ("messages", self.config.compaction.keep_messages)
            summarization = SummarizationMiddleware(
                model=self.shared_model,
                backend=backend,
                trigger=trigger,
                keep=keep,
            )
            tool_middleware = SummarizationToolMiddleware(
                summarization,
                system_prompt="Use the compact_conversation tool when the user asks to compact the conversation.",
            )
            middleware = [summarization, tool_middleware]

        agent_kwargs: dict[str, Any] = {
            "model": self.shared_model,
            "tools": self.shared_mcp_tools or None,
            "system_prompt": "You are a helpful coding assistant. Work in the user's project directory.",
            "subagents": subagents or None,
            "skills": skills,
            "memory": memory,
            "backend": backend,
            "checkpointer": saver,
        }
        if middleware is not None:
            agent_kwargs["middleware"] = middleware

        try:
            return create_deep_agent(**agent_kwargs)
        except Exception as exc:  # noqa: BLE001
            # Clean up saver on failure
            await saver_ctx.__aexit__(type(exc), exc, exc.__traceback__)
            del self._saver_contexts[user_id]
            raise RuntimeError(
                f"Failed to build agent for user {user_id}; check subagent files: {subagent_sources!r}"
            ) from exc

    async def close_all(self) -> None:
        """Close all savers on shutdown."""
        for user_id, saver_ctx in list(self._saver_contexts.items()):
            try:
                await saver_ctx.__aexit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing saver for user %s: %s", user_id, exc)
        self._saver_contexts.clear()
        self._cache.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: init shared model+MCP, init server.db, close on shutdown."""
    cwd = app.state.cwd
    config = app.state.config

    # Ensure server.db exists
    db_path = server_db_path(cwd)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(db_path)) as conn:
        await init_server_db(conn)

    # Build the per-user agent cache
    cache = UserAgentCache(cwd, config)
    await cache.initialize_shared()
    app.state.agent_cache = cache
    app.state.get_agent = cache.get_agent

    yield

    await cache.close_all()


def create_app(cwd: Path, config: Config) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="langchain-harness", lifespan=lifespan)
    app.state.cwd = cwd
    app.state.config = config
    app.include_router(sessions_router)
    return app
