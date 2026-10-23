"""FastAPI app with lifespan-managed shared model, MCP tools, and per-user agent cache."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from langchain_harness.agent_builder import _build_backend, build_agent, build_model
from langchain_harness.config import Config
from langchain_harness.mcp_loader import load_mcp_connections
from langchain_harness.api.auth import build_auth
from langchain_harness.api.routes_sessions import router as sessions_router
from langchain_harness.store import Store, make_store
from langchain_harness.workspaces import (
    get_workspace_memory_paths,
    get_workspace_skills_dir,
    provision_user_workspace,
)

logger = logging.getLogger(__name__)


class UserAgentCache:
    """Manages per-user agent instances and checkpointer lifetimes."""

    def __init__(self, cwd: Path, config: Config, store: Store) -> None:
        self.cwd = cwd
        self.config = config
        self.store = store
        self._cache: dict[str, Any] = {}  # user_id → agent
        self._stacks: dict[str, AsyncExitStack] = {}  # user_id → checkpointer scope
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self.shared_model: Any = None
        self.shared_mcp_tools: list[Any] = []

    async def initialize_shared(self) -> None:
        """Build the shared model and load MCP tools once at startup."""
        self.shared_model = build_model(self.config)

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
        # MultiServerMCPClient is stateless: get_tools() creates a temporary
        # session per call (subprocess spawned and torn down inside the context
        # manager), so there are no persistent stdio processes to close on
        # shutdown.

    async def get_agent(self, user_id: str) -> Any:
        """Get or build the agent for a user. Thread-safe via per-user lock."""
        async with self._global_lock:
            if user_id not in self._locks:
                self._locks[user_id] = asyncio.Lock()

        async with self._locks[user_id]:
            if user_id in self._cache:
                return self._cache[user_id]

            agent = await self._build_user_agent(user_id)
            self._cache[user_id] = agent
            return agent

    async def _build_user_agent(self, user_id: str) -> Any:
        """Build a deep agent for a specific user with their own workspace."""
        workspace = provision_user_workspace(self.cwd, user_id)

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

        # Skills/memory from the provisioned workspace (synced copies)
        skills_dir = get_workspace_skills_dir(workspace)
        skills_override = [str(skills_dir)] if skills_dir is not None else []
        memory_paths = get_workspace_memory_paths(workspace)
        memory_override = [str(p) for p in memory_paths]

        stack = AsyncExitStack()
        try:
            checkpointer = await stack.enter_async_context(
                self.store.checkpointer(user_id)
            )
            agent = await build_agent(
                self.config,
                self.cwd,
                checkpointer=checkpointer,
                enable_shell=enable_shell,
                root_dir=workspace,
                skills_override=skills_override,
                memory_override=memory_override,
                model=self.shared_model,
                tools_override=self.shared_mcp_tools,
                backend=backend,
            )
        except Exception:
            await stack.aclose()
            raise

        self._stacks[user_id] = stack
        return agent

    async def close_all(self) -> None:
        """Close every per-user checkpointer scope on shutdown."""
        for user_id, stack in list(self._stacks.items()):
            try:
                await stack.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing checkpointer for user %s: %s", user_id, exc)
        self._stacks.clear()
        self._cache.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: open the store, init shared model+MCP, close on shutdown."""
    cwd = app.state.cwd
    config = app.state.config

    store = make_store(cwd)
    await store.initialize()
    app.state.db = store

    cache = UserAgentCache(cwd, config, store)
    await cache.initialize_shared()
    app.state.agent_cache = cache
    app.state.get_agent = cache.get_agent

    app.state.auth = build_auth(config.server)
    logger.info(
        "Auth modes (in precedence order): %s", ", ".join(app.state.auth.modes)
    )

    try:
        yield
    finally:
        await cache.close_all()
        await store.close()


def create_app(cwd: Path, config: Config) -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="langchain-harness", lifespan=lifespan)
    app.state.cwd = cwd
    app.state.config = config

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        """Liveness: the process is up and serving HTTP."""
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request) -> JSONResponse:
        """Readiness: startup finished and the state store is reachable."""
        store = getattr(request.app.state, "db", None)
        ready = False
        if store is not None:
            try:
                await store.ping()
                ready = True
            except Exception as exc:  # noqa: BLE001
                logger.debug("Readiness check failed: %s", exc)
        return JSONResponse(
            {"status": "ready" if ready else "starting"},
            status_code=200 if ready else 503,
        )

    app.include_router(sessions_router)
    return app
