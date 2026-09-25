"""Assemble a deep agent from config, discovered project files, and MCP servers."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from deepagents.middleware.summarization import (
    SummarizationMiddleware,
    SummarizationToolMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_harness.config import CompactionConfig, Config
from langchain_harness.discovery import discover_all, discover_agents, discover_mcp_config
from langchain_harness.frontmatter import parse_file
from langchain_harness.mcp_loader import load_mcp_connections

logger = logging.getLogger(__name__)

# Seconds to wait for a single MCP server to return its tool list.
_MCP_TOOLS_TIMEOUT = 30

# Nudge appended to the tool middleware so the model knows when to call it.
_COMPACT_TOOL_NUDGE = (
    "Use the compact_conversation tool when the user asks to compact the conversation."
)

# Known Claude-style model aliases for subagent frontmatter.  Values are passed
# verbatim to ``init_chat_model`` as ``provider:model`` strings.
_MODEL_ALIASES = {
    "sonnet": "anthropic:claude-sonnet-4-5",
    "opus": "anthropic:claude-opus-4-5",
    "haiku": "anthropic:claude-haiku-4-5",
}


def _build_model(config: Config) -> BaseChatModel:
    """Initialize the chat model from the resolved configuration."""
    kwargs: dict[str, Any] = {"model_provider": config.provider}
    if config.base_url is not None:
        kwargs["base_url"] = config.base_url
    if config.api_key is not None:
        kwargs["api_key"] = config.api_key
    return init_chat_model(config.model, **kwargs)


def _resolve_subagent_model(value: str, provider: str, path: Path) -> str | None:
    """Translate a model value from frontmatter into a provider-prefixed spec.

    Returns ``None`` when the key should be dropped (e.g. ``inherit``).
    Bare model IDs are prefixed with the configured provider. Values that are
    still unusable (empty or missing a provider/model part) log a warning and
    return ``None`` so the agent is created without a model override.
    """
    stripped = value.strip()
    normalized = stripped.lower()
    if normalized == "inherit":
        return None
    if normalized in _MODEL_ALIASES:
        return _MODEL_ALIASES[normalized]
    if ":" in stripped:
        provider_part, model_part = stripped.split(":", 1)
        if not provider_part.strip() or not model_part.strip():
            logger.warning(
                "Unrecognised model %r in %s; keeping agent without model override",
                value,
                path,
            )
            return None
        return stripped
    if not stripped:
        logger.warning(
            "Unrecognised model %r in %s; keeping agent without model override",
            value,
            path,
        )
        return None
    prefixed = f"{provider}:{stripped}"
    logger.info("Prefixed bare model %r with provider: %s", value, prefixed)
    return prefixed


def _load_subagents(
    agent_paths: list[Path], provider: str
) -> tuple[list[SubAgent], dict[str, Path]]:
    """Parse agent markdown files into ``SubAgent`` definitions.

    Returns the subagents plus a mapping from subagent name to source file path
    so that downstream failures can report the offending file.
    """
    subagents: list[SubAgent] = []
    source_map: dict[str, Path] = {}
    for path in agent_paths:
        try:
            metadata, body = parse_file(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping agent %s: could not parse frontmatter (%s)", path, exc
            )
            continue

        name = metadata.get("name")
        description = metadata.get("description")
        if not name or not description:
            logger.warning(
                "Skipping agent %s: frontmatter must include 'name' and 'description'", path
            )
            continue

        subagent: SubAgent = {
            "name": str(name),
            "description": str(description),
            "system_prompt": body,
        }
        if "model" in metadata:
            resolved = _resolve_subagent_model(str(metadata["model"]), provider, path)
            if resolved is not None:
                subagent["model"] = resolved

        subagents.append(subagent)
        source_map[str(name)] = path

    return subagents, source_map


def _build_summarization_middleware(
    model: BaseChatModel,
    backend: Any,
    compaction: CompactionConfig,
) -> list[Any] | None:
    """Build compaction middleware when a threshold is configured.

    Returns ``None`` when the user has not set a trigger so that deepagents'
    own defaults remain untouched.
    """
    if compaction.trigger_tokens is not None:
        trigger: tuple[str, int] = ("tokens", compaction.trigger_tokens)
    elif compaction.trigger_messages is not None:
        trigger = ("messages", compaction.trigger_messages)
    else:
        return None

    keep = ("messages", compaction.keep_messages)
    summarization = SummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=trigger,
        keep=keep,
    )
    tool_middleware = SummarizationToolMiddleware(
        summarization,
        system_prompt=_COMPACT_TOOL_NUDGE,
    )
    return [summarization, tool_middleware]


def _build_backend(
    cwd: Path,
    *,
    enable_shell: bool = False,
    root_dir: str | Path | None = None,
) -> Any:
    """Build the agent backend.

    In CLI mode (``enable_shell=True``), uses ``LocalShellBackend`` with full
    shell access — same as before.  In server mode (``enable_shell=False``),
    uses a plain ``FilesystemBackend`` so per-user file privacy is enforced.
    """
    backend_root = str(root_dir) if root_dir is not None else str(cwd)
    if enable_shell:
        return LocalShellBackend(
            root_dir=backend_root, virtual_mode=False, inherit_env=True
        )
    return FilesystemBackend(root_dir=backend_root)


async def build_agent(
    config: Config,
    cwd: Path,
    checkpointer: Any,
    *,
    enable_shell: bool = True,
    root_dir: str | Path | None = None,
    skills_override: list[str] | None = None,
    memory_override: list[str] | None = None,
    model: Any | None = None,
    tools_override: list[BaseTool] | None = None,
    backend: Any | None = None,
) -> Any:
    """Build a harness agent for *cwd*.

    Parameters
    ----------
    config:
        Resolved harness configuration.
    cwd:
        Current working directory (used for discovery).
    checkpointer:
        LangGraph checkpointer for the agent.
    enable_shell:
        If True (default, CLI mode), use ``LocalShellBackend`` with shell.
        If False (server mode), use ``FilesystemBackend`` without shell.
    root_dir:
        Override the backend's root directory (for per-user workspaces).
    skills_override:
        Explicit skills directory paths (skip discovery, use workspace copies).
    memory_override:
        Explicit memory file paths (skip discovery, use workspace copies).
    model:
        Pre-built model (skip model construction, used by server for shared model).
    tools_override:
        Pre-loaded MCP tools (skip MCP loading, used by server for shared tools).
    backend:
        Pre-built backend (skip backend construction).
    """
    if model is not None:
        _model = model
    else:
        _model = _build_model(config)

    if skills_override is not None:
        skills = skills_override if skills_override else None
    else:
        skills_dir, _, _ = discover_all(cwd)
        skills = [str(skills_dir)] if skills_dir is not None else None

    if memory_override is not None:
        memory_paths = [Path(p) for p in memory_override if Path(p).is_file()]
    else:
        _, _, memory_paths = discover_all(cwd)

    _, agent_paths, _ = (
        discover_all(cwd) if skills_override is None and memory_override is None else (None, discover_agents(cwd), [])
    )
    subagents, subagent_sources = _load_subagents(agent_paths, config.provider)

    if tools_override is not None:
        mcp_tools = tools_override
    else:
        connections = load_mcp_connections(discover_mcp_config(cwd))

        async def _load_one_mcp(name: str, connection: dict[str, Any]) -> list[BaseTool]:
            try:
                client = MultiServerMCPClient(
                    {name: connection}, tool_name_prefix=True
                )
                return await asyncio.wait_for(
                    client.get_tools(), timeout=_MCP_TOOLS_TIMEOUT
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load MCP server %r: %s; continuing without it", name, exc
                )
                return []

        mcp_tools: list[BaseTool] = []
        if connections:
            loaded = await asyncio.gather(
                *(_load_one_mcp(name, conn) for name, conn in connections.items())
            )
            for tools in loaded:
                mcp_tools.extend(tools)

    if backend is not None:
        agent_backend = backend
    else:
        agent_backend = _build_backend(cwd, enable_shell=enable_shell, root_dir=root_dir)

    middleware = _build_summarization_middleware(_model, agent_backend, config.compaction)

    agent_kwargs: dict[str, Any] = {
        "model": _model,
        "tools": mcp_tools or None,
        "system_prompt": "You are a helpful coding assistant. Work in the user's project directory.",
        "subagents": subagents or None,
        "skills": skills,
        "memory": [str(p) for p in memory_paths] or None,
        "backend": agent_backend,
        "checkpointer": checkpointer,
    }
    if middleware is not None:
        agent_kwargs["middleware"] = middleware

    try:
        return create_deep_agent(**agent_kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to build agent; check subagent files: {subagent_sources!r}"
        ) from exc
