"""Assemble a deep agent from config, discovered project files, and MCP servers."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from deepagents.backends import LocalShellBackend
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
from langchain_harness.discovery import discover_all, discover_mcp_config
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


async def build_agent(
    config: Config,
    cwd: Path,
    checkpointer: Any,
) -> Any:
    """Build a harness agent for *cwd*."""
    model = _build_model(config)

    skills_dir, agent_paths, memory_paths = discover_all(cwd)

    skills = [str(skills_dir)] if skills_dir is not None else None
    subagents, subagent_sources = _load_subagents(agent_paths, config.provider)

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

    backend = LocalShellBackend(root_dir=str(cwd), virtual_mode=False, inherit_env=True)
    middleware = _build_summarization_middleware(model, backend, config.compaction)

    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": mcp_tools or None,
        "system_prompt": "You are a helpful coding assistant. Work in the user's project directory.",
        "subagents": subagents or None,
        "skills": skills,
        "memory": [str(p) for p in memory_paths] or None,
        "backend": backend,
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
