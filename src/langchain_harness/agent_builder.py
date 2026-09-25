"""Assemble a deep agent from config, discovered project files, and MCP servers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from deepagents.backends import LocalShellBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from langchain_harness.config import Config
from langchain_harness.discovery import discover_all, discover_mcp_config
from langchain_harness.frontmatter import parse_file
from langchain_harness.mcp_loader import load_mcp_connections

logger = logging.getLogger(__name__)

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


def _resolve_subagent_model(value: str, path: Path) -> str | None:
    """Translate a model value from frontmatter into a provider-prefixed spec.

    Returns ``None`` when the key should be dropped (e.g. ``inherit``).
    Logs a warning and returns ``None`` for unrecognised values so the agent
    is still created without a model override.
    """
    normalized = str(value).strip().lower()
    if normalized == "inherit":
        return None
    if normalized in _MODEL_ALIASES:
        return _MODEL_ALIASES[normalized]
    if ":" in str(value):
        return str(value)
    logger.warning(
        "Unrecognised model %r in %s; keeping agent without model override", value, path
    )
    return None


def _load_subagents(agent_paths: list[Path]) -> tuple[list[SubAgent], dict[str, Path]]:
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
            resolved = _resolve_subagent_model(str(metadata["model"]), path)
            if resolved is not None:
                subagent["model"] = resolved

        subagents.append(subagent)
        source_map[str(name)] = path

    return subagents, source_map


async def build_agent(
    config: Config,
    cwd: Path,
) -> Any:
    """Build a harness agent for *cwd*."""
    model = _build_model(config)

    skills_dir, agent_paths, memory_paths = discover_all(cwd)

    skills = [str(skills_dir)] if skills_dir is not None else None
    subagents, subagent_sources = _load_subagents(agent_paths)

    connections = load_mcp_connections(discover_mcp_config(cwd))

    mcp_tools: list[BaseTool] = []
    if connections:
        for name, connection in connections.items():
            try:
                client = MultiServerMCPClient({name: connection}, tool_name_prefix=True)
                tools = await client.get_tools()
                mcp_tools.extend(tools)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load MCP server %r: %s; continuing without it", name, exc
                )

    backend = LocalShellBackend(root_dir=str(cwd), virtual_mode=False, inherit_env=True)
    checkpointer = InMemorySaver()

    try:
        return create_deep_agent(
            model=model,
            tools=mcp_tools or None,
            system_prompt="You are a helpful coding assistant. Work in the user's project directory.",
            subagents=subagents or None,
            skills=skills,
            memory=[str(p) for p in memory_paths] or None,
            backend=backend,
            checkpointer=checkpointer,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to build agent; check subagent files: {subagent_sources!r}"
        ) from exc
