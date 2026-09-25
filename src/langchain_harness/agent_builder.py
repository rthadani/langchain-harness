"""Assemble a deep agent from config, discovered project files, and MCP servers."""

from __future__ import annotations

import logging
from pathlib import Path

from deepagents.backends import LocalShellBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from langchain_harness.config import Config
from langchain_harness.discovery import discover_agents, discover_memory, discover_skills
from langchain_harness.frontmatter import parse_file
from langchain_harness.mcp_loader import load_mcp_connections

logger = logging.getLogger(__name__)


def _build_model(config: Config) -> BaseChatModel:
    """Initialize the chat model from the resolved configuration."""
    provider = config.provider
    model = config.model
    base_url = config.base_url
    api_key = config.api_key

    if base_url and api_key:
        return init_chat_model(
            model, model_provider=provider, base_url=base_url, api_key=api_key
        )
    if base_url:
        return init_chat_model(model, model_provider=provider, base_url=base_url)
    if api_key:
        return init_chat_model(model, model_provider=provider, api_key=api_key)
    return init_chat_model(model, model_provider=provider)


def _load_subagents(agent_paths: list[Path]) -> list[SubAgent]:
    """Parse agent markdown files into ``SubAgent`` definitions."""
    subagents: list[SubAgent] = []
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
            subagent["model"] = str(metadata["model"])

        subagents.append(subagent)

    return subagents


async def build_agent(
    config: Config,
    cwd: Path,
) -> object:
    """Build a harness agent for *cwd*."""
    model = _build_model(config)

    skills_dir = discover_skills(cwd)
    agent_paths = discover_agents(cwd)
    memory_paths = discover_memory(cwd)

    skills = [str(skills_dir)] if skills_dir is not None else None
    subagents = _load_subagents(agent_paths)

    connections = load_mcp_connections(cwd / "mcp.json")

    mcp_tools: list[BaseTool] = []
    try:
        client = MultiServerMCPClient(connections, tool_name_prefix=True)
        mcp_tools = await client.get_tools()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load MCP tools: %s; continuing without MCP", exc)

    backend = LocalShellBackend(root_dir=str(cwd), virtual_mode=False)
    checkpointer = InMemorySaver()

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
