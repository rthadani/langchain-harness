"""Load and normalize MCP server connections from ``mcp.json``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_server(name: str, config: Any) -> dict[str, Any] | None:
    """Convert a single Claude-Code-style server entry to a connection dict."""
    if not isinstance(config, dict):
        logger.warning("MCP server %r has invalid configuration; skipping", name)
        return None

    if "command" in config:
        connection: dict[str, Any] = {
            "transport": "stdio",
            "command": config["command"],
            "args": config.get("args", []),
        }
        if "env" in config:
            connection["env"] = config["env"]
        return connection

    transport = config.get("transport") or config.get("type")
    if transport is None:
        logger.warning(
            "MCP server %r missing transport/type or command; skipping", name
        )
        return None

    transport = str(transport).lower()
    url = config.get("url")
    if url is None:
        logger.warning("MCP server %r missing url; skipping", name)
        return None

    headers = config.get("headers")

    if transport in {"http", "streamable_http", "streamable-http"}:
        connection = {"transport": "streamable_http", "url": url}
    elif transport == "sse":
        connection = {"transport": "sse", "url": url}
    else:
        logger.warning("MCP server %r has unsupported transport %r; skipping", name, transport)
        return None

    if headers is not None:
        connection["headers"] = headers

    return connection


def load_mcp_connections(path: Path) -> dict[str, dict[str, Any]]:
    """Load MCP connections from *path* for ``MultiServerMCPClient``.

    The file may be missing, in which case an empty dict is returned. Malformed
    JSON causes a warning and an empty dict.
    """
    if not path.is_file():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Malformed MCP config %s: %s; continuing without MCP", path, exc)
        return {}

    if not isinstance(raw, dict):
        logger.warning("Malformed MCP config %s: top level is not an object", path)
        return {}

    servers = raw.get("mcpServers", raw)
    if not isinstance(servers, dict):
        logger.warning("Malformed MCP config %s: mcpServers is not an object", path)
        return {}

    connections: dict[str, dict[str, Any]] = {}
    for name, config in servers.items():
        connection = _normalize_server(name, config)
        if connection is not None:
            connections[name] = connection

    return connections
