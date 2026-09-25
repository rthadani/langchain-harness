import json
from pathlib import Path

import pytest

from langchain_harness.mcp_loader import load_mcp_connections


def test_load_stdio(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "stdio-server": {
                        "command": "npx",
                        "args": ["-y", "server"],
                        "env": {"KEY": "value"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    connections = load_mcp_connections(path)
    assert connections == {
        "stdio-server": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "server"],
            "env": {"KEY": "value"},
        }
    }


def test_load_http(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "http-server": {
                        "type": "http",
                        "url": "http://localhost/mcp",
                        "headers": {"Authorization": "Bearer x"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    connections = load_mcp_connections(path)
    assert connections == {
        "http-server": {
            "transport": "streamable_http",
            "url": "http://localhost/mcp",
            "headers": {"Authorization": "Bearer x"},
        }
    }


def test_load_sse_with_transport_alias(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "sse-server": {
                        "transport": "sse",
                        "url": "http://localhost/sse",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    connections = load_mcp_connections(path)
    assert connections == {
        "sse-server": {"transport": "sse", "url": "http://localhost/sse"}
    }


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_mcp_connections(tmp_path / "mcp.json") == {}


def test_malformed_json_warns_and_returns_empty(tmp_path: Path, caplog) -> None:
    path = tmp_path / "mcp.json"
    path.write_text("not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert load_mcp_connections(path) == {}
    assert "Malformed MCP config" in caplog.text


def test_invalid_server_skipped(tmp_path: Path, caplog) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "bad": {"url": "http://x"},
                    "good": {"command": "cat", "args": []},
                }
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        connections = load_mcp_connections(path)
    assert "bad" in caplog.text
    assert connections == {
        "good": {"transport": "stdio", "command": "cat", "args": []}
    }
