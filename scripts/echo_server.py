#!/usr/bin/env python3
"""Tiny example MCP stdio server exposing a single `echo` tool."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("echo-server")


@server.tool()
def echo(message: str) -> str:
    """Echo a message back to the caller."""
    return message


if __name__ == "__main__":
    server.run(transport="stdio")
