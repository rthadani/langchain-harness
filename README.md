# langchain-harness

A small, Claude Code-style harness built on top of [LangChain deepagents](https://github.com/langchain-ai/deepagents). It discovers local skills, subagents, and MCP servers from the current working directory and runs a streaming agent loop against any OpenAI-compatible or Anthropic-compatible LLM endpoint.

## Quick start

```bash
# install dependencies
uv sync --extra dev

# one-shot
HARNESS_PROVIDER=anthropic HARNESS_MODEL=claude-3-5-sonnet-latest HARNESS_API_KEY=... uv run harness "list the files"

# REPL
HARNESS_PROVIDER=openai HARNESS_MODEL=gpt-4o OPENAI_API_KEY=... uv run harness
```

## Configuration

Resolution order: **environment variables > CLI flags > `harness.toml`**.

### Environment variables

- `HARNESS_PROVIDER` — model provider (`anthropic`, `openai`, ...). Defaults to `anthropic`.
- `HARNESS_MODEL` — model name. **Required** if not provided another way.
- `HARNESS_BASE_URL` — custom base URL for OpenAI-compatible endpoints.
- `HARNESS_API_KEY` — explicit API key.
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — used as API-key fallbacks.

### CLI flags

- `--provider`, `--model`, `--base-url`, `--api-key`

### `harness.toml`

```toml
[model]
provider = "openai"
model = "gpt-4o"
base_url = "https://api.example.com/v1"
# api_key = "..."   # better to use env vars
```

## Project-local discovery

All discovery is relative to the **current working directory**:

- **Skills:** `.claude/skills/<name>/SKILL.md` → passed to `create_deep_agent(skills=[...])`.
- **Subagents:** `.claude/agents/*.md` → parsed as `SubAgent` definitions. Frontmatter must contain `name` and `description`; `model` is optional. Any `tools:` frontmatter list is ignored in this version.
- **Memory:** `CLAUDE.md` and `AGENTS.md` → passed as `memory=[...]` paths to `create_deep_agent`.
- **MCP:** `mcp.json` in Claude Code format. Example:

```json
{
  "mcpServers": {
    "example-stdio": {
      "command": "python",
      "args": ["scripts/echo_server.py"]
    },
    "my-server": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": { "NODE_ENV": "production" }
    },
    "http-server": {
      "type": "http",
      "url": "http://localhost:3000/mcp",
      "headers": { "Authorization": "Bearer token" }
    }
  }
}
```

Transport mapping:

- presence of `command` → `stdio`
- `type: "http"` or `transport: "http"` → `streamable_http`
- `type: "sse"` or `transport: "sse"` → `sse`

## Important risks and limitations

- **Auto-run:** The harness runs all tools and shell commands without asking for approval. This is intentional for headless use, but dangerous. Run it only in directories you are willing to modify.
- **No sandbox:** `LocalShellBackend` is configured with `virtual_mode=False` so file operations and shell commands both operate on the real filesystem starting at the current working directory. This matches the unsandboxed behavior of a local assistant.
- **Per-call MCP:** MCP stdio servers are started fresh for each tool invocation. Stateful MCP servers may lose state between calls.
- **Harness profiles:** The underlying `deepagents` package supports harness profiles. This MVP does not expose profile selection; it builds a single default agent.

## Development

```bash
uv run pytest
```
