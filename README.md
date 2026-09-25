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

# resume the latest session
HARNESS_PROVIDER=openai HARNESS_MODEL=gpt-4o OPENAI_API_KEY=... uv run harness --continue

# list persisted sessions
uv run harness sessions
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

[compaction]
# trigger_tokens and trigger_messages are mutually exclusive
trigger_tokens = 12000
keep_messages = 20
```

### Compaction configuration

The harness can optionally replace deepagents' default summarization thresholds with explicit ones. Set **either** `trigger_tokens` **or** `trigger_messages`, not both.

| key | env override | default | meaning |
| --- | --- | --- | --- |
| `trigger_tokens` | `HARNESS_COMPACTION_TRIGGER_TOKENS` | unset | summarize after this many tokens |
| `trigger_messages` | `HARNESS_COMPACTION_TRIGGER_MESSAGES` | unset | summarize after this many messages |
| `keep_messages` | `HARNESS_COMPACTION_KEEP_MESSAGES` | `20` | messages retained after summarization |

When compaction is configured, the harness adds a `compact_conversation` tool to the model.

## Project-local discovery

All discovery is relative to the **current working directory**:

- **Skills:** `.claude/skills/<name>/SKILL.md` → passed to `create_deep_agent(skills=[...])`.
- **Subagents:** `.claude/agents/*.md` → parsed as `SubAgent` definitions. Frontmatter must contain `name` and `description`; `model` is optional. The `model` value can be a provider-prefixed ID (`provider:model`), one of the aliases `sonnet`, `opus`, or `haiku`, or `inherit` to use the parent agent's model. Any `tools:` frontmatter list is ignored in this version.
  - **Subagent model resolution is independent:** a subagent `model:` value is passed straight to deepagents and resolved by the `langchain` provider it names. It does **not** inherit the harness's `base_url` or `api_key`; it hits that provider's default endpoint using its own env-var keys (e.g. `OPENAI_API_KEY` for an `openai:` model).
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

## Sessions and resume

Sessions are persisted in `<cwd>/.harness/sessions.db`. Each REPL or one-shot run gets a fresh `thread_id`, and the harness records a title (first user message, truncated to ~60 characters) after every completed turn.

### CLI

- `harness sessions` — list recent sessions (`id`, `updated_at`, `title`).
- `harness --continue` — resume the latest session.
- `harness --resume <id-or-prefix>` — resume by full id or unambiguous prefix.

### REPL commands

- `/sessions` — list saved sessions.
- `/resume <id-or-prefix>` — switch to a different session mid-REPL.
- `/compact` — ask the model to call the `compact_conversation` tool immediately. The tool's response is shown verbatim; if the conversation is below the configured trigger threshold the tool refuses and the refusal is displayed.

## Important risks and limitations

- **Auto-run:** The harness runs all tools and shell commands without asking for approval. This is intentional for headless use, but dangerous. Run it only in directories you are willing to modify.
- **No sandbox:** `LocalShellBackend` is configured with `virtual_mode=False` so file operations and shell commands both operate on the real filesystem starting at the current working directory. This matches the unsandboxed behavior of a local assistant.
- **Per-call MCP:** MCP stdio servers are started fresh for each tool invocation. Stateful MCP servers may lose state between calls.
- **Harness profiles:** The underlying `deepagents` package supports harness profiles. This MVP does not expose profile selection; it builds a single default agent.
- **Compaction is model-mediated:** `/compact` sends a normal turn and relies on the model to call the `compact_conversation` tool. If the conversation is below the configured trigger threshold the tool refuses and only the refusal text is shown; no summarization happens.
- **Session storage grows unboundedly:** `.harness/sessions.db` accumulates one row per conversation. Prune old rows manually or with a separate job for now.
- **Conversation offloads:** deepagents may write large conversation summaries to `<cwd>/conversation_history/<session>.md` and `<cwd>/large_tool_results/`. These directories appear in the current working directory.
- **Per-call MCP:** MCP stdio servers are started fresh for each tool invocation. Stateful MCP servers may lose state between calls.

## Development

```bash
uv run pytest
```
