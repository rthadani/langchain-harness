# Project-local discovery

All discovery is relative to the **current working directory**. When `--workspace` is set, discovery follows that folder instead.

## Skills

`.claude/skills/<name>/SKILL.md` → passed to `create_deep_agent(skills=[...])`.

In server mode, each user's published skills are layered on top; see [Private published skills and agents](./publishing.md).

## Subagents

`.claude/agents/*.md` → parsed as `SubAgent` definitions. Frontmatter must contain `name` and `description`; `model` is optional. The `model` value can be a provider-prefixed ID (`provider:model`), one of the aliases `sonnet`, `opus`, or `haiku`, or `inherit` to use the parent agent's model. Any `tools:` frontmatter list is ignored in this version.

**Subagent model resolution is independent:** a subagent `model:` value is passed straight to deepagents and resolved by the `langchain` provider it names. It does **not** inherit the harness's `base_url` or `api_key`; it hits that provider's default endpoint using its own env-var keys (e.g. `OPENAI_API_KEY` for an `openai:` model).

Agent files are listed global-first in server mode, and the first definition of a name wins, so a published agent cannot shadow a global one.

## Memory

`CLAUDE.md`, `AGENTS.md`, and `.harness/memory/notes.md` (approved session notes) → passed as `memory=[...]` paths to `create_deep_agent`.

## Guardrails

`GUARDRAILS.md` in the working directory is read host-side at build time and appended to the system prompt on every model call, including subagent calls. It is never read through the agent backend, never copied into user workspaces, and is protected by a deny-write filesystem permission in server mode. It is not a memory source.

The file is per working directory, so one server shares it across all users. Editing it takes effect on the next agent build (restart, or that user's next rebuild), not mid-session. In the container, mount it rather than baking it in; see [Deployment](./deployment.md).

## MCP

`mcp.json` in Claude Code format:

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
