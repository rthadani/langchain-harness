# langchain-harness

A small, Claude Code-style harness built on [LangChain deepagents](https://github.com/langchain-ai/deepagents). It discovers local skills, subagents, and MCP servers from the working directory and runs a streaming agent loop against any OpenAI-compatible or Anthropic-compatible endpoint. It can run as a local CLI or as a multi-user web API with per-user isolation.

- **Local first:** `.claude/skills`, `.claude/agents`, `CLAUDE.md`, `mcp.json` — the same layout you already use.
- **Guardrails:** an operator-owned `GUARDRAILS.md` appended to every model call, including subagents.
- **Measured:** one JSON line per turn, plus an eval that scores sessions against those metrics.
- **Multi-user:** one workspace per user, private published skills and agents, pluggable auth.

## Quick start

```bash
uv sync --extra dev

# one-shot
HARNESS_PROVIDER=anthropic HARNESS_MODEL=claude-3-5-sonnet-latest HARNESS_API_KEY=... uv run harness "list the files"

# REPL
HARNESS_PROVIDER=openai HARNESS_MODEL=gpt-4o OPENAI_API_KEY=... uv run harness

# resume the latest session
uv run harness --continue

# list sessions / review pending memory notes
uv run harness sessions
uv run harness memory --review

# run against a different folder
uv run --project /path/to/langchain-harness harness --workspace ~/my-agents
```

The model is resolved from environment variables, then CLI flags, then `harness.toml`.

## Documentation

| Page | Contents |
| --- | --- |
| [Configuration](docs/configuration.md) | env vars, CLI flags, `harness.toml`, compaction |
| [Project-local discovery](docs/discovery.md) | skills, subagents, memory, guardrails, MCP |
| [Sessions and memory](docs/sessions.md) | resume, REPL commands, note review |
| [Turn metrics and eval](docs/observability.md) | per-turn JSONL, `run_id`, eval, judge model |
| [Server mode](docs/server.md) | API, privacy model, copy-on-build, auth and JWT |
| [Private publishing](docs/publishing.md) | per-user skills and agents published from a session |
| [Deployment](docs/deployment.md) | image, Kubernetes, mounting config instead of baking it |
| [Risks and limitations](docs/risks.md) | read this before running it anywhere real |
| [Development](docs/development.md) | tests, layout, invariants to preserve |

The detailed deployment walkthrough stays in [`deploy/README.md`](deploy/README.md).

## How it works

- Discovery happens relative to the working directory; `--workspace` moves that root.
- Every turn goes through `events.stream_turn_events`, which is where metrics, SSE, and the `run_id` come from.
- Server mode builds one agent per authenticated user, rooted at that user's workspace, and caches it until something changes what it loads.
- Guardrails are injected host-side by middleware on every model call, so workspace content cannot edit or displace them.

## Requirements

Python 3.11+ and [uv](https://docs.astral.sh/uv/). Server mode needs no extra dependencies for `trust` auth; the `jwt` and `postgres` extras are optional.
