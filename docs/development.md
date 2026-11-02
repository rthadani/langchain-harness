# Development

```bash
uv sync --extra dev
uv run pytest
```

Most of the suite runs against fake models and backends, so a normal `uv run pytest` needs no API key.

## Layout

```
src/langchain_harness/
  cli.py            Typer entry point: run, sessions, memory, eval, serve, db
  runner.py         one-shot and REPL loops
  events.py         stream_turn_events: the single path every turn goes through
  tracing.py        per-turn TurnMetrics, callback handler, JSONL sink
  agent_builder.py  builds the deep agent: discovery, guardrails, subagents, MCP
  discovery.py      finds skills, agents, memory, mcp.json
  guardrails.py     host-side GUARDRAILS.md loading and middleware
  frontmatter.py    YAML frontmatter parsing
  mcp_loader.py     mcp.json -> connection configs
  memory.py         summarize, queue notes, eval_session
  published.py      per-user published skills/agents: drafts, validation, store
  workspaces.py     per-user workspace provisioning and copies
  sessions.py       session rows in SQLite
  store.py          SQLite/Postgres store and checkpointer
  config.py         env > CLI > harness.toml resolution
  prompts/          system, eval, summarize, compact prompts
  api/              FastAPI app, auth, SSE, routes, schemas
tests/              pytest suite (asyncio_mode = auto)
deploy/             Dockerfile, entrypoint, k8s manifests, agent profile
docs/               these pages
```

## Behaviour worth knowing before changing it

- `events.stream_turn_events` is the only place a turn is started, so anything per-turn (metrics, tracing, SSE) belongs there rather than in each caller.
- Server mode builds one agent per user and caches it. Anything that changes what an agent loads must invalidate that cache, or it takes effect only after a restart.
- Guardrails are appended by middleware to every model call, including subagents. A new subagent path must attach that middleware or it bypasses them.
- Skills load source-directory order matters: later sources win, which is how global skills shadow user-published ones.
