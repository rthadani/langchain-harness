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

# review pending session notes (approve to make them long-term memory)
uv run harness memory --review

# run against a different working folder (a relative path resolves against the current directory; ~ expands)
uv run --project /path/to/langchain-harness harness --workspace ~/my-agents
```

## Configuration

Resolution order: **environment variables > CLI flags > `harness.toml`**.

### Environment variables

- `HARNESS_PROVIDER` — model provider (`anthropic`, `openai`, ...). Defaults to `anthropic`.
- `HARNESS_MODEL` — model name. **Required** if not provided another way.
- `HARNESS_BASE_URL` — custom base URL for OpenAI-compatible endpoints.
- `HARNESS_API_KEY` — explicit API key.
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — used as API-key fallbacks.
- `HARNESS_DATABASE_URL` — Postgres DSN for server/session storage. When
  unset the server uses SQLite under `<cwd>/.harness`. Applies to `serve` and
  `db init`; local `run`/`sessions`/`memory` still use SQLite.
- `HARNESS_DATABASE_POOL_SIZE` — Postgres connection pool size (default 10).

### CLI flags

- `--provider`, `--model`, `--base-url`, `--api-key`
- `--workspace <path>` — working folder to run in (equivalent to `cd <path> && harness`). Discovery, `harness.toml`, `sessions.db`, and the agent backend all follow it. A relative path resolves against the current directory, and `~` expands. Applies to `run`, `sessions`, and `memory` only (not `serve`).

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

All discovery is relative to the **current working directory**. When `--workspace` is set, discovery follows that folder instead.

- **Skills:** `.claude/skills/<name>/SKILL.md` → passed to `create_deep_agent(skills=[...])`.
- **Subagents:** `.claude/agents/*.md` → parsed as `SubAgent` definitions. Frontmatter must contain `name` and `description`; `model` is optional. The `model` value can be a provider-prefixed ID (`provider:model`), one of the aliases `sonnet`, `opus`, or `haiku`, or `inherit` to use the parent agent's model. Any `tools:` frontmatter list is ignored in this version.
  - **Subagent model resolution is independent:** a subagent `model:` value is passed straight to deepagents and resolved by the `langchain` provider it names. It does **not** inherit the harness's `base_url` or `api_key`; it hits that provider's default endpoint using its own env-var keys (e.g. `OPENAI_API_KEY` for an `openai:` model).
- **Memory:** `CLAUDE.md`, `AGENTS.md`, and `.harness/memory/notes.md` (approved session notes) → passed as `memory=[...]` paths to `create_deep_agent`.
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

Sessions are persisted in `<cwd>/.harness/sessions.db` (or `<workspace>/.harness/sessions.db` when `--workspace` is set). Each REPL or one-shot run gets a fresh `thread_id`, and the harness records a title (first user message, truncated to ~60 characters) after every completed turn.

### CLI

- `harness sessions` — list recent sessions (`id`, `updated_at`, `title`).
- `harness --continue` — resume the latest session.
- `harness --resume <id-or-prefix>` — resume by full id or unambiguous prefix.

### REPL commands

- `/sessions` — list saved sessions.
- `/resume <id-or-prefix>` — switch to a different session mid-REPL.
- `/summarize` — summarize the session so far and queue it for review in `.harness/memory/inbox/`.
- `/compact` — ask the model to call the `compact_conversation` tool immediately. The tool's response is shown verbatim; if the conversation is below the configured trigger threshold the tool refuses and the refusal is displayed.

## Session memory

The harness can turn a session into a durable note that future sessions auto-load as long-term memory. Notes are LLM summaries, queued for human review before they are added.

### Capturing a summary

- One-shot `harness "..."` runs auto-summarize on completion (headless, so there is no prompt to trigger).
- REPL: type `/summarize`.
- Server API: `POST /v1/sessions/{id}/summarize`.

### Reviewing notes

Pending notes are written to `.harness/memory/inbox/<id>.md`.

- `harness memory` — list pending notes.
- `harness memory --review` — interactive approve / skip / delete. Approving appends the note to `.harness/memory/notes.md`; delete discards it; skip leaves it pending.
- `harness memory --workspace <path>` — operate on another working folder.

Approved notes in `.harness/memory/notes.md` are added to the agent's `memory` on every future run. The `.harness/` tree is gitignored, so this memory stays local.

### Server mode

In server mode the inbox lives host-side at `<server-cwd>/.harness/memory/inbox/` — outside every user's workspace. Approved notes are copied into each user's workspace on their next agent build (copy-on-build, like `CLAUDE.md`/`AGENTS.md`); only `notes.md` is synced, never the inbox. Notes approved after a user's agent is already built won't reach that user until the server restarts or the agent is rebuilt.

## Server mode (multi-user web API)

The harness can run as a multi-user web API server with per-user workspace isolation and pluggable auth.

### Quick start

```bash
# initialize the shared store (idempotent; uses Postgres when
# HARNESS_DATABASE_URL is set, otherwise SQLite)
uv run harness db init

# start the server
HARNESS_PROVIDER=anthropic HARNESS_MODEL=claude-3-5-sonnet-latest HARNESS_API_KEY=... uv run harness serve
```

### API surface

Session endpoints require a credential matching one of the configured auth modes (see [Auth](#auth)):

- `GET /healthz` — liveness; no auth. Returns 200 once the process serves HTTP.
- `GET /readyz` — readiness; no auth. Returns 503 until startup finishes, then 200.
- `POST /v1/sessions` — create a new session.
- `GET /v1/sessions` — list the authenticated user's sessions only.
- `GET /v1/sessions/{id}/history` — message history for a session.
- `POST /v1/sessions/{id}/messages` — send a message, stream the response via SSE.
- `POST /v1/sessions/{id}/compact` — stream the compact flow via SSE.
- `POST /v1/sessions/{id}/summarize` — summarize the session and queue it for memory review.
- `DELETE /v1/sessions/{id}` — delete a session.

Sessions in another user's database return `404` naturally — there is no ownership-filtering logic to get wrong.

### Privacy model

Each user gets `<server-cwd>/.harness/users/<user_id>/` containing:

- `workspace/` — the agent's filesystem backend root (a `FilesystemBackend` with no shell tool).
- `conversation_history/` — lands inside the workspace automatically.

With SQLite (the default) each user also has `sessions.db` for checkpoints and
session rows. With Postgres those live in the shared database.

**Shell is off by default** in server mode. The agent uses a plain `FilesystemBackend` that enforces per-user path isolation — the agent cannot access files outside its workspace root.

**Warning: `enable_shell = true` voids per-user privacy.** When shell is enabled, the agent gets a `LocalShellBackend` that can `cd` to any path. This is a deliberate escape hatch for trusted single-user deployments, not a recommended configuration.

**MCP tools are outside the privacy boundary.** MCP servers configured in `mcp.json` are loaded once at startup and shared across all users. They run as host-side processes with whatever access the server process has. Only configure MCP servers that all authenticated users are allowed to use.

### Copy-on-build shared config

Skills (`.claude/skills`) and memory files (`CLAUDE.md`, `AGENTS.md`) are read through the agent's per-user backend, so they must be inside the workspace to be visible. On first agent build (and on rebuild), the harness copies shared config from the server's CWD into the user's workspace:

- `.claude/skills` → `<user-workspace>/.claude/skills`
- `CLAUDE.md` / `AGENTS.md` → `<user-workspace>/`
- `.harness/memory/notes.md` → `<user-workspace>/.harness/memory/notes.md` (approved notes only; the inbox is never copied)

Agent definition files (`.claude/agents/*.md`) and `mcp.json` are parsed host-side, so they stay shared without copying.

### Server configuration

Add a `[server]` section to `harness.toml`:

```toml
[server]
host = "127.0.0.1"
port = 8000
enable_shell = false
auth_modes = ["trust"]  # default
```

Env overrides: `HARNESS_SERVER_HOST`, `HARNESS_SERVER_PORT`, `HARNESS_SERVER_ENABLE_SHELL`, `HARNESS_SERVER_AUTH_MODES`.

### Auth

The server supports two auth modes, listed in precedence order under `[server] auth_modes`. The first mode that recognizes the request wins.

```toml
[server]
auth_modes = ["trust"]  # default
```

Modes:

- `trust` (default): `X-User-Id: <id>` believed as-is. Identity comes entirely from the request header; nothing is persisted about users.
- `jwt`: `Authorization: Bearer <JWT>` verified against the issuer's JWKS via OIDC discovery. The user id is taken from the configured claim. Requires the `jwt` extra: `pip install 'langchain-harness[jwt]'` (or `uv add --extra jwt`).

**Warning:** `trust` mode is only safe behind an authenticating gateway. Anyone who can reach the server's port can claim any user id by sending the `X-User-Id` header, read that user's workspace and sessions, and allocate new workspaces without limit. There is no revocation, no provisioning cap, and no source ownership: identity comes from the header or the JWT claim only, and nothing is persisted about users.

#### JWT configuration

When `auth_modes` includes `jwt`, configure verification in `[server.jwt]`:

```toml
[server]
auth_modes = ["jwt"]

[server.jwt]
issuer = "https://accounts.example.com"
audience = "my-harness-api"
user_claim = "sub"          # claim used as the harness user id
```

JWT mode requires both `issuer` and `audience` to be set; configuration that omits either is rejected at startup. `jwks_url` is optional and only overrides where the JWKS document is fetched from.

Signing keys are fetched from the issuer's OIDC discovery document (`<issuer>/.well-known/openid-configuration`) unless `jwks_url` is set. Only asymmetric algorithms are supported: `RS256`/`RS384`/`RS512`, `ES256`/`ES384`/`ES512`, `PS256`/`PS384`/`PS512`, and `EdDSA`. HMAC (`HS*`) and `none` are forbidden — the verifier uses the issuer's public keys, so accepting a shared-secret algorithm would let anyone forge tokens.

Env overrides: `HARNESS_SERVER_JWT_ISSUER`, `HARNESS_SERVER_JWT_AUDIENCE`, `HARNESS_SERVER_JWT_USER_CLAIM`, `HARNESS_SERVER_JWT_ALGORITHMS`, `HARNESS_SERVER_JWT_JWKS_URL`, `HARNESS_SERVER_JWT_CLOCK_SKEW_SECONDS`, `HARNESS_SERVER_JWT_DISCOVERY_TIMEOUT_SECONDS`.

## Container image and Kubernetes

`make wheel` builds `dist/*.whl` plus a hash-locked `dist/requirements.lock`.
`make image` builds the reference container from that wheel and the chatbot
profile in `deploy/agent/`.

```bash
make wheel
podman build -t registry.example.com/chatbot:1.0.0 .
podman run --rm -p 8000:8000 \
  -e HARNESS_PROVIDER=openai -e HARNESS_MODEL=gpt-4o -e HARNESS_API_KEY=... \
  -v harness-state:/app/.harness langchain-harness:0.1.0
```

Each image bakes one chatbot profile: system prompt, `CLAUDE.md`, `mcp.json`,
and `harness.toml`. Select another profile with
`--build-arg PROFILE_DIR=path/to/profile`. Reference manifests and the state
recommendation are in `deploy/README.md`.

The container defaults to single-replica SQLite under `/app/.harness`, which
needs a ReadWriteOnce volume and `strategy: Recreate`. Do not put that volume
on GCS FUSE.

To run multiple replicas, set `HARNESS_DATABASE_URL` to a Postgres database.
Sessions and LangGraph checkpoints are then shared. The remaining constraint
is workspace files under `users/<id>/workspace/`; to scale past one replica you
must still share those with a ReadWriteMany volume or equivalent shared
filesystem. Reference manifests are in `deploy/README.md`.

## Important risks and limitations

- **Auto-run:** The harness runs all tools and shell commands without asking for approval. This is intentional for headless use, but dangerous. Run it only in directories you are willing to modify.
- **No sandbox:** `LocalShellBackend` is configured with `virtual_mode=False` so file operations and shell commands both operate on the real filesystem starting at the current working directory. This matches the unsandboxed behavior of a local assistant.
- **Per-call MCP:** MCP stdio servers are started fresh for each tool invocation. Stateful MCP servers may lose state between calls.
- **Harness profiles:** The underlying `deepagents` package supports harness profiles. This MVP does not expose profile selection; it builds a single default agent.
- **Compaction is model-mediated:** `/compact` sends a normal turn and relies on the model to call the `compact_conversation` tool. If the conversation is below the configured trigger threshold the tool refuses and only the refusal text is shown; no summarization happens.
- **Session storage grows unboundedly:** `.harness/sessions.db` accumulates one row per conversation. Prune old rows manually or with a separate job for now.
- **Session memory grows unboundedly:** `.harness/memory/notes.md` accumulates one section per approved note. Trim it manually when it gets long.
- **Conversation offloads:** deepagents may write large conversation summaries to `<cwd>/conversation_history/<session>.md` and `<cwd>/large_tool_results/`. These directories appear in the current working directory, or in the workspace folder when `--workspace` is set.
- **Server-mode MCP is shared:** MCP servers configured in `mcp.json` are loaded once and shared across all users. They are outside the per-user privacy boundary.

## Development

```bash
uv run pytest
```
