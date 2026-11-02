# Server mode (multi-user web API)

The harness can run as a multi-user web API server with per-user workspace isolation and pluggable auth.

## Quick start

```bash
# initialize the shared store (idempotent; uses Postgres when
# HARNESS_DATABASE_URL is set, otherwise SQLite)
uv run harness db init

# start the server
HARNESS_PROVIDER=anthropic HARNESS_MODEL=claude-3-5-sonnet-latest HARNESS_API_KEY=... uv run harness serve
```

The server's working directory — `serve --workspace <path>` or `HARNESS_SERVER_WORKSPACE` — decides where `GUARDRAILS.md`, `.claude/skills`, `.claude/agents`, memory, `mcp.json`, and `harness.toml` are read from, and `.harness/` under it must be writable. Without either, it uses the process working directory. See [Deployment](./deployment.md) for the mount pattern.

## API surface

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

Published skill and agent endpoints (also auth-scoped; all operate on the caller's own items only):

- `GET /v1/published` — list your published skills and agents; optional `?kind=skills|agents`.
- `GET /v1/published/items/{kind}/{name}` — read one of your published items, including its body.
- `GET /v1/published/drafts/{draft_id}` — preview a pending draft before confirming it.
- `POST /v1/published/{draft_id}` — confirm and publish a draft. The call is the confirmation; there is no request body.
- `DELETE /v1/published/items/{kind}/{name}` — delete one of your published items.

Sessions in another user's database return `404` naturally — there is no ownership-filtering logic to get wrong. Published items use the same rule: every path is derived from the authenticated user id, so another user's item reads as `404`, not `403`. See [Private published skills and agents](./publishing.md).

## Privacy model

Each user gets `<server-cwd>/.harness/users/<user_id>/` containing:

- `workspace/` — the agent's filesystem backend root (a `FilesystemBackend` with no shell tool).
- `conversation_history/` — lands inside the workspace automatically.

With SQLite (the default) each user also has `sessions.db` for checkpoints and
session rows. With Postgres those live in the shared database.

**Shell is off by default** in server mode. The agent uses a plain `FilesystemBackend` that enforces per-user path isolation — the agent cannot access files outside its workspace root.

**Warning: `enable_shell = true` voids per-user privacy.** When shell is enabled, the agent gets a `LocalShellBackend` that can `cd` to any path. This is a deliberate escape hatch for trusted single-user deployments, not a recommended configuration.

**MCP tools are outside the privacy boundary.** MCP servers configured in `mcp.json` are loaded once at startup and shared across all users. They run as host-side processes with whatever access the server process has. Only configure MCP servers that all authenticated users are allowed to use.

## Copy-on-build shared config

Skills (`.claude/skills`) and memory files (`CLAUDE.md`, `AGENTS.md`) are read through the agent's per-user backend, so they must be inside the workspace to be visible. On first agent build (and on rebuild), the harness copies shared config from the server's working directory into the user's workspace:

- `.claude/skills` → `<user-workspace>/.claude/skills`
- `CLAUDE.md` / `AGENTS.md` → `<user-workspace>/`
- `.harness/memory/notes.md` → `<user-workspace>/.harness/memory/notes.md` (approved notes only; the inbox is never copied)

Agent definition files (`.claude/agents/*.md`) and `mcp.json` are parsed host-side, so they stay shared without copying. `GUARDRAILS.md` is deliberately excluded from the copy: it is loaded host-side so sessions cannot edit or replace it.

## Server configuration

Add a `[server]` section to `harness.toml`:

```toml
[server]
host = "127.0.0.1"
port = 8000
enable_shell = false
auth_modes = ["trust"]  # default
```

Env overrides: `HARNESS_SERVER_HOST`, `HARNESS_SERVER_PORT`, `HARNESS_SERVER_ENABLE_SHELL`, `HARNESS_SERVER_AUTH_MODES`.

## Auth

The server supports two auth modes, listed in precedence order under `[server] auth_modes`. The first mode that recognizes the request wins.

```toml
[server]
auth_modes = ["trust"]  # default
```

Modes:

- `trust` (default): `X-User-Id: <id>` believed as-is. Identity comes entirely from the request header; nothing is persisted about users.
- `jwt`: `Authorization: Bearer <JWT>` verified against the issuer's JWKS via OIDC discovery. The user id is taken from the configured claim. Requires the `jwt` extra: `pip install 'langchain-harness[jwt]'` (or `uv add --extra jwt`).

**Warning:** `trust` mode is only safe behind an authenticating gateway. Anyone who can reach the server's port can claim any user id by sending the `X-User-Id` header, read that user's workspace and sessions, and allocate new workspaces without limit. There is no revocation, no provisioning cap, and no source ownership: identity comes from the header or the JWT claim only, and nothing is persisted about users.

### JWT configuration

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
