# Deploying the harness

The image runs the multi-user FastAPI server (`harness serve`). Each API user
gets an isolated workspace under `/app/.harness/users/<user_id>/`.

## 1. Build the wheel

```bash
make wheel
# dist/langchain_harness-0.1.0-py3-none-any.whl
# dist/requirements.lock
```

`make wheel` builds the wheel and exports the exact pinned dependency set from
`uv.lock`, hashes included. The image installs those pins with
`pip --require-hashes`, so a build cannot silently pick up a new dependency
version.

## 2. Build the image

```bash
podman build -t registry.example.com/chatbots/support:1.0.0 .
# or: make image
```

The profile directory `deploy/agent/` is copied to `/app` and defines the bot:
system prompt, `CLAUDE.md` memory, `mcp.json`, and `harness.toml`. Use a
different directory per chatbot:

```bash
podman build --build-arg PROFILE_DIR=deploy/agent -t my-bot .
```

If your `mcp.json` launches `npx` servers, rebuild with
`--build-arg INSTALL_NODE=1` (Debian's Node may be older than some MCP
servers require) or use HTTP/SSE MCP servers instead.

## 3. Run locally

```bash
podman run --rm -p 8000:8000 \
  -e HARNESS_PROVIDER=openai \
  -e HARNESS_MODEL=gpt-4o \
  -e HARNESS_API_KEY="$OPENAI_API_KEY" \
  -v harness-state:/app/.harness \
  langchain-harness:0.1.0
```

The server defaults to `trust` auth mode, so send the user id via header:

```bash
curl -sN -X POST http://localhost:8000/v1/sessions \
  -H "X-User-Id: alice" \
  -H "Content-Type: application/json" \
  -d '{"title":"first chat"}'
```

## 4. Kubernetes

```bash
# Put the real key in the Secret, not in the committed YAML.
kubectl create secret generic harness-secrets \
  --from-literal=HARNESS_PROVIDER=openai \
  --from-literal=HARNESS_MODEL=gpt-4o \
  --from-literal=HARNESS_API_KEY="$OPENAI_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f deploy/k8s.yaml
```

Probes:

- `/healthz` (liveness) always returns 200 once the process serves HTTP.
- `/readyz` (readiness) returns 503 until the shared model and MCP tools finish
  initializing, then 200.

Neither endpoint requires auth.

## State and scaling

All state lives under `/app/.harness`:

| Path | Contents |
| --- | --- |
| `users/<id>/sessions.db` | LangGraph checkpoints and session rows (SQLite only) |
| `users/<id>/workspace/` | Agent files, conversation history, memory |

### SQLite (default)

**SQLite is single-writer.** LangGraph itself describes `AsyncSqliteSaver` as
not recommended for production. Treat it as one replica, one pod,
`strategy: Recreate`. That is what the reference manifest does.

### Postgres

Set `HARNESS_DATABASE_URL` in the Secret to use Postgres:

```bash
kubectl create secret generic harness-secrets \
  --from-literal=HARNESS_PROVIDER=openai \
  --from-literal=HARNESS_MODEL=gpt-4o \
  --from-literal=HARNESS_API_KEY="$OPENAI_API_KEY" \
  --from-literal=HARNESS_DATABASE_URL="postgresql://user:pass@postgres:5432/harness" \
  --dry-run=client -o yaml | kubectl apply -f -
```

With Postgres, the following are shared across replicas:

- Session rows (`sessions` table)
- LangGraph checkpoints (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes` tables)

The server runs schema creation automatically on startup. You can also run it
manually:

```bash
uv run harness db init
```

Optionally cap the connection pool size (default 10):

```bash
kubectl set env deploy/harness HARNESS_DATABASE_POOL_SIZE=20
```

**Workspace files are still per-pod.** Agent files, conversation history, and
memory live in `users/<id>/workspace/`. To run more than one replica you must
still share that directory with a `ReadWriteMany` volume or an equivalent
shared filesystem. Until that is in place, keep `replicas: 1` even with
Postgres.

### Where to put the state

**Recommended now: a block/file PVC, one replica.** Put the PVC at
`/app/.harness` and set `fsGroup: 1000` so the non-root user can write. Use an
SSD-backed storage class; do not use NFS.

**Do not put SQLite on GCS FUSE.** Cloud Storage FUSE does not support file
locking, and SQLite depends on it; concurrent access can corrupt the database.
Object stores are not a filesystem substitute for the live database.

**If you want GCS involved, use it for backup, not for the live files.** Two
workable shapes:

- Litestream 0.5+ as a sidecar, replicating the whole tree with its directory
  watcher, and restoring into the PVC on startup:

  ```yaml
  dbs:
    - dir: /app/.harness
      pattern: "*.db"
      recursive: true
      replica:
        url: gs://YOUR_BUCKET/langchain-harness
  ```

  Litestream still writes to local disk, so the single-writer rule is
  unchanged. Credentials come from Workload Identity or
  `GOOGLE_APPLICATION_CREDENTIALS`.

- A CronJob that mounts the PVC and snapshots it to GCS (`restic` or `rclone`).
  Quiesce writes or use SQLite's backup API for a consistent copy.

**If you need more than one replica**, SQLite has to go. The touchpoints are
small but real:

- `src/langchain_harness/api/app.py` — swap `AsyncSqliteSaver` for
  `AsyncPostgresSaver` (`langgraph-checkpoint-postgres`).
- `src/langchain_harness/sessions.py` — move the sessions table to Postgres.
- `src/langchain_harness/workspaces.py` — workspace files move to a shared
  filesystem or object store.

Until then, a HorizontalPodAutoscaler would hand two pods the same SQLite file
and corrupt it. Do not add one.

## Auth

The server supports two auth modes, set in `HARNESS_SERVER_AUTH_MODES` or in
`harness.toml` under `[server] auth_modes`. The default is `trust`:

- `trust`: `X-User-Id: <id>` is taken at face value. Nothing is persisted about
  users. This mode must sit behind an authenticating gateway; anyone who can
  reach the port can claim any user id.
- `jwt`: `Authorization: Bearer <JWT>` is verified via OIDC discovery + JWKS.
  The user id comes from `[server.jwt] user_claim` (default `sub`). Requires
  both `issuer` and `audience`.

See the main README for full JWT configuration.

## Configuration reference

Set through the Secret as env vars:

- `HARNESS_PROVIDER` — default `anthropic`.
- `HARNESS_MODEL` — required; the container exits if it is missing.
- `HARNESS_API_KEY` — or the provider-specific `OPENAI_API_KEY` /
  `ANTHROPIC_API_KEY`.
- `HARNESS_BASE_URL` — for OpenAI-compatible gateways.
- `HARNESS_DATABASE_URL` — Postgres DSN. When unset the server uses SQLite
  under `/app/.harness`.
- `HARNESS_DATABASE_POOL_SIZE` — Postgres connection pool size (default 10).
- `HARNESS_SERVER_HOST` / `HARNESS_SERVER_PORT` — the image sets
  `0.0.0.0`; the entrypoint maps a platform-injected `PORT`.
- `HARNESS_SERVER_AUTH_MODES` — e.g. `trust` or `jwt`.
- `HARNESS_SERVER_ENABLE_SHELL` — leave `false`. Enabling it lets agents read
  outside their per-user workspace.
- `HARNESS_COMPACTION_TRIGGER_TOKENS` or `HARNESS_COMPACTION_TRIGGER_MESSAGES`
  — mutually exclusive.

## Notes and gotchas

- MCP subagent models resolve independently of the harness `base_url` and
  `api_key`; they use their own provider env vars.
- The MCP `stdio` servers in `mcp.json` run inside the container. A `python`
  command uses the system interpreter, not the venv.
- `harness.toml` and the prompt files are baked into the image. Changing them
  requires a rebuild; mount a ConfigMap over `/app` only if you accept that it
  hides the baked profile.
- The container runs as UID/GID 1000. Match `runAsUser`/`fsGroup` or the volume
  will be unwritable.
