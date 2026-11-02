# Deployment

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
`--build-arg PROFILE_DIR=path/to/profile`.

## Mount the config instead of baking it

To keep configuration out of the image, mount a directory and point the server at it:

```bash
podman run --rm -p 8000:8000 \
  -v "$PWD/bot-config:/config:ro" \
  -v harness-state:/config/.harness \
  -e HARNESS_PROVIDER=openai -e HARNESS_MODEL=gpt-4o -e HARNESS_API_KEY="$OPENAI_API_KEY" \
  langchain-harness:0.1.0 serve --workspace /config
```

- `--workspace /config` (or `HARNESS_SERVER_WORKSPACE=/config`) makes all discovery read from the mount: `GUARDRAILS.md`, `CLAUDE.md`, `AGENTS.md`, `.claude/skills`, `.claude/agents`, `mcp.json`, `harness.toml`.
- Per-user state lives under `<workspace>/.harness/`, so that path must be writable. The nested `-v harness-state:/config/.harness` volume keeps the config mount read-only while state persists. Mounting the config directory read-write works too.
- Nothing needs rebuilding to change guardrails: edit the mounted file and restart, or wait for the user's next agent build.
- Without `--workspace`, behavior is unchanged: the image's `/app` profile is used.

## Scaling

The container defaults to single-replica SQLite under `/app/.harness`, which
needs a ReadWriteOnce volume and `strategy: Recreate`. Do not put that volume
on GCS FUSE.

To run multiple replicas, set `HARNESS_DATABASE_URL` to a Postgres database.
Sessions and LangGraph checkpoints are then shared. The remaining constraint
is workspace files under `users/<id>/workspace/`; to scale past one replica you
must still share those with a ReadWriteMany volume or equivalent shared
filesystem.

Reference manifests and the full deployment walkthrough are in [`deploy/README.md`](../deploy/README.md).
