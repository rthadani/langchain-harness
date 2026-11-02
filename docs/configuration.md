# Configuration

Resolution order: **environment variables > CLI flags > `harness.toml`**.

## Environment variables

- `HARNESS_PROVIDER` — model provider (`anthropic`, `openai`, ...). Defaults to `anthropic`.
- `HARNESS_MODEL` — model name. **Required** if not provided another way.
- `HARNESS_BASE_URL` — custom base URL for OpenAI-compatible endpoints.
- `HARNESS_API_KEY` — explicit API key.
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — used as API-key fallbacks.
- `HARNESS_DATABASE_URL` — Postgres DSN for server/session storage. When
  unset the server uses SQLite under `<cwd>/.harness`. Applies to `serve` and
  `db init`; local `run`/`sessions`/`memory` still use SQLite.
- `HARNESS_DATABASE_POOL_SIZE` — Postgres connection pool size (default 10).
- `HARNESS_SERVER_WORKSPACE` — default `--workspace` for `serve`; the CLI flag wins.
- `HARNESS_EVAL_MODEL` — judge model for `eval`; see [Turn metrics and eval](./observability.md).
- `HARNESS_APP_VERSION` — build identifier written to metrics and eval records.

## CLI flags

- `--provider`, `--model`, `--base-url`, `--api-key`
- `--workspace <path>` — working folder to run in (equivalent to `cd <path> && harness`). Discovery, `harness.toml`, `sessions.db`, and the agent backend all follow it. A relative path resolves against the current directory, and `~` expands. Applies to `run`, `sessions`, `memory`, `eval`, `db init`, and `serve`.

## `harness.toml`

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

## Compaction configuration

The harness can optionally replace deepagents' default summarization thresholds with explicit ones. Set **either** `trigger_tokens` **or** `trigger_messages`, not both.

| key | env override | default | meaning |
| --- | --- | --- | --- |
| `trigger_tokens` | `HARNESS_COMPACTION_TRIGGER_TOKENS` | unset | summarize after this many tokens |
| `trigger_messages` | `HARNESS_COMPACTION_TRIGGER_MESSAGES` | unset | summarize after this many messages |
| `keep_messages` | `HARNESS_COMPACTION_KEEP_MESSAGES` | `20` | messages retained after summarization |

When compaction is configured, the harness adds a `compact_conversation` tool to the model.

Server-specific configuration lives in [Server mode](./server.md), and the judge model in [Turn metrics and eval](./observability.md).
