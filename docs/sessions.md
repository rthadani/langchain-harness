# Sessions and memory

## Sessions

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
- `/eval` — score the session; see [Turn metrics and eval](./observability.md).

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

### In server mode

The inbox lives host-side at `<server-cwd>/.harness/memory/inbox/` — outside every user's workspace. Approved notes are copied into each user's workspace on their next agent build (copy-on-build, like `CLAUDE.md`/`AGENTS.md`); only `notes.md` is synced, never the inbox. Notes approved after a user's agent is already built won't reach that user until the server restarts or the agent is rebuilt.
