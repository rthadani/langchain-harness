# Important risks and limitations

- **Auto-run:** The harness runs all tools and shell commands without asking for approval. This is intentional for headless use, but dangerous. Run it only in directories you are willing to modify.
- **No sandbox:** `LocalShellBackend` is configured with `virtual_mode=False` so file operations and shell commands both operate on the real filesystem starting at the current working directory. This matches the unsandboxed behavior of a local assistant.
- **Per-call MCP:** MCP stdio servers are started fresh for each tool invocation. Stateful MCP servers may lose state between calls.
- **Harness profiles:** The underlying `deepagents` package supports harness profiles. This MVP does not expose profile selection; it builds a single default agent.
- **Compaction is model-mediated:** `/compact` sends a normal turn and relies on the model to call the `compact_conversation` tool. If the conversation is below the configured trigger threshold the tool refuses and only the refusal text is shown; no summarization happens.
- **Session storage grows unboundedly:** `.harness/sessions.db` accumulates one row per conversation. Prune old rows manually or with a separate job for now.
- **Session memory grows unboundedly:** `.harness/memory/notes.md` accumulates one section per approved note. Trim it manually when it gets long.
- **Conversation offloads:** deepagents may write large conversation summaries to `<cwd>/conversation_history/<session>.md` and `<cwd>/large_tool_results/`. These directories appear in the current working directory, or in the workspace folder when `--workspace` is set.
- **Server-mode MCP is shared:** MCP servers configured in `mcp.json` are loaded once and shared across all users. They are outside the per-user privacy boundary.
- **Prose guardrails are not a security boundary:** `GUARDRAILS.md` reinforces behaviour but cannot guarantee it. With `enable_shell = true` the immutable-guardrail claim does not hold: filesystem permissions are not enforced on execution backends and the agent can edit the file. Rules that must hold absolutely belong in permissions or the tool allowlist.
- **Token counts can read 0:** streaming OpenAI-compatible endpoints often omit usage. Set `stream_usage=True` on the model if cost numbers matter; see [Turn metrics and eval](./observability.md).
