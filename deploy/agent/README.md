# Chatbot profile

This directory is baked into the image at `/app` and defines one chatbot:
its system prompt, durable memory, MCP servers, and harness settings.

Keep per-bot profiles in separate directories and select one at build time:

```
podman build --build-arg PROFILE_DIR=deploy/agent -t my-bot .
```

Everything here is read relative to the server's working directory (`/app`).

## Files

- `prompts/system.md` — the system prompt, wired up by `harness.toml`.
- `CLAUDE.md` — durable memory copied into each user workspace on first use.
- `mcp.json` — MCP servers (Claude Code format). Empty by default.
- `harness.toml` — server, prompt, and compaction settings.

Add `.claude/skills/<name>/SKILL.md` and `.claude/agents/<name>.md` here for
skills and subagents; both are discovered from the profile directory.

Keep secrets out: model credentials come from environment variables
(`HARNESS_API_KEY`), never from a committed file.
