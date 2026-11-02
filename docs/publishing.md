# Private published skills and agents

A user can publish their own skills and subagents from inside a chat session. Artifacts are durable and host-side at `<server-cwd>/.harness/users/<user_id>/published/` (`skills/<name>/SKILL.md`, `agents/<name>.md`), outside the agent-writable workspace, and only that user's agent builds read them. Published skills are copied into the owner's workspace on their next build (`published/skills` → `<workspace>/.claude/user-skills`); a user's published agents are passed to the builder after the global agent files.

## How publishing works

1. The agent calls `draft_skill` / `draft_agent` during a session. The tool only writes a workspace-scoped draft (24h TTL, at most 20 pending) and returns a draft id.
2. `GET /v1/published/drafts/{draft_id}` previews it.
3. `POST /v1/published/{draft_id}` is the explicit confirmation. Nothing is active before that call.
4. `GET /v1/published` lists the caller's items; `DELETE /v1/published/items/{kind}/{name}` removes one.

Publishing renders the frontmatter server-side from the validated `name` and `description` only, so `model`, `tools`, `allowed-tools`, `middleware`, and `permissions` cannot be set by a published file. Names are validated against a strict regex, size-capped, and colliding with a global name is rejected with `409` rather than silently losing.

## Ordering keeps globals authoritative

- Skills load as `[user-skills, global-skills]`. The skills middleware lets later sources win, so a global skill shadows a user skill of the same name.
- Agent files are listed global-first and the first definition of a name wins, so a global agent shadows a published one.

## Visibility and isolation

Every path is derived from the authenticated user id, so another user's item reads as `404`, not `403`. The per-user agent is built from that user's paths only; there is no cross-user code path to get wrong.

## Guardrails

A published agent is a subagent like any other, so it inherits `GuardrailMiddleware`. Publishing cannot weaken [guardrails](./discovery.md#guardrails); content that tries is still subject to the tool and filesystem limits the server enforces.

## Cache invalidation

Publishing or deleting rebuilds the owner's agent on its next request. The old agent's checkpointer scope is retired rather than closed, so an in-flight SSE stream is not interrupted; session history survives because the thread id and the store stay the same.
