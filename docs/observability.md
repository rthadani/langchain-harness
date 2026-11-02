# Turn metrics and eval

## Turn metrics

Every turn — REPL, one-shot, `POST /messages`, `/compact` — logs exactly one JSON line on the `langchain_harness.metrics` logger, so turns can be costed and compared without a database:

```json
{"run_id":"5f1c...","thread_id":"a3b9...","user_id":"alice","model_calls":3,"input_tokens":812,"output_tokens":145,"overhead_calls":0,"overhead_tokens":0,"model_ms":4210.7,"ttft_ms":640.2,"total_ms":5310.4,"tools":[{"name":"search","ms":120.5,"error":null}],"outcome":"ok","error":null,"tool_calls":1,"tool_errors":0}
```

`ttft_ms` is time to first token, `overhead_*` counts compaction-middleware model calls separately from the answer's, and `tools` carries per-call latency plus any error. `outcome` is one of `ok`, `error`, `cancelled` (turn interrupted), or `client_disconnect` (SSE consumer went away).

The `done` event (and its SSE frame) carries `run_id` — the root trace id for the turn — so user feedback or a judge score can be joined back to the metrics line later.

Each line carries `user_id`; hash or drop it before shipping metrics to a shared store. Token counts read 0 when an OpenAI-compatible endpoint omits usage while streaming. `HARNESS_APP_VERSION` overrides the package version in the metadata, so set it per build to tell builds apart.

## Durable sink

When the CLI runs a turn, the same JSON line is also appended to `<workspace>/.harness/metrics/turns.jsonl`. `harness eval <session>` runs in a fresh process that never streams a turn, so it reads the tail of that file (last 200 lines, filtered to the session) rather than the in-memory registry. The registry is bounded to the last 50 turns of the last 200 threads, and metrics failures are logged, never raised: a broken metrics path cannot break a turn.

## Eval inputs and the joined record

`/eval` and `harness eval <session>` receive the last 10 turn metrics of the session as evidence for `tool_use_quality` — tool names and call counts, `tool_errors`, `model_calls`, `ttft_ms`, `total_ms`, `overhead_tokens`. The block is character-capped, so a long session cannot crowd out the transcript. Unfinished turns are skipped, and a session with no metrics gets the same prompt as before.

Each successful eval logs exactly one joined JSON line on the `langchain_harness.eval` logger. It carries no prompt or conversation text:

```json
{"event":"eval","scope":"session","run_id":"5f1c...","run_ids":["5f1c..."],"thread_id":"a3b9...","app_version":"build-42","scores":{"scope_adherence":8,"task_completion":9,"tool_use_quality":7},"verdict":"...","metrics":{"turn_count":1,"totals":{...},"turns":[...]}}
```

`scope` is `session` because the scores judge a session, not one turn: `run_id` is the last turn's run id (the join key back to the metrics line) and `run_ids` lists every turn the judge saw.

## Choosing the judge

The judge defaults to the main model, which means a session graded by the model that produced it. To grade with a different model:

```toml
# harness.toml
[eval]
model = "openai:gpt-4o-mini"
```

or `HARNESS_EVAL_MODEL=anthropic:claude-opus-4-5` (the env var wins over `harness.toml`). A bare id such as `gpt-4o-mini` uses the configured provider. Harness `base_url`/`api_key` are inherited only when the judge names the configured provider; a foreign provider uses its own env-var credentials. Unset means the main model judges.
