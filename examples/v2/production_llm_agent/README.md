# production_llm_agent

The shape of an LLM agent ready for production traffic:

- `model/retry` wraps the model so transient 429 / 5xx errors are
  retried with exponential backoff + jitter.
- `plugin/builtin/Logging` emits structured slog records at every
  callback hook.
- `plugin/builtin/GlobalInstruction` prepends an app-wide safety
  instruction to every LLM request.

This example assembles those pieces against a deterministic stub LLM
that fails twice with a 503 before succeeding — so you can read the
retry log and the structured records the deployment would emit.
