# production_llm_agent

Real Gemini agent wired with the production-grade plumbing: retry on
429/5xx model errors, structured logging at every callback, and an
app-wide safety instruction.

## Run

```
export GOOGLE_API_KEY=...
go run ./examples/v2/production_llm_agent/             # console
go run ./examples/v2/production_llm_agent/ web         # adk-web
```

## Try it

> What's the status of ORD-202?

You'll see in stdout:
- a `before_model` slog record (model = gemini-2.5-flash, agent =
  support_agent)
- the agent's `lookup_order` tool call → `before_tool` / `after_tool`
  records
- if Gemini returns a transient 503, `model_retry` warnings with the
  attempt count and backoff delay

The `GlobalInstruction` plugin prepends the safety string to every
LLM request, so refund/PII/medical guards apply uniformly.

## Wiring breakdown

```go
gemini, _ := gemini.NewModel(...)
wrapped := retry.Wrap(gemini, retry.Config{MaxAttempts: 4, ...})
agent, _ := llmagent.New(llmagent.Config{Model: wrapped, ...})
plugins := []*plugin.Plugin{loggingPlugin, globalInstructionPlugin}
launcher.Execute(ctx, &launcher.Config{
    AgentLoader: agent.NewSingleLoader(agent),
    PluginConfig: runner.PluginConfig{Plugins: plugins},
}, os.Args[1:])
```
