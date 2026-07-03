# AGENTS.md

Context for AI coding agents (Claude Code, Gemini CLI, Cursor, Copilot, etc.)
working in the ADK Go repository. Human contributors should start with
CONTRIBUTING.md.

## Project overview

ADK Go (`google.golang.org/adk/v2`) is an open-source, code-first Go toolkit for
building, evaluating, and deploying AI agents. It is model-agnostic but
optimized for Gemini, and is one of three ADK implementations — Go, Python, and
Java — that share a conceptual model but are independent codebases. Requires
Go 1.25+.

## Setup & core commands

Run from the repo root. These match what CI enforces (CI also passes `-v`):

- Build:       `go build -mod=readonly ./...`
- Test:        `go test -race -mod=readonly -count=1 -shuffle=on ./...`
- Single pkg:  `go test -race ./agent/...`
- Lint:        `golangci-lint run`   (golangci-lint v2; CI pins v2.3.1; config in `.golangci.yml`)
- Tidy check:  `go mod tidy -diff`   (must print nothing)
- Format:      `golangci-lint fmt`   (applies gofumpt + goimports per config)

## Definition of done

A change is complete only when all of these pass locally:

1. `go build` (above) succeeds.
2. `go test` (above) is green.
3. `golangci-lint run` reports no findings.
4. `go mod tidy -diff` prints nothing.
5. New/changed behavior has tests; a bug fix has a test that reproduces the bug.
6. Every new Go file starts with the Apache 2.0 license header (enforced by `goheader`).

## Repository layout

- `agent/`     Agent interface + types (`llmagent`, `workflowagent(s)`, `remoteagent`)
- `runner/`    Execution engine that drives the run loop
- `workflow/`  Node/graph-based workflow engine for multi-agent apps
- `model/`     LLM abstraction (`gemini`, `apigee`)
- `tool/`      Tool/Toolset interface + built-in tools (incl. `skilltoolset/`, `mcptoolset/`)
- `session/`   Conversation state + events
- `memory/`, `artifact/`   Long-term memory and file/data services
- `plugin/`    Cross-cutting lifecycle hooks
- `server/`    HTTP servers (`adkrest` is primary; `adka2a`, `agentengine`)
- `cmd/`       CLI (`adkgo`) and server launchers
- `telemetry/`, `util/`   Public helper packages
- `platform/`  Overridable seams for time & UUID generation (deterministic tests)
- `internal/`  Private packages — NOT public API; `internal/httprr` is vendored
- `examples/`  Runnable example agents (quickstart, tools, a2a, skills, …)

## Conventions & idioms

- **Streaming:** agent runs return `iter.Seq2[*session.Event, error]`; consume
  with `for event, err := range … {}`. Don't collect events into a slice.
- **Interface-first:** public packages expose interfaces (`Agent`, `Tool`,
  `Toolset`, `Service`); concrete impls live in sub-packages or `internal/`.
- **Callbacks over subclassing** (`Before*`/`After*` for Agent/Model/Tool);
  returning non-nil from a `Before` callback short-circuits execution.
- **Errors:** wrap with `fmt.Errorf("…: %w", err)`. Tool confirmation uses
  sentinel errors (e.g. `tool.ErrConfirmationRequired`).
- Prefer an existing helper over a new one; keep packages small and focused.

## Minimal example

```go
model, err := gemini.NewModel(ctx, "gemini-2.5-flash",
    &genai.ClientConfig{APIKey: os.Getenv("GOOGLE_API_KEY")})
// handle err
a, err := llmagent.New(llmagent.Config{
    Name:        "assistant",
    Model:       model,
    Instruction: "You are a helpful assistant.",
    Tools:       []tool.Tool{ /* ... */ },
})
// handle err
r, err := runner.New(runner.Config{
    AppName:           "my-app",
    Agent:             a,
    SessionService:    session.InMemoryService(),
    AutoCreateSession: true,
})
// handle err
msg := genai.NewContentFromText("Hello", genai.RoleUser)
for event, err := range r.Run(ctx, userID, sessionID, msg, agent.RunConfig{}) {
    // handle err; read event.LLMResponse.Content
}
```

See `examples/quickstart` for a full runnable program.

## Extending the framework

- **Add a tool:** wrap a Go function with
  `functiontool.New[Args, Results](cfg, handler)` (Args/Results are structs), or
  implement the `tool.Tool` interface for full control.
- **Add a toolset:** implement `tool.Toolset`; its `Tools(ctx)` may return
  different tools per invocation.
- **Add an agent type:** follow the `agent/workflowagents/*` packages; construct
  agents via `llmagent.New` / `agent.New`, not by implementing `agent.Agent`
  directly.
- **Add cross-cutting behavior:** register a `plugin.New(plugin.Config{...})`
  hook (`Before*`/`After*` for run/agent/model/tool) instead of editing the loop.

## Testing

- Tests run **offline by default**: LLM HTTP traffic is replayed from
  `testdata/*.httprr` via `internal/httprr`. Never add live model or network
  calls to tests.
- To (re)record a package's traffic, supply real credentials (e.g.
  `GOOGLE_API_KEY`) and run `go generate ./<pkg>/...` (it runs
  `go test -httprecord=…`); commit the updated `testdata/*.httprr`.
- Prefer table-driven tests; shared helpers live in `internal/testutil`.

## Boundaries

**Always**
- Run build, tests, lint, and `go mod tidy -diff` before declaring done.
- Keep PRs small and focused — one concern per PR.
- Add or update tests for the code you change.

**Ask first**
- Adding or upgrading a dependency (`go.mod`).
- Changing a high-fan-in package (`session`, `agent`, `model`, `tool`,
  `runner`) — prefer additive, backward-compatible changes.
- Any change to the public API surface, and any breaking change.

**Never**
- Break the public API — keep changes backward-compatible.
- Edit vendored code (`internal/httprr`) or commit secrets / API keys.
- Add tests that make live LLM or network calls.

## PRs & commits

See `CONTRIBUTING.md` for the full process and CLA. Key points for agents:
most PRs (beyond trivial docs/typos) need a linked issue; include a **Testing
Plan**; attach logs or screenshots for behavior changes (Runner output / ADK Web).

## Alignment with adk-python

[adk-python](https://github.com/google/adk-python) is the source of truth for
feature behavior. When porting or validating a feature, check parity with the
Python implementation.

## Resources

- Docs: https://google.github.io/adk-docs/
- Examples: `./examples`
- Java ADK: https://github.com/google/adk-java
