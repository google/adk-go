# AGENTS.md

Context for AI coding agents (Claude Code, Gemini CLI, Cursor, Copilot, etc.)
working in the ADK Go repository. Human contributors should start with
CONTRIBUTING.md.

## Project overview

ADK Go (`google.golang.org/adk/v2`) is an open-source, code-first Go toolkit for
building, evaluating, and deploying AI agents. It is model-agnostic but
optimized for Gemini, and is one of several ADK implementations — Go, Python,
Java, Kotlin, and TypeScript — that share a conceptual model but are independent
codebases. Requires the Go version declared in `go.mod` (currently 1.26.5).

Development happens on `main`, the 2.x line. `v1` is the maintenance branch for
1.x; target it only for fixes that must ship to 1.x. See
[Branches](CONTRIBUTING.md#branches).

## Setup & core commands

This repo is multi-module: the root module `google.golang.org/adk/v2` plus
`plugin/agentanalytics`. Set up a Go workspace first — `go.work` is local-only
and gitignored, and `go work init` fails if one already exists:

```bash
test -f go.work || go work init
go work use -r .
```

Then run from the repo root. The `work` pattern spans every module in the
workspace, while `./...` matches only the module you are standing in:

- Build:       `go build -mod=readonly work`
- Test:        `go test -race -mod=readonly -count=1 -shuffle=on work`
- Single pkg:  `go test -race ./agent/...`
- Lint:        `golangci-lint run`   (per module; v2, CI pins v2.3.1; config in `.golangci.yml`)
- Tidy check:  `go mod tidy -diff`   (per module; must print nothing)
- Format:      `golangci-lint fmt`   (per module; applies gofumpt + goimports per config)

Without a `go.work`, `work` silently falls back to the root module alone and
still exits 0, so confirm the workspace exists before trusting a green run.
`golangci-lint` and `go mod tidy` are per-module either way: run them inside each
submodule too (for example, `cd plugin/agentanalytics`). CI does the same,
running build, test, tidy, and lint once per module, with `-v` on build and test.

## Definition of done

A change is complete only when all of these pass locally:

1. `go build` (above) succeeds.
2. `go test` (above) is green.
3. `golangci-lint run` reports no findings, in every module.
4. `go mod tidy -diff` prints nothing, in every module.
5. New/changed behavior has tests; a bug fix has a test that reproduces the bug.
6. Every new Go file starts with the Apache 2.0 license header (enforced by `goheader`).
7. The root `go.mod` does not require an in-repo submodule (enforced by the
   `guardrail` CI job).

## Repository layout

- `agent/`     Agent interface + types (`llmagent`, `remoteagent`, `workflowagent`;
  `workflowagents/` holds `loopagent`, `parallelagent`, `sequentialagent`)
- `runner/`    Execution engine that drives the run loop
- `workflow/`  Node/graph-based workflow engine for multi-agent apps
- `model/`     LLM abstraction (`gemini`, `apigee`, `openaimodel`)
- `tool/`      Tool/Toolset interface + built-in tools (incl. `skilltoolset/`, `mcptoolset/`)
- `session/`   Conversation state + events
- `memory/`, `artifact/`   Long-term memory and file/data services
- `auth/`      Credentials and auth providers for outbound requests
- `agentregistry/`  Client for Google Cloud Agent Registry (A2A agents, MCP servers, models)
- `plugin/`    Cross-cutting lifecycle hooks; `plugin/agentanalytics` is a separate module
- `server/`    HTTP servers (`adkrest` is primary; `adka2a`, `agentengine`)
- `cmd/`       CLI (`adkgo`) and server launchers
- `telemetry/`, `util/`   Public helper packages
- `platform/`  Overridable seams for time & UUID generation (deterministic tests)
- `internal/`  Private packages — NOT public API; `internal/httprr` is vendored
- `examples/`  Runnable example agents (quickstart, tools, a2a, skills, …)
- `scripts/`   Repo tooling (ADK Web container build and asset refresh)

## Conventions & idioms

- **Streaming:** agent runs return `iter.Seq2[*session.Event, error]`; consume
  with `for event, err := range … {}`. Don't collect events into a slice.
- **Interface-first:** public packages expose interfaces (`Agent`, `Tool`,
  `Toolset`, `Service`); concrete impls live in sub-packages or `internal/`.
- **Callbacks over subclassing** (`Before*`/`After*` for Agent/Model/Tool);
  returning non-nil from a `Before` callback short-circuits execution.
- **Errors:** wrap with `fmt.Errorf("…: %w", err)`. Use `%v` only when
  deliberately not exposing the wrapped error's type. Don't convert existing `%w` to `%v`;
  it might break callers silently. Wrap sentinels first:
  `fmt.Errorf("%w: …: %w", ErrX, err)`. Tool confirmation uses sentinel errors
  (e.g. `tool.ErrConfirmationRequired`).
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

## Multi-module development

See [Multi-Module Development](CONTRIBUTING.md#multi-module-development) in
`CONTRIBUTING.md` for policy, steps to add a new module, and release tagging.

## Testing

- Tests run **offline by default**: LLM HTTP traffic is replayed from
  `testdata/*.httprr` via `internal/httprr`. Never add live model or network
  calls to tests.
- `-httprecord` takes a **regexp matched against the cassette's file path**,
  not a `-run` test-name filter. Keep it as narrow as the set of cassettes you
  mean to replace: recording against a live model produces a different response
  every time, so a broad pattern rewrites unrelated cassettes and buries the
  intended change.
- To re-record **one** cassette — the normal case — supply real credentials
  (e.g. `GOOGLE_API_KEY`) and name it in both flags:
  `go test ./<pkg>/ -run TestFoo -httprecord='TestFoo\.httprr$'`.
  Commit only that `testdata/*.httprr`.
- To re-record a whole package, run `go generate ./<pkg>/...`; each package's
  `//go:generate go test -httprecord=…` directives are scoped so that every
  cassette is recorded exactly once (enforced by
  `TestHTTPRecordDirectivesPartitionCassettes` in `internal`).
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
- Other ADK implementations: [Python](https://github.com/google/adk-python),
  [Java](https://github.com/google/adk-java),
  [Kotlin](https://github.com/google/adk-kotlin),
  [TypeScript](https://github.com/google/adk-js)
