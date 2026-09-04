# AGENTS.md

Context for AI coding agents (Claude Code, Gemini CLI, Cursor, Copilot, etc.)
working in the ADK Go repository. Human contributors should start with
CONTRIBUTING.md.

## Project overview

ADK Go (`google.golang.org/adk/v2`) is an open-source, code-first Go toolkit for
building, evaluating, and deploying AI agents. It is model-agnostic but
optimized for Gemini, and is one of several ADK implementations — Go, Python,
Java, Kotlin, and TypeScript — that share a conceptual model but are independent
codebases. Requires the Go version declared in `go.mod`.

Development happens on `main`, the 2.x line. `v1` is the maintenance branch for
1.x; target it only for fixes that must ship to 1.x. See
[Branches](CONTRIBUTING.md#branches).

## Skills

See [AI-assisted development](CONTRIBUTING.md#ai-assisted-development) in
`CONTRIBUTING.md` for what this repo ships. The rule for agents: task-specific
instructions live in `.agents/skills/<name>/SKILL.md`, and you read the matching
one before starting that kind of work.

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
- Format:      `golangci-lint fmt`   (per module; applies gofumpt + goimports)

Without a `go.work`, `work` silently falls back to the root module alone and
still exits 0, so confirm the workspace exists before trusting a green run.

Install the same `golangci-lint` CI uses. A newer release reports findings CI
does not, which reads as a failure you did not cause:

```bash
go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.3.1
```

`golangci-lint` and `go mod tidy` are per-module, so a run from the repo root
covers only the root module. This loop covers every module the way CI does:

```bash
for m in $(find . -name go.mod -not -path './.git/*' -exec dirname {} \;); do
  ( cd "$m" \
    && go mod tidy -diff \
    && go build -mod=readonly ./... \
    && go test -race -mod=readonly -count=1 -shuffle=on ./... \
    && golangci-lint run ) || echo "FAILED: $m"
done
```

`golangci-lint run` reports formatting problems too, as `gofumpt` and
`goimports` findings, but it does not fix them. `golangci-lint fmt` rewrites
the files in place, and `golangci-lint fmt --diff` shows what it would change
without writing anything.

## Definition of done

A change is complete only when all of these pass locally:

1. `go build` (above) succeeds.
2. `go test` (above) is green.
3. `golangci-lint run` reports no findings, in every module.
4. `go mod tidy -diff` prints nothing, in every module.
5. New/changed behavior has tests, and each one has been seen to fail with the
   source change reverted. See [Before you open a PR](#before-you-open-a-pr).
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
- **Interface-first:** the core abstractions are interfaces — `agent.Agent`,
  `tool.Tool`, `tool.Toolset`, and a separate `Service` interface in each of
  `session`, `artifact` and `memory`. Concrete implementations live in
  sub-packages or `internal/`, except the in-memory ones, which sit beside the
  interface they implement.
- **Callbacks over subclassing** (`Before*`/`After*` for Agent/Model/Tool). A
  `Before` model or tool callback short-circuits the underlying call when it
  returns either a non-nil result or a non-nil error. `BeforeAgentCallback`
  behaves differently: only non-nil content short-circuits, and returning an
  error surfaces that error without stopping the agent from running.
- **Errors:** wrap with `fmt.Errorf("…: %w", err)`. Use `%v` only when
  deliberately not exposing the wrapped error's type. Don't convert existing `%w` to `%v`;
  it might break callers silently. Wrap sentinels first:
  `fmt.Errorf("%w: …: %w", ErrX, err)`. Tool confirmation uses sentinel errors
  (e.g. `tool.ErrConfirmationRequired`).
- Prefer an existing helper over a new one; keep packages small and focused.

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

- **LLM traffic is replayed, not live.** A package with `testdata/*.httprr`
  replays through `internal/httprr` with no flags and no credentials. Each of
  those files is a **cassette**: one recorded request-and-response exchange,
  captured once against a real model and replayed on every run afterwards, so
  tests stay deterministic and need no API key. (The term is used throughout
  this section and in the test name
  `TestHTTPRecordDirectivesPartitionCassettes`, though the `httprr` package
  itself never uses the word.) `session/vertexai` uses a second, unrelated
  system: `rpcreplay` with `testdata/*.replay` files, refreshed by
  `UPDATE_REPLAYS=true`. Never add a live model or network call to a test.
- `-httprecord` takes a **regexp matched against the cassette's file path**,
  not a `-run` test-name filter. Keep it as narrow as the set of cassettes you
  mean to replace: recording against a live model produces a different response
  every time, so a broad pattern rewrites unrelated cassettes and buries the
  intended change.
- To re-record **one** cassette — the normal case — supply real credentials
  (e.g. `GOOGLE_API_KEY`) and name **the cassette file**, which is often not
  the name of a top-level test. Most cassettes here are subtest-derived, so a
  pattern built from the top-level test name matches nothing, records nothing,
  and still exits 0. List the directory first, then name the file exactly:
  ```bash
  ls <pkg>/testdata/*.httprr
  go test ./<pkg>/ -run TestToolCallback \
      -httprecord='TestToolCallback_before_callback_response_used\.httprr$'
  ```
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
- Break the public API — keep changes backward-compatible. An `apidiff` check
  compares every module against the merge base and fails on an incompatible
  change. A `breaking-change` label downgrades that failure to a report, so
  applying it is a deliberate decision to make with a maintainer, not a way
  round a red check.
- Edit vendored code (`internal/httprr`) or commit secrets / API keys.
- Add tests that make live LLM or network calls.

## Before you open a PR

Run the module loop under [Setup & core commands](#setup--core-commands) first.
Then three things no command does for you.

**1. Prove your tests fail without your fix.** Revert your source change, keep
your tests, and run the package. If it stays green your test pins nothing, which
is the most common defect found in review here. Restore the change, then do the
same to each new guard, branch and error path in turn: delete or invert it,
re-run, confirm the suite goes red, put it back. A test you have never seen fail
is not yet a test.

**2. Say what changes for someone already on the current release.** If they
upgrade to a build containing your change, what behaves differently? Write it in
the PR description in plain terms. This includes bug fixes: a fix that alters an
observable result is still a behavior change, and an undeclared one is the
second most common defect found in review. If nothing changes for an existing
user, say that instead.

**3. Title the PR as a Conventional Commit.** `feat:`, `fix:`, `docs:` and so
on, optionally scoped as `fix(runner):`. Release tooling reads the landed
subject to choose the next version and build the release notes, and it skips
anything with no recognized type without reporting an error. A mistitled PR
merges green and then goes missing from the changelog.

## PRs & commits

See `CONTRIBUTING.md` for the full process and CLA. Key points for agents:

- Code follows the
  [Google Go Style Guide](https://google.github.io/styleguide/go/index).
- Most PRs, beyond trivial docs and typo fixes, need a linked issue.
- Include a **Testing Plan**.
- Attach logs or screenshots for behavior changes (Runner output / ADK Web).

## Alignment with adk-python

[adk-python](https://github.com/google/adk-python) is the source of truth for
feature behavior. When porting or validating a feature, check parity with the
Python implementation.

## Resources

- Docs: https://google.github.io/adk-docs/
- Examples: `./examples` — start with `examples/quickstart` for a full runnable
  program
- Other ADK implementations: [Python](https://github.com/google/adk-python),
  [Java](https://github.com/google/adk-java),
  [Kotlin](https://github.com/google/adk-kotlin),
  [TypeScript](https://github.com/google/adk-js)
