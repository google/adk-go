# GitHub Issue Triaging Agent

`githubtriagebot` is an autonomous ADK Go agent that triages open GitHub issues.
For each untriaged issue it:

1. **Sets the issue type** — `Bug`, `Feature`, or `Task`.
2. **Applies one categorization label** from a configurable allowlist
   (`bug`, `enhancement`, `documentation`, `question` by default).

An issue is considered **untriaged** when it has no issue type and/or none of
the allowlisted categorization labels. The agent is instructed to fill in only
what is missing and not to overwrite a type or label that is already set. The
type and label are also validated against an allowlist in Go (not just the
prompt), and the agent may only mutate issues it legitimately targeted.

This is a port of the Python `adk_triaging_agent` sample, re-architected to Go
idioms and to the labels/issue types actually used by `google/adk-go`.

## What it demonstrates

- An `llmagent.New` agent driven by typed `functiontool.New[Args, Result]` tools.
- Running headlessly with a `runner.Runner` + in-memory session, consuming the
  streaming `iter.Seq2[*session.Event, error]` response.
- Calling the GitHub REST API (`go-github`) and GraphQL API (a raw POST through
  the same authenticated client) from inside tools.
- A clean split between pure, table-tested decision logic (`triage.go`) and
  side-effecting I/O (`github.go`).
- Deterministic facts in code, fuzzy judgment in the model: code decides *which*
  issues are untriaged; the model decides *which* type and label fit.

## How it works

| File | Responsibility |
| --- | --- |
| `config.go` | Typed `Config` + env/flag parsing, no global state. |
| `triage.go` | Pure triage predicates (`needsTriage`, allowlist/type validation). |
| `github.go` | GitHub client: GraphQL reads, REST/raw-PATCH writes, dry-run chokepoint. |
| `tools.go` | The three typed agent tools. |
| `prompt.go` + `prompt_instruction.txt` | Embedded, brace-safe instruction with the type/label rubric. |
| `main.go` | Wires model → agent → runner and runs one headless turn. |

The agent has three tools: `list_untriaged_issues`, `change_issue_type`, and
`add_label_to_issue`. In a scheduled sweep it calls `list_untriaged_issues`
(newest first, capped at `ISSUE_COUNT`) and triages each result. For a single
issue (`-issue N`) the details are fetched up front and passed in the prompt.

## The agent loop

If you are new to ADK, this is the core flow to understand (`main.go`):

1. `llmagent.New` is given the model, the rendered instruction, and the tools.
   ADK reflects over each tool's argument struct (via `functiontool.New`) to
   build the JSON schema the model sees — **the Go arg struct is the tool's
   input contract**.
2. `runner.New` binds the agent to a `SessionService`. `runner.Run(...)` returns
   an `iter.Seq2[*session.Event, error]` (a Go 1.23 range-over-func); each
   iteration yields one streamed event or an error.
3. Internally, on each turn the model reads the prompt + tool schemas, may emit a
   tool call, the runner executes the matching Go handler, feeds the result back
   to the model, and loops until the model stops calling tools and returns text.
4. We consume that stream headlessly and keep the last text content as the final
   summary. We also fail loudly (return an error) if any event carries an error.

Two patterns worth copying live in the tool/error handling:

- A tool handler returns a **typed result the model can read**. Validation
  failures (e.g. a disallowed label) are returned as a result with
  `status: "error"` and a **nil Go error**, so the model sees the problem and
  can correct itself. Real I/O failures return a Go `error`.
- `OnToolErrorCallbacks` returns `(nil, nil)` — "observe only": it logs the
  failure (which is otherwise invisible to your program) without replacing the
  result the model receives.

> **Why GraphQL for reads but REST for writes?** Not an ADK convention — GitHub's
> issue *type* is not exposed by the REST API used by `go-github` v66, so reads
> use GraphQL (`issueType { name }`) and the type write is a raw `PATCH`. Labels
> use the regular REST endpoint.

## Prerequisites

- **Go 1.23+** (the run loop uses range-over-func).
- A **`GITHUB_TOKEN`** with `issues: write`. For a personal token: GitHub →
  *Settings* → *Developer settings* → *Personal access tokens* → grant the
  repository **Issues: Read and write** permission. In GitHub Actions the
  built-in `secrets.GITHUB_TOKEN` already has this when `permissions:` grants it.
- Either a **Gemini API key** or **Vertex AI** access (see below).

## Running locally

```bash
export GITHUB_TOKEN=<a token with issues:write>
export GEMINI_API_KEY=<your Gemini API key>   # or use Vertex AI (see below)

# Dry-run a single issue (no writes; logs intended actions).
go run ./examples/githubtriagebot -dry-run -issue 123

# Dry-run a sweep of the backlog.
go run ./examples/githubtriagebot -dry-run

# Act for real (omit -dry-run).
go run ./examples/githubtriagebot -issue 123
```

Instead of an API key you can use Vertex AI via Application Default Credentials:

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=<project>
export GOOGLE_CLOUD_LOCATION=<location>   # e.g. global
```

> **Dry-run is not offline.** `-dry-run` still reads GitHub (search / issue
> fetch) and still calls the model; it only suppresses the writes (label and
> type changes), logging `would …` instead.

A dry-run sweep produces output like:

```
level=INFO msg="running in dry-run mode; no issues will be modified"
level=INFO msg="[dry-run] would set issue type to \"Bug\"" issue=978
level=INFO msg="[dry-run] would add label \"bug\"" issue=978
level=INFO msg="triage complete" summary="Issue #978: ... Type set to Bug; label bug added."
```

## Tests

The package is unit-tested without calling a real LLM or GitHub: pure logic is
table-driven (`triage_test.go`), the GitHub client is exercised with `httptest`
(`github_test.go`), and the tool layer's allowlist/authorization gates are
verified to reject bad input without making any HTTP call (`tools_test.go`).

```bash
go test ./examples/githubtriagebot/...
```

## Configuration

| Variable / flag | Default | Description |
| --- | --- | --- |
| `GITHUB_TOKEN` | — (required) | Token with `issues:write`. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Gemini API key (or use Vertex AI). |
| `OWNER` | `google` | Repository owner. |
| `REPO` | `adk-go` | Repository name. |
| `LLM_MODEL_NAME` | `gemini-3.5-flash` | Model to use. |
| `ALLOWED_LABELS` | `bug,enhancement,documentation,question` | Categorization label allowlist. |
| `ISSUE_COUNT` | `3` | Max issues per scheduled sweep (newest first). |
| `FRESHNESS_WINDOW_DAYS` | `0` (off) | Restrict the sweep to issues created within N days. |
| `ISSUE_TIMEOUT` | `5m` | Bounds a single agent run. |
| `-dry-run` / `DRY_RUN` | `false` | Log intended actions without mutating. |
| `-issue` | `0` | Triage only this issue (0 = sweep). |

## GitHub Actions

`.github/workflows/triage-bot.yml` runs the agent on new issues (`opened`) and
on a 6-hourly schedule, with a manual `workflow_dispatch` that defaults to
dry-run. It uses the built-in `GITHUB_TOKEN`, requires a `GEMINI_API_KEY`
secret, requests least-privilege `permissions` (`issues: write`,
`contents: read`), and is guarded so it only runs on `google/adk-go`.

## Notes & future extensions

- **Issue types** must be enabled at the organization level (they are for the
  `google` org: Bug/Feature/Task).
- **Component labels / owner assignment** from the Python sample are
  intentionally omitted: `google/adk-go` does not yet use component labels, and
  owner routing is out of scope for this version. Both are natural extensions.
