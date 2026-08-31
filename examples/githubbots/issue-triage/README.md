# Issue Triage Bot

An autonomous [ADK Go](https://github.com/google/adk-go) agent that triages open
GitHub issues. For each untriaged issue it:

1. **Sets the issue type** — `Bug`, `Feature`, or `Task`.
2. **Applies one categorization label** from a configurable allowlist
   (`bug`, `enhancement`, `documentation`, `question` by default).

An issue is considered **untriaged** when it has no issue type and/or none of
the allowlisted categorization labels.

## Authority limits

The bot runs a model over an issue title and body, which anyone on the internet
can write, while holding a token that can write issues. So assume the model does
whatever an attacker asks, and read what the Go code still refuses:

- **One agent session per issue.** The work set is chosen in Go, not by a
  model-callable list tool, so two issues' text never share a context.
- **A per-session issue scope.** `withAuditedIssue` binds the session to one
  issue number and every mutating tool checks it first, so issue A's body cannot
  make a tool act on issue B.
- **An atomic need-claim, and a re-read before the write.** The precondition
  ("this field is still empty") and the reservation are taken in one critical
  section, so of several concurrent calls for the same issue exactly one reaches
  the API, and the claim is never re-opened — one write per field per run,
  whatever the response says. Because a sweep can reach its last issue up to
  `SWEEP_TIMEOUT` after reading it, the issue is re-read immediately before the
  write as well, so a field a maintainer filled inside that window is not
  clobbered by a decision taken against a stale snapshot.
- **Allow-listed values.** The type must be one of `Bug`/`Feature`/`Task` and
  the label one of the configured allowlist, matched case-insensitively and sent
  to GitHub in the allowlist's own spelling.
- **Untrusted text inside a nonce fence.** The title and body each go to the
  model inside `[UNTRUSTED:<hex>] … [/UNTRUSTED:<hex>]`, with a fresh
  `crypto/rand` marker per field per issue, so a body cannot write its own
  closing marker and have the text after it read as instructions. A failure to
  draw the nonce aborts the issue rather than falling back to a fixed marker.
- **One dry-run chokepoint that cannot fail open.** Every mutation passes
  through `shouldSkip`, so `-dry-run` cannot be forgotten on a new call site,
  and a malformed `DRY_RUN` aborts at startup rather than falling back to the
  default — which is "act for real".
- **A pinned tool inventory.** A test asserts the exact tool set, so an ungated
  tool reaching the same state cannot be added silently.

## What it demonstrates

- An `llmagent.New` agent driven by typed `functiontool.New[Args, Result]` tools.
- Running headlessly with a `runner.Runner` + in-memory session, consuming the
  streaming `iter.Seq2[*session.Event, error]` response.
- Calling the GitHub REST API (`go-github`) and GraphQL API (a raw POST through
  the same authenticated client) from inside tools.
- A clean split between pure, table-tested decision logic (`triage.go`) and
  side-effecting I/O (`github.go`): deterministic facts in code, fuzzy
  classification in the model.

## The agent loop

If you are new to ADK, this is the core flow (`main.go`):

1. `llmagent.New` is given the model, the rendered instruction, and the tools.
   ADK reflects over each tool's argument struct (via `functiontool.New`) to
   build the JSON schema the model sees — **the Go arg struct is the tool's
   input contract**.
2. `runner.New` binds the agent to a `SessionService`; `runner.Run(...)` returns
   an `iter.Seq2[*session.Event, error]` (a Go 1.23 range-over-func) yielding one
   streamed event or an error per iteration.
3. On each turn the model reads the prompt + tool schemas, may emit a tool call,
   the runner executes the matching Go handler, feeds the result back, and loops
   until the model stops calling tools and returns text.
4. We consume that stream headlessly, keep the last text as the summary, and
   return a non-nil error if any event carried one (so CI fails loudly).

Validation failures (e.g. a disallowed label) are returned to the model as a
result with `status: "error"` and a **nil Go error** so it can self-correct;
real I/O failures return a Go `error`. `OnToolErrorCallbacks` returns
`(nil, nil)` — "observe only" — to log failures that are otherwise invisible.

> **Why GraphQL for reads but REST for writes?** Not an ADK convention — GitHub's
> issue *type* is not exposed by the REST API in `go-github` v66, so reads use
> GraphQL (`issueType { name }`) and the type write is a raw `PATCH`. Labels use
> the regular REST endpoint.

## Running locally

Requires **Go 1.26+** (see `go.mod`). Copy `.env.example` to `.env` and fill it
in (or export the variables), then:

```bash
# Dry-run a single issue (no writes; logs intended actions).
go run . -dry-run -issue 123

# Dry-run a sweep of the backlog.
go run . -dry-run

# Act for real (omit -dry-run).
go run . -issue 123
```

> **Dry-run is not offline.** `-dry-run` still reads GitHub and still calls the
> model; it only suppresses writes, logging `would …` instead.

## Configuration

| Variable / flag | Default | Description |
| --- | --- | --- |
| `GITHUB_TOKEN` | — (required) | Token with `issues: write`. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Gemini API key (or use Vertex AI). |
| `OWNER` | — (required) | Repository owner. |
| `REPO` | — (required) | Repository name. |
| `LLM_MODEL_NAME` | `gemini-flash-latest` | Model to use. |
| `ALLOWED_LABELS` | `bug,enhancement,documentation,question` | Categorization label allowlist. |
| `ISSUE_COUNT` | `3` | Max issues per scheduled sweep (newest first). |
| `FRESHNESS_WINDOW_DAYS` | `0` (off) | Restrict the sweep to issues created within N days. |
| `ISSUE_TIMEOUT` | `5m` | Bounds a single agent run. |
| `SWEEP_TIMEOUT` | `15m` | Bounds the whole run, so N issues cannot multiply into N x `ISSUE_TIMEOUT` and overrun the job timeout. Must be at least `ISSUE_TIMEOUT`. |
| `-dry-run` / `DRY_RUN` | `false` | Log intended actions without mutating. |
| `-issue` | `0` | Triage only this issue (0 = sweep). |

A value the bot cannot parse is a startup error, not a silent fall back to the
default. `DRY_RUN=yes` is the case that matters — `strconv.ParseBool` rejects
it, and defaulting would mean live writes for an operator who asked for a
rehearsal.

Instead of an API key you can use Vertex AI via Application Default Credentials
(`GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`).

## In GitHub Actions

The bot runs from this repository against this repository, driven by
[`.github/workflows/issue-triage-bot.yml`](../../../.github/workflows/issue-triage-bot.yml).
That workflow checks the repo out, installs the Go version from this module's
`go.mod`, and runs `go run .` here. `OWNER`/`REPO` are derived from
`github.repository`, and the built-in `GITHUB_TOKEN` — scoped by the job's
`permissions:` — does the writing, so no PAT or GitHub App is needed.

Three triggers:

- **`issues: [opened]`** triages the new issue immediately.
- **A 6-hourly `schedule`** sweeps the backlog. It is a backstop: the workflow's
  concurrency group is constant, so GitHub keeps at most one run queued and a
  burst of newly opened issues can drop some. The sweep collects those.
- **`workflow_dispatch`** takes an optional issue number and a `dry_run` input
  that **defaults to true**, so a manual run is a rehearsal unless you say
  otherwise.

The concurrency group is deliberately *not* keyed on the issue number. The
no-double-action guard (the need-claim) lives in process memory, so two
concurrent processes would each see an unset field and both write it — which is
exactly what a per-issue group would allow between a scheduled sweep and an
event run for an issue that sweep is already processing.

The job holds `issues: write` and `contents: read`, and nothing else. The
workflow also overrides three of the defaults below, because the sweep budget
has to hold: `ISSUE_COUNT=5`, `ISSUE_TIMEOUT=3m`, `SWEEP_TIMEOUT=18m`, under a
job `timeout-minutes: 25`. Five issues at three minutes each is fifteen, inside
the eighteen-minute process budget, which is itself inside the job limit — so an
ordinary busy sweep finishes rather than reporting an exhausted budget, and a
genuine overrun stops and says what it left instead of being killed silently.

**One thing to confirm on the first live run.**
[GitHub requires *push* access to set an issue's type or
labels](https://docs.github.com/en/rest/issues/issues#update-an-issue) and
silently drops the change otherwise. Whether the job token's `issues: write`
satisfies that has not been confirmed here. The bot reads each write back and
**fails the run** if it did not land, so the gap surfaces loudly rather than
passing silently. Do not reach for `contents: write` as the reflex remedy: it
grants repository push authority to a job whose highest-volume trigger anyone on
the internet can fire, and it is not clear it changes issue-type authority at
all. Confirm what the token can actually do first.

Before enabling the schedule, run the workflow manually once with `dry_run: true`
and then once against a single issue for real.

## Tests

Pure logic is table-driven (`triage_test.go`); the GitHub client is exercised
with `httptest` (`github_test.go`, incl. GraphQL pagination and PR/NOT_FOUND
handling); the tool layer's allowlist, session-scope and need-claim gates are
verified to reject bad input without any HTTP call, and to admit exactly one
writer under concurrency (`tools_test.go`); the sweep, the fence and the run
budget are driven through the real functions via injected seams
(`main_test.go`).

```bash
go test ./...
```

## Notes

- **Issue types** must be enabled at the organization level (they are for the
  `google` org: Bug/Feature/Task). Setting a type — and adding a label — requires
  a token with **push access**; without it GitHub returns success but silently
  drops the change. The bot reads back each write and **fails the run** if the
  type/label was not actually applied, so a permissions gap surfaces loudly
  instead of passing silently. See the Actions section above for what to widen
  if that happens.
- **Component labels / owner assignment** from the original Python
  `adk_triaging_agent` are intentionally omitted. Both are natural extensions,
  but assignment in particular hands an attacker-influenced decision a much
  larger blast radius, so it would need its own allow-list of assignable logins.
- **There is no per-actor rate limit.** Anyone can open issues, and each one
  costs a model call. The bounds that exist are the constant concurrency group
  (one run at a time) and `ISSUE_COUNT` per sweep, which cap the spend but do
  not attribute it. A burst also pushes some event runs out of the queue, and
  because the sweep takes the newest issues first, a sustained burst can leave
  older untriaged issues unreached.
