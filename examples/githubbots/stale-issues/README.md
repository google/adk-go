# Stale-Issue Auditor Bot

An autonomous [ADK Go](https://github.com/google/adk-go) agent that audits open
GitHub issues for staleness. Unlike a timestamp-only "stale bot", it
reconstructs each issue's full conversation history and uses a model to tell the
difference between a maintainer **asking the author a question** (a stale
candidate) and a maintainer **posting a status update** (still active).

It runs in this repository from
[`.github/workflows/stale-issues-bot.yml`](../../../.github/workflows/stale-issues-bot.yml).

## What it demonstrates

- An `llmagent.New` agent driven by typed `functiontool.New[Args, Result]` tools.
- Running headlessly with a `runner.Runner` + in-memory session, consuming the
  streaming `iter.Seq2[*session.Event, error]` response.
- Calling the GitHub REST (`go-github`) and GraphQL APIs from inside tools.
- **Bounded concurrency** with `errgroup`, deterministic decisions
  (`Temperature: 0`), and a clean split between **pure, unit-tested logic**
  (`state.go`) and **side-effecting I/O** (`github.go`).
- Giving an agent write authority over a public issue tracker **without**
  trusting the model — see [Authority limits](#authority-limits) below.

## How it works

For each candidate issue the agent:

1. Calls `get_issue_state`, which issues one GraphQL query (comments,
   description edits, title renames, reopen/label events), replays the history
   to find the **last human actor**, and computes staleness.
2. Follows a decision tree (`prompt_instruction.txt`):
   - **Author/other replied** → remove the stale label (active again); if they
     edited the description silently, alert maintainers.
   - **Maintainer asked a question** and the stale threshold passed → mark stale
     (warning comment + label).
   - **Stale long enough** → close as *not planned*.
   - **Maintainer status update / internal discussion** → no action.

Mutations are ordered for safe re-runs (label before comment; close before
comment) and the bot recognizes its own prior comments to avoid spam.

## Authority limits

The bot reads text any stranger on the internet can write and holds a token
scoped `issues: write`. So the model is treated as fully attacker-controlled,
and the question the design answers is what the Go code still refuses when it
is. Authority that exists only because the prompt asks for it is not authority.

- **Every decision is split.** The mechanical half — arithmetic and state
  checks — is enforced in Go by `stalePredicate`, `closePredicate`,
  `removeStalePredicate` and `alertPredicate`, evaluated against the state the
  bot computed from API metadata, never against what the model asserts. Only the
  genuine judgement (is this maintainer comment actually blocked on the author?)
  is left to the prompt. Injected text can at worst make the bot decline to act.
- **A session is scoped to one issue.** `withAuditedIssue` binds the audited
  issue number to the context and every tool checks it first, so issue A's text
  can never make a tool touch issue B.
- **Each precondition and its claim share one critical section.**
  `claimAction` tests and consumes a per-`(issue, action)` claim under one lock,
  so a duplicate tool call cannot act twice on one observation. Labels are
  idempotent; the comments that accompany them are not.
- **The one attacker-controlled field is fenced.** `last_comment_text` reaches
  the model inside a per-issue `[UNTRUSTED:<hex>] … [/UNTRUSTED:<hex>]` marker
  drawn from `crypto/rand`. Drawing it fails closed: a predictable marker would
  let an attacker write the closing marker themselves. Every other field is
  computed here and is emitted outside the fence.
- **`add_label_to_issue` refuses the stale label.** Marking stale is gated on
  the thresholds through `add_stale_label_and_comment`, and this tool is not, so
  applying the same label here would be a way around that gate: the issue would
  become stale with no warning comment and its close clock already running.
- **The tool set is pinned by a test.** Six tools, asserted exactly, so a
  seventh ungated tool cannot be added without a deliberate decision.
- **The agent and runner are built per issue, not per run.** `runner.Run` writes
  the root agent's mode on its first call, so one agent shared across the
  concurrent sweep is a data race — reproduced under `-race` and pinned by
  `TestConcurrentAuditsDoNotShareAgentState`. The model is shared deliberately:
  it holds the HTTP client and its fields are never written after construction.
- **Three clocks, each answering one question.** `days_since_activity` is
  anything happening on the issue. `days_since_last_actor_action` ages the event
  that decided whose turn it is, so a comment edit by a passer-by does not look
  like the conversation moving. `days_since_author_action` ages the issue
  author's own actions, edits included, because someone who edits their comment
  to add the logs a maintainer asked for has answered. Every Go gate reads the
  last two; none reads the first, and the prompt names the same fields the gates
  do — pointing the model at a different one silently undoes the gate.
- **An unknown stale-label age refuses the close.** When the labelling event has
  scrolled out of the bounded timeline window the age is reported as `-1` and
  the issue waits for a run that can see the event, rather than being closed on
  a guess the author can force by padding the window.

## Running in GitHub Actions

The workflow runs daily and can be dispatched manually. Before it can do
anything, set one repository variable:

| Repository variable | Purpose |
| --- | --- |
| `STALE_BOT_MAINTAINERS` | Comma-separated maintainer logins. The built-in `GITHUB_TOKEN` cannot list collaborators, so the maintainer set has to be supplied. The job **fails loudly** when it is unset, because an empty list makes the bot a silent no-op: with nobody classified as a maintainer, no comment can put the ball in the author's court. |

and one secret:

| Secret | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Authenticates the model. |

Setting `STALE_BOT_MAINTAINERS` is the deliberate step that arms the bot. Until
then the scheduled run fails rather than quietly doing nothing.

A manual dispatch takes an optional `issue` number (blank sweeps the whole
backlog) and a `dry_run` toggle that **defaults to true**. The scheduled run
writes for real.

The job's `timeout-minutes` sits above the bot's own `RUN_BUDGET`, so an
overrun on a large backlog is reported by the process — naming the budget, with
a non-zero exit — instead of the runner killing the job mid-sweep.

## Running locally

Requires **Go 1.26+** (see `go.mod`). Copy `.env.example` to `.env` and fill it
in (set `MAINTAINERS` — without it the bot will never mark issues stale), then:

```bash
# Dry-run the whole backlog (no writes; logs intended actions).
go run . -dry-run

# Dry-run a single issue.
go run . -dry-run -issue 123

# Act for real (omit -dry-run).
go run .
```

> **Dry-run is not offline.** It still reads GitHub and calls the model; it only
> suppresses writes.

## Configuration

| Variable / flag | Default | Description |
| --- | --- | --- |
| `GITHUB_TOKEN` | — (required) | Token with `issues: write`. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Gemini API key (or use Vertex AI). |
| `BOT_LOGIN` | — | The login the bot posts under, so it recognizes its own comments. Set it to `github-actions[bot]` in Actions: `GET /user` is a user-to-server endpoint and the workflow token is an installation token, so discovering the login is refused there. Leave it empty locally to have it resolved from the API. |
| `MAINTAINERS` | — | Comma-separated maintainer logins (the token can't list collaborators). Without it, no comment counts as maintainer activity, so nothing is ever marked stale. |
| `OWNER` | — (required) | Repository owner. |
| `REPO` | — (required) | Repository name. |
| `LLM_MODEL_NAME` | `gemini-flash-latest` | Model to use. |
| `STALE_HOURS_THRESHOLD` | `336` (14d) | Time waiting on the author before warning. |
| `CLOSE_HOURS_AFTER_STALE_THRESHOLD` | `168` (7d) | Time stale before closing. |
| `STALE_LABEL_NAME` | `stale` | Label applied when marking stale. |
| `REQUEST_CLARIFICATION_LABEL` | `request clarification` | Label that flags "waiting on author". |
| `CONCURRENCY_LIMIT` | `3` | Max issues audited in parallel. |
| `MAX_ISSUES` | `100` | Max candidates one sweep may audit. The search is ordered oldest-first, so the remainder is picked up by the next run. |
| `MAX_DESTRUCTIVE_ACTIONS` | `20` | Max issues one run may mark stale or close. Corrective actions (removing a label, alerting a maintainer) are never blocked by it. Hitting the ceiling fails the run. |
| `ISSUE_TIMEOUT` | `5m` | Bounds a single issue's audit. |
| `RUN_BUDGET` | `30m` | Bounds the whole run. Keep it below the workflow's `timeout-minutes`. |
| `-dry-run` / `DRY_RUN` | `false` | Log intended actions without mutating. |
| `-issue` | `0` | Audit only this issue (0 = sweep). |

Instead of an API key you can use Vertex AI via Application Default Credentials
(`GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`).

Every label comparison in the bot is case-insensitive, because GitHub label
names are case-insensitively unique but keep the casing they were created with:
`STALE_LABEL_NAME=stale` has to recognize a label the repository stored as
`Stale`. For the same reason the two configured names must differ
case-insensitively, and `validate()` rejects a configuration where they collide.

A malformed value for any of the numeric or boolean settings fails the run
rather than falling back to the default. `DRY_RUN=yes` is the case that matters:
the default is to write for real, so a silent fall-back would turn a requested
dry run into live labeling, commenting and closing.

## Tests

Pure decision logic is table-driven (`state_test.go`); the GitHub client is
exercised with `httptest` (`github_test.go`); the authority gates, the atomic
claim and the pinned tool set have their own suites (`authority_test.go`,
`claims_test.go`).

```bash
go test -race -count=1 -shuffle=on ./...
```

### End-to-end tests

`e2e_test.go` drives the real Gemini model, the real prompt and the real tools
against a fake GitHub, and asserts on the writes that actually reach it. It
covers every branch of the decision tree plus two prompt-injection attempts.

It is opt-in twice over. The file always compiles, so it cannot rot, but it
never calls a paid API unless you ask it to:

```bash
STALE_BOT_E2E=1 GEMINI_API_KEY=... go test -run TestE2E -timeout 30m .
```

Each scenario first asserts that its fixture produces the `IssueState` the
branch is about, so a drifted fixture reports itself as a fixture bug rather
than as a model failure. A scenario whose model call keeps returning 503 is
reported as skipped, not failed: an unavailable model proves nothing either way.

## Notes

- The `MAINTAINERS` list must be supplied explicitly — the repo-scoped
  `GITHUB_TOKEN` cannot read collaborators/teams at runtime. It is matched on
  the GitHub login, which is mutable: GitHub releases a login for
  re-registration after a rename, so keep the list current when someone changes
  their username.
- Thresholds are policy: the defaults (14d to stale, 7d to close) suit OSS
  volunteer cadence; tune them via the environment.
- Candidates come from a search restricted to issues older than the stale
  threshold, so a silent description edit on a very recent issue is not
  detected.
- Untrusted comment text is capped before it reaches the model, keeping the
  start and the end of a long comment. A maintainer's question is usually the
  last thing in a quoted reply, so a head-only cut would drop it.
- One state the bot cannot resolve on its own: an issue that is stale, whose
  last actor is not a maintainer, and where nobody has acted since the label.
  The close needs a maintainer to have acted last and the removal needs someone
  to have come back, so both refuse. It is reached by a maintainer leaving
  `MAINTAINERS` or renaming, which reclassifies their earlier comment. Nothing
  is written either way, and the run logs a warning naming the issue so a
  maintainer can clear the label.
