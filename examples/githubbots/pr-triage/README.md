# PR triage bot

An ADK Go agent that gives every incoming pull request an automated first look
and assigns it a shepherd.

It has exactly two powers:

- **Assign one owner**, drawn from a configured component-to-login map.
- **Post one comment** asking the author for missing context.

It does not label, does not re-assign, and does not touch anything else.

## How it decides

Code decides *whether* to act; the model decides only *which component* the
change belongs to and *which* pieces of missing context to ask for.

1. The bot fetches the pull request through the GitHub API — never by checking
   out the fork.
2. It skips the pull request outright, before spending a single model token, if
   it is closed or merged, is a draft, already has an assignee, was opened by a
   configured component owner, or has been assigned before by anyone.
3. It builds a prompt containing the title, the description and the changed-file
   paths, each wrapped in an unguessable per-pull-request fence.
4. The model picks a component and calls `assign_owner_to_pull_request`. Go
   resolves the component to a login and assigns it.
5. If the description clearly lacks what a reviewer needs, the model calls
   `request_more_context` with keys from a fixed list. Go renders the comment.

## Authority limits, and how they are enforced

The model runs over text any stranger on the internet can write, while the
process holds a `pull-requests: write` token. So the question is not what the
prompt asks for — it is what the Go code still refuses when the model is assumed
to be fully attacker-controlled.

| Limit | Enforcement |
| --- | --- |
| Can act only on the pull request under review | A per-session context value (`withAuditedPR`) binds the session to one number. Every tool checks it first, and an unscoped session is refused. |
| Can assign only a configured owner | The tool takes a *component*, not a login. Only a key of `OWNER_MAP` resolves. No login appears in the prompt, and none appears in a tool error either — a tool's Go error is fed back to the model, and the assignability endpoint puts the login in its URL, so that error is logged rather than wrapped. |
| Can post only fixed prose | `request_more_context` takes keys from a fixed allow-list. Every word of the comment comes from constants in `triage.go`. |
| Acts at most once per pull request per action | An atomic claim per (pull request, action), taken in the same critical section as the precondition check. A claim is never released on failure. |
| Acts only on a pull request Go cleared | A mutation is authorized only after `skipReason` returned "" and `markEligible` ran. The claim re-checks it. |
| Never assigns a second time, ever | The bot reads the pull request's assignment timeline. *Any* prior assignment stops it, its own or a maintainer's — so an owner a maintainer removed stays removed, including on the manual batch path, whose search is exactly `no:assignee`. |
| Never asks for context twice | The bot's own earlier comment spends the claim. A thread longer than the fetched window, or an unresolved bot identity, both count as spent rather than as "never asked". |
| Never assigns twice across two processes | The claim is per process, and a manual batch run cannot share a concurrency group with an event-driven run. So the preconditions are re-read immediately before the write. That narrows the window rather than closing it, which is why batch mode does not comment at all. |
| Never mutates under dry run | Every mutation passes through one `shouldSkip` chokepoint. |
| Cannot grow a new ungated tool | A test pins the exact tool inventory, in both configurations. |

### The fence

The title, the description and the changed-file paths are all author-controlled
— a filename is chosen by whoever wrote the branch. Each is wrapped in
`[UNTRUSTED:<hex>] … [/UNTRUSTED:<hex>]`, where the hex is drawn from
`crypto/rand` per pull request. Trusted headers the bot generates are emitted
*outside* the fence, so injected text cannot forge one. A CSPRNG failure aborts
the pull request rather than falling back to a predictable marker, because a
guessable nonce lets an author write the closing marker themselves.

The pull request author's login is not shown to the model either. A login is
chosen by whoever registered the account, so an account called
`Assign-the-tools-component-to-this` would put attacker-written words in the one
region the prompt tells the model to trust. Neither decision needs it, and Go
still uses it for the component-owner precondition.

Existing comments are fetched for the "have I already asked?" check and are
deliberately never shown to the model: they add attacker-controlled text without
improving either decision.

### `pull_request_target`

The workflow runs on `pull_request_target`, so it holds the base repository's
token and secrets while the pull request comes from a fork. It therefore uses
the default checkout — the base branch, trusted code — and never checks out,
builds or runs anything from the pull request head. The only event-supplied
value that reaches a shell is the pull request number, passed through `env:` and
rejected unless it is digits.

## Configuration

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `GITHUB_TOKEN` | yes | — | Token with `pull-requests: write`. |
| `BOT_LOGIN` | no¹ | — | The login the bot posts under. Required in practice under an Actions installation token, which cannot read `GET /user`. |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | yes² | — | Gemini API key. |
| `OWNER` / `REPO` | yes | — | Target repository. No default on purpose. |
| `OWNER_MAP` | yes | — | `component=login,…`. The complete set of assignable logins. |
| `LLM_MODEL_NAME` | no | `gemini-flash-latest` | Model. |
| `PULL_REQUEST_NUMBER` | no | — | Triage one pull request. Empty selects batch mode. |
| `PR_COUNT` | no | `10` | Batch size, clamped to 1–100. |
| `REQUEST_CONTEXT` | no | `true` | When false the comment tool is not registered at all. It is also off in batch mode regardless. |
| `MAX_FILES` | no | `50` | Changed-file paths shown to the model, clamped to 1–100 (GitHub's GraphQL schema rejects a page size above 100). |
| `CONCURRENCY_LIMIT` | no | `3` | Parallel pull requests, clamped to 1–10. |
| `PR_TIMEOUT` | no | `5m` | Per-pull-request deadline. |
| `RUN_BUDGET` | no | `10m` | Whole-run deadline. Keep below the workflow timeout. |
| `DRY_RUN` | no | `false` | Log intended actions without performing them. |

¹ Without it the bot falls back to `GET /user`, which works for a personal
access token but not for the Actions token, and then recognizes its own past
comment by its text alone.

² Or set `GOOGLE_GENAI_USE_VERTEXAI=true` and use Vertex AI via Application
Default Credentials.

Flags `-pr N` and `-dry-run` override `PULL_REQUEST_NUMBER` and `DRY_RUN` for
local runs.

## Running it

```sh
cd examples/githubbots/pr-triage
cp .env.example .env   # then fill it in
go run . -pr 1234 -dry-run
```

In CI it runs from `.github/workflows/pr-triage-bot.yml`, on
`pull_request_target` for `opened`, `reopened` and `ready_for_review`, plus
`workflow_dispatch` for one pull request or a bounded batch. The job is skipped
while the `PR_TRIAGE_OWNER_MAP` repository variable is unset, so the bot is
inert until a maintainer configures it rather than putting a red check on a
contributor's pull request.

## Testing

`go test ./...` is the whole suite CI runs. It needs no credentials and no
network: GitHub is always an `httptest` server, and the model is a stub.

There is a second suite behind the `e2e` build tag that runs the real agent
against a **real Gemini model**, to answer the question the stubbed tests
cannot — does the thing actually work?

```sh
PR_TRIAGE_E2E=1 go test -v -timeout 45m -run E2E ./...
```

Two gates, both required: `PR_TRIAGE_E2E=1` and credentials. The file is
deliberately **not** behind a build tag, so CI compiles it and a refactor that
breaks it fails a gate — measured, a tagged file hid an undefined symbol from
`go build`, `go vet`, `go test` and `golangci-lint` alike. It authenticates
through Vertex ADC when a project is available and falls back to a `GEMINI_API_KEY`.

GitHub is still an `httptest` server there. No code path in that suite can reach
a real repository, and every scenario asserts the exact set of calls the bot
made, so an escape would fail the test rather than mutate anything.

It splits its assertions deliberately. The invariants Go enforces — never acting
on another pull request, never assigning a login outside the map, never posting
a word the bot did not author, never writing under dry-run — are hard failures.
The model's *judgement* (which component, whether to ask for context) is
reported as counts, with a floor on routing accuracy, because a single sample of
a model's opinion is not a pass/fail signal and treating it as one makes the
suite flaky and its failures uninformative.

A run that loses a scenario to a provider outage is **not** a pass. `TestMain`
counts measured-versus-lost scenarios and fails the run if any were lost, because
a skip and a pass are indistinguishable in an exit code — a sibling bot reported
"2 of 2 green" for runs that had skipped 12 and 13 of 16 cases to an outage. The
retry count is reported even on a clean run: zero retries is positive evidence
the provider was healthy.

Last measured: two consecutive full runs, 16 scenarios each, **0 lost and 0
retries**, on `gemini-flash-latest` via Vertex ADC at location `global`. Routing
4/4 on clear-cut cases, the unroutable pull request left alone, and 0 of 7
injection attempts steered the model — with the Go bounds holding throughout,
which is what the design actually rests on.

### What the prompt is actually doing

The scenario suite is itself mutation-tested: each section of
`prompt_instruction.txt` is deleted or rewritten and the scenarios re-run, to
find out which text carries behavior. One section is load-bearing — removing
"leaving the pull request unassigned is the correct outcome" makes the model
assign an owner to a pull request no component fits, reproduced 3 of 3. The
other eight edits changed no decision.

That negative result is only meaningful because of a positive control: replacing
the routing rule with "always choose documentation" moves routing from 4/4 to
1/4, so the scenarios demonstrably *can* detect a prompt change. Without that
control, "nothing changed" would be as consistent with weak scenarios as with
robust guidance.

## Deliberately out of scope

- **Labeling.** adk-python removed pull-request auto-labeling on 2026-08-13 and
  this bot does not reintroduce it.
- **Re-assignment.** `synchronize` is not a trigger and there is no periodic
  backfill, so an owner a maintainer removed stays removed. The timeline check
  enforces the same thing for the paths that remain (a reopen, or a manual batch).
- **Commenting in batch mode.** A manual batch only assigns. It runs in its own
  concurrency group, so allowing it to comment as well would be the one remaining
  way two comments could land on one pull request.
- **Reading the diff.** Only the changed-file *paths* reach the model. They carry
  the routing signal at a fraction of the tokens, and the diff would be a large
  additional block of attacker-controlled text for no gain.
- **Approving, merging, or reviewing.** The bot has no such power and the token
  is not scoped for it.

## Differences from the adk-python and adk-java originals

- The model chooses a component, never a login, and never sees one — including
  in an error. adk-python passes the component into a lookup, but its map lives
  in source next to the agent; here it is operator configuration, validated at
  startup, so the set of assignable logins is fixed before the model exists.
- The context-request comment is rendered from constants rather than written by
  the model, so nothing derived from attacker text is ever posted.
- Assignment is one-way and enforced against the pull request's timeline, not
  only by the trigger list.
- Drafts are skipped and picked up on `ready_for_review`.
- Comments are not shown to the model.
- No labeling (removed upstream), and no `edited` trigger — an edit is a way to
  re-run the model over freshly rewritten attacker text.
