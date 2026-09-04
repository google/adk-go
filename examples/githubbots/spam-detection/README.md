# Spam Detection Bot

An autonomous [ADK Go](https://github.com/google/adk-go) agent that moderates a
GitHub repository's issues for spam. It audits open issues for SEO spam,
unsolicited promotion of third-party products or sites, and other off-topic
solicitation. When the model judges an issue's content to be spam it:

1. **Applies the spam label** (`spam` by default), and
2. **Posts one comment** alerting the maintainers, with a short reason.

Two things are worth knowing before anything else:

- **The model only ever sees the issue's own title and body.** Comments are
  never shown to it. That is what stops a stranger leaving one promotional
  comment on your bug report and getting *your* issue labelled as spam.
- **The alert comment contains text the model wrote.** Its one-sentence reason
  is model-authored, which is unusual — the sibling ADK bots publish only labels
  and fixed sentences. It is confined to a fenced code block, so it cannot
  produce a clickable link, render an image, or notify anyone by `@mention`.

## What happens to an issue

Suppose you open an issue on this repository tomorrow morning.

1. **Nothing happens immediately.** There is no trigger on issues being opened
   or commented on. The bot wakes on a schedule, every six hours, and that is
   deliberate: an event trigger would let any passer-by start a model run
   holding a token that can write to your issues.
2. **The sweep picks candidates.** It asks GitHub's search for open issues
   updated in the last day, excluding any already labelled `spam`, and takes at
   most 30 — most recently updated first.
3. **Your issue is fetched**, along with its labels and up to 100 of its
   comments.
4. **Cheap checks run first, and most issues stop here.** The bot skips your
   issue if it already carries the spam label, if the bot has already commented
   on it, or if you are listed as a maintainer. If it stops here, no model is
   called and nothing is spent.
5. **The text for review is assembled — your title and your body, and nothing
   else.** The comments fetched in step 3 are used only to check whether the bot
   already alerted on the thread; they are not shown to the model. Your title
   and body are each truncated to about 1500 characters, wrapped in a marker
   carrying a random number so that text inside cannot pretend the marker has
   ended, and labelled with your GitHub author association (whether you are a
   first-time contributor, a member, and so on) as a hint rather than a verdict.
6. **The model is asked one question: is this spam?** It gets its own private
   session for your issue alone, and its only available action is "flag this
   issue".
7. **If it says no, the run ends.** Nothing is written. Your issue may be
   reviewed again on a later sweep if it is updated.
8. **If it says yes, Go checks the answer before acting.** The flag is rejected
   unless it names your issue and no other; a second flag on the same issue in
   the same run is ignored; and in dry-run mode every write is logged instead of
   sent.
9. **Two things are written, in this order:** a comment saying the issue's title
   and body were flagged, quoting the model's reason inside a code block, and
   then the `spam` label. The comment goes first deliberately — if only one of
   the two can succeed, a notified maintainer is worth more than a label.
10. **The run stops on its own budget.** A single issue gets one minute, the
    whole sweep fifteen, and an overrun exits non-zero so a stuck run is visible
    rather than silently truncated.

Nothing here deletes, hides, edits or closes anything. The bot labels and
comments, and a human decides what to do next.

### What is guaranteed by code rather than asked of the model

Steps 5 and 8 are the load-bearing ones, and they are enforced in Go because a
prompt can be argued with by the text it is reading:

- Only the issue's own title and body reach the model (step 5).
- The model may flag only the issue its session is scoped to, so injected
  instructions cannot redirect it at a different issue (step 8).
- An issue already labelled or already alerted is never re-processed (step 4).
- An issue opened by a configured maintainer is never reviewed (step 4).
- The model's reason is stripped of invisible characters, truncated, and fenced
  before it is posted (step 9).

## What it demonstrates

- An `llmagent.New` agent driven by a single typed `functiontool.New[Args, Result]`
  tool, run **once per issue in its own isolated session** with bounded
  concurrency (`errgroup`) — i.e. code orchestrates the loop and the model only
  classifies.
- A clean split between deterministic work done **in code** and the fuzzy
  judgment delegated to the **model**: the bot fetches the issue, narrows the
  input to the issue's own title and body, filters out maintainer-authored and
  already-handled issues, truncates long text, and annotates the
  author with their GitHub **author association** (a spam-likelihood prior) in
  `spam.go`, then asks the model only "is this spam?" — guided by that signal and
  a few worked examples in the prompt.
- The **zero-waste** optimization from the original Python sample: if nothing
  reviewable remains after filtering (or the issue was already handled), the
  model is never invoked.
- Calling the GitHub REST API (`go-github`) and GraphQL API (a raw POST through
  the same authenticated client) from code.

## The same sequence in ADK terms

The walkthrough above is what a maintainer sees. This is the same run described
in the framework's vocabulary, for anyone reading the code (`main.go`):

1. Code selects the candidate issues (a sweep via the Search API, or a single
   `-issue`).
2. For each issue, in its own goroutine (bounded by `CONCURRENCY_LIMIT`),
   `reviewIssue` binds the session to that one issue number and hands it to
   `runReviewFor`, which fetches the issue and its comments, runs the
   idempotency and filtering logic, and assembles the reviewable text. The
   comments are read only to recognize the bot's own past alert; they are never
   part of what the model judges. Issues with nothing to review are skipped
   without a model call.
3. The agent runs in a fresh, **issue-scoped** session. The prompt carries the
   issue number and the assembled content (clearly fenced and marked untrusted);
   `runner.Run(...)` returns an `iter.Seq2[*session.Event, error]` that yields one
   streamed event or an error per iteration.
4. The model either calls `flag_issue_as_spam(issue_number, detection_reason)`
   or replies "No spam detected." `authorizeIssue` rejects any `issue_number`
   other than the one the session is scoped to.

A rejected tool call (wrong issue) is returned to the model as a result with
`status: "error"` and a **nil Go error**; real I/O failures return a Go `error`,
are recorded, and make the process exit non-zero so scheduled/CI runs fail
loudly.

> **Why embed the content in the prompt instead of a retrieval tool?** Because
> all the deterministic pre-processing (filtering, truncation, and the
> idempotency check that lets us skip the model entirely) happens in code
> before the model runs. Putting the finished, untrusted-marked text in the
> per-issue prompt keeps the model's only job — and its only tool — the spam
> decision itself.

## Running locally

Requires **Go 1.26+** (see `go.mod`). Copy `.env.example` to `.env` and fill it
in (or export the variables), then:

```bash
# Dry-run a single issue (no writes; logs intended actions).
go run . -dry-run -issue 123

# Dry-run a sweep of recent issues.
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
| `BOT_LOGIN` | (empty) | The login the bot posts as, used to recognize its own alert comments. Required under GitHub Actions, where `GET /user` is refused for the built-in token; with a personal access token the bot resolves it itself. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Gemini API key (or use Vertex AI). |
| `OWNER` | — (required) | Repository owner. |
| `REPO` | — (required) | Repository name. |
| `LLM_MODEL_NAME` | `gemini-flash-latest` | Model to use. The workflow pins `gemini-3.6-flash` rather than taking this default: an alias can be repointed with no change here, and the detection rates below are properties of a specific model version. |
| `SPAM_LABEL_NAME` | `spam` | Label applied to flagged issues (must already exist). |
| `MAINTAINERS` | (empty) | Comma-separated logins whose issues are trusted and never reviewed. |
| `ISSUE_COUNT` | `3` | Max issues per scheduled sweep (most-recently-updated first). |
| `CONCURRENCY_LIMIT` | `3` | How many issues to review in parallel. |
| `FRESHNESS_WINDOW_DAYS` | `0` (off) | Restrict the sweep to issues updated within N days. |
| `ISSUE_TIMEOUT` | `5m` | Bounds a single issue review. |
| `RUN_TIMEOUT` | `15m` | Bounds the whole run. The workflow's `timeout-minutes` sits above it, so an overrun is reported here instead of the runner killing the job. |
| `-dry-run` / `DRY_RUN` | `false` | Log intended actions without mutating. |
| `-issue` | `0` | Review only this issue (0 = sweep). |

Instead of an API key you can use Vertex AI via Application Default Credentials
(`GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`).

> **`MAINTAINERS` is optional but recommended.** The built-in Actions
> `GITHUB_TOKEN` cannot list a repo's collaborators, so trusted logins are
> supplied explicitly. With an empty set the bot still works — it just also
> reviews maintainers' own issues (wasting a few tokens and risking a
> false positive); it never misses spam because of it.

## How it runs in CI

`.github/workflows/spam-detection-bot.yml` runs this module directly: it checks
out the repository, installs the Go version from this module's `go.mod`, builds
the bot with `go build -o ./bot .`, and runs `./bot`. The build is a separate
step so the cold compile is outside the bot's own `RUN_TIMEOUT` budget. It needs one repository secret,
`GEMINI_API_KEY`; the GitHub credentials are the built-in `GITHUB_TOKEN`,
narrowed by the job to `contents: read` and `issues: write`.

Triggers:

- **A sweep every six hours** (`cron: '15 */6 * * *'`), covering issues updated
  in the last day, capped at 30. Four consecutive runs therefore cover the same
  issue, so a run that fails or is skipped costs coverage rather than losing it.
- **`workflow_dispatch`**, with an optional `issue` number and a `dry_run`
  toggle that **defaults to true**. The issue input is validated against
  `^[1-9][0-9]*$` in the shell before it reaches the program — a leading zero is
  rejected, because Go's flag package would read `0123` as octal 83 — and it is
  passed through `env:` rather than interpolated into the `run:` script.

**There is deliberately no `issues:` or `issue_comment:` trigger.** An event
trigger lets any drive-by commenter start a model run holding a write-scoped
token, which is the widest attack surface this bot could have. It is also the
shape adk-python removed from its own agent suite. The scheduled sweep is the
backstop instead, which is why its window overlaps.

The job runs only on `google/adk-go` (`if: github.repository == …`), checks out
with `persist-credentials: false`, pins both third-party actions to a full
commit SHA, and takes a constant `concurrency` group so two sweeps can never
overlap — the at-most-once flag claim spans a single process, so two concurrent
runs could each see an issue unlabeled and both post the alert.

## Tests

Pure logic is table-driven (`spam_test.go`: filtering, case-insensitive
maintainer/identity matching, idempotency, truncation, suspect-text assembly,
prompt-injection-resistant signature detection); the GitHub client is exercised
with `httptest` (`github_test.go`, incl. PR exclusion, repo-vs-issue NOT_FOUND
handling, and comment-before-label ordering); the tool layer's issue-scope
authorization, within-run idempotency, and dry-run gates are verified to reject
bad input without any HTTP call (`tools_test.go`).

`agentloop_test.go` drives the real `runReviewFor` — agent loop included —
against a scripted model and an `httptest` GitHub. It is the only place that
proves the session scope set by `reviewIssue` survives the trip through ADK into
the tool: a scripted tool call naming a different issue must be refused at
runtime, with no write reaching GitHub. It also pins that the model never runs
for an already-labeled issue or after a failed nonce draw, and that the prompt
the model actually receives carries the issue text inside a fence keyed to a
fresh 16-character nonce with the authorship header outside it.

`e2e_test.go` runs the same production path against a **real Gemini model**, with
only GitHub faked, and is the only place the spam/not-spam judgement itself and
the prompt that steers it are exercised. It is gated at run time rather than
behind a build tag, so `go vet ./...` and `go test ./...` keep compiling it — a
build tag hid a stale call in it through several green CI runs. Both switches
must be on, so a runner that happens to hold a key still spends nothing.

Thirteen classification scenarios cover both directions. Four of them are
false-positive protection, because labelling a real user as a spammer is the
costlier error and this repository produces exactly the shapes that risk it: an
issue reporting a prompt-injection bug with a sample payload, an issue quoting
the bot's own alert, an issue pasting a fenced agent transcript, and an issue
reporting somebody else's spam.

A run that sheds cases to an unavailable model **fails**, and does so through
the exit code rather than a printed warning, because `go test` without `-v`
discards a passing package's output. A rate over a denominator the run did not
fill is not a measurement. `SPAM_BOT_E2E_FORCE_UNAVAILABLE=1` sheds every call
on demand so that guard can be verified in seconds.

**This bot publishes model-authored prose, and it is the only one of the ADK
GitHub bots that does.** The sibling bots either choose from fixed sentences by
allow-listed key or write no prose at all, so their public output cannot carry
attacker text by construction. This one asks the model to explain, in its own
words, why an issue was flagged, and then posts that under the project's
identity. That is a deliberate trade — a maintainer reading the alert wants
the reason, not a fixed sentence — and it is why the rest of this section
exists.

**The alert comment is treated as hostile output, and the measurement says it
has to be.** The bot writes one thing to a public place under the project's
identity, and the only variable part of it is the model's `detection_reason` —
which the issue body can argue with. Driving the real model with issue text that
asks for a specific payload in the reason, the request for `@torvalds @google`
**reached the published reason in 2 of 9 runs on `gemini-flash-latest`**, and in
0 of 10 on the pinned `gemini-3.6-flash`. Compliance is model-dependent, which
is precisely why the guarantee is not allowed to rest on it. The comment was inert every
time regardless, because every model-authored byte is confined to one
unescapable fenced block, where GitHub does not linkify a URL, render an image,
or notify a mention. Invisible characters are stripped separately, since a
bidirectional override reorders a rendered line from inside a code block just as
well as outside one, and the bot's own identity marker is neutralized so a
steered model cannot publish the one string that makes a later sweep treat a
comment as a prior alert.

Each of those four defences is mutation-checked and each is load-bearing:
removing the fence lets all eight payloads through, removing the invisible-
character strip lets the two hidden-text payloads through, removing the marker
neutralization lets the identity marker through, and removing the fence-escape
substitution lets the reason break out of the block. The 2-in-5 figure is why
this is enforced in Go rather than asked of the prompt.

**The prompt was mutation-tested against those scenarios, and no single section
is load-bearing.** Deleting each of six sections in turn — the comment rule, its
narration clause, "What is NOT spam", the marker-distrust paragraph, the worked
examples, and the definition of spam itself — changed no scenario's outcome, 78
scenario-mutant pairs with zero flips. That is a real measurement rather than a
broken harness: a positive control replacing the whole prompt with "flag
everything" flipped all 8 legitimate scenarios and left all 5 spam scenarios
flagged, so the corpus does detect prompt changes. Two of the six are inert for
a structural reason worth naming — the comment rule and its narration clause ask
the model for something `assembleSuspectText` already enforces by not sending
the text, so there is nothing left for the prompt to get wrong. For the rest the
honest reading is that the sections are redundant with each other and with the
model's own priors on content this obvious, and that a corpus of clear-cut cases
cannot tell a redundant instruction from an unnecessary one.

The obvious explanation is internal redundancy — sections deleted one at a time
would each read as inert while jointly carrying real weight. **That was tested,
and it predicts a flip that does not happen.** Deleting whole groups covering
the same ground changed nothing either: removing the third-party rule, "What is NOT spam" and every
worked example together, 36 of the prompt's 92 lines and all of its restraint
guidance, left all 14 scenarios at 3/3. Removing the definition of spam and the
examples together did the same.

So the accurate reading is narrower than "the prompt is redundant". The model
follows an explicit contrary *instruction* — that is what the positive control
demonstrates, since it replaced the prompt with "every issue is spam" rather
than deleting anything. It does not depend on the prompt's *guidance* to reach
these judgements: on cases this clear-cut its own priors already produce the
same answers. The prompt earns its place on the cases a corpus of obvious
examples does not contain, and this measurement says nothing about those.

**The judgement also lives outside the prompt file, and that had to be deleted
too before any of this counted.** The tool declaration said "Call this only when
the reviewed content is clearly spam" — the core *when to flag* rule, reaching
the model in the function declaration rather than the instruction text, so it
stood in every cell of every mutation above. Removing it as well, on its own and
together with all the restraint guidance, still flipped nothing: 14 of 14
scenarios held at 3/3. The confound was real and the result survives it. Anyone
mutating a prompt should check the tool descriptions first, because a harness
that edits only the instruction file cannot see them.

Two limits apply to every row above. Each cell is three repeats, so only an
effect large enough to flip a scenario outright is visible, and a section — or a
group — shifting behaviour by a fifth would read as inert. And the unit of the
claim is the scenario, so **0 flips in 14 bounds the per-scenario flip rate
below only about 19% at 95% confidence.** That is a weak null. It rules out a
large effect and is consistent with a moderate one, and it should not be read as
"the prompt does nothing".

That reading would also be wrong on other evidence: a sibling bot that deleted
every copy of its own judgement saw 4 of 21 scenarios flip, two of them
injection attempts where the model was talked into acting. So the prompt is what
keeps a model from being turned, and the Go controls bound what a turned model
can reach — different halves of the attack, both load-bearing. This bot's null
says its corpus could not detect the difference, not that the difference is
absent.

```bash
go test ./...

# The end-to-end suite (calls a real model, costs money).
SPAM_BOT_E2E=1 GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=<project> \
  GOOGLE_CLOUD_LOCATION=global go test -run TestE2E -v ./...
```

## Differences from the Python sample

This is adapted from the Python `adk_issue_monitoring_agent`. The behavior
differs in a few deliberate ways:

- **Scan strategy.** Python had an uncapped daily sweep (everything updated in
  the last 24h, via `since`) plus an `INITIAL_FULL_SCAN` of every open issue.
  This bot caps the sweep at `ISSUE_COUNT` (most-recently-updated first). The
  schedule is the only automatic trigger, so raise `ISSUE_COUNT` or shorten the
  cron interval if your issue volume outgrows one sweep's cap.
- **Title review.** The issue title is reviewed in addition to the body (Python
  reviewed only the body), so spam titles are caught.
- **Comments are not judged.** Python fed the issue's comments to the model
  alongside its body. This bot sends the title and body and nothing else. The
  reason is that flagging acts on the *whole issue*: it labels the thread and
  alerts on it. If a comment can support a flag, then any stranger can leave one
  promotional comment on a maintainer's bug report and have the bot label the
  bug report — a moderation action against someone who wrote nothing wrong. The
  issue's author is answerable for the title and the body, so that is the whole
  of the input. It is enforced in `assembleSuspectText` rather than asked of the
  model, because comment bodies are attacker-controlled and a prompt cannot be
  relied on to resist text that argues with it. The cost is real: spam posted as
  a comment on an otherwise legitimate issue is not detected by this bot at all.
- **Code blocks kept.** Python stripped fenced code blocks before review; this
  bot keeps them (bounded by truncation) so spam can't hide inside a ``` fence.
- **Alert comment.** A plain "Maintainers, please review." line rather than
  Python's literal `@maintainers` mention (which only pings if such a team/user
  exists).
- **Concurrency.** Bounded by `errgroup` (`CONCURRENCY_LIMIT`) rather than
  Python's fixed chunk size plus an inter-batch sleep.
- **Idempotency.** The spam **label** is the primary guard; within a run a second
  flag of the same issue is a no-op in code; the bot's own alert comment is a
  best-effort secondary signal that can be missed on threads with more comments
  than the fetch window after the alert (only causing a re-alert if the label was
  also removed).

## Notes

- The **spam label must already exist** in the target repository (the bot adds
  it but does not create it). For `google/adk-go` it does.
- This is a moderation aid, not a verdict: it flags and notifies for human
  review rather than deleting or blocking. It deliberately errs toward inaction —
  the prompt instructs the model not to flag merely unhelpful, off-topic, or
  beginner content.

## Known limitations

- **Spam in comments is not detected.** The model only ever sees the issue's own
  title and body, so a promotional comment left on somebody else's issue is
  invisible to this bot. That is the deliberate price of the rule above: the
  only action available is to label the whole thread, so letting a comment
  support a flag would let any stranger get a maintainer's bug report labelled
  as spam. Moderating comments needs a per-comment action (hiding or deleting
  the comment), which this bot does not have.
- **Truncation padding.** The title and the body are each truncated to ~1500
  runes before review, and padding placed ahead of a spam link pushes it past
  the cutoff. When either is cut the prompt says so, in a trusted line outside
  the fence, so the model is not asked to judge text it was never told was
  partial. A production system would prioritize link-bearing regions; this
  sample keeps the simple bound.
- **A flag labels the whole issue.** There is no narrower action: flagging
  applies the label and posts one alert on the thread. If a maintainer then
  removes the label, the bot's own alert comment keeps the issue out of future
  sweeps — deliberately, so the bot does not overturn a human decision on its
  next run, but it does mean the issue is no longer reviewed. Deleting the bot's
  comment restores it.
- **A failed label leaves the issue alerted but unlabeled.** The alert comment
  is posted before the label, so if one of the two fails it is the label that is
  lost rather than the notification. The next run recognizes its own alert and
  skips the issue, so the label is not applied later: maintainers have been
  told and the issue is visible, but it will not carry the label.
- **Every installed GitHub App is reviewed like a user** unless it is named in
  `MAINTAINERS`. Skipping accounts by their `[bot]` suffix would extend
  unconditional trust to every App installed on the repository. That includes
  the bot's own login: under Actions it is `github-actions[bot]`, shared with
  every workflow in the repository, so an issue opened under it is reviewed like
  any other, and only a comment carrying the bot's own alert marker counts as a
  prior alert.
- **Evasion by appended instruction-shaped text is a property of the model, and
  the pinned model does not have it.** The attack is to follow obvious spam with
  prose asserting that the untrusted region has closed and that what precedes it
  was reviewed and approved. Measured on two models, same day, same code, same
  harness:

  | model | spam **with** the injection | the same spam without it |
  | --- | --- | --- |
  | `gemini-3.6-flash` (pinned in the workflow) | detected **50 of 50** | 50 of 50 |
  | `gemini-flash-latest` (the code default) | detected **18 of 50** | 50 of 50 |

  So on the model this bot actually ships with, the attack did not work once in
  50 attempts. That is an upper bound rather than immunity: 0 failures in 50
  puts the true rate below roughly 6% at 95% confidence, and it is a fact about
  one model version, which is exactly why the workflow pins one instead of
  following a floating alias.

  The 18-of-50 row is kept because it is the more useful number. It shows what
  this design costs when the model is weaker, and the Go controls were unaffected
  throughout it: no write ever landed on an issue other than the one under
  review, and dry-run suppressed every write. A missed detection is not a loss
  of authority.

  Neutralizing the forged marker in Go (`defangFenceMarkers`) removes one
  capability from the attacker but does not by itself fix the weak-model case,
  because the persuasion is carried by the prose rather than by the marker. It
  does have to be paired with `normalize`, which strips invisible characters
  from untrusted text *before* the marker is matched — the matcher is a literal
  regex, so one zero-width space inside the word (`[/UNTRUSTED<ZWSP>:`) slipped
  a forged boundary past it while a model still read it as a boundary. Four of
  five variants got through before that was wired up.

  **An inversion was tried and is not shipped.** Rather than only scrubbing the
  instruction-shaped text, the bot counted it and reported the count to the
  model in the trusted header, so the content could not attack the classifier
  while its presence still counted as evidence of spam. On `gemini-flash-latest`
  detection went from 18/50 to **23/50** — directionally positive in both paired
  runs, but z≈1.0 (p≈0.31), which is not distinguishable from noise; separating
  it would need roughly 370 samples per arm. It cost a hand-maintained list of
  trigger phrases that is itself a false-positive surface, so it was reverted.
  Nothing in the measurement says the idea is wrong, only that this measurement
  could not show it working — and on the pinned model there is nothing left for
  it to fix.

  Provenance, because these numbers are only comparable to each other: all of
  them were taken through Vertex AI (`GOOGLE_CLOUD_LOCATION=global`) on one day,
  with every sample answered, 0 dropped and 0 transient failures recovered. A
  run that sheds samples fails rather than quoting a rate over a denominator it
  did not fill. Re-measure with
  `SPAM_BOT_E2E=1 E2E_SAMPLES=25 LLM_MODEL_NAME=<model> go test -run TestE2EInstructionEvasionRate`.
- **The nonce is pinned to a CSPRNG by reading the source, not the output.** A
  predictable fence marker would let an attacker pre-write the closing marker
  and escape the fence, and no test of the returned value can detect that —
  `math/rand` produces hex tokens that are unique and well-distributed exactly
  like the real ones. Swapping it in compiles and passes everything else,
  `-race` included, so the check parses `main.go` instead.
- **Author association is a prior, not proof.** It nudges borderline calls; it
  is not a substitute for reading the content, and spam from an established
  account is still flagged on its merits.
