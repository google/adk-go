# Release Docs Analysis Bot

An [ADK Go](https://github.com/google/adk-go) agent that answers "what
documentation needs updating?" after a release ships.

It diffs a release tag against the one before it, analyzes the changed files in
bounded groups, and files **one** GitHub issue listing the documentation updates
the changes imply.

## What it demonstrates

- An `llmagent.New` agent whose only tool **mutates nothing**. The model records
  findings into an in-memory collector; Go decides afterwards whether an issue is
  filed, where, and with what title.
- A clean split between deterministic work done **in code** — tag resolution,
  duplicate detection, diff bounding, grouping, sanitization, issue assembly —
  and the one fuzzy judgement delegated to the **model**: "does this change make
  the documentation wrong?"
- Bounding an untrusted, unbounded input (a release diff) so it fits a context
  window and a budget, and saying in the output exactly what was left out.

## The agent loop

1. Code resolves the tag pair: explicit tags, or the newest release and the one
   before it.
2. Code checks whether an issue for that exact tag pair already exists. If so the
   run stops here, before a single model token is spent.
3. Code fetches the compare diff and applies the caps (`MAX_FILES`,
   `MAX_PATCH_BYTES`, `MAX_COMMITS`), then splits the files into groups of
   `FILES_PER_GROUP`.
4. For each group, in its own session scoped to that `(release, group)` pair, the
   agent analyzes the fenced diff and calls
   `record_documentation_findings(release, group_index, findings)`.
5. Code assembles the issue body from the recorded findings and files it.

## Authority limits, and how they are enforced

The input is a code diff, so the attacker is a contributor who puts text in a
commit message, a file name, or a code comment. Assume the model is fully
steered by that text, then ask what the Go code still refuses.

| Limit | Enforced by |
| --- | --- |
| The model cannot silently skip a group | Each group's outcome is recorded. A group that failed, was never reached, or finished without calling the tool at all is counted and named in the issue. A model that calls the tool and reports an empty list is a genuine "nothing to suggest" answer and is not distinguishable from one steered into saying so — see Known limitations. |
| The model cannot file an issue | It has no tool that writes to GitHub. `Issues.Create` is called by Go after the loop. |
| The model cannot choose the target repository or the title | Both come from configuration and `issueTitle`, not from tool arguments. |
| The model cannot attribute findings to another group or release | `authorizeGroup` checks the tool's `release` and `group_index` against the session scope; an unscoped session records nothing. |
| The model cannot record a group's findings twice | `recorder.record` claims the group's single slot in the same critical section that reads it. |
| The model cannot write an unbounded issue body | `MAX_FINDINGS_PER_GROUP` caps the count, `maxFindingFieldRunes` caps each field, and the body is truncated to GitHub's 65536-byte limit with a notice. |
| The model cannot write arbitrary values into structured fields | `kind` is allow-listed to a fixed set; `doc_file` must match a restricted path pattern with no `..`. |
| Model text cannot escape into Markdown | Every model-authored field is rendered inside a fenced block, and ` ``` `, `<!--` and `-->` are neutralized first. GitHub does not notify `@mentions` inside a fence. |
| Model text cannot escape into another parser | Control characters, bidirectional marks and the line/paragraph separators are stripped **before** the Markdown sequences are neutralized. The other order is exploitable: a zero-width character between two backticks and a third hides the fence from the replacer, and the strip then reassembles it. |
| Model text in a log cannot become a command | The tool-error callback logs argument NAMES, not values, and folds the error text to one line with no control characters. The Actions runner scans a step's stderr for `::` commands exactly as it scans stdout. |
| A model whose output is entirely unrenderable is not mistaken for one with nothing to say | Findings that sanitization empties, and findings the per-group cap drops, are counted and reported in the issue. Neither count reaches `complete()`, because a counter the model controls must not decide that an issue is filed. |
| Contributor text cannot forge trusted context | Each blob sits inside an unguessable per-run `[UNTRUSTED:<hex>]` fence drawn from `crypto/rand`; a draw failure aborts the group rather than falling back to a guessable marker. |
| A malformed tag cannot reshape an API path | `validTag` allow-lists tags at config load, again in `Compare`, and once more in the workflow's shell. |
| Nothing is written under `dry_run` | Every mutation passes `shouldSkip`, the single chokepoint. A test drives the whole program under `dry_run` and asserts zero write requests. |
| Model text cannot reach the Actions runner's command parser | The dry-run render writes to stdout, which the runner scans for lines beginning `::`. `escapeWorkflowCommands` defuses exactly those lines. |

## Exactly one issue per release

There is no per-issue session to scope here — the bot files an issue rather than
acting on one — so the bound is **per release tag pair**, in three parts:

1. **Before any analysis**, `FindExistingIssue` looks for an issue this bot
   already filed for the same tag pair. It runs two probes and treats a hit from
   either as a duplicate:
   - a **list** probe over the target repository's most recent issues, which is
     immediately consistent and so catches the case that actually produces
     duplicates — a re-run moments after a successful run;
   - a **search** probe by the deterministic title, which reaches issues older
     than the list probe's bound but is only eventually consistent.

   An error from either probe aborts the run. A probe that failed proves nothing,
   and filing on that basis is how a duplicate gets created.
2. **Immediately before the write**, both probes run again inside the claim,
   against the caller's own base tag rather than one re-derived from the release
   key. The check in step 1 happened before the analysis loop and is minutes
   stale by then, which is long enough for a concurrent run to have filed.
3. **Within a run**, `claimRelease` takes an atomic claim keyed by the tag pair,
   in the same critical section that reads the previous outcome.
4. **Across runs**, the workflow's `concurrency` group serializes every run of
   the workflow. The group is a constant rather than the release tag, because the
   tag pair is resolved inside the program: a key built from the trigger's inputs
   would put a release run and a manual dispatch that resolves to the same
   release into different groups and run them side by side.

An issue counts as "already filed" only when its **first line** is the exact
marker **and** its author is this bot. Both halves matter:

- Matching the marker anywhere in the body would let model-authored text inside
  one issue name a different tag pair and suppress that release.
- Accepting any author would let a stranger open an issue carrying the marker and
  suppress the release that way.

When the identity lookup fails — the built-in Actions token cannot read its own
user — authorship falls back to "written by a GitHub App", which still excludes
every ordinary account. The residual gap is an App installed on the target
repository, which costs a suppressed issue rather than a wrong mutation.

## Choosing the tag pair

`ListReleases` returns releases ordered by `created_at`, which GitHub documents
as *"the date of the commit used for the release, and not the date when the
release was drafted or published"*. On a repository with a maintenance branch
that order interleaves release lines: the live `google/adk-go` listing runs
v2.3.0, v1.6.0, v2.2.0, v2.1.0, v1.5.1, v2.0.0, … Taking "the next entry" as the
base would diff v2.3.0 against v1.6.0.

So the base is selected rather than read off the list. It is the greatest
non-prerelease version strictly below the head, among releases published no
later than the head. Both halves are needed: the version comparison rules out
v1.6.0 as a base for v2.3.0, and the publication cutoff rules it out as a base
for v2.0.0, which shipped six weeks earlier. A prerelease is never a base,
because diffing a release against its own release candidate covers only the
rc-to-final delta.

An explicit `-start-tag` bypasses all of this, and is the documented answer when
the derivation cannot give a sensible pair.

## Filing into another repository

adk-python files this issue into **`google/adk-docs`**, a different repository,
using a dedicated PAT. **This bot defaults to filing into the repository it
diffed**, because that is the only repository the built-in Actions
`GITHUB_TOKEN` can write to.

`TARGET_OWNER` / `TARGET_REPO` point it elsewhere. Doing so requires a token with
`issues: write` on the target, supplied as a secret in place of `github.token` in
the workflow. **This repository may not have such a token**, and setting one up
is a decision for whoever adopts the bot, not something the code can paper over.
The bot logs a warning when the target differs from the source.

## Running locally

Requires **Go 1.26+** (see `go.mod`). Copy `.env.example` to `.env` and fill it
in (or export the variables), then:

```bash
# Dry-run the most recent release against the one before it.
go run . -dry-run

# Dry-run an explicit range (also how you backfill).
go run . -dry-run -start-tag v2.2.0 -end-tag v2.3.0

# File the issue for real.
go run . -start-tag v2.2.0 -end-tag v2.3.0
```

> **Dry-run is not offline.** It still reads GitHub and still calls the model. It
> suppresses the write and prints the issue body it would have filed.

## Configuration

| Variable / flag | Default | Description |
| --- | --- | --- |
| `GITHUB_TOKEN` | — (required) | Token with `issues: write` on the **target** repository. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — | Gemini API key (or use Vertex AI). |
| `OWNER` | — (required) | Source repository owner. |
| `REPO` | — (required) | Source repository name. |
| `TARGET_OWNER` | `OWNER` | Owner of the repository the issue is filed in. |
| `TARGET_REPO` | `REPO` | Repository the issue is filed in. |
| `LLM_MODEL_NAME` | `gemini-flash-latest` | Model to use. |
| `START_TAG` / `-start-tag` | (derived) | Base tag. Empty = the release before the head tag. |
| `END_TAG` / `-end-tag` | (derived) | Head tag. Empty = the most recent release. |
| `MAX_FILES` | `60` | Changed files analyzed. The rest are reported as omitted. |
| `MAX_PATCH_BYTES` | `8000` | Bytes of one file's patch read. |
| `MAX_COMMITS` | `100` | Commit subjects included in the prompt. |
| `FILES_PER_GROUP` | `5` | Files per model call. |
| `MAX_FINDINGS_PER_GROUP` | `10` | Suggestions one group may record. |
| `RUN_BUDGET` | `15m` | Bounds the whole analysis loop. |
| `GROUP_TIMEOUT` | `4m` | Bounds one group's model call. |
| `-dry-run` / `DRY_RUN` | `false` | Render the issue without filing it. |

Instead of an API key you can use Vertex AI via Application Default Credentials
(`GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`).

## GitHub Actions

`.github/workflows/release-docs-bot.yml` runs on `release: [published]` and on
`workflow_dispatch` with explicit `start_tag` / `end_tag` inputs for a retry or a
backfill. The manual trigger's `dry_run` input defaults to **true**; a
`release: published` run is always a real run.

The job needs `issues: write` and a `GEMINI_API_KEY` secret. Without the secret
the run fails at configuration load with a message naming what is missing.

## Tests

```bash
go test -race -count=1 -shuffle=on ./...
```

Pure logic is table-driven (`release_test.go`: tag validation, version-based tag
selection, diff bounding, grouping, the finding allow-lists, fence containment,
marker matching, issue assembly and its coverage disclosures). The GitHub client
is exercised with `httptest` (`github_test.go`: draft and prerelease handling,
compare bounding and cross-page deduplication, both duplicate probes, impostor
and pull-request rejection, the per-release claim under concurrency, and the
re-probe before the write). The tool layer's group scoping, per-group claim and
volume cap are verified without any HTTP call (`tools_test.go`).

`agent_test.go` builds the real `llmagent`, `runner` and `functiontool` chain
against a stub `model.LLM`. It is what proves the authority model rather than
assuming it: that ADK carries the session scope through to the tool at all, that
the model is offered exactly one tool and that tool's exact argument schema, that
a model naming another group is refused end to end, that a nonce failure aborts
before any model call, and that the issue the program files is one both duplicate
probes can find. `main_test.go` drives the real `runWith` to prove a release that
already has an issue reaches neither the diff nor a write.

## Known limitations

- **The analysis reads the diff, not the documentation.** It suggests what
  *might* need updating; it cannot tell you the docs already say the right thing.
  Every finding needs a human to check it.
- **The list probe is bounded** to the most recent 300 issues in the target
  repository. Beyond that, duplicate detection rests on the search probe alone,
  which is eventually consistent.
- **Duplicate detection trusts an authored-by-a-GitHub-App fallback.** The
  built-in Actions token cannot read its own user, so the bot cannot resolve its
  own login and accepts any App-authored issue carrying the marker on line one.
  An App installed on the target repository could therefore suppress one
  release's issue. It cannot cause a wrong write.
- **The comparison is fetched over a bounded number of pages.** A release larger
  than that reports its file and commit totals as lower bounds, and says so.
- **A model can be steered into reporting nothing for a group.** Text in a commit
  message or a code comment can tell the model its group needs no documentation
  change, and no Go check can tell that apart from an honest "nothing here" —
  the judgement being delegated is exactly that question. It costs a missing
  suggestion, never a wrong write. What IS caught is a group whose tool never
  fired at all, which the issue reports.
- **A prerelease head is skipped by the workflow, not by the program.** Running
  `go run .` with `-end-tag v2.4.0-rc.1` will analyze and file for it, and the
  final v2.4.0 issue will then overlap it.
- **Truncation is by position, not importance.** With more than `MAX_FILES`
  changed files, it is the first ones the compare API returns that are analyzed,
  not the most documentation-relevant. The issue says how many were skipped.
- **No cross-group awareness.** Each group is analyzed independently, so two
  groups can produce overlapping suggestions. adk-python's version passes a
  running summary between groups; this one keeps the sessions isolated instead.
