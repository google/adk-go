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
| The model cannot file an issue | It has no tool that writes to GitHub. `Issues.Create` is called by Go after the loop. |
| The model cannot choose the target repository or the title | Both come from configuration and `issueTitle`, not from tool arguments. |
| The model cannot attribute findings to another group or release | `authorizeGroup` checks the tool's `release` and `group_index` against the session scope; an unscoped session records nothing. |
| The model cannot record a group's findings twice | `recorder.record` claims the group's single slot in the same critical section that reads it. |
| The model cannot write an unbounded issue body | `MAX_FINDINGS_PER_GROUP` caps the count, `maxFindingFieldRunes` caps each field, and the body is truncated to GitHub's 65536-byte limit with a notice. |
| The model cannot write arbitrary values into structured fields | `kind` is allow-listed to a fixed set; `doc_file` must match a restricted path pattern with no `..`. |
| Model text cannot escape into Markdown | Every model-authored field is rendered inside a fenced block, and ` ``` `, `<!--` and `-->` are neutralized first. GitHub does not notify `@mentions` inside a fence. |
| Contributor text cannot forge trusted context | Each blob sits inside an unguessable per-run `[UNTRUSTED:<hex>]` fence drawn from `crypto/rand`; a draw failure aborts the group rather than falling back to a guessable marker. |
| A malformed tag cannot reshape an API path | `validTag` allow-lists tags at config load, again in `Compare`, and once more in the workflow's shell. |
| Nothing is written under `dry_run` | Every mutation passes `shouldSkip`, the single chokepoint. |

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
2. **Within a run**, `claimRelease` takes an atomic claim keyed by the tag pair,
   in the same critical section that reads the previous outcome.
3. **Across runs**, the workflow's `concurrency` group serializes two runs for the
   same release. This is a convenience, not the guarantee — the probes are.

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

Pure logic is table-driven (`release_test.go`: tag validation, diff bounding,
grouping, the finding allow-lists, fence containment, marker matching, issue
assembly). The GitHub client is exercised with `httptest` (`github_test.go`:
draft filtering, compare bounding and cross-page deduplication, both duplicate
probes, impostor and pull-request rejection, the per-release claim). The tool
layer's group scoping, per-group claim and volume cap are verified without any
HTTP call (`tools_test.go`), and `main_test.go` drives the real `runWith` to
prove a release that already has an issue reaches neither the diff nor a write.

## Known limitations

- **The analysis reads the diff, not the documentation.** It suggests what
  *might* need updating; it cannot tell you the docs already say the right thing.
  Every finding needs a human to check it.
- **The list probe is bounded** to the most recent 300 issues in the target
  repository. Beyond that, duplicate detection rests on the search probe alone,
  which is eventually consistent.
- **Truncation is by position, not importance.** With more than `MAX_FILES`
  changed files, it is the first ones the compare API returns that are analyzed,
  not the most documentation-relevant. The issue says how many were skipped.
- **No cross-group awareness.** Each group is analyzed independently, so two
  groups can produce overlapping suggestions. adk-python's version passes a
  running summary between groups; this one keeps the sessions isolated instead.
