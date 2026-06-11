# GitHub Stale-Issue Auditor (ADK Go example)

An autonomous agent that audits open GitHub issues for staleness. Unlike a
timestamp-only "stale bot", it reconstructs each issue's full conversation
history and uses an LLM to tell the difference between a maintainer **asking a
question** (a stale candidate) and a maintainer **posting a status update**
(still active).

## What this example demonstrates

- An [`llmagent`](../../agent/llmagent) driven by typed
  [`functiontool`](../../tool/functiontool) tools.
- Running an agent headlessly with a [`runner`](../../runner) and an in-memory
  session — no interactive launcher.
- Calling the GitHub REST **and** GraphQL APIs from inside tools
  (`github.com/google/go-github` for REST; a raw GraphQL POST through the same
  authenticated client).
- Bounded concurrency with `errgroup`, deterministic decisions
  (`Temperature: 0`), and a clean separation between **pure, unit-tested logic**
  (`state.go`) and **side-effecting I/O** (`github.go`).

## How it works

For each candidate issue the agent:

1. Calls `get_issue_state`, which issues one GraphQL query (comments,
   description edits, title renames, reopen/label events), replays the history
   to find the **last human actor**, and computes staleness.
2. Follows a decision tree (see `prompt_instruction.txt`):
   - **Author/other replied** → remove the stale label (issue is active again);
     if they edited the description silently, alert maintainers.
   - **Maintainer asked a question** and the threshold passed → mark stale.
   - **Stale long enough** → close as *not planned*.
   - **Maintainer status update / internal discussion** → no action.

## Configuration

| Env var | Default | Description |
|---|---|---|
| `GITHUB_TOKEN` | — (required) | Token for GitHub API. In Actions, the built-in `github-actions[bot]` token. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | — (required) | Gemini (AI Studio) API key. |
| `OWNER` | `google` | Repository owner. |
| `REPO` | `adk-go` | Repository name. |
| `MAINTAINERS` | _(empty)_ | Comma-separated maintainer logins. The default `github-actions[bot]` token is repo-scoped and can read neither collaborators nor org **team** membership at runtime, so the maintainer set is supplied here. Mirror it from [`@google/adk-go-maintainers-team`](https://github.com/orgs/google/teams/adk-go-maintainers-team) and keep it in sync manually. |
| `LLM_MODEL_NAME` | `gemini-3.5-flash` | Gemini model. |
| `STALE_HOURS_THRESHOLD` | `336` (14 days) | Hours an issue may wait on the author after a maintainer's request before it is marked stale. |
| `CLOSE_HOURS_AFTER_STALE_THRESHOLD` | `168` (7 days) | Hours an issue may remain stale (after the warning comment) before closing. |
| `STALE_LABEL_NAME` | `stale` | Label applied to stale issues (must already exist). |
| `REQUEST_CLARIFICATION_LABEL` | `request clarification` | Label added when a maintainer's question marks an issue stale (must already exist). |
| `CONCURRENCY_LIMIT` | `3` | Issues audited in parallel. |
| `ISSUE_TIMEOUT` | `5m` | Per-issue audit timeout (Go duration). |

Flags: `-dry-run` (log intended actions without mutating) and `-issue N` (audit
a single issue and skip the search step).

## Run locally

```bash
export GITHUB_TOKEN="<a token with repo/issues access>"
export GEMINI_API_KEY="<your AI Studio key>"
export OWNER=google REPO=adk-go
# Mirror of @google/adk-go-maintainers-team (see note in the config table).
export MAINTAINERS="baptmont,dpasiukevich,foxfrikses,hanorik,hyangah,jjsasha63,karolpiotrowicz,kdroste-google,mazas-google,wolo-lab,yarolegovich"

# Safe: analyze one issue without making any changes.
go run ./examples/githubstalebot -dry-run -issue 123

# Audit all stale candidates (will comment/label/close):
go run ./examples/githubstalebot
```

## Run as a GitHub Action

Add one secret — `GEMINI_API_KEY` — under *Settings → Secrets and variables →
Actions*. The GitHub token is the auto-provided `github-actions[bot]`; the
workflow grants it write access via a least-privilege `permissions:` block:

```yaml
permissions:
  issues: write
  contents: read
```

A ready-to-use workflow lives at
[`.github/workflows/stale-bot.yml`](../../.github/workflows/stale-bot.yml).

> [!IMPORTANT]
> Committing that workflow makes the bot **live**: on its schedule it will
> comment on, label, and close real issues. Note the dry-run asymmetry — the
> manual `workflow_dispatch` trigger defaults to dry-run, but **scheduled runs
> are not dry** (they act for real). Before enabling the cron:
> 1. create the `stale` **and** `request clarification` labels in the repository, and
> 2. trigger the workflow manually (`workflow_dispatch`, dry-run on) and review the logs.

## Files

Start with `doc.go` for the thesis, then `main.go` for the ADK wiring
(model → tools → agent → runner) and `state.go` for the domain logic.

| File | Responsibility |
|---|---|
| `doc.go` | Package overview — what this sample demonstrates. |
| `main.go` | Model/agent/runner wiring and the concurrent audit loop. |
| `config.go` | Typed configuration from env + flags. |
| `state.go` | **Pure** history reconstruction + staleness logic (unit-tested). |
| `github.go` | GitHub REST + GraphQL client; dry-run handling; bot identity. |
| `tools.go` | Typed `functiontool` definitions. |
| `prompt.go` / `prompt_instruction.txt` | Embedded, pre-rendered agent instruction. |
