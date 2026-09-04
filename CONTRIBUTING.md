# How to contribute

We'd love to accept your patches and contributions to this project.

-   [How to contribute](#how-to-contribute)
-   [Branches](#branches)
    -   [Backporting to `v1`](#backporting-to-v1)
-   [Multi-Module Development](#multi-module-development)
-   [Before you begin](#before-you-begin)
    -   [Sign our Contributor License Agreement](#sign-our-contributor-license-agreement)
    -   [Review our community guidelines](#review-our-community-guidelines)
    -   [Code reviews](#code-reviews)
-   [Contribution workflow](#contribution-workflow)
    -   [Finding Issues to Work On](#finding-issues-to-work-on)
    -   [Requirement for PRs](#requirement-for-prs)
    -   [Large or Complex Changes](#large-or-complex-changes)
    -   [Testing Requirements](#testing-requirements)
    -   [Unit Tests](#unit-tests)
    -   [Manual End-to-End (E2E) Tests](#manual-end-to-end-e2e-tests)
    -   [Documentation](#documentation)
    -   [Alignment with adk-python](#alignment-with-adk-python)
-   [AI-assisted development](#ai-assisted-development)

## Branches

ADK Go uses two long-lived branches:

-   **`main`** — the actively developed 2.x line. This is the default branch and
    the base for new pull requests.
-   **`v1`** — the maintenance branch for the 1.x line. Target this branch only
    for fixes that need to ship to 1.x.

The `v1` branch is a snapshot of the 1.x line, branched from `main` before the
2.0 work landed. `main` then continued forward as the 2.x line (the 2.0 release
was merged into it), so its history is unbroken — old clones fast-forward
cleanly. There is no need to re-sync or rename anything locally:

```bash
git switch main
git pull            # fast-forwards onto the 2.x line
```

To work on a 1.x fix, base your branch on `v1`:

```bash
git switch -c my-fix origin/v1
```

### Backporting to `v1`

Three labels are in play, and only one of them does anything:

| Label       | Meaning                                                        |
| ----------- | -------------------------------------------------------------- |
| `v1-needed` | This change still owes a 1.x equivalent. **Drives the queue.**  |
| `v2`        | Informational: the PR targets `main`.                           |
| `v1`        | Informational: the PR targets the `v1` branch.                  |

Add `v1-needed` while the context is fresh — at review time, not later. It is
the only signal the automation reads, so a fix that should reach 1.x without it
is a fix nobody backports.

Merging a `v1-needed` PR opens its backport by itself: the
[`Backport to v1`](.github/workflows/backport.yml) workflow replays the squash
commit onto a branch cut from `v1` and opens a pull request against it. **One
backport PR per original PR** — a change that conflicts costs its own backport
and nothing else. The workflow works off the label queue rather than the merge
event, so a PR labelled *after* it merged is picked up by the nightly run, and
re-running on one that is already backported is a no-op rather than an error.

The workflow runs `scripts/backport.sh`, which is also usable directly — to
check the queue, or to work through a conflict the automation could not:

```bash
scripts/backport.sh --list         # what is pending?
scripts/backport.sh 1301           # replay one PR into a scratch worktree
scripts/backport.sh --pr           # replay the whole queue and open the PRs
```

It replays each commit on a detached HEAD in a scratch worktree under `$TMPDIR`,
so your working tree and your branches are left alone — it does add and remove
its own worktree in your clone, and prunes stale worktree registrations while
doing so. Nothing leaves the machine without `--pr`.

The script rewrites the module path (`google.golang.org/adk/v2` →
`google.golang.org/adk`) in each patch before applying it. That difference is
otherwise the main source of cherry-pick conflicts, because it puts every Go
file's import block out of sync between the two branches.

A PR leaves the queue when its change reaches `v1` — matched on the
`(cherry picked from commit <sha>)` trailer the script writes, not on anything
parsed out of a title — or while a backport pull request for it is open. Close
that pull request and delete its `backport/v1/pr-<n>` branch and the PR is
queued again, which is how you regenerate a backport that went stale. A branch
left behind by a run that died does not suppress anything: the next run replays
it.

Backport PRs get the usual CI, because the `pull_request` triggers in `go.yml`
and `apidiff.yml` filter on the base branch and list `v1` — but **the runs start
held**. A pull request opened by `github-actions[bot]` gets its workflows in an
approval-required state, so open the backport PR and click **Approve workflows
to run** in the merge box; anyone with write access can.

That is what running on the built-in `GITHUB_TOKEN` costs, and it is worth
paying: no long-lived credential lives in the repository, and the click lands on
a pull request someone has to review anyway. The one repository setting it needs
is "Allow GitHub Actions to create and approve pull requests", under
Settings → Actions → General.

Authorship follows the same mechanism as today, with one visible change. Each
replayed commit keeps its original author, and squash-merging the backport PR
makes the PR owner the author of the commit that lands on `v1`, recording the
original author as a `Co-authored-by:` trailer. Because these pull requests are
opened by `github-actions[bot]`, that owner is now the bot: `git blame` on `v1`
will point at it, and the human author is the trailer.

Two things the tooling cannot do for you:

-   **A clean apply is not a correct backport.** `v1` may lack a helper or a
    refactor that `main` already had, so a patch can apply and still not
    compile. Build and test the backport branch before merging it.
-   **Dependency changes need judgement.** The 1.x dependency set differs from
    main's, so `go.mod` hunks often reject. Re-run with `--skip-gomod` and
    `go mod tidy` the module instead.

When a patch conflicts, a local run applies what it can and leaves `.rej` files
in the worktree to work from. The unattended run does not push a half-applied
tree: it comments on the original PR asking for a manual backport, once, and
leaves the PR queued so a later `v1` change can still let it land on its own.

## Multi-Module Development

**Policy**: New integrations with heavy or optional dependencies must be created as separate Go modules.

**Local Development**: Contributors should use `go work init && go work use -r .` to set up their local workspaces.

**Steps to Add a New Module (e.g., `plugin/myplugin`)**:
1. Navigate into the directory: `cd <module_directory_path>`
2. Initialize the module: `go mod init google.golang.org/adk/<module_directory_path>`
3. Add your Go code, dependencies, and tests.
4. Tidy the module: `go mod tidy`
5. Return to the repo root.
6. Tidy the root module: `go mod tidy`
7. Add the module to your workspace: `go work use ./<module_directory_path>`
8. Verify everything builds and tests from the root: `go build work && go test work`. The CI will automatically pick up the new module on the PR.

**Release Tagging**:
- **Core Module**: Tags remain `vX.Y.Z` (e.g., `v2.1.0`).
- **Submodules**: Tags are prefixed with the full module path directory, e.g., `plugin/agentanalytics/v0.1.0`. This is the standard Go way to version modules not at the repo root.
- **go get / go install**: Consumers will use:
  - `go get google.golang.org/adk/v2@v2.1.0`
  - `go get google.golang.org/adk/plugin/agentanalytics@v0.1.0`
- **Version Coupling**: Each submodule's `go.mod` will specify the minimum version of `google.golang.org/adk/v2` it depends on. Submodules can be released independently of the core module and each other.
- **go.work Impact**: `go.work` is for local development only and does not affect how modules are versioned, tagged, or fetched by consumers.

## Before you begin

### Sign our Contributor License Agreement

All submissions to this project need to follow Google’s [Contributor
License Agreement (CLA)](https://cla.developers.google.com/about), which
covers any original work of authorship included in the submission. This
doesn't prohibit the use of coding assistance tools, including tool-,
AI-, or machine-generated code, as long as these submissions abide by the
CLA's requirements.

You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Contribution workflow

### Finding Issues to Work On

-   Browse issues labeled **`good first issue`** (newcomer-friendly) or **`help
    wanted`** (general contributions).
-   For other issues, please kindly ask before contributing to avoid
    duplication.

### Requirement for PRs

-   Code must follow [Google Go Style Guide](https://google.github.io/styleguide/go/index).
-   All PRs, other than small documentation or typo fixes, should have an Issue
    associated. If a relevant issue doesn't exist, please create one first or
    you may instead describe the bug or feature directly within the PR
    description, following the structure of our issue templates.
-   Small, focused PRs. Keep changes minimal—one concern per PR.
-   Use [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`,
    `fix:`, `docs:` and so on, optionally scoped as `fix(runner):` — in the PR
    title, and in the commit subject too on a single-commit PR, where that is
    what lands instead. Release tooling reads the landed subject to pick the
    next version and build the release notes, and silently skips anything with
    no recognized type.
-   For bug fixes or features, please provide logs or screenshots after the fix
    is applied to help reviewers better understand the fix.
-   Please include a `testing plan` section in your PR to talk about how you
    will test. This will save time for PR review. See `Testing Requirements`
    section for more details.

### Large or Complex Changes

For substantial features or architectural revisions:

-   Open an Issue First: Outline your proposal, including design considerations
    and impact.
-   Gather Feedback: Discuss with maintainers and the community to ensure
    alignment and avoid duplicate work.

### Testing Requirements

To maintain code quality and prevent regressions, all code changes must include
comprehensive tests and verifiable end-to-end (E2E) evidence.

#### Unit Tests

Please add or update unit tests for your change.

Requirements for unit tests:

-   Cover new features, edge cases, error conditions, and typical
    use cases.
-   Fast and isolated.
-   Written clearly with descriptive names.
-   Free of external dependencies (use mocks or fixtures as needed).
-   Aim for high readability and maintainability; include comments for complex
    scenarios.

#### Manual End-to-End (E2E) Tests

Manual E2E tests ensure integrated flows work as intended. Your tests should
cover all scenarios. Sometimes, it's also good to ensure relevant functionality
is not impacted.

Depending on your change:

-   **ADK Web:**

    -   Capture and attach relevant screenshots demonstrating the UI/UX changes
        or outputs.
    -   Label screenshots clearly in your PR description.

-   **Runner:**

    -   Provide testing setup. For example, the agent definition, and the
        runner setup.
    -   Include the command used and console output showing test results.
    -   Highlight sections of the log that directly relate to your change.

## AI-assisted development

This repo ships skills for AI coding agents (Antigravity, Gemini CLI, Claude
Code, and others) in `.agents/skills/`. Compatible tools load them on their own;
otherwise read the relevant `SKILL.md` before starting that kind of work.

-   **`adk-sample-creator`** — authoring or reworking a runnable example under
    `examples/`: directory layout, `main.go` anatomy, the README template with
    its diagram and transcript, and the checks to run before opening the PR.

`AGENTS.md` carries the rest of the project context an agent needs.

# ADK Web

## Refreshing the embedded web bundle

The web UI ships as a prebuilt Angular bundle, committed under
`cmd/launcher/web/webui/distr/` and embedded into the server binary. It is built
from a pinned revision of [adk-web](https://github.com/google/adk-web), not from
the latest `main`.

-   Run `./scripts/adk-web/update-adk-web.sh` to rebuild the bundle from the
    pinned revision. It reports the previous and the new upstream commit, and a
    URL comparing the two.
-   The committed bundle predates the pin and carries no provenance file, so
    the pin does not yet describe what is in `distr/`. The first refresh will
    therefore move the UI forward to the pinned revision rather than reproduce
    the current bundle. Review that refresh as a version bump. It reads the
    previous commit out of that missing file, so it reports the previous commit
    as unknown and prints no compare URL. Every refresh after it prints both.
-   Run `docker run -it adk-web-builder:latest sh -c "<COMMAND>"` to start the container and debug the build, e.g.:
    -   `docker run -it adk-web-builder:latest sh -c "ls -alh dist/agent_framework_web/browser"` to view the built files.
    -   `docker run -it adk-web-builder:latest sh -c "npm run build"` to debug the build output.

Please leave the revision pinned. It used to track `main`, so one refresh
silently pulled in an upstream change that moved 30 API endpoints, and most of
the UI returned 404 for two months.

See [scripts/adk-web/README.md](scripts/adk-web/README.md) for how to bump the
pin deliberately, and for the checklist to apply when reviewing a refresh. A
refresh also needs the UI-to-server route contract test to pass:

```bash
go test ./server/adkrest/ -run TestUIContract
```

### Documentation

For any changes that impact user-facing documentation (guides, API reference,
tutorials), please open a PR in the
[adk-docs](https://github.com/google/adk-docs) repository to update the relevant
parts before or alongside your code PR.

### Alignment with adk-python
We lean on [adk-python](https://github.com/google/adk-python) for being the source of truth and one should refer to adk-python for validation.
