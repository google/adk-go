# How to contribute

We'd love to accept your patches and contributions to this project.

-   [How to contribute](#how-to-contribute)
-   [Branches](#branches)
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

## Branches

ADK Go uses two long-lived branches:

-   **`main`** — the actively developed 2.x line. This is the default branch and
    the base for new pull requests.
-   **`v1`** — the maintenance branch for the 1.x line. Target this branch only
    for fixes that need to ship to 1.x.

As part of the ADK Go 2.0 release the previous `main` was renamed to `v1`, and
the `v2` development branch became `main`. GitHub automatically retargets open
pull requests and migrates branch protection, but **existing local clones must
be re-synced** — running `git pull` on an old `main` reports diverged history
because the branch now points at a different line.

To move a local clone whose `main` still tracks the old 1.x line over to the new
layout:

```bash
git fetch origin --prune
git branch -m main v1            # your old "main" is the v1 (1.x) line
git branch -u origin/v1 v1       # track origin/v1
git switch -c main origin/main   # check out the new main (2.x)
git remote set-head origin -a
```

If you don't need the 1.x line locally, reset your `main` to the new one instead
(this discards local `main` commits that aren't pushed elsewhere):

```bash
git fetch origin --prune
git switch main
git reset --hard origin/main
git remote set-head origin -a
```

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

# ADK Web

## Updating ADK web version to latest

-   Run `./scripts/adk-web/update-adk-web.sh` to update the web UI to the latest version from [GitHub](https://github.com/google/adk-web).
-   Run `docker run -it adk-web-builder:latest sh -c "<COMMAND>"` to start the container and debug the build, e.g.:
    -   `docker run -it adk-web-builder:latest sh -c "ls -alh dist/agent_framework_web/browser"` to view the built files.
    -   `docker run -it adk-web-builder:latest sh -c "npm run build"` to debug the build output.

### Documentation

For any changes that impact user-facing documentation (guides, API reference,
tutorials), please open a PR in the
[adk-docs](https://github.com/google/adk-docs) repository to update the relevant
parts before or alongside your code PR.

### Alignment with adk-python
We lean on [adk-python](https://github.com/google/adk-python) for being the source of truth and one should refer to adk-python for validation.
