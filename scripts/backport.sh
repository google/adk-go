#!/usr/bin/env bash
#
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Backport merged `main` pull requests onto the `v1` maintenance branch.
#
# A PR into `main` is labelled `v1-needed` when it also has to ship to 1.x.
# This script drains that queue: one backport pull request per labelled PR.
#
# The one systematic obstacle is the module path: `main` is
# `google.golang.org/adk/v2`, `v1` is `google.golang.org/adk`, so every Go
# file's import block differs and a plain cherry-pick conflicts on any patch
# that touches imports. That difference is a pure string rewrite, so the patch
# is rewritten before it is applied and the common case lands cleanly. Genuine
# semantic drift still conflicts, and is left for a human.
#
# Each PR is replayed on its own branch, cut fresh from v1, in its own throwaway
# worktree. One conflict costs one backport, not the run; the current checkout
# is never touched.

set -euo pipefail

# inherit_errexit needs bash 4.4; macOS still ships 3.2 as /bin/bash. Checked
# before it is used, so the failure names the requirement instead of printing a
# bare `shopt: invalid option` and exiting 1.
if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4))); then
  printf 'error: bash >= 4.4 required, found %s.\n' "${BASH_VERSION}" >&2
  printf '  On macOS: brew install bash, then run with that bash.\n' >&2
  exit 1
fi
# Without this, a `die` inside a command substitution is silently ignored and
# the caller carries on with an empty result.
shopt -s inherit_errexit

readonly REPO="${ADK_REPO:-google/adk-go}"
readonly MAIN_BRANCH="main"
readonly V1_BRANCH="v1"
readonly V2_MODULE="google.golang.org/adk/v2"
readonly V1_MODULE="google.golang.org/adk"

# `v1-needed` marks a merged main PR as still owing a 1.x equivalent and is what
# the queue filters on. `v1` is informational, and goes on the PRs this opens.
readonly BACKPORT_LABEL="v1-needed"
readonly V1_LABEL="v1"

# Every branch this script pushes lives under here. Also the guard: nothing
# outside this prefix is ever pushed.
readonly BRANCH_PREFIX="backport/v1/pr-"

# Hidden marker on the conflict comment, so a nightly re-run recognises its own
# earlier message and does not post it again.
readonly COMMENT_MARKER="<!-- adk-backport-conflict -->"

list_only=false
open_pr=false
skip_gomod=false
declare -a prs=()

info() { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() {
  printf '\033[1;31merror:\033[0m %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Backport merged '${MAIN_BRANCH}' PRs labelled '${BACKPORT_LABEL}' onto '${V1_BRANCH}'.

Usage:
  scripts/backport.sh --list             Show what is pending.
  scripts/backport.sh [PR...]            Replay locally; nothing is pushed.
  scripts/backport.sh --pr [PR...]       Replay, push, and open the PRs.

With no PR numbers, the whole pending queue is taken. One pull request is
opened per backported PR: a conflict on one does not stop the others.

Options:
  -l, --list        List the pending queue and exit.
  -p, --pr          Push each branch and open its pull request. Without this
                    nothing leaves the machine.
      --skip-gomod  Leave go.mod / go.sum out of the patch. The 1.x dependency
                    set differs from main's, so those hunks often reject on
                    their own; run 'go mod tidy' on the result.
  -h, --help        This text.

Environment:
  ADK_REPO          Override the repository (default ${REPO}).
  ADK_REMOTE        Override the git remote to use.
EOF
}

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "$1 is required but not installed"; }

# Prints the remote pointing at REPO.
detect_remote() {
  if [[ -n "${ADK_REMOTE:-}" ]]; then
    printf '%s\n' "${ADK_REMOTE}"
    return
  fi
  # Anchored to the end of the URL: a substring match would also accept a remote
  # pointing at google/adk-go-experimental.
  local remote url
  while read -r remote url _; do
    url="${url%.git}"
    url="${url%/}"
    if [[ "${url}" == *"/${REPO}" || "${url}" == *":${REPO}" ]]; then
      printf '%s\n' "${remote}"
      return
    fi
  done < <(git remote -v | awk '$3 == "(fetch)"')
  die "no git remote points at ${REPO}; set ADK_REMOTE to choose one"
}

# True when this source commit is already on v1.
#
# Keyed off the "(cherry picked from commit <sha>)" trailer that this script
# writes on every backport, so the question is answered by an exact match on
# something the automation owns. Reading PR numbers out of commit subjects and
# titles instead would mean parsing prose: "Fixes #1152" names an issue, and one
# v1 commit lists seven PR numbers precisely because they were *not* backported.
# The --grep only narrows: it matches the phrase anywhere in a message, and a
# squash merge copies contributor commit bodies onto v1 verbatim, so a v1 commit
# that merely quotes the line would otherwise suppress an unrelated backport.
# Each candidate is then confirmed against the exact trailer line.
already_backported() {
  local remote="$1" sha="$2" candidates candidate
  local trailer="(cherry picked from commit ${sha})"
  # Captured rather than piped straight into the loop: a git failure inside a
  # process substitution is invisible, and would read here as "no candidates",
  # meaning not yet backported -- which opens a duplicate pull request. Same
  # failure mode as the `gh pr list` one below, so it fails closed the same way.
  if ! candidates="$(git log --format=%H --fixed-strings \
    --grep="cherry picked from commit ${sha}" "${remote}/${V1_BRANCH}")"; then
    die "could not search ${V1_BRANCH} for an existing backport of ${sha}"
  fi
  [[ -n "${candidates}" ]] || return 1
  while read -r candidate; do
    [[ -n "${candidate}" ]] || continue
    if git show -s --format=%B "${candidate}" | grep -Fxq "${trailer}"; then
      return 0
    fi
  done <<<"${candidates}"
  return 1
}

# True when a backport branch for this PR is already pushed, meaning a backport
# PR is open or was opened and its branch kept. Deleting the branch re-queues
# the PR, which is the escape hatch when one goes stale.
in_flight() {
  local pr="$1" open_prs
  # An open pull request, not merely a branch. Branch-existence conflated "a
  # backport is under way" with "one was attempted and the run died", and only
  # the second is silent: a job killed between the push and `gh pr create`
  # leaves a branch that would exclude its PR from every later run. Keyed on the
  # head ref, which this script owns.
  #
  # Fails loudly rather than guessing, the same way the queue listing does: a
  # wrong answer here either duplicates a pull request or drops one silently.
  open_prs="$(gh pr list --repo "${REPO}" --head "${BRANCH_PREFIX}${pr}" \
    --state open --json number --jq 'length')" ||
    die "could not check for an open backport of #${pr}"
  [[ "${open_prs}" != "0" ]]
}

# Prints "number<TAB>merge-sha<TAB>title" for each pending PR, oldest merge
# first, so backports land on v1 in the order they landed on main.
pending_queue() {
  local remote="$1" listing
  listing="$(gh pr list --repo "${REPO}" --base "${MAIN_BRANCH}" --state merged \
    --label "${BACKPORT_LABEL}" --limit 200 \
    --json number,title,mergedAt,mergeCommit \
    --jq 'sort_by(.mergedAt)[] | "\(.number)\t\(.mergeCommit.oid // "")\t\(.title)"')" ||
    die "could not list merged ${MAIN_BRANCH} pull requests"

  local number sha title
  while IFS=$'\t' read -r number sha title; do
    [[ -n "${number}" ]] || continue
    if [[ -z "${sha}" ]]; then
      warn "PR #${number} has no merge commit; skipping"
      continue
    fi
    already_backported "${remote}" "${sha}" && continue
    in_flight "${number}" && continue
    printf '%s\t%s\t%s\n' "${number}" "${sha}" "${title}"
  done <<<"${listing}"
}

# Comments on the original PR asking for a manual backport, at most once.
comment_conflict() {
  local pr="$1" sha="$2" existing me
  # Authored-by filter: the marker is public, so without it anyone can post it
  # on a pull request before it ever conflicts and permanently suppress the only
  # signal a human gets. Accepts the bot (how CI posts) or whoever is
  # authenticated (how a local --pr run posts), and nobody else.
  # gh's --jq takes no --arg, so both values are interpolated. The marker is a
  # readonly constant; the login has to be validated, because on an API error
  # `gh api --jq` prints the response body to *stdout* -- 2>/dev/null does not
  # suppress it and `|| echo ''` only appends to it. Unvalidated, a failed call
  # puts a JSON blob into the filter, jq refuses to parse it, `gh pr view` exits
  # non-zero into the handler below, and the conflict comment silently never
  # posts. That is the CI path: GITHUB_TOKEN is an installation token and
  # GET /user needs user-to-server auth. Anything that is not a login is
  # dropped, and an empty login matches no author.
  me="$(gh api user --jq '.login' 2>/dev/null)" || me=''
  [[ "${me}" =~ ^[A-Za-z0-9-]{1,39}$ ]] || me=''
  existing="$(gh pr view "${pr}" --repo "${REPO}" --json comments \
    --jq '[.comments[]
      | select(.author.login == "github-actions[bot]" or .author.login == "'"${me}"'")
      | select(.body | contains("'"${COMMENT_MARKER}"'"))] | length')" ||
    {
      warn "could not read comments on PR #${pr}; not commenting"
      return 0
    }
  if [[ "${existing}" != "0" ]]; then
    info "  conflict already reported on PR #${pr}; not commenting again"
    return 0
  fi

  gh pr comment "${pr}" --repo "${REPO}" --body "${COMMENT_MARKER}
Automatic backport to \`${V1_BRANCH}\` failed: this change does not replay
cleanly, which usually means 1.x lacks something 2.x already had.

To finish it by hand:

\`\`\`bash
scripts/backport.sh ${pr}
\`\`\`

That leaves the conflicts as \`.rej\` files in a scratch worktree and prints its
path. Finish it in that worktree — **do not re-run the script**, which starts
from a clean replay and discards what you resolved:

\`\`\`bash
cd <the worktree it printed>
# resolve the conflicts, then drop the .rej files git apply left behind:
find . -name '*.rej' -delete
# -A, so files the patch *added* are included: apply --reject leaves them
# untracked, and -u would stage only what git already knew about:
git add -A
git commit -m '<subject>

(cherry picked from commit ${sha})'
git push <remote> HEAD:refs/heads/${BRANCH_PREFIX}${pr}
gh pr create --repo ${REPO} --base ${V1_BRANCH} --head ${BRANCH_PREFIX}${pr}
\`\`\`

This is reported once. The backport is still queued and will be retried, so if a
later \`${V1_BRANCH}\` change makes it apply, it lands without further action." >/dev/null ||
    warn "could not comment on PR #${pr}"
}

# Replays one PR onto its own branch and, with --pr, opens the backport PR.
#
# Returns 0 on success, 1 on conflict, 2 when the patch is genuinely empty, and
# 3 for anything the tooling could not do -- a missing commit, a merge commit
# that does not carry the whole PR, a worktree, patch, commit, push or PR call
# that failed. The distinction is what matters: a conflict is a normal outcome
# that a human resolves and leaves the run green, while a 3 goes red.
backport_one() {
  local remote="$1" pr="$2" sha="$3"
  local branch="${BRANCH_PREFIX}${pr}"
  local worktree="${TMPDIR:-/tmp}/adk-backport/${branch//\//-}"
  local subject author date patch message

  if ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then
    warn "commit ${sha} for PR #${pr} is not in this clone; fetch and retry"
    return 3
  fi

  subject="$(git log -1 --format=%s "${sha}")"
  info "PR #${pr}: ${subject}"

  # A squash merge puts the whole pull request in one commit, which is what the
  # replay below assumes. A rebase merge does not: mergeCommit.oid is then only
  # the last commit of the branch, so a multi-commit PR would be backported in
  # part, apply cleanly, and clear itself from the queue on the trailer -- a
  # silent half-backport. Every merge on main so far is a squash, but the
  # repository allows rebase merges, so it is asserted rather than assumed.
  # Both sides are sorted and compared under LC_ALL=C. jq sorts by codepoint and
  # sort(1) by locale, and the two disagree wherever a name starts with a
  # non-alphanumeric -- `.github/...` against `CONTRIBUTING.md` under
  # en_US.UTF-8. comm then reports present files as missing and this guard
  # refuses an ordinary squash merge. CI never saw it because the runner is
  # C.UTF-8; a workstation is not, and this path is the one CONTRIBUTING and the
  # conflict comment both point people at.
  #
  # quotePath off for the same reason: git C-escapes a non-ASCII path where the
  # API returns it raw, which breaks the comparison on the runner too.
  #
  # Paginated: `gh pr view --json files` stops at 100 (#1109 has 547), which
  # made the comparison quieter than it read.
  # Fails closed. A guard against a silent half-backport must not disable itself
  # silently: if the file list cannot be fetched, the merge shape is unknown,
  # and unknown is not the same as fine.
  # Counts first, so the common case costs one request. Paginating the file list
  # up front turned a 547-file pull request into ~19 requests, each of which can
  # now fail into the refusal below -- the cheap check and the fail-closed
  # behaviour compose into a new way for a green run to go red. A squash carries
  # the whole pull request, so the counts match; a rebase merge records only the
  # last commit, so they do not. The full list is fetched only once they differ,
  # to name what is missing.
  #
  # Residual, and it is wider than the counts: this compares file *names*, so a
  # rebase whose last commit touches the same files as the whole branch clears
  # both the count check and the comm below. Comparing lists rather than content
  # is what allows that; the count step neither introduced it nor widened it.
  # What covers it is the replay itself -- that last commit's patch is written
  # against a parent that is not on v1, so it would almost certainly fail to
  # apply rather than land silently.
  local pr_changed commit_files commit_count pr_files missing
  if ! pr_changed="$(gh pr view "${pr}" --repo "${REPO}" --json changedFiles \
    --jq '.changedFiles')" || [[ ! "${pr_changed}" =~ ^[0-9]+$ ]]; then
    warn "  could not count the files of PR #${pr}, so the merge shape cannot be
  checked. Refusing rather than risking a partial backport of a rebase merge."
    return 3
  fi
  commit_files="$(git -c core.quotePath=false show --name-only --format= "${sha}" |
    LC_ALL=C sort -u)"
  commit_count="$(printf '%s' "${commit_files}" | grep -c '^' || true)"
  [[ -n "${commit_files}" ]] || commit_count=0

  if [[ "${commit_count}" -ne "${pr_changed}" ]]; then
    if ! pr_files="$(gh api --paginate "repos/${REPO}/pulls/${pr}/files" \
      --jq '.[].filename')"; then
      warn "  PR #${pr} reports ${pr_changed} changed files but merge commit
  ${sha:0:8} carries ${commit_count}, and the file list could not be fetched to
  say which. Refusing rather than risking a partial backport."
      return 3
    fi
    pr_files="$(printf '%s\n' "${pr_files}" | LC_ALL=C sort -u)"
    missing="$(LC_ALL=C comm -23 <(printf '%s\n' "${pr_files}") <(printf '%s\n' "${commit_files}"))"
    if [[ -n "${missing}" ]]; then
      warn "  merge commit ${sha:0:8} does not carry the whole of PR #${pr}
  Missing: $(printf '%s' "${missing}" | tr '\n' ' ')
  This is what a rebase merge looks like, where the recorded merge commit is
  only the last commit of the branch. Backporting it would ship part of the
  change and then mark the PR done. Cherry-pick the range by hand instead."
      return 3
    fi
  fi

  # Clear any leftover from an earlier run. Removing the directory alone is not
  # enough: git keeps the worktree registered in .git/worktrees and then refuses
  # to re-add the path as "missing but already registered".
  git worktree remove --force "${worktree}" 2>/dev/null || true
  rm -rf "${worktree}"
  git worktree prune
  mkdir -p "$(dirname "${worktree}")"

  # Deliberately detached, and it stays that way: nothing here creates a local
  # branch. A named branch would have to be resolved again at push time, and if
  # the name already existed in this clone the push would send that instead of
  # the replay. HEAD is the only thing committed to and the only thing pushed.
  #
  # Checked rather than assumed: with `|| status=$?` at the call site, errexit
  # is inert for this whole function, so a failure here would otherwise fall
  # through to `git apply` and be reported to the contributor as a conflict in
  # their patch.
  if ! git worktree add --quiet --detach "${worktree}" "${remote}/${V1_BRANCH}"; then
    warn "  could not create worktree ${worktree}"
    return 3
  fi

  local -a pathspec=()
  if [[ "${skip_gomod}" == true ]]; then
    # :(top) anchors to the repository root; without it a run from a
    # subdirectory silently narrows the patch to that subtree.
    pathspec=(-- ':(top)' ':(top,exclude)go.mod' ':(top,exclude)go.sum'
      ':(top,exclude)*/go.mod' ':(top,exclude)*/go.sum')
  fi

  # BSD mktemp wants a template, so give it one on every platform.
  if ! patch="$(mktemp "${TMPDIR:-/tmp}/adk-backport-patch.XXXXXX")"; then
    warn "  could not create a temporary file for the patch"
    cleanup_worktree "${worktree}"
    return 3
  fi

  # A main squash commit has a single parent, so `git show` is the whole change.
  # Rewriting the module path makes the context lines match the v1 tree, which
  # is what lets the patch apply directly.
  git show --binary --format= "${sha}" "${pathspec[@]}" |
    sed "s|${V2_MODULE}|${V1_MODULE}|g" |
    # Put back the one place that rewrite is wrong: a go.mod require line reads
    # `google.golang.org/adk/v2 v2.0.0`, and dropping the /v2 leaves
    # `google.golang.org/adk v2.0.0`, which the go tool rejects outright
    # ("should be v0 or v1, not v2"). Left alone the hunk simply fails to
    # apply, which is the honest outcome: the 1.x dependency set differs.
    sed -E "s|${V1_MODULE} (v2[0-9.])|${V2_MODULE} \1|g" >"${patch}"

  # Checked, and checked before the emptiness test. errexit is inert in this
  # function (the call site is `|| status=$?`), so a failed `git show` would
  # otherwise leave an empty file, be read as "nothing to backport", and be
  # tallied as a benign skip on a green run. Of the four return codes 2 is the
  # one that quietly absorbs tooling failure, so it has to be earned.
  local -a pipe_status=("${PIPESTATUS[@]}")
  local stage
  for stage in "${pipe_status[@]}"; do
    if [[ "${stage}" -ne 0 ]]; then
      warn "  could not build the patch for #${pr} (pipeline status: ${pipe_status[*]})"
      rm -f "${patch}"
      cleanup_worktree "${worktree}"
      return 3
    fi
  done

  if [[ ! -s "${patch}" ]]; then
    warn "  empty patch; nothing to backport"
    rm -f "${patch}"
    cleanup_worktree "${worktree}"
    return 2
  fi

  if ! git -C "${worktree}" apply --index "${patch}" 2>/dev/null; then
    warn "  does not apply cleanly"
    if [[ "${open_pr}" == true ]]; then
      # Unattended: the worktree is worthless to anyone, so take it away.
      rm -f "${patch}"
      cleanup_worktree "${worktree}"
      comment_conflict "${pr}" "${sha}"
    else
      # By hand: leave the .rej files where they can be worked on.
      git -C "${worktree}" apply --reject "${patch}" || true
      rm -f "${patch}"
      cat >&2 <<EOF

  Conflicts are in ${worktree}, as *.rej files. Finish them there -- do not
  re-run the script, which starts from a clean replay and discards this:

    cd ${worktree}
    # resolve the conflicts, then drop the .rej files git apply left behind:
    find . -name '*.rej' -delete
    # -A, so files the patch *added* are included; the delete above is what
    # keeps .rej files off ${V1_BRANCH}:
    git add -A
    git commit -m '<subject>

(cherry picked from commit ${sha})'
    git push ${remote} HEAD:refs/heads/${branch}
    gh pr create --repo ${REPO} --base ${V1_BRANCH} --head ${branch}

  The worktree is detached on purpose, so the push needs the HEAD:refs/heads/
  form: there is no local branch of that name to push by name.

EOF
    fi
    return 1
  fi
  rm -f "${patch}"

  author="$(git log -1 --format='%an <%ae>' "${sha}")"
  date="$(git log -1 --format=%aD "${sha}")"

  # Command substitution strips trailing newlines, so appending a blank line
  # here always leaves exactly one before the trailer. It matters: around one in
  # eight main squash commits has no body, and without the separator git folds
  # the trailer into the subject and the PR title ends in a bare 40-char SHA.
  message="$(git log -1 --format=%B "${sha}" | sed "s|${V2_MODULE}|${V1_MODULE}|g")

(cherry picked from commit ${sha})"

  if ! git -C "${worktree}" commit --quiet --author="${author}" --date="${date}" \
    --message="${message}"; then
    warn "  could not commit the replay in ${worktree}"
    cleanup_worktree "${worktree}"
    return 3
  fi
  info "  applied cleanly"

  if [[ "${open_pr}" != true ]]; then
    info "  replay is ready in ${worktree} (nothing pushed)"
    info "  push it with: git -C ${worktree} push ${remote} HEAD:refs/heads/${branch}"
    return 0
  fi

  open_backport_pr "${remote}" "${pr}" "${branch}" "${worktree}" "${subject}"
}

# Pushes the branch and opens its pull request.
open_backport_pr() {
  local remote="$1" pr="$2" branch="$3" worktree="$4" subject="$5" url

  # The invariant is held by construction: `branch` is the readonly prefix plus
  # an integer. This is an assertion against a future refactor passing something
  # else, not the thing that makes it safe. The second guard is load-bearing: a
  # branch with no commits pushes fine and then fails at PR creation, leaving an
  # orphan behind. Both read HEAD, which is what the push sends.
  if [[ "${branch}" != "${BRANCH_PREFIX}"* ]]; then
    warn "refusing to push '${branch}': not a ${BRANCH_PREFIX} branch"
    cleanup_worktree "${worktree}"
    return 3
  fi
  if [[ "$(git -C "${worktree}" rev-list --count "${remote}/${V1_BRANCH}..HEAD")" -eq 0 ]]; then
    warn "nothing to push for #${pr}: HEAD is ${V1_BRANCH}"
    cleanup_worktree "${worktree}"
    return 3
  fi

  info "  pushing ${branch}"
  if ! git -C "${worktree}" push --quiet "${remote}" "HEAD:refs/heads/${branch}"; then
    warn "could not push ${branch}"
    cleanup_worktree "${worktree}"
    return 3
  fi

  local err
  err="$(mktemp "${TMPDIR:-/tmp}/adk-backport-err.XXXXXX")"
  url="$(gh pr create --repo "${REPO}" --base "${V1_BRANCH}" --head "${branch}" \
    --title "${subject}" --body "Backports #${pr} from \`${MAIN_BRANCH}\` to \`${V1_BRANCH}\`.

Import paths were rewritten from \`${V2_MODULE}\` to \`${V1_MODULE}\`; the change
is otherwise identical to the original.

A clean apply is not a correct backport — 1.x can lack a helper that main
already had — so read the diff rather than just the checks.

If the merge box says the workflows are awaiting approval, click **Approve
workflows to run**: pull requests opened by \`github-actions[bot]\` start their
CI held.

Generated by \`scripts/backport.sh\`." 2>"${err}")" || {
    # The one failure that is a repository setting rather than a mistake, and
    # the error for it reads as a bare permissions problem otherwise.
    if grep -q 'not permitted to create or approve pull requests' "${err}"; then
      warn "this repository does not allow GitHub Actions to create pull requests.
  Enable Settings -> Actions -> General -> 'Allow GitHub Actions to create and
  approve pull requests' (it may be inherited from the organization)."
    else
      warn "could not open the PR for #${pr}: $(tr '\n' ' ' <"${err}")"
    fi
    rm -f "${err}"
    # Take the branch back down. in_flight keys on an open pull request now, so
    # a leftover branch no longer excludes #${pr} -- but it would collide with
    # the next attempt's push, so it still goes.
    if git -C "${worktree}" push --quiet "${remote}" --delete "${branch}" 2>/dev/null; then
      info "  removed ${branch}; #${pr} stays queued and will be retried"
    else
      warn "could not delete ${branch}; the next attempt for #${pr} will have to
  force past it or you can remove it by hand:
    git push ${remote} --delete ${branch}"
    fi
    cleanup_worktree "${worktree}"
    return 3
  }
  rm -f "${err}"
  info "  ${url}"

  # Labelled separately: `gh pr create --label` fails outright on a label the
  # repository does not have, and the pull request is already open by now.
  gh pr edit "${url##*/}" --repo "${REPO}" --add-label "${V1_LABEL}" >/dev/null ||
    warn "could not add the '${V1_LABEL}' label; add it by hand"

  # Explicit: as the last statement, cleanup_worktree's status would become this
  # function's, so a worktree that refused to go away would report a landed
  # backport as a conflict and tell the contributor to redo it by hand.
  cleanup_worktree "${worktree}"
  return 0
}

# No branch to delete: the worktree is detached throughout, so the only ref this
# script ever writes is the remote one it pushes.
cleanup_worktree() {
  local worktree="$1"
  git worktree remove --force "${worktree}" 2>/dev/null || rm -rf "${worktree}"
}

main() {
  require_cmd git
  require_cmd gh
  gh auth status >/dev/null 2>&1 || die "gh is not authenticated; run 'gh auth login'"
  git rev-parse --git-dir >/dev/null 2>&1 || die "not inside a git repository"

  local remote
  remote="$(detect_remote)"
  info "using remote '${remote}' for ${REPO}"

  # Spell the refspecs out. A CI checkout configures a narrow remote.fetch
  # covering only the branch it cloned, and under it a bare `git fetch <remote>
  # v1` updates FETCH_HEAD without ever writing refs/remotes/<remote>/v1, which
  # everything below refers to.
  git fetch --quiet "${remote}" \
    "+refs/heads/${MAIN_BRANCH}:refs/remotes/${remote}/${MAIN_BRANCH}" \
    "+refs/heads/${V1_BRANCH}:refs/remotes/${remote}/${V1_BRANCH}" ||
    die "failed to fetch ${MAIN_BRANCH} and ${V1_BRANCH} from ${remote}"

  git merge-base "${remote}/${MAIN_BRANCH}" "${remote}/${V1_BRANCH}" >/dev/null 2>&1 ||
    die "${MAIN_BRANCH} and ${V1_BRANCH} have no common ancestor here; a shallow
clone cannot be used, check out with fetch-depth: 0"

  # Explicit numbers are filtered out of the same queue rather than resolved
  # separately, so a hand-run and an unattended run cannot disagree about what
  # is pending or what a PR's merge commit is. Asking for one that is already
  # backported is then a no-op, which is what makes a re-run safe.
  local queue
  queue="$(pending_queue "${remote}")"
  if [[ ${#prs[@]} -gt 0 && -n "${queue}" ]]; then
    local wanted
    wanted="$(printf '%s\n' "${prs[@]}")"
    queue="$(awk -F'\t' 'NR==FNR {want[$0]; next} $1 in want' \
      <(printf '%s\n' "${wanted}") <(printf '%s\n' "${queue}"))"
  fi

  if [[ -z "${queue}" ]]; then
    if [[ ${#prs[@]} -gt 0 ]]; then
      # Asked about specific numbers: say something about those, not about the
      # queue as a whole. This is the command the conflict comment tells a
      # contributor to run, so "nothing pending" reads as though their PR does
      # not need a backport at all.
      info "nothing to do for ${prs[*]/#/#}. A PR is skipped when it is already
  on '${V1_BRANCH}', when its '${BRANCH_PREFIX}<n>' branch is already pushed, or
  when it is not a merged '${MAIN_BRANCH}' PR labelled '${BACKPORT_LABEL}'.
  Run --list to see the whole queue."
    else
      info "nothing pending: no merged '${MAIN_BRANCH}' PR labelled '${BACKPORT_LABEL}' is missing from '${V1_BRANCH}'"
    fi
    return 0
  fi

  if [[ "${list_only}" == true ]]; then
    info "pending backports (oldest merge first):"
    while IFS=$'\t' read -r number _ title; do
      printf '  #%-6s %s\n' "${number}" "${title}"
    done <<<"${queue}"
    return 0
  fi

  local number sha title status
  local -a done_prs=() conflicted=() empty=() broken=()
  while IFS=$'\t' read -r number sha title; do
    status=0
    backport_one "${remote}" "${number}" "${sha}" || status=$?
    case "${status}" in
      0) done_prs+=("${number}") ;;
      1) conflicted+=("${number}") ;;
      2) empty+=("${number}") ;;
      *) broken+=("${number}") ;;
    esac
  done <<<"${queue}"

  info "backported ${#done_prs[@]}, conflicted ${#conflicted[@]}, empty ${#empty[@]}, failed ${#broken[@]}"
  [[ ${#conflicted[@]} -eq 0 ]] ||
    warn "needs a manual backport: ${conflicted[*]/#/#}"

  # A conflict is a real outcome rather than a broken run: the PRs that applied
  # have their pull requests, and the comment on the original is the signal. A
  # push or PR call that failed is different -- nobody has been told, so the run
  # has to go red.
  [[ ${#broken[@]} -eq 0 ]] ||
    die "could not open backports for: ${broken[*]/#/#}"
  return 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -l | --list) list_only=true ;;
    -p | --pr) open_pr=true ;;
    --skip-gomod) skip_gomod=true ;;
    -h | --help)
      usage
      exit 0
      ;;
    -*) die "unknown option: $1" ;;
    *)
      [[ "$1" =~ ^[0-9]+$ ]] || die "not a PR number: $1"
      prs+=("$1")
      ;;
  esac
  shift
done

main
