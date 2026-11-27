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
# Sync new commits from `main` onto `v3` via a single rolling pull request.
#
# Because this repository disallows merge commits (`allow_merge_commit: false`),
# every sync PR lands on `v3` as a squash or rebase commit, leaving git's
# merge-base at the original branch point. To avoid false conflicts on files
# synced in earlier runs, each sync commit carries a `Synced-from-main: <sha>`
# trailer. The next run performs a 3-way merge of `v3` and `main` using that
# exact `main` commit as the merge base.
#
# On a clean merge with `--pr`, the script pushes one commit to `sync/main-to-v3`
# and opens (or updates) a pull request targeting `v3` so a maintainer can
# approve workflows, review, and merge it. Re-runs when nothing changed are
# no-ops so CI and approvals on an open sync PR stay intact, and manual commits
# or conflict resolutions pushed to `sync/main-to-v3` are never overwritten.

set -euo pipefail

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4))); then
  printf 'error: bash >= 4.4 required, found %s.\n' "${BASH_VERSION}" >&2
  printf '  On macOS: brew install bash, then run with that bash.\n' >&2
  exit 1
fi
shopt -s inherit_errexit

readonly REPO="${ADK_REPO:-google/adk-go}"
readonly MAIN_BRANCH="main"
readonly V3_BRANCH="v3"
readonly SYNC_BRANCH="sync/main-to-v3"
readonly TRAILER_KEY="Synced-from-main"
readonly MAX_LISTED_COMMITS=50
readonly BOT_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"

open_pr=false
status_only=false

declare -a tmp_files=()
cleanup_tmp() {
  if [[ ${#tmp_files[@]} -gt 0 ]]; then
    rm -f "${tmp_files[@]}"
  fi
}
trap cleanup_tmp EXIT

make_tmp() {
  local f
  f="$(mktemp "${TMPDIR:-/tmp}/adk-sync-v3.XXXXXX")"
  tmp_files+=("${f}")
  printf '%s\n' "${f}"
}

info() { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() {
  printf '\033[1;31merror:\033[0m %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Sync new commits from '${MAIN_BRANCH}' onto '${V3_BRANCH}' via a rolling PR.

Usage:
  scripts/sync-v3.sh --status    Show last synced commit, pending count, and merge check.
  scripts/sync-v3.sh             Compute the merge locally; on conflict, populate a worktree.
  scripts/sync-v3.sh --pr        Push '${SYNC_BRANCH}' and open or update the PR against '${V3_BRANCH}'.

Options:
  -s, --status  Report sync status without pushing or opening a pull request.
  -p, --pr      Push '${SYNC_BRANCH}' and open or update its pull request.
  -h, --help    This text.

Environment:
  ADK_REPO      Override the repository (default ${REPO}).
  ADK_REMOTE    Override the git remote to use.
EOF
}

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "$1 is required but not installed"; }

detect_remote() {
  if [[ -n "${ADK_REMOTE:-}" ]]; then
    printf '%s\n' "${ADK_REMOTE}"
    return
  fi
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

# Extracts the last valid Synced-from-main SHA from a commit's trailer block.
extract_synced_sha() {
  local ref="$1"
  git show -s --format="%(trailers:key=${TRAILER_KEY},valueonly=true)" "${ref}" 2>/dev/null |
    awk '
      { sub(/\r$/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, "") }
      /^[0-9a-f]{40}$/ { sha = $0 }
      END { print sha }
    '
}

# True when ref's subject matches the automated sync commit title.
is_sync_commit_subject() {
  local ref="$1" subject
  subject="$(git show -s --format=%s "${ref}" 2>/dev/null || true)"
  [[ "${subject}" =~ ^chore\(${V3_BRANCH}\):\ sync\ ${MAIN_BRANCH}\ \([0-9a-f]+\) ]]
}

# True when remote_sync_sha looks like a single automated sync commit with no
# human fixups or manual conflict resolutions stacked on top.
is_automated_sync_branch() {
  local remote="$1" remote_sync_sha="$2"
  local parent count synced_sha committer_email
  parent="$(git rev-parse "${remote_sync_sha}^1" 2>/dev/null || true)"
  [[ -n "${parent}" ]] || return 1
  # Must be a single commit directly on top of an ancestor of v3.
  git merge-base --is-ancestor "${parent}" "${remote}/${V3_BRANCH}" 2>/dev/null || return 1
  count="$(git rev-list --count "${parent}..${remote_sync_sha}" 2>/dev/null || echo 0)"
  [[ "${count}" -eq 1 ]] || return 1
  is_sync_commit_subject "${remote_sync_sha}" || return 1
  synced_sha="$(extract_synced_sha "${remote_sync_sha}")"
  [[ -n "${synced_sha}" ]] || return 1
  # If a human pushed a manual resolution locally (their own committer email),
  # treat it as human-owned unless we are running with the same non-bot identity
  # and the tree matches the clean automated merge (handled earlier by callers).
  committer_email="$(git show -s --format=%ce "${remote_sync_sha}" 2>/dev/null || true)"
  if [[ "${committer_email}" == "${BOT_EMAIL}" ]]; then
    return 0
  fi
  # For local test/manual runs that created an automated sync commit with the
  # current git user, confirm the commit tree equals a clean merge of its own
  # parent + synced_sha; a manually edited conflict resolution or fixup won't.
  local expected_base expected_tree actual_tree
  expected_base="$(find_last_synced_main_at "${remote}" "${parent}")"
  expected_tree="$(git merge-tree --write-tree --merge-base="${expected_base}" "${parent}" "${synced_sha}" 2>/dev/null || true)"
  actual_tree="$(git rev-parse "${remote_sync_sha}^{tree}" 2>/dev/null || true)"
  [[ -n "${expected_tree}" && "${expected_tree}" == "${actual_tree}" ]]
}

# Finds the most recent commit on `main` already incorporated into target_ref.
#
# 1. Valid `Synced-from-main: <sha>` footer trailers on sync commits in
#    common..target_ref, verified to be ancestors of `main`; when multiple
#    exist, picks the descendant-most commit on `main` so an older SHA can
#    never rewind the merge-base.
# 2. Bootstrap fallback for squash-merged history before the trailer existed:
#    the newest commit in common..target_ref whose tree matches a commit in
#    common..main.
# 3. Git merge-base of `main` and target_ref.
find_last_synced_main_at() {
  local remote="$1" target_ref="$2"
  local main_ref="${remote}/${MAIN_BRANCH}"
  local common
  common="$(git merge-base "${main_ref}" "${target_ref}" 2>/dev/null)" ||
    die "${MAIN_BRANCH} and ${target_ref} have no common ancestor here; fetch full history (fetch-depth: 0)"

  local candidates candidate sha
  local -a valid_shas=()
  if ! candidates="$(git log --format=%H --fixed-strings \
    --grep="${TRAILER_KEY}: " "${common}..${target_ref}")"; then
    die "could not search ${target_ref} for ${TRAILER_KEY} trailers"
  fi
  if [[ -n "${candidates}" ]]; then
    while read -r candidate; do
      [[ -n "${candidate}" ]] || continue
      is_sync_commit_subject "${candidate}" || continue
      sha="$(extract_synced_sha "${candidate}")"
      if [[ -n "${sha}" ]] &&
        git cat-file -e "${sha}^{commit}" 2>/dev/null &&
        git merge-base --is-ancestor "${sha}" "${main_ref}" 2>/dev/null; then
        valid_shas+=("${sha}")
      fi
    done <<<"${candidates}"
  fi

  if [[ ${#valid_shas[@]} -gt 0 ]]; then
    # Pick the commit furthest along main (first in reverse-chronological/topo
    # walk of main that appears in valid_shas).
    local newest
    newest="$(awk '
      NR == FNR { want[$1] = 1; next }
      $1 in want { print $1; exit }
    ' <(printf '%s\n' "${valid_shas[@]}") <(git rev-list "${common}..${main_ref}"))"
    if [[ -n "${newest}" ]]; then
      printf '%s\n' "${newest}"
      return 0
    fi
  fi

  local main_trees target_trees matched
  main_trees="$(git log --format='%T %H' "${common}..${main_ref}")"
  target_trees="$(git log --format='%T %H' "${common}..${target_ref}")"
  if [[ -n "${main_trees}" && -n "${target_trees}" ]]; then
    matched="$(awk '
      NR == FNR {
        if (!($1 in main_for_tree)) {
          main_for_tree[$1] = $2
        }
        next
      }
      $1 in main_for_tree {
        print main_for_tree[$1]
        exit
      }
    ' <(printf '%s\n' "${main_trees}") <(printf '%s\n' "${target_trees}"))"
    if [[ -n "${matched}" ]]; then
      printf '%s\n' "${matched}"
      return 0
    fi
  fi

  printf '%s\n' "${common}"
}

format_commit_list() {
  local base_sha="$1" head_sha="$2" total="$3"
  local list
  list="$(git log --format='- %h %s' --max-count="${MAX_LISTED_COMMITS}" "${base_sha}..${head_sha}")"
  printf '%s\n' "${list}"
  if ((total > MAX_LISTED_COMMITS)); then
    printf -- '- ... and %d more commit(s)\n' "$((total - MAX_LISTED_COMMITS))"
  fi
}

# Fetches SYNC_BRANCH if it exists on the remote and prints its SHA (or empty).
fetch_remote_sync_sha() {
  local remote="$1"
  if git ls-remote --exit-code --heads "${remote}" "${SYNC_BRANCH}" >/dev/null 2>&1; then
    git fetch --quiet "${remote}" "+refs/heads/${SYNC_BRANCH}:refs/remotes/${remote}/${SYNC_BRANCH}" ||
      die "failed to fetch ${SYNC_BRANCH} from ${remote}"
    git rev-parse "${remote}/${SYNC_BRANCH}"
  else
    git update-ref -d "refs/remotes/${remote}/${SYNC_BRANCH}" 2>/dev/null || true
  fi
}

comment_conflict() {
  local pr="$1" v3_sha="$2" main_sha="$3" conflicts="$4" messages="$5"
  local marker="<!-- adk-sync-v3-conflict:${v3_sha}:${main_sha} -->"
  local existing me

  me="$(gh api user --jq '.login' 2>/dev/null)" || me=''
  [[ "${me}" =~ ^[A-Za-z0-9-]{1,39}$ ]] || me=''

  existing="$(gh pr view "${pr}" --repo "${REPO}" --json comments \
    --jq '[.comments[]
      | select(.author.login == "github-actions" or .author.login == "github-actions[bot]" or .author.login == "'"${me}"'")
      | select(.body | contains("'"${marker}"'"))] | length')" ||
    {
      warn "could not read comments on PR #${pr}; not commenting"
      return 0
    }
  if [[ "${existing}" != "0" ]]; then
    info "conflict already reported on PR #${pr} for ${v3_sha:0:9}..${main_sha:0:9}"
    return 0
  fi

  local detail_block
  if [[ -n "${conflicts}" ]]; then
    detail_block="$(printf '%s\n' "${conflicts}" | awk '{ printf "- `%s`\n", $0 }')"
  elif [[ -n "${messages}" ]]; then
    local fence='```'
    detail_block="$(printf '%s\n%s\n%s' "${fence}" "${messages}" "${fence}")"
  else
    detail_block="- (no conflicted paths reported by git merge-tree)"
  fi

  gh pr comment "${pr}" --repo "${REPO}" --body "${marker}
Automatic sync from \`${MAIN_BRANCH}\` (\`${main_sha:0:9}\`) onto \`${V3_BRANCH}\` (\`${v3_sha:0:9}\`) hit merge conflicts:

${detail_block}

To resolve by hand:

\`\`\`bash
scripts/sync-v3.sh
\`\`\`

That prepares a detached worktree under \`\$TMPDIR/adk-sync-v3\` with unmerged conflict entries. Resolve the conflicts, commit with \`${TRAILER_KEY}: ${main_sha}\`, and push to \`${SYNC_BRANCH}\`." >/dev/null ||
    warn "could not comment on PR #${pr}"
}

prepare_conflict_worktree() {
  local remote="$1" v3_sha="$2" main_sha="$3" conflict_tree="$4" stage_lines="$5" conflicts="$6" messages="$7" remote_sync_sha="$8"
  local worktree="${TMPDIR:-/tmp}/adk-sync-v3"

  git worktree remove --force "${worktree}" 2>/dev/null || true
  rm -rf "${worktree}"
  git worktree prune
  mkdir -p "$(dirname "${worktree}")"

  git worktree add --quiet --detach "${worktree}" "${v3_sha}" ||
    die "could not create worktree ${worktree}"
  git -C "${worktree}" read-tree --reset -u "${conflict_tree}" ||
    die "could not populate conflict tree in ${worktree}"

  # Restore higher-order unmerged index stages (1/2/3) so `git status` shows
  # real UU/DU/UD conflicts and `git commit` refuses unresolved files (including
  # modify/delete and binary conflicts that carry no `<<<<<<<` markers).
  if [[ -n "${stage_lines}" ]]; then
    local clear_lines
    clear_lines="$(printf '%s\n' "${stage_lines}" | awk -F'\t' '{ print $2 }' | sort -u |
      awk '{ printf "0 0000000000000000000000000000000000000000\t%s\n", $0 }')"
    printf '%s\n%s\n' "${clear_lines}" "${stage_lines}" |
      git -C "${worktree}" update-index --index-info ||
      die "could not record unmerged conflict stages in ${worktree}"
  fi

  local lease_opt="--force-with-lease=refs/heads/${SYNC_BRANCH}:${remote_sync_sha}"

  cat >&2 <<EOF

Conflicting paths:
$(if [[ -n "${conflicts}" ]]; then printf '%s\n' "${conflicts}" | sed 's/^/  /'; else printf '  (see conflict messages below)\n'; fi)
$(if [[ -n "${messages}" ]]; then printf '\n%s\n' "${messages}"; fi)

Conflicts are recorded in ${worktree} (check 'git status' for UU/DU/UD entries). Resolve them there:

  cd ${worktree}
  git status
  # resolve every unmerged path (git add / git rm), then commit:
  git commit -m 'chore(${V3_BRANCH}): sync ${MAIN_BRANCH} (${main_sha:0:9})

${TRAILER_KEY}: ${main_sha}'
  git push ${lease_opt} ${remote} HEAD:refs/heads/${SYNC_BRANCH}
  # If no sync PR is open yet:
  gh pr create --repo ${REPO} --base ${V3_BRANCH} --head ${SYNC_BRANCH} --title 'chore(${V3_BRANCH}): sync ${MAIN_BRANCH} (${main_sha:0:9})'

EOF
}

main() {
  require_cmd git
  if [[ "${open_pr}" == true ]]; then
    require_cmd gh
    require_cmd jq
    gh auth status >/dev/null 2>&1 || die "gh is not authenticated; run 'gh auth login'"
  fi
  git rev-parse --git-dir >/dev/null 2>&1 || die "not inside a git repository"

  local remote
  remote="$(detect_remote)"
  info "using remote '${remote}' for ${REPO}"

  git fetch --quiet "${remote}" \
    "+refs/heads/${MAIN_BRANCH}:refs/remotes/${remote}/${MAIN_BRANCH}" \
    "+refs/heads/${V3_BRANCH}:refs/remotes/${remote}/${V3_BRANCH}" ||
    die "failed to fetch ${MAIN_BRANCH} and ${V3_BRANCH} from ${remote}"

  local remote_sync_sha
  remote_sync_sha="$(fetch_remote_sync_sha "${remote}")"

  local main_sha v3_sha v3_tree
  main_sha="$(git rev-parse "${remote}/${MAIN_BRANCH}")"
  v3_sha="$(git rev-parse "${remote}/${V3_BRANCH}")"
  v3_tree="$(git rev-parse "${v3_sha}^{tree}")"

  local last_synced
  last_synced="$(find_last_synced_main_at "${remote}" "${remote}/${V3_BRANCH}")"
  info "last synced ${MAIN_BRANCH} commit: ${last_synced:0:9} (tip is ${main_sha:0:9})"

  local commit_count
  commit_count="$(git rev-list --count "${last_synced}..${main_sha}")"
  if [[ "${commit_count}" -eq 0 ]]; then
    info "${V3_BRANCH} is already up to date with ${MAIN_BRANCH} (${main_sha:0:9})"
    return 0
  fi

  info "${commit_count} commit(s) on ${MAIN_BRANCH} since ${last_synced:0:9}"

  # If an open sync PR already carries a commit on top of v3_sha (for example,
  # a maintainer's manual conflict resolution or fixup commit), respect it
  # before attempting a fresh automated merge that could re-report a resolved
  # conflict or overwrite human work.
  local existing_pr=""
  if [[ "${open_pr}" == true ]]; then
    existing_pr="$(gh pr list --repo "${REPO}" --base "${V3_BRANCH}" \
      --head "${SYNC_BRANCH}" --state open --json number --jq '.[0].number // ""')" ||
      die "could not check for an open sync PR"
    if [[ -n "${existing_pr}" && -n "${remote_sync_sha}" ]]; then
      if ! is_automated_sync_branch "${remote}" "${remote_sync_sha}"; then
        local manual_synced
        manual_synced="$(extract_synced_sha "${remote_sync_sha}")"
        if [[ "${manual_synced}" == "${main_sha}" ]] &&
          git merge-base --is-ancestor "${v3_sha}" "${remote_sync_sha}" 2>/dev/null; then
          info "PR #${existing_pr} already has a manual resolution for ${main_sha:0:9} on ${v3_sha:0:9}; leaving it untouched"
          return 0
        fi
        die "PR #${existing_pr} on '${SYNC_BRANCH}' has manual commits; refusing to force-overwrite them (merge or update PR #${existing_pr} by hand)"
      fi
    fi
  fi

  local merge_out merge_rc=0
  merge_out="$(git -c core.quotePath=false merge-tree --write-tree --messages \
    --merge-base="${last_synced}" "${v3_sha}" "${main_sha}" 2>&1)" || merge_rc=$?

  if [[ "${merge_rc}" -gt 1 ]]; then
    die "git merge-tree failed (exit ${merge_rc}): ${merge_out}"
  fi

  if [[ "${merge_rc}" -eq 1 ]]; then
    local conflict_tree stage_lines conflicts messages
    conflict_tree="$(printf '%s\n' "${merge_out}" | sed -n '1p')"
    # In `git merge-tree --write-tree --messages`, line 1 is the tree OID,
    # followed by `<mode> <oid> <stage>\t<path>` lines until the first blank
    # line, followed by informational conflict messages.
    stage_lines="$(printf '%s\n' "${merge_out}" | awk 'NR == 1 { next } /^$/ { exit } { print }')"
    conflicts="$(printf '%s\n' "${stage_lines}" | awk -F'\t' 'NF >= 2 { print $2 }' | LC_ALL=C sort -u)"
    messages="$(printf '%s\n' "${merge_out}" | awk 'NR == 1 { next } !seen && /^$/ { seen = 1; next } seen { print }')"

    warn "merge from ${MAIN_BRANCH} (${main_sha:0:9}) into ${V3_BRANCH} (${v3_sha:0:9}) has conflicts:"
    if [[ -n "${conflicts}" ]]; then
      printf '%s\n' "${conflicts}" | sed 's/^/  /' >&2
    fi
    if [[ -n "${messages}" ]]; then
      printf '%s\n' "${messages}" >&2
    fi

    if [[ "${status_only}" == true ]]; then
      return 1
    fi

    if [[ "${open_pr}" == true ]]; then
      if [[ -n "${existing_pr}" ]]; then
        comment_conflict "${existing_pr}" "${v3_sha}" "${main_sha}" "${conflicts}" "${messages}"
      fi
      die "sync from ${MAIN_BRANCH} to ${V3_BRANCH} has merge conflicts; resolve locally with scripts/sync-v3.sh"
    fi

    prepare_conflict_worktree "${remote}" "${v3_sha}" "${main_sha}" \
      "${conflict_tree}" "${stage_lines}" "${conflicts}" "${messages}" "${remote_sync_sha}"
    return 1
  fi

  local merged_tree
  merged_tree="$(printf '%s\n' "${merge_out}" | sed -n '1p')"
  [[ "${merged_tree}" =~ ^[0-9a-f]{40,64}$ ]] ||
    die "unexpected git merge-tree output: ${merge_out}"

  if [[ "${merged_tree}" == "${v3_tree}" ]]; then
    info "${V3_BRANCH} already matches ${MAIN_BRANCH} (${main_sha:0:9}); nothing to sync"
    return 0
  fi

  local commit_list
  commit_list="$(format_commit_list "${last_synced}" "${main_sha}" "${commit_count}")"

  if [[ "${status_only}" == true ]]; then
    info "merge check: clean (${commit_count} commit(s) to sync)"
    printf '%s\n' "${commit_list}"
    return 0
  fi

  local title="chore(${V3_BRANCH}): sync ${MAIN_BRANCH} (${main_sha:0:9})"
  local msg_file body_file
  msg_file="$(make_tmp)"
  body_file="$(make_tmp)"

  cat >"${msg_file}" <<EOF
${title}

Syncs ${commit_count} commit(s) from ${MAIN_BRANCH} (${last_synced:0:9}..${main_sha:0:9}) into ${V3_BRANCH}:

${commit_list}

${TRAILER_KEY}: ${main_sha}
EOF

  cat >"${body_file}" <<EOF
Nightly sync of \`${MAIN_BRANCH}\` (\`${last_synced:0:9}..${main_sha:0:9}\`, ${commit_count} commit(s)) into \`${V3_BRANCH}\`.

${commit_list}

If workflows are awaiting approval, select **Approve workflows to run** before reviewing and merging.

Generated by \`scripts/sync-v3.sh\`.
EOF

  if [[ "${open_pr}" != true ]]; then
    local commit_sha
    commit_sha="$(git commit-tree "${merged_tree}" -p "${v3_sha}" -F "${msg_file}")"
    info "merge is clean: created local commit ${commit_sha:0:9} on top of ${v3_sha:0:9} (nothing pushed)"
    info "push it with: git push --force-with-lease=refs/heads/${SYNC_BRANCH}:${remote_sync_sha} ${remote} ${commit_sha}:refs/heads/${SYNC_BRANCH}"
    return 0
  fi

  if [[ -n "${existing_pr}" && -n "${remote_sync_sha}" ]]; then
    local remote_parent remote_tree remote_trailer
    remote_parent="$(git rev-parse "${remote_sync_sha}^1" 2>/dev/null || true)"
    remote_tree="$(git rev-parse "${remote_sync_sha}^{tree}" 2>/dev/null || true)"
    remote_trailer="$(extract_synced_sha "${remote_sync_sha}")"
    if [[ "${remote_parent}" == "${v3_sha}" && "${remote_tree}" == "${merged_tree}" && "${remote_trailer}" == "${main_sha}" ]]; then
      info "PR #${existing_pr} is already up to date (${main_sha:0:9} on ${v3_sha:0:9}); nothing to push"
      return 0
    fi
  fi

  local commit_sha
  commit_sha="$(git commit-tree "${merged_tree}" -p "${v3_sha}" -F "${msg_file}")" ||
    die "could not create sync commit"

  info "pushing ${SYNC_BRANCH} (${commit_sha:0:9})"
  git push --quiet --force-with-lease="refs/heads/${SYNC_BRANCH}:${remote_sync_sha}" \
    "${remote}" "${commit_sha}:refs/heads/${SYNC_BRANCH}" ||
    die "could not push ${SYNC_BRANCH}"

  if [[ -n "${existing_pr}" ]]; then
    local patch_json
    patch_json="$(make_tmp)"
    jq -n --arg title "${title}" --rawfile body "${body_file}" \
      '{title: $title, body: $body}' >"${patch_json}"
    gh api "repos/${REPO}/pulls/${existing_pr}" --method PATCH --input "${patch_json}" >/dev/null ||
      die "could not update PR #${existing_pr}"
    info "updated PR #${existing_pr} for ${MAIN_BRANCH} (${main_sha:0:9})"
  else
    local err url
    err="$(make_tmp)"
    if ! url="$(gh pr create --repo "${REPO}" --base "${V3_BRANCH}" --head "${SYNC_BRANCH}" \
      --title "${title}" --body-file "${body_file}" 2>"${err}")"; then
      local msg
      if grep -q 'not permitted to create or approve pull requests' "${err}"; then
        msg="this repository does not allow GitHub Actions to create pull requests; enable it in Settings -> Actions -> General"
      else
        msg="could not open sync PR: $(tr '\n' ' ' <"${err}")"
      fi
      if [[ -z "${remote_sync_sha}" ]]; then
        git push --quiet "${remote}" --delete "${SYNC_BRANCH}" 2>/dev/null ||
          warn "could not clean up orphan branch ${SYNC_BRANCH}"
      fi
      die "${msg}"
    fi
    info "opened ${url}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s | --status) status_only=true ;;
    -p | --pr) open_pr=true ;;
    -h | --help)
      usage
      exit 0
      ;;
    *) die "unknown option: $1" ;;
  esac
  shift
done

main
