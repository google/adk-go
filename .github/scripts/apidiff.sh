#!/usr/bin/env bash
# Reports incompatible changes to the exported API of the repository's modules.
#
# Compares every module against a base revision and fails when apidiff calls a
# change incompatible. Adding to the API is always allowed.
#
# Usage: apidiff.sh <target-ref>
#
# Why this exists: a trailing variadic parameter added to an exported function
# changes that function's type. Every ordinary call site still compiles, so
# neither the build nor the tests notice, while any caller holding the function
# as a value stops compiling. That shipped once.
set -euo pipefail

TARGET_REF="${1:?usage: apidiff.sh <target-ref>}"

# Compare against the merge base, not the target's tip. A branch cut before a
# recent change to the target would otherwise be reported as having removed
# whatever the target gained in the meantime, which is a false positive and the
# quickest way to get a check like this ignored.
BASE_REF="$(git merge-base HEAD "$TARGET_REF")"
echo "Target $TARGET_REF, comparing against merge base ${BASE_REF:0:12}"

WORK="$(mktemp -d)"
BASE_TREE="$WORK/base"
cleanup() {
  git worktree remove --force "$BASE_TREE" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT
mkdir -p "$WORK/api"

# Every module in the repository, as a path relative to the root.
modules() {
  find . -not -path '*/.git/*' -name go.mod -exec dirname {} \; | sort
}

# The module path declared by a directory. GOWORK=off because inside a
# workspace `go list -m` reports every module in the workspace, not this one.
module_path() { (cd "$1" && GOWORK=off go list -m); }

snapshot_for() { echo "$WORK/api/$(echo "$1" | tr '/' '_').api"; }

# apidiff narrates every internal package it skips, on stderr, which buries the
# one line that says what actually went wrong under eighty that do not.
report_error() { grep -v '^Ignoring internal package ' "$WORK/err" | sed 's/^/    /' || true; }

# apidiff is asked for a whole module at a time. -m loads every package in the
# module in one pass and skips internal/ by itself, where invoking it once per
# package costs minutes on a module this size and reports the same thing.
#
# It exits non-zero only when it could not do its job; finding an incompatible
# change is a successful run that prints to stdout. Keeping those two apart is
# the difference between "nothing broke" and "nothing was compared", which look
# identical from the outside.
failed=0
breaks=0

echo "Collecting the API at the base revision"
git worktree add --detach --quiet "$BASE_TREE" "$BASE_REF"
(
  cd "$BASE_TREE"
  go work init >/dev/null 2>&1 || true
  go work use -r . >/dev/null 2>&1 || true
)

while IFS= read -r dir; do
  mod="$(module_path "$dir")"
  snap="$(snapshot_for "$mod")"

  # A module this change introduces has no baseline, which is not a problem.
  if [ ! -d "$BASE_TREE/$dir" ]; then
    echo "  $mod is new; nothing to compare against"
    continue
  fi

  if (cd "$BASE_TREE/$dir" && apidiff -m -w "$snap" "$mod") 2>"$WORK/err"; then
    echo "  $mod"
  else
    echo "  ERROR: could not read the API of $mod at the base revision"
    report_error
    failed=1
  fi
done < <(modules)

echo "Comparing HEAD against it"
while IFS= read -r dir; do
  mod="$(module_path "$dir")"
  snap="$(snapshot_for "$mod")"
  [ -f "$snap" ] || continue

  if ! out="$( (cd "$dir" && apidiff -m "$snap" "$mod") 2>"$WORK/err" )"; then
    echo "  ERROR: could not read the API of $mod at HEAD"
    report_error
    failed=1
    continue
  fi

  # apidiff prints an "Incompatible changes:" section only when there are some.
  if printf '%s' "$out" | grep -q '^Incompatible changes:'; then
    echo
    echo "=== $mod ==="
    printf '%s\n' "$out" | sed -n '/^Incompatible changes:/,/^$/p'
    breaks=1
  fi
done < <(modules)

# A module that could not be read was not checked at all. Reporting that as a
# clean run is the one outcome worse than not having the check, so it fails
# here, and the breaking-change label below does not cover it.
if [ "$failed" -ne 0 ]; then
  cat <<'MSG'

The comparison did not complete: the errors above mean some module's API was
never read, so nothing about it was verified. Fix those before reading this
check as "no breaking changes".
MSG
  exit 1
fi

if [ "$breaks" -eq 0 ]; then
  echo "No incompatible changes."
  exit 0
fi

cat <<'MSG'

Incompatible API changes found.

Adding to the API is always allowed; the changes above remove something or
change its type. Note that adding a variadic parameter to an existing exported
function counts, even though every call site still compiles.

If the break is deliberate, label the pull request "breaking-change". The
comparison still runs and still prints what changed, so the break stays on the
record, but it stops failing the check.
MSG

if [ "${ALLOW_BREAKING:-false}" = "true" ]; then
  echo
  echo 'Labelled "breaking-change", so this is reported rather than enforced.'
  exit 0
fi
exit 1
