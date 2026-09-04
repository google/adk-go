#!/usr/bin/env bash
# Checks that a subject line is a Conventional Commit whose type is one the
# Release Please config for this release line actually recognizes.
#
# Usage: check-conventional-subject.sh <release-please-config> <what> <subject>
#
#   what     what the subject is, quoted back in the error ("PR title", ...)
#   subject  the line to check; only the first line is considered
#
# Why this exists: Release Please derives the next version and the changelog
# from the conventional-commit subjects that land on the release branch. A
# subject with no recognized type is not an error there — it is silently
# dropped, so the change ships while being invisible in the release notes and
# without moving the version. Merging is the last moment that is cheap to fix.
#
# The accepted types come out of the config rather than being repeated here, so
# this cannot drift from the set that actually shapes the changelog.
set -euo pipefail

CONFIG="${1:?usage: check-conventional-subject.sh <release-please-config> <what> <subject>}"
WHAT="${2:?usage: check-conventional-subject.sh <release-please-config> <what> <subject>}"
SUBJECT="$(printf '%s' "${3-}" | head -n 1)"

# Every type the changelog knows about, hidden sections included: those are
# legitimate subjects, they just do not show up in the released notes. `revert`
# is part of the Conventional Commits spec but is not a changelog section, so it
# has to be added back by hand.
TYPES="$(jq -r '[.packages["."]["changelog-sections"][].type] | join("|")' "$CONFIG")"
[ -n "$TYPES" ] || { echo "no changelog-sections in $CONFIG" >&2; exit 1; }
TYPES="${TYPES}|revert"

# type(optional/scope)!: description
if printf '%s' "$SUBJECT" | grep -qE "^($TYPES)(\([a-zA-Z0-9._/-]+\))?!?: .+"; then
  echo "OK ($WHAT): $SUBJECT"
  exit 0
fi

cat >&2 <<EOF
::error title=Not a Conventional Commit::${WHAT}: ${SUBJECT}

The $WHAT must start with a type, an optional scope, and a colon:

  feat: add the thing
  fix(runner): stop leaking the context
  docs: correct the example
  feat!: drop the legacy signature      (breaking change)

Allowed types (from $CONFIG): ${TYPES//|/, }

Only feat, fix and perf move the version. The rest are recorded but neither
released nor listed. A subject with no type at all is dropped entirely, so the
change would ship without appearing anywhere in the release notes.
EOF
exit 1
