// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Command releasedocsbot is an ADK Go agent that answers "what documentation
// needs updating?" after a release ships.
//
// It diffs a release tag against the one before it, analyzes the changed files
// in bounded groups, and files a single GitHub issue listing the documentation
// updates the changes imply. It runs from a GitHub Actions workflow on
// release: [published], and can be re-run or backfilled with explicit tags via
// workflow_dispatch.
//
// # Authority
//
// The model holds exactly one tool, record_documentation_findings, and that
// tool mutates nothing: it appends sanitized entries to an in-memory collector.
// Creating the issue is done by Go after the analysis loop, with the target
// repository, the title and the duplicate-detection marker all computed from
// configuration and API metadata. A fully steered model therefore cannot file
// an issue, cannot choose where one is filed, and cannot decide that one is
// filed at all. Its influence is bounded to the prose inside a fenced block in
// a body the program wrote.
//
// # One issue per release
//
// There is no per-issue session to scope here, so the bound is per release. It
// has three parts:
//
//   - Before any model token is spent, the bot looks for an issue it already
//     filed for the same tag pair, identified by an exact marker on the first
//     line of the body and by the issue's author. Two probes run: an
//     immediately-consistent scan of the target repository's recent issues,
//     which is what catches a re-run moments after a successful run, and a
//     search-index query by the deterministic title, which covers issues older
//     than the scan's bound. A hit from either means the release is done.
//   - Both probes run again immediately before the write, because the check
//     above happened before the analysis loop and is minutes stale by then.
//   - Within a run, an atomic claim keyed by the tag pair is taken in the same
//     critical section that reads the previous outcome, so two callers cannot
//     both pass on one observation.
//   - The workflow serializes every run of itself, so two cannot analyze one
//     release side by side.
//   - Each model session is scoped to one (release, file group) pair, and the
//     recording tool refuses any call naming a different one.
//
// # Untrusted input
//
// The input is a code diff, so the attacker is a contributor who puts text in a
// commit message, a file name, or a code comment. Every contributor-authored
// blob is wrapped in a per-run [UNTRUSTED:nonce] fence drawn from crypto/rand,
// and the nonce draw fails closed. The diff is bounded by file count, by bytes
// per file, and by commit count, and whatever those caps drop is stated in the
// issue rather than silently omitted.
//
// The issue body is largely model-authored. Every model-authored field is
// sanitized (value allow-lists on the kind and the documentation path, a rune
// cap on the free text) and rendered inside a fenced block, with the sequences
// that could close that fence or open an HTML comment neutralized first. The
// dry-run render additionally defuses any line the GitHub Actions runner would
// read as a workflow command, because that is a different parser.
//
// The bot also reports what it did NOT analyze. A file group that failed, that
// the run budget never reached, or that finished without the model calling the
// tool at all is counted and named in the issue, so a partial analysis cannot
// be mistaken for a complete one.
//
// What that does NOT catch, and cannot: a model that calls the tool and reports
// an empty list is indistinguishable in Go from one that genuinely found
// nothing, because the judgement being delegated is precisely "is there
// anything here?". Injected text that makes the model report nothing for its
// own group is therefore a real, unremovable limit of the design. It costs a
// missing suggestion, never a wrong write.
//
// # Filing into another repository
//
// adk-python files this issue into google/adk-docs, which needs a cross-repo
// token. This bot defaults to filing into the repository it diffed, which the
// built-in Actions GITHUB_TOKEN can write to. TARGET_OWNER/TARGET_REPO point it
// elsewhere, and doing so requires a token this repository may not have; see
// README.md.
package main
