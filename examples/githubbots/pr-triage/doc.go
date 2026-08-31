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

// Command githubprtriagebot gives every incoming pull request an automated
// first look and assigns it a shepherd.
//
// It has exactly two powers, and nothing else:
//
//   - assign one owner, drawn from a configured component-to-login map;
//   - post one comment asking the author for missing context.
//
// It deliberately does not label. adk-python removed pull-request auto-labeling
// on security grounds and this bot does not reinstate it. It also never
// re-assigns: assignment is one-way per pull request, so an owner a maintainer
// removed stays removed.
//
// # Authority limits
//
// The model chooses a component name and a set of missing-context items. It
// cannot write free text to GitHub and it cannot name a GitHub login:
//
//   - assign_owner_to_pull_request takes a component, which must be a key of the
//     configured owner map. Go resolves the login. Logins never appear in the
//     prompt at all, so an injected instruction has no login to aim at, and an
//     invented component is rejected before any HTTP call.
//   - request_more_context takes items from a fixed allow-list of constants. Go
//     renders the comment body from those constants, so no model-authored (and
//     therefore attacker-influenced) prose is ever posted.
//
// Both tools are additionally bound by a per-session pull-request scope
// (withAuditedPR), an atomic one-shot claim taken in the same critical section
// as the Go-computed precondition check, and a single dry-run chokepoint.
//
// # Trust model
//
// The workflow runs on pull_request_target, which executes with the base
// repository's token and secrets against a pull request from a fork. The
// workflow therefore uses the default checkout of the base branch and never
// checks out, builds or executes the fork's code; the pull request is read
// entirely through the GitHub API.
//
// Everything the author controls -- title, body and the paths of the files they
// added -- is wrapped in an unguessable per-pull-request nonce fence
// ([UNTRUSTED:hex] ... [/UNTRUSTED:hex]) drawn from crypto/rand, with the
// trusted headers the bot generates emitted outside it. A draw failure aborts
// the review rather than falling back to a predictable marker.
//
// # Deterministic pre-processing
//
// Code, not the model, decides which pull requests are eligible. Before a single
// token is spent the bot skips a pull request that is closed or merged, is a
// draft, already has an assignee, was opened by a configured component owner, or
// that this bot has assigned once before (established from the pull request's
// assignment timeline, so a maintainer's later un-assignment is never undone).
// Existing comments are fetched for the "have I already asked?" check and are
// deliberately never shown to the model.
//
// Triggers are pull_request_target on opened, reopened and ready_for_review,
// plus workflow_dispatch for one pull request or a bounded batch. synchronize is
// excluded and there is no periodic backfill. See README.md.
package main
