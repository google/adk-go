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

// Command githubspambot is an autonomous ADK Go agent that moderates a GitHub
// repository's issues for spam.
//
// It audits open issues for SEO spam, unsolicited promotion of third-party
// products or sites, and other off-topic solicitation. When the model judges an
// issue's content to be spam it applies a configurable label (default "spam")
// and posts a single comment alerting the maintainers.
//
// The bot is code-orchestrated: code finds the candidate issues and runs the
// LLMAgent once per issue in its own isolated session, with bounded concurrency
// via errgroup. Each issue's session is scoped to that issue (withAuditedIssue),
// so injected instructions in the (untrusted) issue text cannot make a tool act
// on a different issue. The deterministic pre-processing is done
// entirely in code and the finished, untrusted-marked text is embedded in the
// per-issue prompt; the model's only tool is flag_issue_as_spam. Before the
// model is ever invoked, the bot:
//   - narrows the input to the issue's OWN title and body. Comments are fetched
//     to recognize the bot's past alerts and are never sent to the model:
//     flagging labels the whole thread, so any content the decision may rest on
//     is content a stranger could use to get somebody else's issue labelled;
//   - skips issues already labeled spam or already carrying the bot's alert
//     comment;
//   - skips issues opened by a maintainer. A bot account is skipped only by
//     being named in MAINTAINERS: skipping every "[bot]" login by suffix would
//     extend unconditional trust to every GitHub App installed on the
//     repository, and an issue opened under the bot's own (shared) Actions
//     identity is reviewed like any other;
//   - truncates the title and the body independently, saying in a trusted line
//     of the prompt when either was cut (it does not strip fenced code blocks,
//     so spam cannot hide inside a ``` fence);
//   - annotates the author with their GitHub author association
//     (e.g. FIRST_TIME_CONTRIBUTOR), which the prompt uses as a spam-likelihood
//     prior alongside a few worked examples.
//
// If nothing reviewable remains, the issue is skipped without spending a single
// model token (the "zero-waste" optimization from the original Python sample).
// Only the fuzzy spam classification is delegated to the model. A -dry-run flag
// logs intended actions without mutating anything.
//
// Idempotency: a re-run never duplicates work. The spam label is the primary
// guard (the sweep excludes already-labeled issues and the per-issue check skips
// them); within a run, flagging an issue twice is a no-op in code; the bot's own
// alert comment is a best-effort secondary signal, recognized by an invisible
// marker only this bot emits (see github.go for its bound).
//
// Each review builds its own agent, runner and session service. ADK initializes
// mutable state on the agent during the first run, so a shared runner races
// under concurrency.
//
// Deliberate differences from the Python adk_issue_monitoring_agent original:
// the scheduled sweep is capped (ISSUE_COUNT, most-recently-updated first)
// rather than Python's uncapped 24h "since" sweep / INITIAL_FULL_SCAN; the issue
// title is reviewed in addition to the body; comments are NOT reviewed (Python
// fed them to the model, which lets a stranger's comment get somebody else's
// issue labelled); fenced code blocks are kept (Python
// stripped them) so spam cannot hide in a code fence; the alert is plain text
// rather than an @maintainers mention; and bounded concurrency replaces Python's
// inter-batch sleep. See README.md for the full list.
//
// Detection is best-effort against a determined evader: a spammer who appends
// instruction-shaped prose to their spam escapes it about two times in three,
// against a control that is caught every time (measured; see README.md, which
// also records an inversion that was tried against this and not shipped). The
// Go controls are what hold under that, not the classification.
//
// The agent runs from .github/workflows/spam-detection-bot.yml on a six-hourly
// schedule and on manual dispatch, using the built-in GITHUB_TOKEN. There is
// deliberately no issue or issue_comment trigger: an event trigger would let any
// commenter start a model run holding a write-scoped token. See README.md.
package main
