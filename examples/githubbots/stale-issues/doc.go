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

// Command stale-issues is an example ADK Go agent that audits open GitHub
// issues for staleness.
//
// Unlike a timestamp-only "stale bot", this sample reconstructs the full
// conversation history of each issue (comments, description edits, title
// renames, reopen and label events) via a single GraphQL query, replays it to
// find the last human actor, and then lets an LLMAgent decide what to do using
// a small set of typed function tools. The LLM distinguishes a maintainer
// asking a question (a stale candidate) from a maintainer posting a status
// update (still active).
//
// It demonstrates:
//   - building an [llmagent.New] agent driven by typed [functiontool.New] tools,
//   - running it headlessly with a [runner.Runner] and an in-memory session,
//   - calling the GitHub REST and GraphQL APIs from a tool,
//   - bounded concurrency with errgroup, and
//   - granting an agent write authority over a public issue tracker without
//     trusting the model with any part of it.
//
// The agent runs on a schedule from .github/workflows/stale-issues-bot.yml
// using the built-in GITHUB_TOKEN. See README.md for setup and for the
// authority limits the Go code enforces regardless of what the model asks for.
package main
