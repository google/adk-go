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

// Package workflowstate hands a few facts about a rehydrated workflow run to
// the packages that drive it, without widening the workflow package's public
// API. It deliberately imports nothing: the workflow package installs the hooks
// below from its own init, and any package that reads them already imports
// workflow, so they are non-nil by the time they are called.
//
// The arguments are typed any because naming *workflow.RunState here would be
// an import cycle — workflow imports this package.
package workflowstate

// ActionableInterruptIDs reports the interrupt IDs a rehydrated run recognises,
// mapped to whether Resume can still do something with them. An ID absent from
// the map answers nothing in that run. The live/spent rule is the workflow
// package's to state, not this one's — see RunState.actionableInterruptIDs,
// which is the single definition; restating it here is how the two came to
// disagree.
//
// The distinction matters to a caller deciding whether a turn is a resume. A
// spent ID on its own is a replay, and routing the turn to Resume on its
// strength alone would fail the whole turn with ErrNothingToResume.
//
// The argument is a *workflow.RunState. For anything else the caller gets
// nothing recognised, which is the safe reading: a caller that recognises
// nothing starts a fresh run rather than resuming on an ID it cannot vouch
// for. An empty map and a nil map read identically at the only call site,
// which indexes it.
//
// It is a var so the workflow package can install the real implementation from
// its own init without this package importing it. The stand-in below is what
// makes the var safe to call unconditionally: a nil func here would be a panic
// on a request path for any future importer that does not transitively pull in
// workflow, with no compile-time signal. Only that init assigns to it in
// production code — there is no synchronisation, and that write happens during
// package initialisation, before any goroutine can read it. A test that
// substitutes it must not run in parallel with another that reads it.
var ActionableInterruptIDs = func(runState any) map[string]bool { return map[string]bool{} }
