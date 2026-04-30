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

// Package workflow provides graph-based agent orchestration. Mirrors the
// adk-python google.adk.workflow subsystem.
//
// A Workflow is a directed graph of Nodes. Edges connect nodes; routing
// values on edges enable conditional branching. The engine schedules nodes
// as they become ready, runs each in its own goroutine, and multiplexes
// emitted events into a single iter.Seq2 returned to the caller.
//
// A Workflow is itself an agent.Agent and can be used anywhere an Agent
// is accepted (e.g. runner.Config.Agent or runner.Config.RootNode).
//
// The phased delivery (see plans/i-want-you-to-glistening-lightning.md)
// builds this engine incrementally. Phase 2A ships:
//   - Node interface + Base embeddable struct
//   - Workflow struct + New constructor
//   - Edge / Route / Connect / RouteMap / START sentinel
//   - Schema interface + JSONSchemaFor[T] / JSONSchemaRaw helpers
//   - RetryConfig with default values matching adk-python
//   - NodeOpt functional-options for wrappers
package workflow
