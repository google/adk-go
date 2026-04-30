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

package session

import (
	"time"

	"google.golang.org/genai"
)

// NodeInfo describes a workflow node's contribution to an event. The workflow
// engine writes it on every event emitted by a node so that resume (lazy scan
// of session events) can reconstruct node state without a separate
// persistence channel.
//
// Non-workflow agents do not produce events with a NodeInfo set.
type NodeInfo struct {
	// Path is the hierarchical node path that produced the event, e.g.
	// "wf@1/classify@1" or "wf@1/branchA@1/llm_agent@2".
	Path string

	// OutputFor lists ancestor paths whose output this event also represents.
	// Used when a terminal node's output is delegated upward.
	OutputFor []string

	// Route is the routing value emitted by the node, if any. Mirrored from
	// EventActions.Route for convenience during scan.
	Route any

	// Output is the typed output value produced by the node. JSON-serializable.
	Output any

	// Interrupt is true if this event represents a human-in-the-loop interrupt
	// (a RequestInput). The InterruptID identifies it; resume looks for a
	// matching ResolvedID elsewhere in the event log.
	Interrupt bool

	// InterruptID is the unique identifier of an emitted interrupt. Mirrored
	// onto Event.LongRunningToolIDs for the existing function-call wire
	// protocol.
	InterruptID string

	// ResolvedID, when set, indicates this event resolves a prior interrupt
	// with the matching id (typically a FunctionResponse event the user
	// supplied on resume).
	ResolvedID string
}

// EventCompaction marks an event as a summary spanning a timestamp range of
// older events. The contents-builder uses the latest valid compaction in
// place of subsumed raw events when constructing an LLM prompt.
//
// The sliding-window and token-threshold compaction triggers in the runner
// produce events with Author="user" and Actions.Compaction set.
type EventCompaction struct {
	StartTimestamp   time.Time
	EndTimestamp     time.Time
	CompactedContent *genai.Content
}

// TaskRequest is the data shape produced by RequestTaskTool when a coordinator
// agent delegates a task to a sub-agent.
type TaskRequest struct {
	AgentName string
	Input     map[string]any
}

// TaskResult is the data shape produced by FinishTaskTool when a task agent
// signals completion with output.
type TaskResult struct {
	AgentName string
	Output    map[string]any
}
