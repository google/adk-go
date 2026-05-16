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

package workflow

import "google.golang.org/adk/session"

// NodeStatus is the lifecycle status of a node in the workflow graph.
//
// The status is the engine's single source of truth for what a node
// is doing. It mediates between the trigger buffer (what wants to
// run), the in-process task table (what is currently running), and
// the persisted node history (what has run).
type NodeStatus uint8

const (
	// NodeInactive means the node has not been touched yet. This is
	// the zero value.
	NodeInactive NodeStatus = iota

	// NodePending means the node is ready to run. It may be queued
	// because its input has arrived (consumed from the trigger
	// buffer) or because it is being retried after a failure. The
	// scheduler may keep a NodePending node from starting if the
	// engine's max-concurrency cap is reached.
	NodePending

	// NodeRunning means the engine has started a task for this node.
	// The task is in flight in the current process. A NodeRunning
	// entry that has no live task in the run state (e.g. after a
	// process restart) must be re-scheduled.
	NodeRunning

	// NodeCompleted means the node finished and produced its output.
	// This is a terminal status for normal execution, but a node in
	// NodeCompleted may still be re-triggered by a fresh entry in
	// the trigger buffer (this is what enables loops as graph
	// cycles).
	NodeCompleted

	// NodeWaiting means the node is paused. Two cases share this
	// status:
	//
	//   1. Human-in-the-loop: the node yielded a RequestInput and
	//      is blocked until a function-response payload resumes it.
	//   2. Fan-in (WaitForOutput=true, e.g. JoinNode): the node ran
	//      but did not yet produce its final output because not all
	//      predecessors have triggered it.
	//
	// While any node is NodeWaiting the workflow does not finalize.
	NodeWaiting

	// NodeFailed means the node returned an error and the retry
	// policy (if any) has been exhausted. Terminal.
	NodeFailed

	// NodeCancelled means the node was cancelled, typically because
	// a sibling node failed and the engine cancelled all running
	// tasks. Terminal.
	NodeCancelled
)

// NodeState is the per-node lifecycle record. A RunState holds one
// of these for every node the engine has touched.
//
// JSON-marshallable: NodeState is persisted across pause/resume
// turns via session.State (see persistence.go). The Input and
// PendingRequest.Payload fields are typed any and must therefore
// be JSON-encodable for the persisted state to round-trip. Nodes
// that need to carry binary data across a pause should store the
// bytes via agent.Artifacts and stash a URI string in place of
// the bytes, mirroring how the Live API surfaces audio.
type NodeState struct {
	// Status is the current lifecycle position. See NodeStatus.
	Status NodeStatus `json:"status"`

	// Input is the value most recently handed to the node's Run
	// method. It is set when the node is scheduled.
	Input any `json:"input,omitempty"`

	// TriggeredBy is the name of the upstream node that produced
	// the current Input. Empty for the initial START activation.
	TriggeredBy string `json:"triggeredBy,omitempty"`

	// PendingRequest, when non-nil, carries the human-input request
	// the node emitted before pausing. Non-nil iff Status ==
	// NodeWaiting and the wait was caused by a human-input request
	// (as opposed to a fan-in barrier).
	PendingRequest *session.RequestInput `json:"pendingRequest,omitempty"`

	// ResumedInputs accumulates response payloads for re-entry-mode
	// nodes that yield RequestInput more than once during a single
	// activation lifecycle. Each Resume call adds the new response
	// keyed by its InterruptID. The map is exposed to the node via
	// ctx.ResumedInput on every re-entry activation, so the node
	// can observe responses to all prior requests, not only the
	// most recent one.
	//
	// Cleared when the node transitions to NodeCompleted; absent
	// for handoff-mode nodes (where successors receive the response
	// as input and the asker never re-runs).
	ResumedInputs map[string]any `json:"resumedInputs,omitempty"`
}

// RunState is the per-invocation lifecycle state for a workflow
// run. It rides in session.State across pause and resume turns so
// the workflow can pick up where a previous invocation left off.
type RunState struct {
	// Nodes is the per-node lifecycle map. Absent entries are
	// inactive.
	Nodes map[string]*NodeState `json:"nodes,omitempty"`
}

// NewRunState returns an empty state with the Nodes map
// initialised so callers can write to it without a nil check.
func NewRunState() *RunState {
	return &RunState{Nodes: map[string]*NodeState{}}
}

// EnsureNode returns the NodeState for the given node name,
// creating an inactive entry if none exists. The returned pointer
// is owned by the state and may be mutated in place.
func (s *RunState) EnsureNode(name string) *NodeState {
	if s.Nodes == nil {
		s.Nodes = map[string]*NodeState{}
	}
	ns, ok := s.Nodes[name]
	if !ok {
		ns = &NodeState{Status: NodeInactive}
		s.Nodes[name] = ns
	}
	return ns
}
