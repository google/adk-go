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

import (
	"iter"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

// sliceEvents adapts a []*session.Event to session.Events for tests.
type sliceEvents []*session.Event

func (e sliceEvents) Len() int                { return len(e) }
func (e sliceEvents) At(i int) *session.Event { return e[i] }
func (e sliceEvents) All() iter.Seq[*session.Event] {
	return func(yield func(*session.Event) bool) {
		for _, ev := range e {
			if !yield(ev) {
				return
			}
		}
	}
}

// fakeSession backs ReconstructRunState, which reads only Events(); the
// rest are stubs.
type fakeSession struct {
	events session.Events
}

func (s fakeSession) ID() string                { return "test-session" }
func (s fakeSession) AppName() string           { return "test-app" }
func (s fakeSession) UserID() string            { return "test-user" }
func (s fakeSession) State() session.State      { return nil }
func (s fakeSession) Events() session.Events    { return s.events }
func (s fakeSession) LastUpdateTime() time.Time { return time.Time{} }

func modelEvent(path, text string, messageAsOutput bool) *session.Event {
	ev := &session.Event{
		NodeInfo: &session.NodeInfo{Path: path, MessageAsOutput: messageAsOutput},
	}
	ev.LLMResponse.Content = &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{{Text: text}},
	}
	return ev
}

// Resume derives output from the model message when an event is
// flagged MessageAsOutput with no explicit Output (adk-python parity).
func TestCollectNodeOutputs_MessageAsOutput(t *testing.T) {
	nodes := map[string]Node{"talky": newDummyNode("talky")}

	events := sliceEvents{modelEvent("talky", "Hello, world!", true)}

	outputs, completed := collectNodeOutputs(events, nodes, "")

	if got, want := outputs["talky"], "Hello, world!"; got != want {
		t.Errorf("outputs[talky] = %#v, want %q", got, want)
	}
	if !completed["talky"] {
		t.Errorf("completed[talky] = false, want true")
	}
}

func TestCollectNodeOutputs_MessageNotFlagged(t *testing.T) {
	nodes := map[string]Node{"talky": newDummyNode("talky")}

	events := sliceEvents{modelEvent("talky", "Hello, world!", false)}

	outputs, _ := collectNodeOutputs(events, nodes, "")

	if _, ok := outputs["talky"]; ok {
		t.Errorf("outputs[talky] = %#v, want absent", outputs["talky"])
	}
}

func TestCollectNodeOutputs_ExplicitOutputWins(t *testing.T) {
	nodes := map[string]Node{"talky": newDummyNode("talky")}

	ev := modelEvent("talky", "from message", true)
	ev.Output = "explicit"
	events := sliceEvents{ev}

	outputs, _ := collectNodeOutputs(events, nodes, "")

	if got, want := outputs["talky"], "explicit"; got != want {
		t.Errorf("outputs[talky] = %#v, want %q", got, want)
	}
}

// A delegated child's output is attributed on resume to the static
// owners of every path in OutputFor, so a delegating ancestor recovers
// it without re-emitting (adk-python output_for parity).
func TestCollectNodeOutputs_OutputForAttributesAncestors(t *testing.T) {
	nodes := map[string]Node{
		"child": newDummyNode("child"),
		"outer": newDummyNode("outer"),
	}

	ev := &session.Event{
		Output: "delegated",
		NodeInfo: &session.NodeInfo{
			Path:      "child/gc@1",
			OutputFor: []string{"child/gc@1", "outer/child@1"},
		},
	}

	outputs, _ := collectNodeOutputs(sliceEvents{ev}, nodes, "")

	if got, want := outputs["child"], "delegated"; got != want {
		t.Errorf("outputs[child] = %#v, want %q", got, want)
	}
	if got, want := outputs["outer"], "delegated"; got != want {
		t.Errorf("outputs[outer] = %#v, want %q (ancestor not attributed)", got, want)
	}
}

func TestEventNodeName(t *testing.T) {
	nodes := map[string]Node{
		"nodeA":  newDummyNode("nodeA"),
		"parent": newDummyNode("parent"),
		"child":  newDummyNode("child"),
	}

	tests := []struct {
		name string
		ev   *session.Event
		want string
	}{
		{
			name: "nil NodeInfo falls back to Author",
			ev:   &session.Event{Author: "authorNode"},
			want: "authorNode",
		},
		{
			name: "empty Path falls back to Author",
			ev:   &session.Event{Author: "authorNode", NodeInfo: &session.NodeInfo{Path: ""}},
			want: "authorNode",
		},
		{
			name: "static node path",
			ev:   &session.Event{Author: "authorNode", NodeInfo: &session.NodeInfo{Path: "nodeA"}},
			want: "nodeA",
		},
		{
			name: "node path with invocation ID",
			ev:   &session.Event{Author: "authorNode", NodeInfo: &session.NodeInfo{Path: "nodeA@1"}},
			want: "nodeA",
		},
		{
			name: "hierarchical path matches static parent",
			ev:   &session.Event{Author: "authorNode", NodeInfo: &session.NodeInfo{Path: "parent/child@1"}},
			want: "parent",
		},
		{
			name: "hierarchical path matches child when parent unknown",
			ev:   &session.Event{Author: "authorNode", NodeInfo: &session.NodeInfo{Path: "unknown/child@2"}},
			want: "child",
		},
		{
			name: "path segments not in nodes map falls back to Author",
			ev:   &session.Event{Author: "authorNode", NodeInfo: &session.NodeInfo{Path: "unknown/other@1"}},
			want: "authorNode",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := eventNodeName(tc.ev, nodes)
			if got != tc.want {
				t.Errorf("eventNodeName() = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestReconstructRunState_InvocationScope verifies rehydration is scoped
// to one logical run: a stable interrupt ID resolved in an earlier
// invocation must not shadow the same ID freshly raised in the current
// invocation (the examples/workflow/hitl_simple re-run bug). It drives
// the public ReconstructRunState so the production resume path is
// exercised end to end, not just the internal scan helper.
func TestReconstructRunState_InvocationScope(t *testing.T) {
	ask := newDummyNode("ask")
	wf, err := New("hitl-rerun", []Edge{{From: Start, To: ask}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	raise := func(invID string) *session.Event {
		return &session.Event{
			Author:             "ask",
			InvocationID:       invID,
			LongRunningToolIDs: []string{"iid"},
		}
	}
	resolve := func(invID string) *session.Event {
		ev := &session.Event{Author: "user", InvocationID: invID}
		ev.Content = &genai.Content{Parts: []*genai.Part{{
			FunctionResponse: &genai.FunctionResponse{
				ID:       "iid",
				Response: map[string]any{"payload": "first"},
			},
		}}}
		return ev
	}

	// Both runs reuse interrupt ID "iid": run 1 (inv1) raised and
	// resolved, run 2 (inv2) freshly raised.
	sess := fakeSession{events: sliceEvents{raise("inv1"), resolve("inv1"), raise("inv2")}}

	// Scanning inv2, the prior run's resolution is out of scope, so the
	// reused ID counts as unresolved and "ask" rehydrates as still waiting.
	state, err := wf.ReconstructRunState(sess, "inv2")
	if err != nil {
		t.Fatalf("ReconstructRunState(inv2) error = %v", err)
	}
	if ns := nodeState(t, state, "ask"); ns.Status != NodeWaiting ||
		len(ns.Interrupts) != 1 || ns.Interrupts[0] != "iid" {
		t.Errorf("inv2 ask = %+v, want NodeWaiting on interrupt [iid]", ns)
	}

	// inv1's own resolution is in scope, so the same node resolves —
	// proving the scope isolates runs rather than merely hiding history.
	state, err = wf.ReconstructRunState(sess, "inv1")
	if err != nil {
		t.Fatalf("ReconstructRunState(inv1) error = %v", err)
	}
	if ns := nodeState(t, state, "ask"); ns.Status != NodeCompleted || len(ns.Interrupts) != 0 {
		t.Errorf("inv1 ask = %+v, want NodeCompleted with no pending interrupts", ns)
	}
}

func nodeState(t *testing.T, state *RunState, name string) *NodeState {
	t.Helper()
	if state == nil {
		t.Fatalf("ReconstructRunState returned nil state, want node %q", name)
	}
	ns := state.Nodes[name]
	if ns == nil {
		t.Fatalf("no NodeState for %q; nodes = %+v", name, state.Nodes)
	}
	return ns
}
