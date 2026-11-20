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
	"fmt"
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// The workflow reads a node name from a path segment the way
// session.NodeInfo.Name does: the text before the segment's last "@". These
// tests use node names that contain "@", which a first-"@" split truncates.

func TestNestedWorkflow_TerminalNameWithAt(t *testing.T) {
	for _, tc := range []struct {
		name     string
		terminal string
	}{
		{name: "one_at", terminal: "fetch@v2"},
		{name: "two_ats", terminal: "fetch@v2@b"},
		{name: "trailing_at", terminal: "fetch@"},
		// The segment "@1" starts with its last "@"; the name is "".
		{name: "empty_name", terminal: ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			echo := func(agent.Context, any) (string, error) { return "out", nil }
			inner, err := NewWorkflowNode("inner", Chain(Start, NewFunctionNode(tc.terminal, echo, NodeConfig{})))
			if err != nil {
				t.Fatal(err)
			}
			var got any
			after := NewFunctionNode("after", func(_ agent.Context, in any) (string, error) {
				got = in
				return fmt.Sprint(in), nil
			}, NodeConfig{})
			wf := mustNew(t, Chain(Start, inner, after))
			for _, err := range wf.Run(newSeededMockCtx(t)) {
				if err != nil {
					t.Fatal(err)
				}
			}
			if got != "out" {
				t.Errorf("after received %#v, want %q", got, "out")
			}
		})
	}
}

// A sibling "ask" makes a first-"@" split of "ask@v2@1" resolve to the
// wrong graph node instead of to no node.
func TestReconstructRunState_InterruptOwnerNameWithAt(t *testing.T) {
	ask := NewDynamicNode("ask@v2", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		return ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "approval", Message: "approve"})
	}, NodeConfig{})
	plain := NewFunctionNode("ask", func(agent.Context, any) (string, error) { return "plain", nil }, NodeConfig{})
	wf, err := New("root", []Edge{{From: Start, To: ask}, {From: Start, To: plain}})
	if err != nil {
		t.Fatal(err)
	}
	var events sliceEvents
	for ev, err := range wf.Run(newMockCtx(t)) {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}
	state, err := wf.ReconstructRunState(fakeSession{events: events}, "")
	if err != nil {
		t.Fatal(err)
	}
	if got := nodeState(t, state, "ask@v2").Status; got != NodeWaiting {
		t.Errorf("ask@v2 status = %v, want %v", got, NodeWaiting)
	}
	if got := nodeState(t, state, "ask"); got.Status != NodeCompleted || got.Output != "plain" {
		t.Errorf("ask state = %+v, want completed with output %q", got, "plain")
	}
}

func TestReconstructRunState_CompletedNodeNameWithAt(t *testing.T) {
	ask := NewDynamicNode("ask", func(ctx agent.Context, _ any, emit func(*session.Event) error) (any, error) {
		return ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "approval", Message: "approve"})
	}, NodeConfig{})
	fetch := NewFunctionNode("fetch@v2", func(agent.Context, any) (string, error) { return "out", nil }, NodeConfig{})
	wf, err := New("root", []Edge{{From: Start, To: ask}, {From: Start, To: fetch}})
	if err != nil {
		t.Fatal(err)
	}
	var events sliceEvents
	for ev, err := range wf.Run(newMockCtx(t)) {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}
	state, err := wf.ReconstructRunState(fakeSession{events: events}, "")
	if err != nil {
		t.Fatal(err)
	}
	if got := nodeState(t, state, "fetch@v2"); got.Status != NodeCompleted || got.Output != "out" {
		t.Errorf("fetch@v2 state = %+v, want completed with output %q", got, "out")
	}
}

// A node named "" has the segment "@1", which starts with its last "@". The
// Author "wf" (the agent name a runner stamps) keeps the Author fallback
// from naming the node.
func TestReconstructRunState_EmptyNodeName(t *testing.T) {
	wf, err := New("root", []Edge{{From: Start, To: newDummyNode("")}, {From: Start, To: newDummyNode("child")}})
	if err != nil {
		t.Fatal(err)
	}
	events := sliceEvents{
		{Author: "wf", NodeInfo: &session.NodeInfo{Path: "root@1/@1"}, LongRunningToolIDs: []string{"approval"}},
		// Output of a dynamic child of "", not delegated to it: no graph
		// node owns it, although a static node is also named "child".
		{Author: "wf", Output: "discarded", NodeInfo: &session.NodeInfo{Path: "root@1/@1/child@1"}},
	}
	state, err := wf.ReconstructRunState(fakeSession{events: events}, "")
	if err != nil {
		t.Fatal(err)
	}
	if got := nodeState(t, state, "").Status; got != NodeWaiting {
		t.Errorf("status of node \"\" = %v, want %v", got, NodeWaiting)
	}
	if got, ok := state.Nodes["child"]; ok {
		t.Errorf("child state = %+v, want none", got)
	}
}
