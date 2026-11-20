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

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// An empty edge list builds the empty workflow, whose run yields nothing:
// no START event, no completion marker and no error.
func TestNew_EmptyEdges(t *testing.T) {
	tests := []struct {
		name  string
		edges []Edge
	}{
		{name: "nil", edges: nil},
		{name: "empty slice", edges: []Edge{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			w, err := New("w", tc.edges)
			if err != nil {
				t.Fatalf("New() error = %v, want nil", err)
			}

			items := 0
			for _, err := range w.Run(newMockCtx(t)) {
				items++
				if err != nil {
					t.Errorf("Run() yielded error %v", err)
				}
			}

			if items != 0 {
				t.Errorf("Run() yielded %d items, want 0", items)
			}
		})
	}
}

// An empty nested workflow produces no output: the enclosing graph records
// the node's completion with a nil output and runs its successor with nil.
func TestNewWorkflowNode_EmptyEdges(t *testing.T) {
	inner, err := NewWorkflowNode("inner", nil)
	if err != nil {
		t.Fatalf("NewWorkflowNode(\"inner\", nil) error = %v, want nil", err)
	}
	after := NewFunctionNode("after", func(_ agent.Context, in any) (string, error) {
		return fmt.Sprintf("after got %v", in), nil
	}, NodeConfig{})
	outer, err := New("outer", Chain(Start, inner, after))
	if err != nil {
		t.Fatalf("New() error = %v, want nil", err)
	}

	type item struct {
		Path   string
		Output any
	}
	var got []item
	for ev, err := range outer.Run(newMockCtx(t)) {
		if err != nil {
			t.Fatalf("Run() yielded error %v", err)
		}
		var path string
		if ev.NodeInfo != nil {
			path = ev.NodeInfo.Path
		}
		got = append(got, item{Path: path, Output: ev.Output})
	}

	want := []item{
		{Path: "outer@1/inner@1", Output: nil},
		{Path: "outer@1/after@1", Output: "after got <nil>"},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Run() events mismatch (-want +got):\n%s", diff)
	}
}

// The empty workflow has nothing to resume, even when the history holds an
// event whose path names a node the graph does not have.
func TestReconstructRunState_EmptyWorkflow(t *testing.T) {
	wf, err := New("wf", nil)
	if err != nil {
		t.Fatalf("New() error = %v, want nil", err)
	}
	sess := fakeSession{events: sliceEvents{{
		InvocationID:       "inv1",
		LongRunningToolIDs: []string{"i1"},
		NodeInfo:           &session.NodeInfo{Path: "wf@1/ghost@1"},
	}}}

	state, err := wf.ReconstructRunState(sess, "inv1")
	if state != nil || err != nil {
		t.Fatalf("ReconstructRunState() = (%v, %v), want (nil, nil)", state, err)
	}

	items := 0
	for range wf.Resume(agent.Promote(newMockCtx(t)), state, map[string]any{"i1": "x"}) { //nolint:errcheck // the test counts the items; their errors are not under test
		items++
	}
	if items != 0 {
		t.Errorf("Resume() yielded %d items, want 0", items)
	}
}
