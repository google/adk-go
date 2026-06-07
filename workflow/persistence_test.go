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

	"google.golang.org/adk/session"
)

// sliceEvents adapts a []*session.Event to the session.Events interface
// for tests; the session package exposes no constructor for a raw slice.
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

// TestCollectNodeOutputs_OutputWithNilNodeInfo guards that an output
// event with a nil NodeInfo (a non-workflow output event) is attributed
// via Author without dereferencing NodeInfo.OutputFor and panicking.
func TestCollectNodeOutputs_OutputWithNilNodeInfo(t *testing.T) {
	nodesByName := map[string]Node{"nodeA": &dummyNode{name: "nodeA"}}

	events := sliceEvents{
		{Author: "nodeA", Output: "result-A", NodeInfo: nil},
	}

	outputs, completed := collectNodeOutputs(events, nodesByName)

	if got, want := outputs["nodeA"], "result-A"; got != want {
		t.Errorf("outputs[nodeA] = %v, want %v", got, want)
	}
	if !completed["nodeA"] {
		t.Errorf("completed[nodeA] = false, want true")
	}
}

// TestCollectNodeOutputs_DelegatedOutputAttributedToAncestors verifies a
// single delegated event (WithUseAsOutput) recovers output for the static
// owners of every path in OutputFor, so delegating ancestors resume
// without a re-emitted event.
func TestCollectNodeOutputs_DelegatedOutputAttributedToAncestors(t *testing.T) {
	nodesByName := map[string]Node{
		"parent": &dummyNode{name: "parent"},
		"child":  &dummyNode{name: "child"},
	}

	events := sliceEvents{
		{
			Author: "child",
			Output: "delegated",
			NodeInfo: &session.NodeInfo{
				Path:      "child/grandchild@1",
				OutputFor: []string{"child/grandchild@1", "parent/child@1"},
			},
		},
	}

	outputs, _ := collectNodeOutputs(events, nodesByName)

	if got, want := outputs["child"], "delegated"; got != want {
		t.Errorf("outputs[child] = %v, want %v", got, want)
	}
	if got, want := outputs["parent"], "delegated"; got != want {
		t.Errorf("outputs[parent] = %v, want %v (delegated output not attributed to ancestor)", got, want)
	}
}
