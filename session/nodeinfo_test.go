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

package session_test

import (
	"testing"

	"google.golang.org/adk/v2/session"
)

// The accessors split a node path as adk-python's _NodePathBuilder does:
// segments at "/", the run id at the last "@" of a segment. Every row was
// taken from adk-python itself.
func TestNodeInfoPathAccessors(t *testing.T) {
	tests := []struct {
		path        string
		name        string
		runID       string
		parentRunID string
		parentOK    bool
	}{
		{path: "workflow@10/child_node@20", name: "child_node", runID: "20", parentRunID: "10", parentOK: true},
		{path: "workflow/child_node", name: "child_node"},
		{path: ""},
		{path: "wf@/x@1", name: "x", runID: "1", parentOK: true},
		{path: "a@b@c", name: "a@b", runID: "c"},
		{path: "x@1/", parentRunID: "1", parentOK: true},
		{path: "/x@2", name: "x", runID: "2"},
		{path: "a@1/b@2/c@3", name: "c", runID: "3", parentRunID: "2", parentOK: true},
		{path: "a@1/b/c@3", name: "c", runID: "3"},
		{path: "a@1/b@2/c/d@4", name: "d", runID: "4"},
		{path: "a@1/b@2/c@3/d@4", name: "d", runID: "4", parentRunID: "3", parentOK: true},
		{path: "x@", name: "x"},
		{path: "@", parentOK: false},
		{path: "@/x@1", name: "x", runID: "1", parentOK: true},
	}
	for _, tc := range tests {
		t.Run(tc.path, func(t *testing.T) {
			ni := &session.NodeInfo{Path: tc.path}
			if got := ni.Name(); got != tc.name {
				t.Errorf("Name() = %q, want %q", got, tc.name)
			}
			if got := ni.RunID(); got != tc.runID {
				t.Errorf("RunID() = %q, want %q", got, tc.runID)
			}
			if got, ok := ni.ParentRunID(); got != tc.parentRunID || ok != tc.parentOK {
				t.Errorf("ParentRunID() = (%q, %v), want (%q, %v)", got, ok, tc.parentRunID, tc.parentOK)
			}
		})
	}
}

// Event.NodeInfo is nil outside a workflow, so the accessors accept a nil
// receiver.
func TestNodeInfoAccessorsNilReceiver(t *testing.T) {
	var ni *session.NodeInfo
	if got := ni.Name(); got != "" {
		t.Errorf("Name() = %q, want \"\"", got)
	}
	if got := ni.RunID(); got != "" {
		t.Errorf("RunID() = %q, want \"\"", got)
	}
	if got, ok := ni.ParentRunID(); got != "" || ok {
		t.Errorf("ParentRunID() = (%q, %v), want (\"\", false)", got, ok)
	}
}
