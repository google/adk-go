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

package llminternal

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func subBranchEvent(author, branch string, content *genai.Content) *session.Event {
	return &session.Event{Author: author, Branch: branch, LLMResponse: model.LLMResponse{Content: content}}
}

func TestIsToolSubBranchEvent(t *testing.T) {
	ids := map[string]struct{}{"fc1": {}}
	tests := []struct {
		name        string
		agentBranch string
		eventBranch string
		agentName   string
		want        bool
	}{
		{"root agent, tool branch", "", "translator@fc1", "coordinator", true},
		{"root agent, nested below tool branch", "", "translator@fc1.inner@1", "coordinator", true},
		{"branched agent, tool branch below it", "wf.coord@1", "wf.coord@1.translator@fc1", "coordinator", true},
		{"own branch carries a call id", "translator@fc1", "translator@fc1", "translator", false},
		{"own events below own branch", "", "translator@fc1", "translator", false},
		{"workflow node branch keyed by counter", "", "worker@1", "coordinator", false},
		{"unknown id", "", "translator@fc2", "coordinator", false},
		{"no run id", "", "parallel.a", "coordinator", false},
		{"event on an ancestor branch", "parent.translator@fc1", "parent", "coordinator", false},
		{"unbranched event", "", "", "coordinator", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ev := subBranchEvent("x", tc.eventBranch, nil)
			if got := isToolSubBranchEvent(tc.agentBranch, ev, tc.agentName, ids); got != tc.want {
				t.Errorf("isToolSubBranchEvent(%q, branch %q, %q) = %v, want %v",
					tc.agentBranch, tc.eventBranch, tc.agentName, got, tc.want)
			}
		})
	}
}

// A root agent that ran a single_turn sub-agent as a tool must get only the
// tool's function response, not the sub-agent's own transcript from the
// tool's sub-branch.
func TestBuildContents_SkipsToolSubBranchEvents(t *testing.T) {
	call := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "fc1", Name: "translator", Args: map[string]any{"request": "hello"}}}}}
	resp := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "fc1", Name: "translator", Response: map[string]any{"result": "hola"}}}}}
	events := []*session.Event{
		subBranchEvent("user", "", genai.NewContentFromText("turn1", genai.RoleUser)),
		subBranchEvent("coordinator", "", call),
		subBranchEvent("translator", "translator@fc1", genai.NewContentFromText("hola", genai.RoleModel)),
		subBranchEvent("coordinator", "", resp),
		subBranchEvent("coordinator", "", genai.NewContentFromText("done1", genai.RoleModel)),
		subBranchEvent("user", "", genai.NewContentFromText("turn2", genai.RoleUser)),
	}
	want := []*genai.Content{
		genai.NewContentFromText("turn1", genai.RoleUser),
		call,
		resp,
		genai.NewContentFromText("done1", genai.RoleModel),
		genai.NewContentFromText("turn2", genai.RoleUser),
	}

	t.Run("default", func(t *testing.T) {
		got, err := buildContentsDefault("coordinator", "", "", events, false, nil)
		if err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("contents mismatch (-want +got):\n%s", diff)
		}
	})

	// With no history, the sub-agent's reply must not become the start of
	// the current turn, or the call and its response would be cut off.
	t.Run("current turn only", func(t *testing.T) {
		got, err := buildContentsCurrentTurnContextOnly("coordinator", "", "", events[:4], false, nil)
		if err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(want[:3], got); diff != "" {
			t.Errorf("contents mismatch (-want +got):\n%s", diff)
		}
	})
}
