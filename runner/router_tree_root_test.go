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

package runner_test

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
)

// TestFindAgentToRun_TransferabilityWalksPastRunnerRoot checks which agent
// answers the second turn when the runner starts at mid, mid transferred the
// first turn to its sub-agent leaf, and mid has a parent above the runner. As
// adk-python's is_transferable_across_agent_tree walks parent_agent to the
// root of the tree (agents/_agent_router.py), leaf keeps the conversation only
// when every agent above it, the one above the runner included, is an LLM
// agent that allows transfer to its parent. Fails when the walk stops at the
// runner's agent: leaf then answers below a SequentialAgent too.
func TestFindAgentToRun_TransferabilityWalksPastRunnerRoot(t *testing.T) {
	tests := []struct {
		name      string
		newParent func(t *testing.T, mid agent.Agent)
		want      string
	}{
		{
			name: "sequential agent above the runner",
			newParent: func(t *testing.T, mid agent.Agent) {
				if _, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{mid}}}); err != nil {
					t.Fatal(err)
				}
			},
			want: "mid: mid 2",
		},
		{
			name: "llm agent above the runner",
			newParent: func(t *testing.T, mid agent.Agent) {
				if _, err := llmagent.New(llmagent.Config{Name: "top", SubAgents: []agent.Agent{mid}}); err != nil {
					t.Fatal(err)
				}
			},
			want: "leaf: leaf 2",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			leaf, err := llmagent.New(llmagent.Config{Name: "leaf", Model: &testutil.MockModel{Responses: []*genai.Content{
				genai.NewContentFromText("leaf 1", genai.RoleModel),
				genai.NewContentFromText("leaf 2", genai.RoleModel),
			}}})
			if err != nil {
				t.Fatal(err)
			}
			transfer := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID: "t1", Name: "transfer_to_agent", Args: map[string]any{"agent_name": "leaf"},
			}}}}
			mid, err := llmagent.New(llmagent.Config{Name: "mid", SubAgents: []agent.Agent{leaf}, Model: &testutil.MockModel{Responses: []*genai.Content{
				transfer,
				genai.NewContentFromText("mid 2", genai.RoleModel),
			}}})
			if err != nil {
				t.Fatal(err)
			}
			tt.newParent(t, mid)
			r, err := runner.NewInMemory("app", mid)
			if err != nil {
				t.Fatal(err)
			}

			var got []string
			for turn, text := range []string{"first", "second"} {
				got = nil
				for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText(text, genai.RoleUser), agent.RunConfig{}) {
					if err != nil {
						t.Fatalf("turn %d: Run() error = %v", turn+1, err)
					}
					if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text != "" {
						got = append(got, ev.Author+": "+ev.Content.Parts[0].Text)
					}
				}
			}
			if len(got) != 1 || got[0] != tt.want {
				t.Errorf("second turn text events = %q, want [%q]", got, tt.want)
			}
		})
	}
}

// TestFindAgentToRun_RunTreeParentWinsOverLaterParent checks that the
// transferability walk reads the parent from the tree being run when an agent
// is also listed in a SequentialAgent constructed later, which takes over the
// parent link agent.New records. Fails when the walk reads only those links:
// it then reaches the SequentialAgent, and root answers the second turn
// instead of leaf.
func TestFindAgentToRun_RunTreeParentWinsOverLaterParent(t *testing.T) {
	leaf, err := llmagent.New(llmagent.Config{Name: "leaf", Model: &testutil.MockModel{Responses: []*genai.Content{
		genai.NewContentFromText("leaf 1", genai.RoleModel),
		genai.NewContentFromText("leaf 2", genai.RoleModel),
	}}})
	if err != nil {
		t.Fatal(err)
	}
	transfer := &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
		ID: "t1", Name: "transfer_to_agent", Args: map[string]any{"agent_name": "leaf"},
	}}}}
	root, err := llmagent.New(llmagent.Config{Name: "root", SubAgents: []agent.Agent{leaf}, Model: &testutil.MockModel{Responses: []*genai.Content{
		transfer,
		genai.NewContentFromText("root 2", genai.RoleModel),
	}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "other", SubAgents: []agent.Agent{leaf}}}); err != nil {
		t.Fatal(err)
	}
	r, err := runner.NewInMemory("app", root)
	if err != nil {
		t.Fatal(err)
	}

	var got []string
	for turn, text := range []string{"first", "second"} {
		got = nil
		for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText(text, genai.RoleUser), agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("turn %d: Run() error = %v", turn+1, err)
			}
			if ev.Content != nil && len(ev.Content.Parts) > 0 && ev.Content.Parts[0].Text != "" {
				got = append(got, ev.Author+": "+ev.Content.Parts[0].Text)
			}
		}
	}
	if want := "leaf: leaf 2"; len(got) != 1 || got[0] != want {
		t.Errorf("second turn text events = %q, want [%q]", got, want)
	}
}
