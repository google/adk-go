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

package llmagent_test

import (
	"fmt"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
)

// TestGlobalInstruction_RunStartedBelowRootUsesTreeRoot checks that a run
// started at a sub-agent takes the global instruction from the root of the
// sub-agent's tree, above the runner's agent, as adk-python reads it from
// agent.root_agent (flows/llm_flows/prompt/_instructions.py). Fails when the
// root is taken from the run's parent map alone, which ends at the runner's
// agent, or when the walk stops at the first parent above it.
func TestGlobalInstruction_RunStartedBelowRootUsesTreeRoot(t *testing.T) {
	tests := []struct {
		name string
		// levels is the number of agents between the runner's agent and top.
		levels int
	}{
		{name: "runner at a child of the root", levels: 0},
		{name: "runner at a grandchild of the root", levels: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			topModel := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("ok", genai.RoleModel)}}
			start := newLLMAgent(t, llmagent.Config{
				Name:                     "start",
				Instruction:              "start instruction",
				DisallowTransferToParent: true,
				DisallowTransferToPeers:  true,
			})
			child := start
			for i := range tt.levels {
				child = newLLMAgent(t, llmagent.Config{Name: fmt.Sprintf("between%d", i), SubAgents: []agent.Agent{child}})
			}
			newLLMAgent(t, llmagent.Config{
				Name:              "top",
				Model:             topModel,
				GlobalInstruction: "global from top",
				SubAgents:         []agent.Agent{child},
			})
			r, err := runner.NewInMemory("app", start)
			if err != nil {
				t.Fatal(err)
			}

			for _, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("Run() error = %v", err)
				}
			}

			if len(topModel.Requests) != 1 {
				t.Fatalf("model got %d requests, want 1", len(topModel.Requests))
			}
			var system string
			if cfg := topModel.Requests[0].Config; cfg != nil && cfg.SystemInstruction != nil {
				for _, part := range cfg.SystemInstruction.Parts {
					system += part.Text
				}
			}
			if want := "global from top\n\nstart instruction"; !strings.HasPrefix(system, want) {
				t.Errorf("system instruction = %q, want prefix %q", system, want)
			}
		})
	}
}
