// Copyright 2025 Google LLC
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

package agent_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/types"
	"google.golang.org/genai"
)

func TestNewSequentialAgent(t *testing.T) {
	type args struct {
		maxIterations uint
		subAgents     []types.Agent
	}

	tests := []struct {
		name       string
		args       args
		wantEvents []*types.Event
		wantErr    bool
	}{
		{
			name: "ok",
			args: args{
				maxIterations: 0,
				subAgents:     []types.Agent{newCustomAgent(0), newCustomAgent(1)},
			},
			wantEvents: []*types.Event{
				{
					Author: "custom_agent_0",
					LLMResponse: &types.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{
								genai.NewPartFromText("hello 0"),
							},
							Role: genai.RoleModel,
						},
					},
				},
				{
					Author: "custom_agent_1",
					LLMResponse: &types.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{
								genai.NewPartFromText("hello 1"),
							},
							Role: genai.RoleModel,
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			agent, err := agent.NewLoopAgent("test_agent", tt.args.maxIterations, agent.WithSubAgents(tt.args.subAgents...))
			if (err != nil) != tt.wantErr {
				t.Errorf("NewLoopAgent() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			var gotEvents []*types.Event

			for event, err := range newTestAgentRunner(t, agent).Run(t, "session_id", "user input") {
				if err != nil {
					t.Errorf("got unexpected error: %v", err)
				}

				if tt.args.maxIterations == 0 && len(gotEvents) == len(tt.wantEvents) {
					break
				}

				gotEvents = append(gotEvents, event)
			}

			if len(tt.wantEvents) != len(gotEvents) {
				t.Fatalf("Unexpected event length, got: %v, want: %v", len(gotEvents), len(tt.wantEvents))
			}

			for i, gotEvent := range gotEvents {
				tt.wantEvents[i].Time = gotEvent.Time
				if diff := cmp.Diff(tt.wantEvents[i], gotEvent); diff != "" {
					t.Errorf("event[i] mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}
