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
	"context"
	"fmt"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/types"

	"google.golang.org/genai"
)

func TestNewLoopAgent(t *testing.T) {
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
			name: "infinite loop",
			args: args{
				maxIterations: 0,
				subAgents:     []types.Agent{newCustomAgent(0)},
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
			},
		},
		{
			name: "loop agent with max iterations",
			args: args{
				maxIterations: 1,
				subAgents:     []types.Agent{newCustomAgent(0)},
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
					t.Errorf("event[%v] mismatch (-want +got):\n%s", i, diff)
				}
			}
		})
	}
}

func newCustomAgent(id int) *customAgent {
	return &customAgent{
		id: id,
		spec: &types.AgentSpec{
			Name: fmt.Sprintf("custom_agent_%v", id),
		},
	}
}

// TODO: create test util allowing to create custom agents, agent trees for
type customAgent struct {
	id          int
	spec        *types.AgentSpec
	callCounter int
}

func (a *customAgent) Spec() *types.AgentSpec { return a.spec }

func (a *customAgent) Run(context.Context, *types.InvocationContext) iter.Seq2[*types.Event, error] {
	return func(yield func(*types.Event, error) bool) {
		a.callCounter++

		yield(&types.Event{
			Author: a.spec.Name,
			LLMResponse: &types.LLMResponse{
				Content: genai.NewContentFromText(fmt.Sprintf("hello %v", a.id), genai.RoleModel),
			},
		}, nil)
	}
}
