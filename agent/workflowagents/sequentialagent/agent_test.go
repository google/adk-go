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

package sequentialagent_test

import (
	"context"
	"fmt"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

func TestNewSequentialAgent(t *testing.T) {
	type args struct {
		maxIterations uint
		subAgents     []agent.Agent
	}

	sameAgent := newSequentialAgent(t, []agent.Agent{newCustomAgent(t, 1), newCustomAgent(t, 2)}, "same_agent")

	tests := []struct {
		name           string
		args           args
		wantEvents     []*session.Event
		wantErr        bool
		wantErrMessage string
	}{
		{
			name: "ok",
			args: args{
				maxIterations: 0,
				subAgents:     []agent.Agent{newCustomAgent(t, 0), newCustomAgent(t, 1)},
			},
			wantEvents: []*session.Event{
				{
					Author: "custom_agent_0",
					LLMResponse: model.LLMResponse{
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
					LLMResponse: model.LLMResponse{
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
		{
			name: "ok with inner sequential",
			args: args{
				maxIterations: 0,
				subAgents:     []agent.Agent{newCustomAgent(t, 0), newSequentialAgent(t, []agent.Agent{newCustomAgent(t, 1), newCustomAgent(t, 2)}, "test_agent1"), newCustomAgent(t, 3)},
			},
			wantEvents: []*session.Event{
				{
					Author: "custom_agent_0",
					LLMResponse: model.LLMResponse{
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
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{
								genai.NewPartFromText("hello 1"),
							},
							Role: genai.RoleModel,
						},
					},
				},
				{
					Author: "custom_agent_2",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{
								genai.NewPartFromText("hello 2"),
							},
							Role: genai.RoleModel,
						},
					},
				},
				{
					Author: "custom_agent_3",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Parts: []*genai.Part{
								genai.NewPartFromText("hello 3"),
							},
							Role: genai.RoleModel,
						},
					},
				},
			},
		},
		{
			name: "err with inner sequential with same name as root",
			args: args{
				maxIterations: 0,
				subAgents:     []agent.Agent{newCustomAgent(t, 0), newSequentialAgent(t, []agent.Agent{newCustomAgent(t, 1), newCustomAgent(t, 2)}, "test_agent1"), newCustomAgent(t, 3)},
			},
			wantErr:        true,
			wantErrMessage: `failed to create agent tree: agent names must be unique in the agent tree, found duplicate: "test_agent"`,
		},
		{
			name: "err with 2 levels of inner sequential with same name as root ",
			args: args{
				maxIterations: 0,
				subAgents: []agent.Agent{newCustomAgent(t, 0), newSequentialAgent(t, []agent.Agent{
					newSequentialAgent(t, []agent.Agent{newCustomAgent(t, 1), newCustomAgent(t, 2)}, "test_agent1"),
				}, "test_agent"), newCustomAgent(t, 3)},
			},
			wantErr:        true,
			wantErrMessage: `failed to create agent tree: agent names must be unique in the agent tree, found duplicate: "test_agent"`,
		},
		{
			name: "err with 2 levels of inner sequential with same name as parent ",
			args: args{
				maxIterations: 0,
				subAgents: []agent.Agent{newCustomAgent(t, 0), newSequentialAgent(t, []agent.Agent{
					newSequentialAgent(t, []agent.Agent{newCustomAgent(t, 1), newCustomAgent(t, 2)}, "test_agent1"),
				}, "test_agent1"), newCustomAgent(t, 3)},
			},
			wantErr:        true,
			wantErrMessage: `failed to create agent tree: agent names must be unique in the agent tree, found duplicate: "test_agent1"`,
		},
		{
			name: "err with repeated inner sequential",
			args: args{
				maxIterations: 0,
				subAgents:     []agent.Agent{newCustomAgent(t, 0), sameAgent, sameAgent, newCustomAgent(t, 3)},
			},
			wantErr:        true,
			wantErrMessage: `failed to create base agent: error creating agent: subagent "same_agent" appears multiple times in subAgents`,
		},
		{
			name: "err with repeated inner sequential in two levels",
			args: args{
				maxIterations: 0,
				subAgents: []agent.Agent{
					newCustomAgent(t, 0), newSequentialAgent(t, []agent.Agent{sameAgent}, "test_agent1"),
					sameAgent, newCustomAgent(t, 3),
				},
			},
			wantErr:        true,
			wantErrMessage: `failed to create agent tree: "same_agent" agent cannot have >1 parents, found: "test_agent1", "test_agent"`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := t.Context()

			sequentialAgent, err := sequentialagent.New(sequentialagent.Config{
				AgentConfig: agent.Config{
					Name:      "test_agent",
					SubAgents: tt.args.subAgents,
				},
			})
			if err != nil {
				if !tt.wantErr {
					t.Errorf("NewSequentialAgent() error = %v, wantErr %v", err, tt.wantErr)
				}
				if diff := cmp.Diff(tt.wantErrMessage, err.Error()); diff != "" {
					t.Errorf("err message mismatch (-want +got):\n%s", diff)
				}
				return
			}

			var gotEvents []*session.Event

			sessionService := session.InMemoryService()

			agentRunner, err := runner.New(runner.Config{
				AppName:        "test_app",
				Agent:          sequentialAgent,
				SessionService: sessionService,
			})
			if err != nil {
				if !tt.wantErr {
					t.Fatalf("NewSequentialAgent() error = %v, wantErr %v", err, tt.wantErr)
				}
				if diff := cmp.Diff(tt.wantErrMessage, err.Error()); diff != "" {
					t.Fatalf("err message mismatch (-want +got):\n%s", diff)
				}
				return
			}

			_, err = sessionService.Create(ctx, &session.CreateRequest{
				AppName:   "test_app",
				UserID:    "user_id",
				SessionID: "session_id",
			})
			if err != nil {
				t.Fatal(err)
			}

			// run twice, the second time it will need to determine which agent to use, and we want to get the same result
			gotEvents = make([]*session.Event, 0)
			for range 2 {
				for event, err := range agentRunner.Run(ctx, "user_id", "session_id", genai.NewContentFromText("user input", genai.RoleUser), agent.RunConfig{}) {
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
					tt.wantEvents[i].Timestamp = gotEvent.Timestamp
					if diff := cmp.Diff(tt.wantEvents[i], gotEvent, cmpopts.IgnoreFields(session.Event{}, "ID", "Timestamp", "InvocationID"),
						cmpopts.IgnoreFields(session.EventActions{}, "StateDelta")); diff != "" {
						t.Errorf("event[i] mismatch (-want +got):\n%s", diff)
					}
				}
			}
		})
	}
}

func newCustomAgent(t *testing.T, id int) agent.Agent {
	t.Helper()

	a, err := llmagent.New(llmagent.Config{
		Name:  fmt.Sprintf("custom_agent_%v", id),
		Model: &FakeLLM{id: id, callCounter: 0},
	})
	if err != nil {
		t.Fatal(err)
	}

	return a
}

func newSequentialAgent(t *testing.T, subAgents []agent.Agent, name string) agent.Agent {
	t.Helper()

	sequentialAgent, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      name,
			SubAgents: subAgents,
		},
	})
	if err != nil {
		t.Fatalf("NewSequentialAgent() error = %v", err)
	}

	return sequentialAgent
}

// FakeLLM is a mock implementation of model.LLM for testing.
type FakeLLM struct {
	id          int
	callCounter int
}

func (f *FakeLLM) Name() string {
	return "fake-llm"
}

func (f *FakeLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		f.callCounter++

		yield(&model.LLMResponse{
			Content: genai.NewContentFromText(fmt.Sprintf("hello %v", f.id), genai.RoleModel),
		}, nil)
	}
}
