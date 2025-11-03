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
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/genai"
)

func TestNewSequentialAgent(t *testing.T) {
	type args struct {
		maxIterations uint
		subAgents     []agent.Agent
	}

	tests := []struct {
		name       string
		args       args
		wantEvents []*session.Event
		wantErr    bool
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
			if (err != nil) != tt.wantErr {
				t.Errorf("NewSequentialAgent() error = %v, wantErr %v", err, tt.wantErr)
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
				t.Fatal(err)
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
			for i := 0; i < 2; i++ {
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
