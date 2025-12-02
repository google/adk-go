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

package loopagent_test

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
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

func TestNewLoopAgent(t *testing.T) {
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
			name: "infinite loop",
			args: args{
				maxIterations: 0,
				subAgents:     []agent.Agent{newCustomAgent(t, 0)},
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
			},
		},
		{
			name: "loop agent with max iterations",
			args: args{
				maxIterations: 1,
				subAgents:     []agent.Agent{newCustomAgent(t, 0)},
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
			},
		},
		{
			name: "loop agent with max iterations and 2 sub agents",
			args: args{
				maxIterations: 1,
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
			name: "loop with escalate function returns sumarization",
			args: args{
				maxIterations: 2,
				subAgents:     []agent.Agent{newLmmAgentWithFunctionCall(t, 0, false), newCustomAgent(t, 1)},
			},
			wantEvents: []*session.Event{
				{
					Author: "custom_agent_0",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromFunctionCall("exampleFunction", make(map[string]any), genai.RoleModel),
					},
				},
				{
					Author: "custom_agent_0",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromFunctionResponse("exampleFunction", make(map[string]any), genai.RoleUser),
					},
					Actions: session.EventActions{
						Escalate: true,
					},
				},
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
			},
		},
		{
			name: "loop with escalate function returns sumarization",
			args: args{
				maxIterations: 2,
				subAgents:     []agent.Agent{newLmmAgentWithFunctionCall(t, 0, true), newCustomAgent(t, 1)},
			},
			wantEvents: []*session.Event{
				{
					Author: "custom_agent_0",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromFunctionCall("exampleFunction", make(map[string]any), genai.RoleModel),
					},
				},
				{
					Author: "custom_agent_0",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromFunctionResponse("exampleFunction", make(map[string]any), genai.RoleUser),
					},
					Actions: session.EventActions{
						Escalate:          true,
						SkipSummarization: true,
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := t.Context()

			loopAgent, err := loopagent.New(loopagent.Config{
				MaxIterations: tt.args.maxIterations,
				AgentConfig: agent.Config{
					Name:      "test_agent",
					SubAgents: tt.args.subAgents,
				},
			})
			if (err != nil) != tt.wantErr {
				t.Errorf("NewLoopAgent() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			var gotEvents []*session.Event

			sessionService := session.InMemoryService()

			agentRunner, err := runner.New(runner.Config{
				AppName:        "test_app",
				Agent:          loopAgent,
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

			ignoreFields := []cmp.Option{
				cmpopts.IgnoreFields(session.Event{}, "ID", "InvocationID", "Timestamp"),
				cmpopts.IgnoreFields(session.EventActions{}, "StateDelta"),
				cmpopts.IgnoreFields(genai.FunctionCall{}, "ID"),
				cmpopts.IgnoreFields(genai.FunctionResponse{}, "ID"),
			}

			for i, gotEvent := range gotEvents {
				tt.wantEvents[i].Timestamp = gotEvent.Timestamp
				if diff := cmp.Diff(tt.wantEvents[i], gotEvent, ignoreFields...); diff != "" {
					t.Errorf("event[%v] mismatch (-want +got):\n%s", i, diff)
				}
			}
		})
	}
}

func newCustomAgent(t *testing.T, id int) agent.Agent {
	t.Helper()

	customAgent := &customAgent{
		id: id,
	}

	a, err := agent.New(agent.Config{
		Name: fmt.Sprintf("custom_agent_%v", id),
		Run:  customAgent.Run,
	})
	if err != nil {
		t.Fatal(err)
	}

	return a
}

// TODO: create test util allowing to create custom agents, agent trees for
type customAgent struct {
	id          int
	callCounter int
}

func (a *customAgent) Run(agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		a.callCounter++

		yield(&session.Event{
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText(fmt.Sprintf("hello %v", a.id), genai.RoleModel),
			},
		}, nil)
	}
}

type EmptyArgs struct{}

func exampleFunctionThatEscalates(ctx tool.Context, myArgs EmptyArgs) (map[string]string, error) {
	ctx.Actions().Escalate = true
	ctx.Actions().SkipSummarization = false
	return map[string]string{}, nil
}

func exampleFunctionThatEscalatesAndSkips(ctx tool.Context, myArgs EmptyArgs) (map[string]string, error) {
	ctx.Actions().Escalate = true
	ctx.Actions().SkipSummarization = true
	return map[string]string{}, nil
}

func newLmmAgentWithFunctionCall(t *testing.T, id int, skipSummarization bool) agent.Agent {
	t.Helper()

	exampleFunction := exampleFunctionThatEscalates
	if skipSummarization {
		exampleFunction = exampleFunctionThatEscalatesAndSkips
	}

	exampleFunctionThatEscalatesTool, err := functiontool.New(functiontool.Config{
		Name:        "exampleFunction",
		Description: "Call this function to escalate\n",
	}, exampleFunction)
	if err != nil {
		t.Fatalf("error creating exampleFunction tool: %s", err)
	}

	customAgent, err := llmagent.New(llmagent.Config{
		Name:  fmt.Sprintf("custom_agent_%v", id),
		Model: &FakeLLM{id: id, callCounter: 0, skipSummarization: skipSummarization},
		Tools: []tool.Tool{exampleFunctionThatEscalatesTool},
	})
	if err != nil {
		t.Fatal(err)
	}

	return customAgent
}

// FakeLLM is a mock implementation of model.LLM for testing.
type FakeLLM struct {
	id                int
	callCounter       int
	skipSummarization bool
}

func (f *FakeLLM) Name() string {
	return "fake-llm"
}

func (f *FakeLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		f.callCounter++

		if len(req.Contents) == 1 {
			if !yield(&model.LLMResponse{
				Content: genai.NewContentFromFunctionCall("exampleFunction", make(map[string]any), genai.RoleModel),
			}, nil) {
				return
			}
		} else {
			if !yield(&model.LLMResponse{
				Content: genai.NewContentFromText(fmt.Sprintf("hello %v", f.id), genai.RoleModel),
			}, nil) {
				return
			}
		}
	}
}
