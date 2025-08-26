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

package parallelagent_test

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/llm"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/sessionservice"

	"google.golang.org/genai"
)

func TestNewParallelAgent(t *testing.T) {
	tests := []struct {
		name          string
		maxIterations uint
		agentError    error // the agent will return this error
		cancelContext bool
		wantEvents    []*session.Event
		wantErr       bool
	}{
		{
			name:          "subagents complete run",
			maxIterations: 2,
			wantEvents: func() []*session.Event {
				var res []*session.Event
				for agentID := 1; agentID <= 2; agentID++ {
					for responseCount := 1; responseCount <= 2; responseCount++ {
						res = append(res, &session.Event{
							Author: fmt.Sprintf("sub%d", agentID),
							LLMResponse: &llm.Response{
								Content: &genai.Content{
									Parts: []*genai.Part{
										genai.NewPartFromText(fmt.Sprintf("hello %d", agentID)),
									},
									Role: genai.RoleModel,
								},
							},
						})
					}
				}
				return res
			}(),
		},
		{
			name:          "handle ctx cancel", // terminates infinite agent loop
			maxIterations: 0,
			cancelContext: true,
			wantErr:       true,
		},
		{
			name:          "agent returns error",
			maxIterations: 0,
			agentError:    fmt.Errorf("agent error"),
			wantErr:       true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			ctx := t.Context()

			agent := newParallelAgent(t, tt.maxIterations, tt.agentError)

			var gotEvents []*session.Event

			sessionService := sessionservice.Mem()

			agentRunner, err := runner.New("test_app", agent, sessionService)
			if err != nil {
				t.Fatal(err)
			}

			_, err = sessionService.Create(ctx, &sessionservice.CreateRequest{
				AppName:   "test_app",
				UserID:    "user_id",
				SessionID: "session_id",
			})
			if err != nil {
				t.Fatal(err)
			}

			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			if tt.cancelContext {
				go func() {
					time.Sleep(5 * time.Millisecond)
					cancel()
				}()
			}

			for event, err := range agentRunner.Run(ctx, "user_id", "session_id", genai.NewContentFromText("user input", genai.RoleUser), &runner.RunConfig{}) {
				if tt.wantErr != (err != nil) {
					if tt.cancelContext && err == nil {
						// In case of context cancellation some events can be processed before cancel is applied.
						continue
					}
					t.Errorf("got unexpected error: %v", err)
				}

				gotEvents = append(gotEvents, event)
			}

			if tt.wantEvents != nil {
				eventCompareFunc := func(e1, e2 *session.Event) int {
					if e1.Author <= e2.Author {
						return -1
					}
					if e1.Author == e2.Author {
						return 0
					}
					return 1
				}

				slices.SortFunc(tt.wantEvents, eventCompareFunc)
				slices.SortFunc(gotEvents, eventCompareFunc)

				if diff := cmp.Diff(tt.wantEvents, gotEvents); diff != "" {
					t.Errorf("events mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

// newParallelAgent creates parallel agent with 2 subagents emitting maxIterations events or infinitely if maxIterations==0.
func newParallelAgent(t *testing.T, maxIterations uint, agentErr error) agent.Agent {
	subAgents := []agent.Agent{
		must(loopagent.New(loopagent.Config{
			MaxIterations: maxIterations,
			AgentConfig: agent.Config{
				Name: "loop1",
				SubAgents: []agent.Agent{
					must(agent.New(agent.Config{
						Name: "sub1",
						Run:  customRun(1, agentErr),
					},
					)),
				},
			},
		})),
		must(loopagent.New(loopagent.Config{
			MaxIterations: maxIterations,
			AgentConfig: agent.Config{
				Name: "loop2",
				SubAgents: []agent.Agent{
					must(agent.New(agent.Config{
						Name: "sub2",
						Run:  customRun(2, agentErr),
					},
					)),
				},
			},
		})),
	}

	agent, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:      "test_agent",
			SubAgents: subAgents,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	return agent
}

func must[T agent.Agent](a T, err error) T {
	if err != nil {
		panic(err)
	}
	return a
}

func customRun(id int, agentErr error) func(agent.Context) iter.Seq2[*session.Event, error] {
	return func(agent.Context) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			time.Sleep(2 * time.Millisecond)
			if agentErr != nil {
				yield(nil, agentErr)
				return
			}
			yield(&session.Event{
				LLMResponse: &llm.Response{
					Content: genai.NewContentFromText(fmt.Sprintf("hello %v", id), genai.RoleModel),
				},
			}, nil)
		}
	}
}
