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
	rand "math/rand/v2"
	"slices"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/model"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
)

func TestNewParallelAgent(t *testing.T) {
	tests := []struct {
		name          string
		maxIterations uint
		numSubAgents  int
		agentError    error // one of the subAgents will return this error
		cancelContext bool
		wantEvents    []*session.Event
		wantErr       bool
	}{
		{
			name:          "subagents complete run",
			maxIterations: 2,
			numSubAgents:  3,
			wantEvents: func() []*session.Event {
				var res []*session.Event
				for agentID := 1; agentID <= 3; agentID++ {
					for responseCount := 1; responseCount <= 2; responseCount++ {
						res = append(res, &session.Event{
							Author: fmt.Sprintf("sub%d", agentID),
							LLMResponse: model.LLMResponse{
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
			// one agent returns error, other agents run infinitely
			name:          "agent returns error",
			maxIterations: 0,
			numSubAgents:  100,
			agentError:    fmt.Errorf("agent error"),
			wantErr:       true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			ctx := t.Context()

			parallelAgent := newParallelAgent(t, tt.maxIterations, tt.numSubAgents, tt.agentError)

			var gotEvents []*session.Event

			sessionService := session.InMemoryService()

			agentRunner, err := runner.New(runner.Config{
				AppName:        "test_app",
				Agent:          parallelAgent,
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

			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			if tt.cancelContext {
				go func() {
					time.Sleep(5 * time.Millisecond)
					cancel()
				}()
			}

			for event, err := range agentRunner.Run(ctx, "user_id", "session_id", genai.NewContentFromText("user input", genai.RoleUser), agent.RunConfig{}) {
				if tt.wantErr != (err != nil) {
					if tt.cancelContext && err == nil {
						// In case of context cancellation some events can be processed before cancel is applied.
						continue
					}
					if tt.agentError != nil && err == nil {
						// In case of agent error some events from other agents can be processed before error is returned.
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
func newParallelAgent(t *testing.T, maxIterations uint, numSubAgents int, agentErr error) agent.Agent {
	var subAgents []agent.Agent

	for i := 1; i <= numSubAgents; i++ {
		subAgents = append(subAgents, must(loopagent.New(loopagent.Config{
			MaxIterations: maxIterations,
			AgentConfig: agent.Config{
				Name: fmt.Sprintf("loop_agent_%d", i),
				SubAgents: []agent.Agent{
					must(agent.New(agent.Config{
						Name: fmt.Sprintf("sub%d", i),
						Run:  customRun(i, nil),
					},
					)),
				},
			},
		})))
	}

	if agentErr != nil {
		subAgents = append(subAgents, must(agent.New(agent.Config{
			Name: "error_agent",
			Run:  customRun(-1, agentErr),
		})))
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

func customRun(id int, agentErr error) func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			time.Sleep((time.Duration(rand.IntN(5) + 1)) * time.Millisecond)
			if agentErr != nil {
				yield(nil, agentErr)
				return
			}
			yield(&session.Event{
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromText(fmt.Sprintf("hello %v", id), genai.RoleModel),
				},
			}, nil)
		}
	}
}
