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
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/internal/httprr"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

const modelName = "gemini-2.0-flash-exp"

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

func TestParallelAgentWithTools(t *testing.T) {
	agent1 := createAgentWithGemini(t, "agent1")
	agent2 := createAgentWithGemini(t, "agent2")

	parallelAgent, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:      "parallel_test",
			SubAgents: []agent.Agent{agent1, agent2},
		},
	})
	if err != nil {
		t.Fatalf("Failed to create parallel agent: %v", err)
	}

	runner := testutil.NewTestAgentRunner(t, parallelAgent)
	stream := runner.Run(t, "test_session", "Search for AI news")

	events, err := testutil.CollectEvents(stream)
	if err != nil {
		t.Fatalf("Agent run failed: %v", err)
	}

	if len(events) < 2 {
		t.Errorf("Expected at least 2 events from parallel agents, got %d", len(events))
	}

	// Count FunctionCall and FunctionResponse events per branch
	branchCalls := make(map[string]int)
	branchResponses := make(map[string]int)

	for _, ev := range events {
		branch := ev.Branch
		if ev.LLMResponse.Content != nil {
			for _, part := range ev.LLMResponse.Content.Parts {
				if part.FunctionCall != nil {
					branchCalls[branch]++
				}
				if part.FunctionResponse != nil {
					branchResponses[branch]++
				}
			}
		}
	}

	for branch, calls := range branchCalls {
		responses := branchResponses[branch]
		if calls > responses {
			t.Errorf("Branch %s: session has %d FunctionCalls but only %d FunctionResponses. "+
				"This indicates race condition: agent read session before FunctionResponse was appended.",
				branch, calls, responses)
		}
	}
}

func createAgentWithGemini(t *testing.T, name string) agent.Agent {
	t.Helper()

	searchTool, err := functiontool.New(
		functiontool.Config{
			Name:        fmt.Sprintf("search_tool_%s", name),
			Description: "Search for information on the web",
		},
		func(ctx tool.Context, args struct{ Query string }) (string, error) {
			return fmt.Sprintf("search result for '%s' from %s", args.Query, name), nil
		},
	)
	if err != nil {
		t.Fatalf("Failed to create search tool: %v", err)
	}

	analyzeTool, err := functiontool.New(
		functiontool.Config{
			Name:        fmt.Sprintf("analyze_tool_%s", name),
			Description: "Analyze data and return insights",
		},
		func(ctx tool.Context, args struct{ Data string }) (string, error) {
			return fmt.Sprintf("analysis result for '%s' from %s", args.Data, name), nil
		},
	)
	if err != nil {
		t.Fatalf("Failed to create analyze tool: %v", err)
	}

	model := newGeminiModelForTest(t, modelName, name)

	a, err := llmagent.New(llmagent.Config{
		Name:        name,
		Description: fmt.Sprintf("Test agent %s that searches for information", name),
		Model:       model,
		Tools:       []tool.Tool{searchTool, analyzeTool},
		Instruction: "Use the search tool to find information, then provide a brief response.",
	})
	if err != nil {
		t.Fatalf("Failed to create agent %s: %v", name, err)
	}

	return a
}

func newGeminiModelForTest(t *testing.T, modelName string, agentName string) model.LLM {
	t.Helper()

	trace := filepath.Join("testdata", fmt.Sprintf("%s_%s.httprr",
		strings.ReplaceAll(t.Name(), "/", "_"), agentName))

	apiKey := "fakeKey"
	transport, recording := newGeminiTestTransport(t, trace)
	if recording {
		apiKey = ""
	}

	model, err := gemini.NewModel(t.Context(), modelName, &genai.ClientConfig{
		HTTPClient: &http.Client{Transport: transport},
		APIKey:     apiKey,
	})
	if err != nil {
		t.Fatalf("Failed to create Gemini model: %v", err)
	}
	return model
}

func newGeminiTestTransport(t *testing.T, rrfile string) (http.RoundTripper, bool) {
	t.Helper()
	rr, err := testutil.NewGeminiTransport(rrfile)
	if err != nil {
		t.Fatal(err)
	}
	recording, _ := httprr.Recording(rrfile)
	return rr, recording
}
