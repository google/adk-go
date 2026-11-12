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
	"errors"
	"fmt"
	"iter"
	rand "math/rand/v2"
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagent"
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/httprr"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/workflow"
)

const modelName = "gemini-2.5-flash"

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

				// IDs and timestamps are filled in when the events are run and
				// appended, so they cannot be expressed in a fixture. This test
				// is about which events came out and who authored them.
				if diff := cmp.Diff(tt.wantEvents, gotEvents, cmpopts.IgnoreFields(session.Event{}, "ID", "Timestamp")); diff != "" {
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
			time.Sleep(time.Duration(rand.IntN(5)+1) * time.Millisecond)
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

// TestParallelAgent_AwaitsSubAgentTeardownOnEarlyStop verifies that when the
// consumer stops early, Run does not return until every sub-agent goroutine has
// finished its deferred teardown. Otherwise teardown can outlive the run and
// touch an already-ended context.
func TestParallelAgent_AwaitsSubAgentTeardownOnEarlyStop(t *testing.T) {
	t.Parallel()

	const numSubAgents = 3
	// Each sub-agent teardown announces it began (buffered so it never blocks) and
	// then blocks on releaseTeardown, so the test can prove Run waits for teardown
	// without relying on wall-clock timing.
	teardownStarted := make(chan struct{}, numSubAgents)
	releaseTeardown := make(chan struct{})
	var teardownFinished atomic.Int32

	var subAgents []agent.Agent
	for i := 1; i <= numSubAgents; i++ {
		subAgents = append(subAgents, must(agent.New(agent.Config{
			Name: fmt.Sprintf("sub%d", i),
			Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					defer func() {
						teardownStarted <- struct{}{}
						<-releaseTeardown
						teardownFinished.Add(1)
					}()
					for {
						if !yield(&session.Event{
							LLMResponse: model.LLMResponse{
								Content: genai.NewContentFromText(fmt.Sprintf("hello %d", i), genai.RoleModel),
							},
						}, nil) {
							return
						}
					}
				}
			},
		})))
	}

	parallelAgent, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:      "test_agent",
			SubAgents: subAgents,
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	agentRunner, err := runner.New(runner.Config{
		AppName:           "test_app",
		Agent:             parallelAgent,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	runReturned := make(chan struct{})
	go func() {
		defer close(runReturned)
		for _, err := range agentRunner.Run(t.Context(), "user_id", "session_id", genai.NewContentFromText("user input", genai.RoleUser), agent.RunConfig{}) {
			if err != nil {
				t.Errorf("Run() yielded error: %v", err)
			}
			break // stop after the first event
		}
	}()

	// Always release gated teardowns so no goroutine is left blocked, even if an
	// assertion fails. Safe to call twice (same goroutine as the explicit release).
	released := false
	release := func() {
		if !released {
			released = true
			close(releaseTeardown)
		}
	}
	defer release()

	// Every sub-agent must reach teardown once the consumer stops early.
	for range numSubAgents {
		select {
		case <-teardownStarted:
		case <-runReturned:
			t.Fatal("Run returned before all sub-agent teardowns started")
		}
	}

	// Teardowns are now gated; Run must still be blocked waiting for them.
	select {
	case <-runReturned:
		t.Fatal("Run returned while sub-agent teardowns were still in progress")
	default:
	}

	release()
	<-runReturned

	if got := teardownFinished.Load(); got != numSubAgents {
		t.Errorf("teardownFinished = %d, want %d", got, numSubAgents)
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
		func(ctx agent.Context, args struct{ Query string }) (string, error) {
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
		func(ctx agent.Context, args struct{ Data string }) (string, error) {
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

func newGeminiModelForTest(t *testing.T, modelName, agentName string) model.LLM {
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

// TestParallelAgent_PropagatesContextError verifies that if the context is canceled,
// the iterator yields the error from errgroup.Wait().
func TestParallelAgent_PropagatesContextError(t *testing.T) {
	t.Parallel()

	// Create a sub-agent that yields an event and then waits.
	// We want to trigger runSubAgent returning ctx.Err().
	subAgent := must(agent.New(agent.Config{
		Name: "yielder",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				// Yield one event so we engage runSubAgent logic
				if !yield(&session.Event{
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("hello", genai.RoleModel),
					},
				}, nil) {
					return
				}

				// Wait for context cancellation
				<-ctx.Done()
			}
		},
	}))

	parallelAgent, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:      "parallel_agent",
			SubAgents: []agent.Agent{subAgent},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	spy := &spyAgent{Agent: parallelAgent}

	ctx, cancel := context.WithCancel(t.Context())

	sessionService := session.InMemoryService()
	_, _ = sessionService.Create(ctx, &session.CreateRequest{
		AppName:   "test_app",
		UserID:    "user_id",
		SessionID: "session_id",
	})

	r, err := runner.New(runner.Config{
		AppName:        "test_app",
		Agent:          spy,
		SessionService: sessionService,
	})
	if err != nil {
		t.Fatal(err)
	}

	go func() {
		// Wait a tiny bit to ensure we started
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	for range r.Run(ctx, "user_id", "session_id", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
		// Simulate processing delay so that ackChan takes time,
		// increasing chance runSubAgent is blocked on ackChan when cancel happens?
		time.Sleep(100 * time.Millisecond)
	}

	if spy.yieldedError == nil {
		t.Fatal("Expected parallelAgent to yield an error (e.g. context canceled), but it yielded nil")
	}

	t.Logf("Yielded error: %v", spy.yieldedError)
}

type spyAgent struct {
	agent.Agent
	yieldedError error
}

func (s *spyAgent) Run(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	next := s.Agent.Run(ctx)
	return func(yield func(*session.Event, error) bool) {
		for event, err := range next {
			if err != nil {
				s.yieldedError = err
			}
			if !yield(event, err) {
				return
			}
		}
	}
}

func TestParallelAgent_StateSync(t *testing.T) {
	ctx := t.Context()

	var gotValue any
	var gotErr error

	subAgent, err := agent.New(agent.Config{
		Name: "test_subagent",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := &session.Event{
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("hello", genai.RoleModel),
					},
					Actions: session.EventActions{
						StateDelta: map[string]any{"test_key": "test_value"},
					},
				}
				yield(event, nil)
			}
		},
		AfterAgentCallbacks: []agent.AfterAgentCallback{
			func(c agent.Context) (*genai.Content, error) {
				gotValue, gotErr = c.State().Get("test_key")
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	parallelAgent, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:      "test_parallel_agent",
			SubAgents: []agent.Agent{subAgent},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

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

	for _, err := range agentRunner.Run(ctx, "user_id", "session_id", genai.NewContentFromText("user input", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
	}

	if gotErr != nil {
		t.Fatalf("expected to get value from state, got error: %v", gotErr)
	}
	if gotValue != "test_value" {
		t.Fatalf("expected state value 'test_value', got %v", gotValue)
	}
}

// adk-python's ParallelAgent stops its remaining branches as soon as a direct
// sub-agent yields an event with Actions.escalate (parallel_agent.py,
// _asks_this_agent_to_exit; asserted by
// test_run_async_short_circuits_other_agents_on_escalate_action). The Go
// agent kept every sibling running to completion.
func TestParallelAgent_EscalateStopsSiblings(t *testing.T) {
	var resumed atomic.Bool
	escalator, err := agent.New(agent.Config{
		Name: "escalator",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				if !yield(&session.Event{
					LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("escalating", genai.RoleModel)},
					Actions:     session.EventActions{Escalate: true},
				}, nil) {
					return
				}
				// adk-python never resumes the escalating branch.
				resumed.Store(true)
				yield(&session.Event{
					LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("after escalate", genai.RoleModel)},
				}, nil)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// slow never yields on its own; only a cancel ends it.
	var cancelled atomic.Bool
	slow, err := agent.New(agent.Config{
		Name: "slow",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				<-ctx.Done()
				cancelled.Store(true)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	parallelAgent, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{escalator, slow}},
	})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "test_app", Agent: parallelAgent, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}

	// The deadline only bounds a regression: without the fix, slow keeps the
	// run open until this context ends.
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	var authors []string
	for ev, err := range r.Run(ctx, "user_id", "session_id", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		authors = append(authors, ev.Author)
	}
	if ctx.Err() != nil {
		t.Fatalf("run ended only at the test deadline: sibling kept running after escalate, want it cancelled")
	}
	if !cancelled.Load() {
		t.Errorf("slow was not cancelled after escalate")
	}
	if resumed.Load() {
		t.Errorf("escalating branch was resumed after its escalate event, want it stopped")
	}
	if want := []string{"escalator"}; !slices.Equal(authors, want) {
		t.Errorf("events authored by %v, want %v", authors, want)
	}
}

// When a sibling fails before a direct sub-agent escalates, the caller must
// still get the failure. adk-python queues the failure when it happens and
// raises it before the later escalation is handled (parallel_agent.py).
func TestParallelAgent_SiblingErrorSurvivesEscalate(t *testing.T) {
	boom := errors.New("boom")
	// The escalator waits for the failure to cancel its context and then
	// escalates anyway, so the failure always comes first. The parallel agent
	// then picks between the cancel and the escalation at random, so repeat.
	for i := range 50 {
		failing, err := agent.New(agent.Config{
			Name: "failing",
			Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) { yield(nil, boom) }
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		escalator, err := agent.New(agent.Config{
			Name: "escalator",
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					<-ctx.Done()
					yield(&session.Event{
						LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("escalating", genai.RoleModel)},
						Actions:     session.EventActions{Escalate: true},
					}, nil)
				}
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		parallelAgent, err := parallelagent.New(parallelagent.Config{
			AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{failing, escalator}},
		})
		if err != nil {
			t.Fatal(err)
		}
		r, err := runner.New(runner.Config{AppName: "test_app", Agent: parallelAgent, SessionService: session.InMemoryService(), AutoCreateSession: true})
		if err != nil {
			t.Fatal(err)
		}

		sawErr := false
		for _, err := range r.Run(t.Context(), "user_id", "session_id", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
			if errors.Is(err, boom) {
				sawErr = true
			}
		}
		if !sawErr {
			t.Fatalf("run %d: sibling error was not returned after an escalate", i)
		}
	}
}

// A sibling's error must reach the caller even when a direct sub-agent
// escalates while that sibling is still unwinding: errgroup only records the
// error once the sibling's Run has returned.
func TestParallelAgent_ErrorDuringSiblingTeardownSurvivesEscalate(t *testing.T) {
	boom := errors.New("boom")
	failed := make(chan struct{})
	failing, err := agent.New(agent.Config{
		Name: "failing",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(nil, boom)
				close(failed)
				// Teardown that lasts until the run cancels it.
				<-ctx.Done()
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	escalator, err := agent.New(agent.Config{
		Name: "escalator",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				<-failed
				yield(&session.Event{
					LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("escalating", genai.RoleModel)},
					Actions:     session.EventActions{Escalate: true},
				}, nil)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	parallelAgent, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{failing, escalator}},
	})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "test_app", Agent: parallelAgent, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}

	// The deadline only bounds a regression where the run never cancels the
	// failing sibling's teardown.
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	sawErr := false
	for ev, err := range r.Run(ctx, "user_id", "session_id", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if errors.Is(err, boom) {
			sawErr = true
		}
		if ev != nil && ev.Actions.Escalate {
			t.Errorf("escalate event yielded after a sibling error, want the error instead")
		}
	}
	if ctx.Err() != nil {
		t.Fatalf("run ended only at the test deadline")
	}
	if !sawErr {
		t.Errorf("sibling error was not returned")
	}
}

// A caller cancel that comes after a sibling's failure must not hide that
// failure when a direct sub-agent then escalates.
func TestParallelAgent_CancelAfterSiblingErrorKeepsError(t *testing.T) {
	boom := errors.New("boom")
	// The escalator sees the failure, the caller cancels, then the escalator
	// escalates anyway. The parallel agent then picks between the cancel and
	// the escalation at random, so repeat.
	for i := range 50 {
		ctx, cancel := context.WithCancel(t.Context())
		failing, err := agent.New(agent.Config{
			Name: "failing",
			Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) { yield(nil, boom) }
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		escalator, err := agent.New(agent.Config{
			Name: "escalator",
			Run: func(ictx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					<-ictx.Done()
					cancel()
					yield(&session.Event{
						LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("escalating", genai.RoleModel)},
						Actions:     session.EventActions{Escalate: true},
					}, nil)
				}
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		parallelAgent, err := parallelagent.New(parallelagent.Config{
			AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{failing, escalator}},
		})
		if err != nil {
			t.Fatal(err)
		}
		r, err := runner.New(runner.Config{AppName: "test_app", Agent: parallelAgent, SessionService: session.InMemoryService(), AutoCreateSession: true})
		if err != nil {
			t.Fatal(err)
		}

		sawErr := false
		for _, err := range r.Run(ctx, "user_id", "session_id", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
			if errors.Is(err, boom) {
				sawErr = true
			}
		}
		cancel()
		if !sawErr {
			t.Fatalf("run %d: sibling error was hidden by a later caller cancel", i)
		}
	}
}

// An escalation from a deeper agent ends that agent's own parent workflow, not
// this one. adk-python only reacts to escalations authored by a direct
// sub-agent (parallel_agent.py, _asks_this_agent_to_exit).
func TestParallelAgent_NestedEscalateKeepsSiblings(t *testing.T) {
	escalator, err := agent.New(agent.Config{
		Name: "escalator",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(&session.Event{
					LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("escalating", genai.RoleModel)},
					Actions:     session.EventActions{Escalate: true},
				}, nil)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	inner, err := loopagent.New(loopagent.Config{
		AgentConfig: agent.Config{Name: "inner", SubAgents: []agent.Agent{escalator}},
	})
	if err != nil {
		t.Fatal(err)
	}
	// slow yields only once the caller has seen the nested escalation, so the
	// test separates the two behaviors without depending on timing.
	seenEscalate := make(chan struct{})
	slow, err := agent.New(agent.Config{
		Name: "slow",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				select {
				case <-seenEscalate:
				case <-ctx.Done():
					return
				}
				yield(&session.Event{
					LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("slow finished", genai.RoleModel)},
				}, nil)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	parallelAgent, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{Name: "par", SubAgents: []agent.Agent{inner, slow}},
	})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{AppName: "test_app", Agent: parallelAgent, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}

	// The deadline only bounds a regression where the nested escalation is
	// never yielded and slow waits for it.
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	var authors []string
	for ev, err := range r.Run(ctx, "user_id", "session_id", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatal(err)
		}
		authors = append(authors, ev.Author)
		if ev.Author == "escalator" && ev.Actions.Escalate {
			close(seenEscalate)
		}
	}
	if !slices.Contains(authors, "escalator") {
		t.Errorf("events authored by %v: the nested escalation was never yielded", authors)
	}
	if !slices.Contains(authors, "slow") {
		t.Errorf("events authored by %v: sibling was stopped by a nested escalate, want it to finish", authors)
	}
}

// An error yielded by an agent nested below a direct sub-agent, for example
// inside an inner ParallelAgent, must be returned when another direct
// sub-agent escalates after it, even if the agents in between have not
// returned the error yet. Each case runs once; with the fix the outcome does
// not depend on timing.
func TestParallelAgent_NestedErrorSurvivesEscalate(t *testing.T) {
	boom := errors.New("boom")
	parallel := func(name string, subAgents ...agent.Agent) agent.Agent {
		return must(parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: name, SubAgents: subAgents}}))
	}
	sequential := func(name string, subAgents ...agent.Agent) agent.Agent {
		return must(sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: name, SubAgents: subAgents}}))
	}
	loop := func(name string, subAgents ...agent.Agent) agent.Agent {
		return must(loopagent.New(loopagent.Config{MaxIterations: 1, AgentConfig: agent.Config{Name: name, SubAgents: subAgents}}))
	}
	nestings := []struct {
		name string
		wrap func(failing agent.Agent) agent.Agent
	}{
		{"inner parallel", func(f agent.Agent) agent.Agent { return parallel("inner", f) }},
		{"sequential then parallel", func(f agent.Agent) agent.Agent { return sequential("seq", parallel("inner", f)) }},
		{"loop then parallel", func(f agent.Agent) agent.Agent { return loop("loop", parallel("inner", f)) }},
		{"loop, sequential, parallel", func(f agent.Agent) agent.Agent {
			return loop("loop", sequential("seq", parallel("inner", f)))
		}},
		{"parallel, sequential, parallel", func(f agent.Agent) agent.Agent {
			return parallel("mid", sequential("seq", parallel("inner", f)))
		}},
	}
	cleanups := []struct {
		name string
		run  func(ctx agent.InvocationContext)
	}{
		{"no cleanup", func(agent.InvocationContext) {}},
		{"50ms cleanup", func(agent.InvocationContext) { time.Sleep(50 * time.Millisecond) }},
		// Cleanup that lasts until the run cancels it.
		{"cleanup until canceled", func(ctx agent.InvocationContext) { <-ctx.Done() }},
	}
	for _, nesting := range nestings {
		for _, cleanup := range cleanups {
			t.Run(nesting.name+"/"+cleanup.name, func(t *testing.T) {
				failed := make(chan struct{})
				failing := must(agent.New(agent.Config{
					Name: "failing",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							defer cleanup.run(ctx)
							yield(nil, boom)
							close(failed)
						}
					},
				}))
				escalator := must(agent.New(agent.Config{
					Name: "escalator",
					Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							<-failed
							yield(&session.Event{
								LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("escalating", genai.RoleModel)},
								Actions:     session.EventActions{Escalate: true},
							}, nil)
						}
					},
				}))
				outer := parallel("outer", nesting.wrap(failing), escalator)
				r, err := runner.New(runner.Config{AppName: "test_app", Agent: outer, SessionService: session.InMemoryService(), AutoCreateSession: true})
				if err != nil {
					t.Fatal(err)
				}

				// The deadline only bounds a regression where the run never
				// cancels the failing agent's cleanup.
				ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
				defer cancel()
				sawErr := false
				for ev, err := range r.Run(ctx, "user_id", "session_id", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
					if errors.Is(err, boom) {
						sawErr = true
					}
					if ev != nil && ev.Actions.Escalate {
						t.Errorf("escalate event yielded after a nested error, want the error instead")
					}
				}
				if ctx.Err() != nil {
					t.Fatalf("run ended only at the test deadline")
				}
				if !sawErr {
					t.Errorf("nested error was not returned")
				}
			})
		}
	}
}

// An inner ParallelAgent that ends on its own escalation cancels its other
// branches. The error a canceled branch yields then never leaves the inner
// agent, so it must not replace a later escalation of the outer one.
func TestParallelAgent_CanceledNestedBranchErrorKeepsEscalate(t *testing.T) {
	parallel := func(name string, subAgents ...agent.Agent) agent.Agent {
		return must(parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: name, SubAgents: subAgents}}))
	}
	sequential := func(name string, subAgents ...agent.Agent) agent.Agent {
		return must(sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: name, SubAgents: subAgents}}))
	}
	escalate := func(text string) *session.Event {
		return &session.Event{
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleModel)},
			Actions:     session.EventActions{Escalate: true},
		}
	}
	for _, nesting := range []struct {
		name string
		wrap func(inner agent.Agent) agent.Agent
	}{
		{"inner parallel", func(inner agent.Agent) agent.Agent { return inner }},
		{"sequential then parallel", func(inner agent.Agent) agent.Agent { return sequential("seq", inner) }},
	} {
		t.Run(nesting.name, func(t *testing.T) {
			started := make(chan struct{})
			waitingFailed := make(chan struct{})
			waiting := must(agent.New(agent.Config{
				Name: "waiting",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						defer close(waitingFailed)
						close(started)
						<-ctx.Done()
						// As a model call cut off by cancellation does.
						yield(nil, ctx.Err())
					}
				},
			}))
			innerEsc := must(agent.New(agent.Config{
				Name: "inner_esc",
				Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						<-started
						yield(escalate("inner done"), nil)
					}
				},
			}))
			outerEsc := must(agent.New(agent.Config{
				Name: "outer_esc",
				Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						<-waitingFailed
						yield(escalate("outer done"), nil)
					}
				},
			}))
			outer := parallel("outer", nesting.wrap(parallel("inner", innerEsc, waiting)), outerEsc)
			r, err := runner.New(runner.Config{AppName: "test_app", Agent: outer, SessionService: session.InMemoryService(), AutoCreateSession: true})
			if err != nil {
				t.Fatal(err)
			}

			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			sawOuterEsc := false
			for ev, err := range r.Run(ctx, "user_id", "session_id", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					t.Errorf("run yielded error %v, want the outer_esc escalation", err)
				}
				if ev != nil && ev.Author == "outer_esc" && ev.Actions.Escalate {
					sawOuterEsc = true
				}
			}
			if ctx.Err() != nil {
				t.Fatalf("run ended only at the test deadline")
			}
			if !sawOuterEsc {
				t.Errorf("outer_esc escalation was not yielded")
			}
		})
	}
}

// An agent between two ParallelAgents that catches the inner one's error and
// carries on keeps that error from the outer one, so it must not replace an
// escalation of the outer one.
func TestParallelAgent_CaughtNestedErrorKeepsEscalate(t *testing.T) {
	boom := errors.New("boom")
	failing := must(agent.New(agent.Config{
		Name: "failing",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) { yield(nil, boom) }
		},
	}))
	inner := must(parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "inner", SubAgents: []agent.Agent{failing}}}))
	recovered := make(chan struct{})
	catcher := must(agent.New(agent.Config{
		Name:      "catcher",
		SubAgents: []agent.Agent{inner},
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				defer close(recovered)
				for ev, err := range inner.Run(ctx) {
					if err != nil {
						yield(&session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("fallback", genai.RoleModel)}}, nil)
						return
					}
					if !yield(ev, nil) {
						return
					}
				}
			}
		},
	}))
	escalator := must(agent.New(agent.Config{
		Name: "escalator",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				<-recovered
				yield(&session.Event{
					LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("escalating", genai.RoleModel)},
					Actions:     session.EventActions{Escalate: true},
				}, nil)
			}
		},
	}))
	outer := must(parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "outer", SubAgents: []agent.Agent{catcher, escalator}}}))
	r, err := runner.New(runner.Config{AppName: "test_app", Agent: outer, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}

	sawEsc := false
	for ev, err := range r.Run(t.Context(), "user_id", "session_id", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Errorf("run yielded error %v, want the escalation", err)
		}
		if ev != nil && ev.Author == "escalator" && ev.Actions.Escalate {
			sawEsc = true
		}
	}
	if !sawEsc {
		t.Errorf("escalation was not yielded")
	}
}

// An agent that runs a ParallelAgent itself and catches or retries its error
// keeps that error from the outer ParallelAgent, even when the inner one is
// also declared as a step of a SequentialAgent or LoopAgent under the outer
// one. Each shape must end with the escalation and no error.
func TestParallelAgent_ErrorCaughtAtRunTimeKeepsEscalate(t *testing.T) {
	newAgent := func(name string, run func(agent.InvocationContext, func(*session.Event, error) bool)) agent.Agent {
		return must(agent.New(agent.Config{Name: name, Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) { run(ctx, yield) }
		}}))
	}
	text := func(s string) *session.Event {
		return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(s, genai.RoleModel)}}
	}
	escalation := func(s string) *session.Event {
		ev := text(s)
		ev.Actions.Escalate = true
		return ev
	}
	// newInner returns a ParallelAgent named name whose only branch fails on
	// its first run and succeeds after.
	newNamedInner := func(name string) agent.Agent {
		var runs atomic.Int32
		flaky := newAgent("flaky", func(_ agent.InvocationContext, yield func(*session.Event, error) bool) {
			if runs.Add(1) == 1 {
				yield(nil, errors.New("boom"))
				return
			}
			yield(text("ok"), nil)
		})
		return must(parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: name, SubAgents: []agent.Agent{flaky}}}))
	}
	newInner := func() agent.Agent { return newNamedInner("inner") }
	// retryingNode runs a in a workflow node that tries it twice.
	retryingNode := func(t *testing.T, a agent.Agent) *workflow.AgentNode {
		node, err := workflow.NewAgentNode(a, workflow.NodeConfig{RetryConfig: &workflow.RetryConfig{MaxAttempts: 2}})
		if err != nil {
			t.Fatal(err)
		}
		return node
	}
	// newNamedCatcher runs inner without declaring it and turns its error
	// into fallback, then closes caught.
	newNamedCatcher := func(name string, inner agent.Agent, fallback *session.Event, caught chan struct{}) agent.Agent {
		return newAgent(name, func(ctx agent.InvocationContext, yield func(*session.Event, error) bool) {
			defer close(caught)
			for ev, err := range inner.Run(ctx) {
				if err != nil {
					yield(fallback, nil)
					return
				}
				if !yield(ev, nil) {
					return
				}
			}
		})
	}
	newCatcher := func(inner agent.Agent, fallback *session.Event, caught chan struct{}) agent.Agent {
		return newNamedCatcher("catcher", inner, fallback, caught)
	}

	tests := []struct {
		name string
		// build returns the outer ParallelAgent's sub-agent that holds the
		// inner one, and a channel closed once the inner error was handled.
		build func(t *testing.T) (agent.Agent, <-chan struct{})
	}{
		{"catcher in a SequentialAgent that lists the inner agent", func(t *testing.T) (agent.Agent, <-chan struct{}) {
			inner := newInner()
			caught := make(chan struct{})
			return must(sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{
				Name: "seq", SubAgents: []agent.Agent{newCatcher(inner, text("fallback"), caught), inner},
			}})), caught
		}},
		{"catcher with an escalating fallback in a LoopAgent that lists the inner agent", func(t *testing.T) (agent.Agent, <-chan struct{}) {
			inner := newInner()
			caught := make(chan struct{})
			return must(loopagent.New(loopagent.Config{AgentConfig: agent.Config{
				Name: "loop", SubAgents: []agent.Agent{newCatcher(inner, escalation("fallback"), caught), inner},
			}})), caught
		}},
		{"workflow agent retrying the inner agent in a SequentialAgent that lists it", func(t *testing.T) (agent.Agent, <-chan struct{}) {
			inner := newInner()
			node := retryingNode(t, inner)
			wf := must(workflowagent.New(workflowagent.Config{Name: "wf", Edges: workflow.Chain(workflow.Start, node)}))
			done := make(chan struct{})
			// The workflow agent ends after its retry succeeded.
			signal := newAgent("signal", func(agent.InvocationContext, func(*session.Event, error) bool) { close(done) })
			return must(sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{
				Name: "seq", SubAgents: []agent.Agent{wf, signal, inner},
			}})), done
		}},
		{"workflow agent retrying an inner agent it declares", func(t *testing.T) (agent.Agent, <-chan struct{}) {
			inner := newInner()
			node := retryingNode(t, inner)
			wf := must(workflowagent.New(workflowagent.Config{Name: "wf", SubAgents: []agent.Agent{inner}, Edges: workflow.Chain(workflow.Start, node)}))
			done := make(chan struct{})
			signal := newAgent("signal", func(agent.InvocationContext, func(*session.Event, error) bool) { close(done) })
			return must(sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{
				Name: "seq", SubAgents: []agent.Agent{wf, signal},
			}})), done
		}},
	}
	tests = append(tests, []struct {
		name  string
		build func(t *testing.T) (agent.Agent, <-chan struct{})
	}{
		// A wrapper may keep the name of the agent it wraps, so the relay
		// has to tell agents apart by identity, not by name.
		{"wrapper named like the undeclared inner agent it catches", func(t *testing.T) (agent.Agent, <-chan struct{}) {
			caught := make(chan struct{})
			return newNamedCatcher("inner", newInner(), text("fallback"), caught), caught
		}},
		{"same-name wrapper in a SequentialAgent", func(t *testing.T) (agent.Agent, <-chan struct{}) {
			caught := make(chan struct{})
			return must(sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{
				Name: "seq", SubAgents: []agent.Agent{newNamedCatcher("inner", newInner(), text("fallback"), caught)},
			}})), caught
		}},
		// A SequentialAgent or LoopAgent the catcher runs itself gets no
		// relay to pass on.
		{"catcher running the inner agent through an undeclared SequentialAgent", func(t *testing.T) (agent.Agent, <-chan struct{}) {
			caught := make(chan struct{})
			hidden := must(sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "hidden", SubAgents: []agent.Agent{newInner()}}}))
			return newCatcher(hidden, text("fallback"), caught), caught
		}},
		{"catcher running the inner agent through an undeclared LoopAgent", func(t *testing.T) (agent.Agent, <-chan struct{}) {
			caught := make(chan struct{})
			hidden := must(loopagent.New(loopagent.Config{MaxIterations: 1, AgentConfig: agent.Config{Name: "hidden", SubAgents: []agent.Agent{newInner()}}}))
			return newCatcher(hidden, text("fallback"), caught), caught
		}},
	}...)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			holder, handled := tt.build(t)
			escalator := newAgent("escalator", func(_ agent.InvocationContext, yield func(*session.Event, error) bool) {
				<-handled
				yield(escalation("done"), nil)
			})
			outer := must(parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "outer", SubAgents: []agent.Agent{holder, escalator}}}))
			r, err := runner.New(runner.Config{AppName: "test_app", Agent: outer, SessionService: session.InMemoryService(), AutoCreateSession: true})
			if err != nil {
				t.Fatal(err)
			}

			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			sawEsc := false
			for ev, err := range r.Run(ctx, "user_id", "session_id", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					t.Errorf("run yielded error %v, want the escalation", err)
				}
				if ev != nil && ev.Author == "escalator" && ev.Actions.Escalate {
					sawEsc = true
				}
			}
			if ctx.Err() != nil {
				t.Fatalf("run ended only at the test deadline")
			}
			if !sawEsc {
				t.Errorf("escalation was not yielded")
			}
		})
	}
}

// A LoopAgent runs its later passes with a different context, and still passes
// the relay on there. An error from the inner ParallelAgent on the second pass
// must replace the escalation while the failing branch is cleaning up.
func TestParallelAgent_LoopLaterPassErrorSurvivesEscalate(t *testing.T) {
	boom := errors.New("boom")
	failed := make(chan struct{})
	var passes atomic.Int32
	failing := must(agent.New(agent.Config{
		Name: "failing",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				if passes.Add(1) == 1 {
					yield(&session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("pass one", genai.RoleModel)}}, nil)
					return
				}
				// Cleanup that lasts until the run cancels it.
				defer func() { <-ctx.Done() }()
				yield(nil, boom)
				close(failed)
			}
		},
	}))
	escalator := must(agent.New(agent.Config{
		Name: "escalator",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				<-failed
				yield(&session.Event{
					LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("escalating", genai.RoleModel)},
					Actions:     session.EventActions{Escalate: true},
				}, nil)
			}
		},
	}))
	inner := must(parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "inner", SubAgents: []agent.Agent{failing}}}))
	loop := must(loopagent.New(loopagent.Config{MaxIterations: 3, AgentConfig: agent.Config{Name: "loop", SubAgents: []agent.Agent{inner}}}))
	outer := must(parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "outer", SubAgents: []agent.Agent{loop, escalator}}}))
	r, err := runner.New(runner.Config{AppName: "test_app", Agent: outer, SessionService: session.InMemoryService(), AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	sawErr := false
	for ev, err := range r.Run(ctx, "user_id", "session_id", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if errors.Is(err, boom) {
			sawErr = true
		}
		if ev != nil && ev.Actions.Escalate {
			t.Errorf("escalate event yielded after the second pass failed, want the error instead")
		}
	}
	if ctx.Err() != nil {
		t.Fatalf("run ended only at the test deadline")
	}
	if !sawErr {
		t.Errorf("error from the second pass was not returned")
	}
}

// appendHookService calls hook before appending each event.
type appendHookService struct {
	session.Service
	hook func(*session.Event)
}

func (s *appendHookService) AppendEvent(ctx context.Context, sess session.Session, ev *session.Event) error {
	s.hook(ev)
	return s.Service.AppendEvent(ctx, sess, ev)
}

// A ParallelAgent seals its error record before it yields its own escalation.
// A branch that fails while the caller is still handling that escalation does
// not report up, because this agent returns the escalation, not the error.
func TestParallelAgent_ErrorDuringOwnEscalationIsNotReported(t *testing.T) {
	escAppending := make(chan struct{})
	failingDone := make(chan struct{})
	newAgent := func(name string, run func(func(*session.Event, error) bool)) agent.Agent {
		return must(agent.New(agent.Config{Name: name, Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) { run(yield) }
		}}))
	}
	escalate := func(text string) *session.Event {
		return &session.Event{
			LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleModel)},
			Actions:     session.EventActions{Escalate: true},
		}
	}
	innerEsc := newAgent("inner_esc", func(yield func(*session.Event, error) bool) { yield(escalate("inner done"), nil) })
	failing := newAgent("failing", func(yield func(*session.Event, error) bool) {
		defer close(failingDone)
		<-escAppending
		yield(nil, errors.New("boom"))
	})
	outerEsc := newAgent("outer_esc", func(yield func(*session.Event, error) bool) {
		<-failingDone
		yield(escalate("outer done"), nil)
	})
	inner := must(parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "inner", SubAgents: []agent.Agent{innerEsc, failing}}}))
	outer := must(parallelagent.New(parallelagent.Config{AgentConfig: agent.Config{Name: "outer", SubAgents: []agent.Agent{inner, outerEsc}}}))
	// Hold inner's escalation in the caller until the sibling has failed.
	svc := &appendHookService{Service: session.InMemoryService(), hook: func(ev *session.Event) {
		if ev.Author == "inner_esc" && ev.Actions.Escalate {
			close(escAppending)
			<-failingDone
		}
	}}
	r, err := runner.New(runner.Config{AppName: "test_app", Agent: outer, SessionService: svc, AutoCreateSession: true})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	sawOuterEsc := false
	for ev, err := range r.Run(ctx, "user_id", "session_id", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Errorf("run yielded error %v, want the outer_esc escalation", err)
		}
		if ev != nil && ev.Author == "outer_esc" && ev.Actions.Escalate {
			sawOuterEsc = true
		}
	}
	if ctx.Err() != nil {
		t.Fatalf("run ended only at the test deadline")
	}
	if !sawOuterEsc {
		t.Errorf("outer_esc escalation was not yielded")
	}
}

// TestParallelAgent_PropagatesIsolationScope verifies that every sub-agent runs
// with the parent's isolation scope. A workflow node that wraps a parallel
// agent stamps the node's scope on every event the branches emit, and the LLM
// history filter keeps only events whose scope matches exactly. A branch that
// ran unscoped would therefore drop its own earlier turns from its prompt.
func TestParallelAgent_PropagatesIsolationScope(t *testing.T) {
	const scope = "scope-1"

	var (
		mu        sync.Mutex
		gotScopes = map[string]string{}
		gotBranch = map[string]string{}
	)
	record := func(name string) agent.Agent {
		return must(agent.New(agent.Config{
			Name: name,
			Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					mu.Lock()
					gotScopes[name] = ctx.IsolationScope()
					gotBranch[name] = ctx.Branch()
					mu.Unlock()
				}
			},
		}))
	}

	parallel := must(parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:      "parallel",
			SubAgents: []agent.Agent{record("a"), record("b")},
		},
	}))

	created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{
		AppName: "app", UserID: "user", SessionID: "session",
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Session:        created.Session,
		Agent:          parallel,
		IsolationScope: scope,
		InvocationID:   "inv-1",
	})

	for _, err := range parallel.Run(ctx) {
		if err != nil {
			t.Fatal(err)
		}
	}

	wantScopes := map[string]string{"a": scope, "b": scope}
	if diff := cmp.Diff(wantScopes, gotScopes); diff != "" {
		t.Errorf("sub-agent isolation scopes mismatch (-want +got):\n%s", diff)
	}
	wantBranch := map[string]string{"a": "parallel.a", "b": "parallel.b"}
	if diff := cmp.Diff(wantBranch, gotBranch); diff != "" {
		t.Errorf("sub-agent branches mismatch (-want +got):\n%s", diff)
	}
}

// TestParallelAgent_ScopedBranchSeesItsOwnHistory checks the user-visible
// effect: an LLM branch of a scoped parallel agent must see its own earlier,
// scoped turn in the prompt it sends to the model.
func TestParallelAgent_ScopedBranchSeesItsOwnHistory(t *testing.T) {
	const scope = "scope-1"
	ctx := t.Context()

	mock := &testutil.MockModel{Responses: []*genai.Content{
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	branchAgent := must(llmagent.New(llmagent.Config{Name: "a", Model: mock}))
	parallel := must(parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{Name: "parallel", SubAgents: []agent.Agent{branchAgent}},
	}))

	service := session.InMemoryService()
	created, err := service.Create(ctx, &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "session"})
	if err != nil {
		t.Fatal(err)
	}
	// An earlier turn of this branch, stamped with the node's scope the way
	// workflow's agent node stamps every event a wrapped agent emits.
	if err := service.AppendEvent(ctx, created.Session, &session.Event{
		ID:             "earlier",
		InvocationID:   "inv-1",
		Author:         "a",
		Branch:         "parallel.a",
		IsolationScope: scope,
		LLMResponse: model.LLMResponse{
			Content: genai.NewContentFromText("earlier scoped answer", genai.RoleModel),
		},
	}); err != nil {
		t.Fatal(err)
	}
	got, err := service.Get(ctx, &session.GetRequest{AppName: "app", UserID: "user", SessionID: "session"})
	if err != nil {
		t.Fatal(err)
	}

	invCtx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{
		Session:        got.Session,
		Agent:          parallel,
		IsolationScope: scope,
		InvocationID:   "inv-1",
		RunConfig:      &agent.RunConfig{},
	})
	for _, err := range parallel.Run(invCtx) {
		if err != nil {
			t.Fatal(err)
		}
	}

	if len(mock.Requests) != 1 {
		t.Fatalf("model got %d requests, want 1", len(mock.Requests))
	}
	var texts []string
	for _, c := range mock.Requests[0].Contents {
		for _, p := range c.Parts {
			texts = append(texts, p.Text)
		}
	}
	if !slices.Contains(texts, "earlier scoped answer") {
		t.Errorf("branch prompt is missing its own scoped history; prompt texts: %q", texts)
	}
}
