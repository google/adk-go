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
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
)

// stubTool is a minimal tool.Tool for tests that only need a named entry
// in state.Tools.
type stubTool struct{ name string }

func (s stubTool) Name() string        { return s.name }
func (s stubTool) Description() string { return s.name }
func (s stubTool) IsLongRunning() bool { return false }

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
					Actions: session.EventActions{
						StateDelta:    map[string]any{},
						ArtifactDelta: map[string]int64{},
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
					Actions: session.EventActions{
						StateDelta:    map[string]any{},
						ArtifactDelta: map[string]int64{},
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
					Actions: session.EventActions{
						StateDelta:    map[string]any{},
						ArtifactDelta: map[string]int64{},
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
					Actions: session.EventActions{
						StateDelta:    map[string]any{},
						ArtifactDelta: map[string]int64{},
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
					Actions: session.EventActions{
						StateDelta:    map[string]any{},
						ArtifactDelta: map[string]int64{},
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
					Actions: session.EventActions{
						StateDelta:    map[string]any{},
						ArtifactDelta: map[string]int64{},
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
					if diff := cmp.Diff(tt.wantEvents[i], gotEvent, cmpopts.IgnoreFields(session.Event{}, "ID", "Timestamp", "InvocationID")); diff != "" {
						t.Errorf("event[i] mismatch (-want +got):\n%s", diff)
					}
				}
			}
		})
	}
}

func newCustomAgent(t *testing.T, id int) agent.Agent {
	return newCustomAgentWithTools(t, id, nil)
}

func newCustomAgentWithTools(t *testing.T, id int, tools []tool.Tool) agent.Agent {
	t.Helper()

	a, err := llmagent.New(llmagent.Config{
		Name:  fmt.Sprintf("custom_agent_%v", id),
		Model: &FakeLLM{id: id, callCounter: 0},
		Tools: tools,
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

type mockLiveAgent struct {
	agent.Agent
	runLiveFn func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
}

func (m *mockLiveAgent) RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
	return m.runLiveFn(ctx)
}

type dummyLiveSession struct {
	sendChan chan agent.LiveRequest
	closed   bool
}

func (d *dummyLiveSession) Send(req agent.LiveRequest) error {
	d.sendChan <- req
	return nil
}

func (d *dummyLiveSession) Close() error {
	d.closed = true
	return nil
}

func mustAgent(a agent.Agent, err error) agent.Agent {
	if err != nil {
		panic(err)
	}
	return a
}

type mockInvocationContext struct {
	agent.InvocationContext
	agent        agent.Agent
	invocationID string
	ctx          context.Context
}

func (m *mockInvocationContext) Agent() agent.Agent {
	return m.agent
}

func (m *mockInvocationContext) InvocationID() string {
	return m.invocationID
}

func (m *mockInvocationContext) Context() context.Context {
	return m.ctx
}

func (m *mockInvocationContext) Deadline() (time.Time, bool) { return m.ctx.Deadline() }
func (m *mockInvocationContext) Done() <-chan struct{}       { return m.ctx.Done() }
func (m *mockInvocationContext) Err() error                  { return m.ctx.Err() }
func (m *mockInvocationContext) Value(key any) any           { return m.ctx.Value(key) }

func TestSequentialAgent_RunLive_Injection(t *testing.T) {
	existingTaskCompleted := stubTool{name: "task_completed"}
	subAgent1 := newCustomAgentWithTools(t, 1, []tool.Tool{existingTaskCompleted})
	subAgent2 := newCustomAgent(t, 2)

	sequentialAgent, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "seq_agent",
			SubAgents: []agent.Agent{subAgent1, subAgent2},
		},
	})
	if err != nil {
		t.Fatalf("failed to create sequential agent: %v", err)
	}

	// Before RunLive, sub-agents do not have the task_completed tool
	if llmAgent1, ok := subAgent1.(llminternal.Agent); ok {
		state := llminternal.Reveal(llmAgent1)
		if len(state.Tools) != 1 || state.Tools[0].Name() != existingTaskCompleted.Name() {
			t.Fatalf("sub-agent 1 lost its pre-existing task_completed tool before RunLive: %#v", state.Tools)
		}
	}

	// Call RunLive (it will prepare/inject but will fail/return error when executing due to nil/mock context,
	// which is perfectly fine since the injection happens beforehand).
	// Let's pass a mock context that returns seqAgent as the Agent.
	invCtx := &mockInvocationContext{
		agent:        sequentialAgent,
		invocationID: "test_id",
		ctx:          t.Context(),
	}

	liveAgent, ok := sequentialAgent.(interface {
		RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
	})
	if !ok {
		t.Fatalf("sequential agent does not implement RunLive")
	}

	_, _, _ = liveAgent.RunLive(invCtx)

	// After RunLive initiation, the sub-agents MUST have the task_completed tool injected!
	if llmAgent1, ok := subAgent1.(llminternal.Agent); ok {
		state := llminternal.Reveal(llmAgent1)
		if len(state.Tools) != 1 || state.Tools[0].Name() != existingTaskCompleted.Name() {
			t.Errorf("sub-agent 1 pre-existing task_completed tool changed after RunLive: %#v", state.Tools)
		}
		if strings.Contains(state.Instruction, taskCompletedInstructionMarker) {
			t.Errorf("sub-agent 1 instruction unexpectedly received task_completed suffix")
		}
	}
}

// taskCompletedInstructionMarker is a stable fragment of the instruction
// suffix RunLive appends alongside the task_completed tool. The test
// counts it instead of hardcoding the whole sentence.
const taskCompletedInstructionMarker = "call the task_completed function to exit so the next agents can take over"

// TestSequentialAgent_RunLive_ConcurrentInjectionIsRaceFree is the
// regression test for the data race in RunLive's task_completed
// injection: llminternal.Reveal(llmAgent) returns the sub-agent's shared,
// persistent state, and the unguarded read-then-append/read-then-concat
// used to race under any two concurrent RunLive calls dispatched onto the
// same sub-agent (e.g. two simultaneous live/streaming sessions sharing
// an agent tree built once at startup) -- confirmed with `go test -race`
// before the fix. It also raced itself into double-injecting the tool
// whenever both goroutines observed hasTaskCompleted == false before
// either had written it back.
//
// Beyond "injected exactly once", the test pins the ordering guarantee
// the LiveModeInjection latch is there for: each goroutine reads state.Tools
// right after its own RunLive returns and must observe the completed
// injection (exactly one task_completed tool, never zero). A plain
// "already done" marker would pass the final count assertion but let a
// losing caller return before the winner's write is visible -- that read
// is what fails under -race without the barrier.
//
// The pipeline has three sub-agents so RunLive's injection loop runs more
// than one iteration, and each sub-agent carries its own State with its own
// LiveModeInjection latch: the guard has to hold independently for every one.
func TestSequentialAgent_RunLive_ConcurrentInjectionIsRaceFree(t *testing.T) {
	subAgents := []agent.Agent{
		newCustomAgent(t, 1), newCustomAgent(t, 2), newCustomAgent(t, 3),
	}

	sequentialAgent, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "seq_agent",
			SubAgents: subAgents,
		},
	})
	if err != nil {
		t.Fatalf("failed to create sequential agent: %v", err)
	}

	liveAgent, ok := sequentialAgent.(interface {
		RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
	})
	if !ok {
		t.Fatalf("sequential agent does not implement RunLive")
	}

	states := make([]*llminternal.State, len(subAgents))
	for i, sub := range subAgents {
		llmAgent, ok := sub.(llminternal.Agent)
		if !ok {
			t.Fatalf("sub-agent %d does not implement llminternal.Agent", i)
		}
		states[i] = llminternal.Reveal(llmAgent)
	}

	countTaskCompleted := func(s *llminternal.State) int {
		n := 0
		for _, tl := range s.Tools {
			if tl != nil && tl.Name() == "task_completed" {
				n++
			}
		}
		return n
	}

	const concurrent = 20
	// seenByCaller[i][j]: what caller i saw in sub-agent j's Tools.
	seenByCaller := make([][]int, concurrent)
	var wg sync.WaitGroup
	wg.Add(concurrent)
	for i := range concurrent {
		go func() {
			defer wg.Done()
			invCtx := &mockInvocationContext{agent: sequentialAgent, invocationID: "test_id", ctx: t.Context()}
			_, _, _ = liveAgent.RunLive(invCtx)
			// RunLive has returned, so every sub-agent's Once has completed:
			// these reads are ordered after the winners' writes.
			seen := make([]int, len(states))
			for j, s := range states {
				seen[j] = countTaskCompleted(s)
			}
			seenByCaller[i] = seen
		}()
	}
	wg.Wait()

	for i, seen := range seenByCaller {
		for j, n := range seen {
			if n != 1 {
				t.Errorf("caller %d saw task_completed %d times in sub-agent %d's Tools after RunLive returned, want exactly 1", i, n, j)
			}
		}
	}

	for j, s := range states {
		if n := countTaskCompleted(s); n != 1 {
			t.Errorf("sub-agent %d: task_completed injected %d times across %d concurrent RunLive calls, want exactly 1", j, n, concurrent)
		}
		// The tool is useless without the instruction telling the model to
		// call it; two goroutines can both append the suffix and still leave
		// one tool in the slice, so the suffix count is a distinct check.
		if n := strings.Count(s.Instruction, taskCompletedInstructionMarker); n != 1 {
			t.Errorf("sub-agent %d: task_completed instruction suffix appears %d times, want exactly 1", j, n)
		}
	}
}

// TestSequentialAgent_RunLive_PreexistingToolAndNilSkip pins the two
// branches of RunLive's injection loop that no other test reaches: the
// idempotence check that leaves a sub-agent's own task_completed tool (and
// the instruction) alone, and the untyped-nil skip in the same loop.
// Deleting the "does it already have task_completed" scan and injecting
// unconditionally otherwise keeps the whole package green, because the
// LiveInjection guard fires exactly once either way.
func TestSequentialAgent_RunLive_PreexistingToolAndNilSkip(t *testing.T) {
	subAgent, err := llmagent.New(llmagent.Config{
		Name:  "sub_with_task_completed",
		Model: &FakeLLM{id: 1},
		Tools: []tool.Tool{nil, stubTool{name: "task_completed"}}, // nil exercises the skip
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}

	seq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "seq_agent", SubAgents: []agent.Agent{subAgent}},
	})
	if err != nil {
		t.Fatalf("sequentialagent.New: %v", err)
	}
	liveAgent, ok := seq.(interface {
		RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
	})
	if !ok {
		t.Fatalf("sequential agent does not implement RunLive")
	}

	state := llminternal.Reveal(subAgent.(llminternal.Agent))
	toolsBefore := len(state.Tools)
	instrBefore := state.Instruction

	_, _, _ = liveAgent.RunLive(&mockInvocationContext{agent: seq, invocationID: "test_id", ctx: t.Context()})

	if len(state.Tools) != toolsBefore {
		t.Errorf("Tools grew from %d to %d; a pre-existing task_completed must suppress the append", toolsBefore, len(state.Tools))
	}
	got := 0
	for _, tl := range state.Tools {
		if tl != nil && tl.Name() == "task_completed" {
			got++
		}
	}
	if got != 1 {
		t.Errorf("task_completed tool count = %d, want 1 (the sub-agent's own, untouched)", got)
	}
	if state.Instruction != instrBefore {
		t.Errorf("Instruction changed; the hand-off suffix must not be appended when task_completed already exists")
	}
	if n := strings.Count(state.Instruction, taskCompletedInstructionMarker); n != 0 {
		t.Errorf("instruction suffix appears %d times, want 0", n)
	}
}

func TestSequentialAgent_RunLive_SequentialOrchestration(t *testing.T) {
	ctx := t.Context()

	sendChan1 := make(chan agent.LiveRequest, 10)
	sendChan2 := make(chan agent.LiveRequest, 10)

	subSess1 := &dummyLiveSession{sendChan: sendChan1}
	subSess2 := &dummyLiveSession{sendChan: sendChan2}

	agent1 := mustAgent(agent.New(agent.Config{Name: "sub_agent_1"}))
	liveAgent1 := &mockLiveAgent{
		Agent: agent1,
		runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
			iterFn := func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "sub_agent_1"
				yield(ev, nil)
			}
			return subSess1, iterFn, nil
		},
	}

	agent2 := mustAgent(agent.New(agent.Config{Name: "sub_agent_2"}))
	liveAgent2 := &mockLiveAgent{
		Agent: agent2,
		runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
			iterFn := func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "sub_agent_2"
				yield(ev, nil)
			}
			return subSess2, iterFn, nil
		},
	}

	seqAgent, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "seq_agent",
			SubAgents: []agent.Agent{liveAgent1, liveAgent2},
		},
	})
	if err != nil {
		t.Fatalf("failed to create sequential agent: %v", err)
	}

	invCtx := &mockInvocationContext{
		agent:        seqAgent,
		invocationID: "test_inv_id",
		ctx:          ctx,
	}

	liveAgent, ok := seqAgent.(interface {
		RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
	})
	if !ok {
		t.Fatalf("sequential agent does not implement RunLive")
	}

	sess, seqIter, err := liveAgent.RunLive(invCtx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}

	next, stop := iter.Pull2(seqIter)
	defer stop()

	// Consume first sub-agent event
	ev1, err1, ok := next()
	if !ok || err1 != nil {
		t.Fatalf("expected first event, got ok=%v, err=%v", ok, err1)
	}
	if ev1.Author != "sub_agent_1" {
		t.Errorf("expected event from sub_agent_1, got %s", ev1.Author)
	}

	// Now seqSess should route to subSess1
	req1 := agent.LiveRequest{Content: genai.NewContentFromText("to agent 1", "")}
	if err := sess.Send(req1); err != nil {
		t.Fatalf("failed to Send to sess: %v", err)
	}
	gotReq1 := <-sendChan1
	if gotReq1.Content.Parts[0].Text != "to agent 1" {
		t.Errorf("expected request to subSess1, got: %v", gotReq1)
	}

	// The subSess1 completes, transitioning to agent2
	ev2, err2, ok := next()
	if !ok || err2 != nil {
		t.Fatalf("expected second event, got ok=%v, err=%v", ok, err2)
	}
	if ev2.Author != "sub_agent_2" {
		t.Errorf("expected event from sub_agent_2, got %s", ev2.Author)
	}

	// Now seqSess should route to subSess2
	req2 := agent.LiveRequest{Content: genai.NewContentFromText("to agent 2", "")}
	if err := sess.Send(req2); err != nil {
		t.Fatalf("failed to Send to sess: %v", err)
	}
	gotReq2 := <-sendChan2
	if gotReq2.Content.Parts[0].Text != "to agent 2" {
		t.Errorf("expected request to subSess2, got: %v", gotReq2)
	}

	// Verify that subSess1 is closed
	if !subSess1.closed {
		t.Errorf("expected sub_agent_1 session to be closed after transition")
	}

	// The subSess2 completes
	_, _, ok = next()
	if ok {
		t.Errorf("expected iterator to be exhausted")
	}

	// Verify subSess2 is closed
	if !subSess2.closed {
		t.Errorf("expected sub_agent_2 session to be closed at the end")
	}
}
