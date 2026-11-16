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
	"errors"
	"fmt"
	"iter"
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
	"google.golang.org/adk/v2/agent/workflowagents/loopagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
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

func (m *mockInvocationContext) WithICDelta(d *agent.InvocationContextDelta) agent.InvocationContext {
	cp := *m
	if d != nil {
		if d.Agent != nil {
			cp.agent = *d.Agent
		}
		if d.Context != nil {
			cp.ctx = *d.Context
		}
	}
	return &cp
}

func TestSequentialAgent_RunLive_Injection(t *testing.T) {
	subAgent1 := newCustomAgent(t, 1)
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
		for _, tool := range state.Tools {
			if tool.Name() == "task_completed" {
				t.Errorf("sub-agent 1 already has task_completed tool before RunLive")
			}
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
		hasTaskCompleted := false
		for _, tool := range state.Tools {
			if tool.Name() == "task_completed" {
				hasTaskCompleted = true
				break
			}
		}
		if !hasTaskCompleted {
			t.Errorf("sub-agent 1 does not have task_completed tool injected after RunLive")
		}
	} else {
		t.Fatalf("subAgent1 is not an llminternal.Agent")
	}
}

func TestSequentialAgent_RunLive_NestedSequentialInjection(t *testing.T) {
	subAgentA := newCustomAgent(t, 1)
	subAgentB := newCustomAgent(t, 2)

	innerSeq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "inner_seq",
			SubAgents: []agent.Agent{subAgentB},
		},
	})
	if err != nil {
		t.Fatalf("failed to create inner sequential agent: %v", err)
	}

	outerSeq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "outer_seq",
			SubAgents: []agent.Agent{subAgentA, innerSeq},
		},
	})
	if err != nil {
		t.Fatalf("failed to create outer sequential agent: %v", err)
	}

	invCtx := &mockInvocationContext{
		agent:        outerSeq,
		invocationID: "test_id",
		ctx:          t.Context(),
	}

	liveAgent, ok := outerSeq.(interface {
		RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
	})
	if !ok {
		t.Fatalf("sequential agent does not implement RunLive")
	}

	_, _, _ = liveAgent.RunLive(invCtx)

	// subAgentB inside innerSeq must have task_completed injected!
	if llmAgentB, ok := subAgentB.(llminternal.Agent); ok {
		state := llminternal.Reveal(llmAgentB)
		hasTaskCompleted := false
		for _, tool := range state.Tools {
			if tool.Name() == "task_completed" {
				hasTaskCompleted = true
				break
			}
		}
		if !hasTaskCompleted {
			t.Errorf("nested sub-agent B does not have task_completed tool injected after RunLive")
		}
	} else {
		t.Fatalf("subAgentB is not an llminternal.Agent")
	}
}

func TestSequentialAgent_RunLive_NonLiveCompositeNotMutated(t *testing.T) {
	subAgentA := newCustomAgent(t, 1)
	subAgentB := newCustomAgent(t, 2)

	// A composite agent that does not implement liveRunner (standard agent without RunLive)
	nonLiveComposite, err := agent.New(agent.Config{
		Name:      "non_live_composite",
		SubAgents: []agent.Agent{subAgentB},
	})
	if err != nil {
		t.Fatalf("failed to create non_live_composite agent: %v", err)
	}

	outerSeq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "outer_seq",
			SubAgents: []agent.Agent{subAgentA, nonLiveComposite},
		},
	})
	if err != nil {
		t.Fatalf("failed to create outer sequential agent: %v", err)
	}

	invCtx := &mockInvocationContext{
		agent:        outerSeq,
		invocationID: "test_id",
		ctx:          t.Context(),
	}

	liveAgent, ok := outerSeq.(interface {
		RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
	})
	if !ok {
		t.Fatalf("sequential agent does not implement RunLive")
	}

	_, _, _ = liveAgent.RunLive(invCtx)

	// subAgentB inside nonLiveComposite must NOT have task_completed injected!
	if llmAgentB, ok := subAgentB.(llminternal.Agent); ok {
		state := llminternal.Reveal(llmAgentB)
		for _, tool := range state.Tools {
			if tool.Name() == "task_completed" {
				t.Errorf("nested sub-agent B inside non-live composite must not have task_completed tool injected")
			}
		}
	} else {
		t.Fatalf("subAgentB is not an llminternal.Agent")
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
	var gotReq1 agent.LiveRequest
	select {
	case gotReq1 = <-sendChan1:
	case <-ctx.Done():
		t.Fatalf("timeout waiting for request to subSess1: %v", ctx.Err())
	}
	if gotReq1.Content.Parts[0].Text != "to agent 1" {
		t.Errorf("expected request to subSess1, got: %v", gotReq1)
	}

	// The subSess1 completes, transitioning to agent2
	ev2, err2, ok := next()
	if !ok || err2 != nil {
		t.Fatalf("expected second event, got ok=%v, err=%v", ok, err2)
	}
	if ev2.Author != "sub_agent_2" {
		t.Fatalf("expected event from sub_agent_2, got %s", ev2.Author)
	}

	// Now seqSess should route to subSess2
	req2 := agent.LiveRequest{Content: genai.NewContentFromText("to agent 2", "")}
	if err := sess.Send(req2); err != nil {
		t.Fatalf("failed to Send to sess: %v", err)
	}
	var gotReq2 agent.LiveRequest
	select {
	case gotReq2 = <-sendChan2:
	case <-ctx.Done():
		t.Fatalf("timeout waiting for request to subSess2: %v", ctx.Err())
	}
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

func TestSequentialAgent_RunLive_NestedSequentialOrchestration(t *testing.T) {
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

	innerSeq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "inner_seq",
			SubAgents: []agent.Agent{liveAgent2},
		},
	})
	if err != nil {
		t.Fatalf("failed to create inner sequential agent: %v", err)
	}

	outerSeq, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{
			Name:      "outer_seq",
			SubAgents: []agent.Agent{liveAgent1, innerSeq},
		},
	})
	if err != nil {
		t.Fatalf("failed to create outer sequential agent: %v", err)
	}

	invCtx := &mockInvocationContext{
		agent:        outerSeq,
		invocationID: "test_inv_id",
		ctx:          ctx,
	}

	liveAgent, ok := outerSeq.(interface {
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

	// Route to subSess1
	req1 := agent.LiveRequest{Content: genai.NewContentFromText("to agent 1", "")}
	if err := sess.Send(req1); err != nil {
		t.Fatalf("failed to Send to sess: %v", err)
	}
	var gotReq1 agent.LiveRequest
	select {
	case gotReq1 = <-sendChan1:
	case <-ctx.Done():
		t.Fatalf("timeout waiting for request to subSess1: %v", ctx.Err())
	}
	if gotReq1.Content.Parts[0].Text != "to agent 1" {
		t.Errorf("expected request to subSess1, got: %v", gotReq1)
	}

	// The subSess1 completes, transitioning to nested innerSeq -> agent2
	ev2, err2, ok := next()
	if !ok || err2 != nil {
		t.Fatalf("expected second event, got ok=%v, err=%v", ok, err2)
	}
	if ev2.Author != "sub_agent_2" {
		t.Fatalf("expected event from sub_agent_2, got %s", ev2.Author)
	}

	// Route to subSess2
	req2 := agent.LiveRequest{Content: genai.NewContentFromText("to agent 2", "")}
	if err := sess.Send(req2); err != nil {
		t.Fatalf("failed to Send to sess: %v", err)
	}
	var gotReq2 agent.LiveRequest
	select {
	case gotReq2 = <-sendChan2:
	case <-ctx.Done():
		t.Fatalf("timeout waiting for request to subSess2: %v", ctx.Err())
	}
	if gotReq2.Content.Parts[0].Text != "to agent 2" {
		t.Errorf("expected request to subSess2, got: %v", gotReq2)
	}

	// Verify that subSess1 is closed
	if !subSess1.closed {
		t.Errorf("expected sub_agent_1 session to be closed after transition")
	}

	// Sequence completes
	_, _, ok = next()
	if ok {
		t.Errorf("expected iterator to be exhausted")
	}

	// Verify subSess2 is closed
	if !subSess2.closed {
		t.Errorf("expected sub_agent_2 session to be closed at the end")
	}
}

// failingLLM fails every call, as a model does on a quota or auth error.
type failingLLM struct {
	calls int
}

func (f *failingLLM) Name() string {
	return "failing-llm"
}

func (f *failingLLM) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		f.calls++
		yield(nil, fmt.Errorf("model unavailable"))
	}
}

func TestSequentialAgentStopsOnSubAgentError(t *testing.T) {
	firstLLM := &failingLLM{}
	first, err := llmagent.New(llmagent.Config{Name: "first", Model: firstLLM})
	if err != nil {
		t.Fatal(err)
	}
	secondLLM := &FakeLLM{id: 1}
	second, err := llmagent.New(llmagent.Config{Name: "second", Model: secondLLM})
	if err != nil {
		t.Fatal(err)
	}
	r, err := runner.New(runner.Config{
		AppName:           "test_app",
		Agent:             newSequentialAgent(t, []agent.Agent{first, second}, "pipeline"),
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Keep reading after an error, as the run_sse handler and the console
	// launcher do.
	var errs []error
	for _, err := range r.Run(t.Context(), "user_id", "session_id", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "model unavailable") {
		t.Errorf("got errors %v, want exactly one from the first agent's model", errs)
	}
	if firstLLM.calls != 1 {
		t.Errorf("first agent's model called %d times, want 1", firstLLM.calls)
	}
	if secondLLM.callCounter != 0 {
		t.Errorf("second agent's model called %d times after the first agent failed, want 0", secondLLM.callCounter)
	}
}

// TestSequentialAgent_RunLive_StopsOnSubAgentError is the live counterpart of
// TestSequentialAgentStopsOnSubAgentError: a sub-agent whose live stream yields
// an error ends the sequence, and the next sub-agent is never started.
func TestSequentialAgent_RunLive_StopsOnSubAgentError(t *testing.T) {
	firstSess := &dummyLiveSession{sendChan: make(chan agent.LiveRequest, 1)}
	first := &mockLiveAgent{
		Agent: mustAgent(agent.New(agent.Config{Name: "first"})),
		runLiveFn: func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
			return firstSess, func(yield func(*session.Event, error) bool) {
				yield(nil, fmt.Errorf("live connection failed"))
			}, nil
		},
	}
	secondStarted := false
	second := &mockLiveAgent{
		Agent: mustAgent(agent.New(agent.Config{Name: "second"})),
		runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
			secondStarted = true
			return &dummyLiveSession{sendChan: make(chan agent.LiveRequest, 1)}, func(yield func(*session.Event, error) bool) {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Author = "second"
				yield(ev, nil)
			}, nil
		},
	}

	seqAgent, err := sequentialagent.New(sequentialagent.Config{
		AgentConfig: agent.Config{Name: "pipeline", SubAgents: []agent.Agent{first, second}},
	})
	if err != nil {
		t.Fatal(err)
	}
	liveAgent, ok := seqAgent.(interface {
		RunLive(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error)
	})
	if !ok {
		t.Fatal("sequential agent does not implement RunLive")
	}
	_, seqIter, err := liveAgent.RunLive(&mockInvocationContext{agent: seqAgent, invocationID: "inv", ctx: t.Context()})
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}

	// Keep reading after an error, as the runner does.
	var errs []error
	var authors []string
	for ev, err := range seqIter {
		if err != nil {
			errs = append(errs, err)
			continue
		}
		authors = append(authors, ev.Author)
	}

	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "live connection failed") {
		t.Errorf("got errors %v, want exactly one from the first sub-agent", errs)
	}
	if secondStarted {
		t.Errorf("second sub-agent was started after the first one failed; events from: %v", authors)
	}
	if !firstSess.closed {
		t.Error("first sub-agent's live session was not closed after its error")
	}
}

// TestSequentialAgent_RunWithoutSessionOrFunctionResponse checks that an
// ordinary turn does not read the session to look for a resume target.
func TestSequentialAgent_RunWithoutSessionOrFunctionResponse(t *testing.T) {
	ran := false
	sub, err := agent.New(agent.Config{Name: "sub", Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(func(*session.Event, error) bool) { ran = true }
	}})
	if err != nil {
		t.Fatal(err)
	}
	root, err := sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "root", SubAgents: []agent.Agent{sub}}})
	if err != nil {
		t.Fatal(err)
	}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: root})
	for _, err := range root.Run(ctx) {
		if err != nil {
			t.Fatal(err)
		}
	}
	if !ran {
		t.Error("sub-agent did not run")
	}
}

// Lookups return the agents the caller built, not the base agent that
// sequentialagent.New wraps, as adk-python's find_agent returns the tree's own
// objects.
func TestSequentialAgent_FindAgentPreservesWrapperIdentity(t *testing.T) {
	// root ("shared") -> child ("shared") -> grandchild ("grandchild_seq")
	grandchild := newSequentialAgent(t, nil, "grandchild_seq")
	child := newSequentialAgent(t, []agent.Agent{grandchild}, "shared")
	root := newSequentialAgent(t, []agent.Agent{child}, "shared")

	tests := []struct {
		name string
		got  agent.Agent
		want agent.Agent
	}{
		{name: "FindAgent prefers self over a same-named descendant", got: root.FindAgent("shared"), want: root},
		{name: "FindSubAgent returns the same-named descendant", got: root.FindSubAgent("shared"), want: child},
		{name: "FindSubAgent excludes self", got: child.FindSubAgent("shared"), want: nil},
		{name: "FindAgent returns a nested sequential agent", got: root.FindAgent("grandchild_seq"), want: grandchild},
		{name: "FindSubAgent returns nil for an unknown name", got: root.FindSubAgent("unknown"), want: nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.got != tc.want {
				t.Errorf("got %p (%T), want %p (%T)", tc.got, tc.got, tc.want, tc.want)
			}
		})
	}
}

func TestSequentialAgent_FindAgentReturnsSequentialAgent(t *testing.T) {
	child := newCustomAgent(t, 0)
	seq := newSequentialAgent(t, []agent.Agent{child}, "seq")
	llmParent, err := llmagent.New(llmagent.Config{Name: "llm_parent", SubAgents: []agent.Agent{seq}})
	if err != nil {
		t.Fatal(err)
	}
	baseParent, err := agent.New(agent.Config{Name: "base_parent", SubAgents: []agent.Agent{seq}})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name       string
		root       agent.Agent
		targetName string
		want       agent.Agent
	}{
		{name: "own name", root: seq, targetName: "seq", want: seq},
		{name: "through llm agent parent", root: llmParent, targetName: "seq", want: seq},
		{name: "through base agent parent", root: baseParent, targetName: "seq", want: seq},
		{name: "sub-agent", root: seq, targetName: "custom_agent_0", want: child},
		{name: "unknown name", root: seq, targetName: "unknown", want: nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.root.FindAgent(tc.targetName); got != tc.want {
				t.Errorf("FindAgent(%q) = %T(%p), want %T(%p)", tc.targetName, got, got, tc.want, tc.want)
			}
		})
	}
}

// pausingAgent returns an agent named name that yields one event with a
// long-running call to "ask" (ID "call-<name>"), then one text event per entry
// of tail. runs counts its runs.
func pausingAgent(t *testing.T, name string, runs *atomic.Int32, tail ...string) agent.Agent {
	t.Helper()
	return signalingPausingAgent(t, name, runs, make(chan struct{}), tail...)
}

// signalingPausingAgent is pausingAgent that also closes yielded once the
// consumer has taken the call event, which in a runner means the session
// holds it.
func signalingPausingAgent(t *testing.T, name string, runs *atomic.Int32, yielded chan struct{}, tail ...string) agent.Agent {
	t.Helper()
	return mustAgent(agent.New(agent.Config{Name: name, Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			runs.Add(1)
			if !yield(longRunningCallEvent(ctx, name), nil) {
				return
			}
			close(yielded)
			for _, text := range tail {
				if !yield(textEvent(ctx, name, text), nil) {
					return
				}
			}
		}
	}}))
}

// textAgent returns an agent named name that yields the text "<name> text".
// runs counts its runs.
func textAgent(t *testing.T, name string, runs *atomic.Int32) agent.Agent {
	t.Helper()
	return mustAgent(agent.New(agent.Config{Name: name, Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			runs.Add(1)
			yield(textEvent(ctx, name, name+" text"), nil)
		}
	}}))
}

func longRunningCallEvent(ctx agent.InvocationContext, author string) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = author
	ev.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{ID: "call-" + author, Name: "ask", Args: map[string]any{}}},
	}}
	ev.LongRunningToolIDs = []string{"call-" + author}
	return ev
}

func textEvent(ctx agent.InvocationContext, author, text string) *session.Event {
	ev := session.NewEvent(ctx, ctx.InvocationID())
	ev.Author = author
	ev.Content = genai.NewContentFromText(text, genai.RoleModel)
	return ev
}

// describeEvent renders ev as "author:" followed by "FC(name)" for each
// function call, "FR(name)" for each function response and the text of each
// text part, or as "author:END" for an end-of-agent event.
func describeEvent(ev *session.Event) string {
	if ev.Actions.EndOfAgent != nil && *ev.Actions.EndOfAgent {
		return ev.Author + ":END"
	}
	var b strings.Builder
	if ev.Content != nil {
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionCall != nil:
				fmt.Fprintf(&b, "FC(%s)", p.FunctionCall.Name)
			case p.FunctionResponse != nil:
				fmt.Fprintf(&b, "FR(%s)", p.FunctionResponse.Name)
			default:
				b.WriteString(p.Text)
			}
		}
	}
	return ev.Author + ":" + b.String()
}

// runResumable runs root for one turn on a resumable runner and returns the
// described events and the errors, reading on after an error as the runner's
// callers do.
func runResumable(t *testing.T, root agent.Agent) ([]string, []string) {
	t.Helper()
	r, err := runner.New(runner.Config{
		AppName:           "test_app",
		Agent:             root,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
		Resumable:         true,
	})
	if err != nil {
		t.Fatal(err)
	}
	var events, errs []string
	for ev, err := range r.Run(t.Context(), "user_id", "session_id", genai.NewContentFromText("go", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			errs = append(errs, err.Error())
			continue
		}
		events = append(events, describeEvent(ev))
	}
	return events, errs
}

// scriptedLLM replies with its responses in order and counts its requests.
type scriptedLLM struct {
	mu        sync.Mutex
	responses []*genai.Content
	requests  int
}

func (m *scriptedLLM) Name() string { return "scripted" }

func (m *scriptedLLM) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.mu.Lock()
		m.requests++
		if len(m.responses) == 0 {
			m.mu.Unlock()
			yield(nil, errors.New("scripted model has no response left"))
			return
		}
		content := m.responses[0]
		m.responses = m.responses[1:]
		m.mu.Unlock()
		yield(&model.LLMResponse{Content: content, FinishReason: genai.FinishReasonStop}, nil)
	}
}

func TestSequentialAgent_ResumablePauseSkipsLaterSubAgents(t *testing.T) {
	var aRuns, bRuns, cRuns atomic.Int32
	root := newSequentialAgent(t, []agent.Agent{
		textAgent(t, "a", &aRuns),
		pausingAgent(t, "b", &bRuns),
		textAgent(t, "c", &cRuns),
	}, "root")

	events, errs := runResumable(t, root)

	if diff := cmp.Diff([]string{"root:", "a:a text", "root:", "b:FC(ask)"}, events); diff != "" {
		t.Errorf("events (-want +got):\n%s", diff)
	}
	if len(errs) != 0 {
		t.Errorf("errors = %v, want none", errs)
	}
	if got := cRuns.Load(); got != 0 {
		t.Errorf("c ran %d times after b paused the invocation, want 0", got)
	}
}

// TestSequentialAgent_ResumablePauseLetsPausedSubAgentFinish checks that the
// sub-agent that pauses still yields the rest of its events, as adk-python's
// SequentialAgent drains it (agents/sequential_agent.py): the ordinary tool's
// response reaches the caller, and then the sub-agent's next step pauses
// before calling the model, because the long-running call is unanswered
// (flows/llm_flows/base_llm_flow.py). The next sub-agent does not run.
func TestSequentialAgent_ResumablePauseLetsPausedSubAgentFinish(t *testing.T) {
	type args struct{}
	longRunning, err := functiontool.New(functiontool.Config{Name: "ask_human", Description: "asks", IsLongRunning: true},
		func(agent.Context, args) (map[string]any, error) { return nil, nil })
	if err != nil {
		t.Fatal(err)
	}
	ordinary, err := functiontool.New(functiontool.Config{Name: "lookup", Description: "looks up"},
		func(agent.Context, args) (map[string]any, error) { return map[string]any{"v": 1}, nil })
	if err != nil {
		t.Fatal(err)
	}
	llm := &scriptedLLM{responses: []*genai.Content{
		{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "lr-1", Name: "ask_human", Args: map[string]any{}}},
			{FunctionCall: &genai.FunctionCall{ID: "fc-2", Name: "lookup", Args: map[string]any{}}},
		}},
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	first, err := llmagent.New(llmagent.Config{Name: "first", Model: llm, Tools: []tool.Tool{longRunning, ordinary}})
	if err != nil {
		t.Fatal(err)
	}
	var afterRuns atomic.Int32
	root := newSequentialAgent(t, []agent.Agent{first, textAgent(t, "after", &afterRuns)}, "root")

	events, errs := runResumable(t, root)

	want := []string{"root:", "first:FC(ask_human)FC(lookup)", "first:FR(lookup)"}
	if diff := cmp.Diff(want, events); diff != "" {
		t.Errorf("events (-want +got):\n%s", diff)
	}
	if len(errs) != 0 {
		t.Errorf("errors = %v, want none", errs)
	}
	if llm.requests != 1 {
		t.Errorf("first's model got %d requests, want 1", llm.requests)
	}
	if got := afterRuns.Load(); got != 0 {
		t.Errorf("after ran %d times after first paused the invocation, want 0", got)
	}
}

// TestSequentialAgent_PauseInDescendantStopsSequence checks that a pause
// raised below a nested shell agent also stops the outer sequence.
func TestSequentialAgent_PauseInDescendantStopsSequence(t *testing.T) {
	tests := []struct {
		name string
		// inner builds the first sub-agent of the outer sequence, which holds
		// the pausing agent "p"; pRuns counts p's runs and otherRuns those of
		// the agents beside it that must not run.
		inner      func(t *testing.T, pRuns, otherRuns *atomic.Int32) agent.Agent
		wantEvents []string
	}{
		{
			name: "sequential",
			inner: func(t *testing.T, pRuns, otherRuns *atomic.Int32) agent.Agent {
				return newSequentialAgent(t, []agent.Agent{pausingAgent(t, "p", pRuns), textAgent(t, "q", otherRuns)}, "inner")
			},
			wantEvents: []string{"root:", "inner:", "p:FC(ask)"},
		},
		{
			name: "loop",
			inner: func(t *testing.T, pRuns, _ *atomic.Int32) agent.Agent {
				return mustAgent(loopagent.New(loopagent.Config{
					AgentConfig:   agent.Config{Name: "inner", SubAgents: []agent.Agent{pausingAgent(t, "p", pRuns)}},
					MaxIterations: 3,
				}))
			},
			wantEvents: []string{"root:", "inner:", "p:FC(ask)"},
		},
		{
			name: "parallel",
			inner: func(t *testing.T, pRuns, _ *atomic.Int32) agent.Agent {
				yielded := make(chan struct{})
				return mustAgent(parallelagent.New(parallelagent.Config{
					AgentConfig: agent.Config{Name: "inner", SubAgents: []agent.Agent{
						signalingPausingAgent(t, "p", pRuns, yielded),
						gatedTextAgent(t, "s", yielded),
					}},
				}))
			},
			// The sibling branch finishes before the outer sequence stops.
			wantEvents: []string{"root:", "inner:", "p:FC(ask)", "s:s text"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var pRuns, otherRuns, afterRuns atomic.Int32
			root := newSequentialAgent(t, []agent.Agent{tc.inner(t, &pRuns, &otherRuns), textAgent(t, "after", &afterRuns)}, "root")

			events, errs := runResumable(t, root)

			if diff := cmp.Diff(tc.wantEvents, events); diff != "" {
				t.Errorf("events (-want +got):\n%s", diff)
			}
			if len(errs) != 0 {
				t.Errorf("errors = %v, want none", errs)
			}
			if got := pRuns.Load(); got != 1 {
				t.Errorf("p ran %d times, want 1", got)
			}
			if got := otherRuns.Load(); got != 0 {
				t.Errorf("p's later sibling ran %d times, want 0", got)
			}
			if got := afterRuns.Load(); got != 0 {
				t.Errorf("after ran %d times after p paused the invocation, want 0", got)
			}
		})
	}
}

// gatedTextAgent returns an agent named name that yields "<name> text" once
// gate is closed.
func gatedTextAgent(t *testing.T, name string, gate <-chan struct{}) agent.Agent {
	t.Helper()
	return mustAgent(agent.New(agent.Config{Name: name, Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			select {
			case <-ctx.Done():
				yield(nil, ctx.Err())
			case <-gate:
				yield(textEvent(ctx, name, name+" text"), nil)
			}
		}
	}}))
}

func TestSequentialAgent_ErrorAfterPausingEventIsYielded(t *testing.T) {
	b := mustAgent(agent.New(agent.Config{Name: "b", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			if !yield(longRunningCallEvent(ctx, "b"), nil) {
				return
			}
			yield(nil, errors.New("boom"))
		}
	}}))
	var cRuns atomic.Int32
	root := newSequentialAgent(t, []agent.Agent{b, textAgent(t, "c", &cRuns)}, "root")

	events, errs := runResumable(t, root)

	if diff := cmp.Diff([]string{"root:", "b:FC(ask)"}, events); diff != "" {
		t.Errorf("events (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff([]string{"boom"}, errs); diff != "" {
		t.Errorf("errors (-want +got):\n%s", diff)
	}
	if got := cRuns.Load(); got != 0 {
		t.Errorf("c ran %d times, want 0", got)
	}
}

// TestSequentialAgent_ContextWithoutRunConfigDoesNotPause checks that a
// sequence run outside a runner, with no run config, is not resumable and so
// runs every sub-agent.
func TestSequentialAgent_ContextWithoutRunConfigDoesNotPause(t *testing.T) {
	var pRuns, cRuns atomic.Int32
	root := newSequentialAgent(t, []agent.Agent{pausingAgent(t, "p", &pRuns), textAgent(t, "c", &cRuns)}, "root")
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: root})

	var events []string
	for ev, err := range root.Run(ctx) {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, describeEvent(ev))
	}

	if diff := cmp.Diff([]string{"p:FC(ask)", "c:c text"}, events); diff != "" {
		t.Errorf("events (-want +got):\n%s", diff)
	}
}

// TestSequentialAgent_LongRunningIDOfAnotherCallDoesNotPause checks that an
// event pauses the sequence only when one of its own function calls is long
// running: b's call "call-b" is not in its LongRunningToolIDs, so adk-python
// does not pause (agents/invocation_context.py, should_pause_invocation) and
// c runs.
func TestSequentialAgent_LongRunningIDOfAnotherCallDoesNotPause(t *testing.T) {
	var cRuns atomic.Int32
	b := mustAgent(agent.New(agent.Config{Name: "b", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			ev := longRunningCallEvent(ctx, "b")
			ev.LongRunningToolIDs = []string{"other"}
			yield(ev, nil)
		}
	}}))
	root := newSequentialAgent(t, []agent.Agent{b, textAgent(t, "c", &cRuns)}, "root")

	events, errs := runResumable(t, root)

	if diff := cmp.Diff([]string{"root:", "b:FC(ask)", "root:", "c:c text", "root:END"}, events); diff != "" {
		t.Errorf("events (-want +got):\n%s", diff)
	}
	if len(errs) != 0 {
		t.Errorf("errors = %v, want none", errs)
	}
	if got := cRuns.Load(); got != 1 {
		t.Errorf("c ran %d times, want 1", got)
	}
}
