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
	"google.golang.org/adk/internal/llminternal"
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
				ev := session.NewEvent(ctx.InvocationID())
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
				ev := session.NewEvent(ctx.InvocationID())
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
