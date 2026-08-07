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

package agent

import (
	"context"
	"iter"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestAgentCallbacks(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		beforeAgent  []BeforeAgentCallback
		afterAgent   []AfterAgentCallback
		wantLLMCalls int
		wantEvents   []*session.Event
	}{
		{
			name: "before agent callback runs, no llm calls",
			beforeAgent: []BeforeAgentCallback{
				func(ctx Context) (*genai.Content, error) {
					return genai.NewContentFromText("hello from before_agent_callback", genai.RoleModel), nil
				},
			},
			wantEvents: []*session.Event{
				{
					Author: "test",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("hello from before_agent_callback", genai.RoleModel),
					},
					Actions: session.EventActions{
						StateDelta:    map[string]any{},
						ArtifactDelta: map[string]int64{},
					},
				},
			},
		},
		{
			name: "no callback effect if callbacks return nil",
			beforeAgent: []BeforeAgentCallback{
				func(ctx Context) (*genai.Content, error) {
					return nil, nil
				},
			},
			afterAgent: []AfterAgentCallback{
				func(Context) (*genai.Content, error) {
					return nil, nil
				},
			},
			wantLLMCalls: 1,
			wantEvents: []*session.Event{
				{
					Author: "test",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("hello", genai.RoleModel),
					},
				},
			},
		},
		{
			name: "after agent callback create a new event with new content",
			afterAgent: []AfterAgentCallback{
				func(Context) (*genai.Content, error) {
					return genai.NewContentFromText("hello from after_agent_callback", genai.RoleModel), nil
				},
			},
			wantLLMCalls: 1,
			wantEvents: []*session.Event{
				{
					Author: "test",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("hello", genai.RoleModel),
					},
				},
				{
					Author: "test",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("hello from after_agent_callback", genai.RoleModel),
					},
					Actions: session.EventActions{
						StateDelta:    map[string]any{},
						ArtifactDelta: map[string]int64{},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			custom := &customAgent{}

			testAgent, err := New(Config{
				Name:                 "test",
				BeforeAgentCallbacks: tt.beforeAgent,
				Run:                  custom.Run,
				AfterAgentCallbacks:  tt.afterAgent,
			})
			if err != nil {
				t.Fatalf("failed to create agent: %v", err)
			}

			ctx := &invocationContext{
				Context:       t.Context(),
				agent:         testAgent,
				session:       &mockSession{sessionID: "test-session"},
				endInvocation: new(atomic.Bool),
			}
			var gotEvents []*session.Event
			for event, err := range testAgent.Run(ctx) {
				if err != nil {
					t.Fatalf("unexpected error from the agent: %v", err)
				}

				gotEvents = append(gotEvents, event)
			}

			if tt.wantLLMCalls != custom.callCounter {
				t.Errorf("unexpected want_llm_calls, got: %v, want: %v", custom.callCounter, tt.wantLLMCalls)
			}

			if len(tt.wantEvents) != len(gotEvents) {
				t.Errorf("unexpected event lengths, got: %v, want: %v", len(gotEvents), len(tt.wantEvents))
			}

			for i, gotEvent := range gotEvents {
				if diff := cmp.Diff(tt.wantEvents[i], gotEvent, cmpopts.IgnoreFields(session.Event{}, "ID", "Timestamp", "InvocationID")); diff != "" {
					t.Errorf("diff in the events: got event[%d]: %v, want: %v, diff: %v", i, gotEvent, tt.wantEvents[i], diff)
				}
			}
		})
	}
}

func TestEndInvocation_EndsBeforeMainCall(t *testing.T) {
	custom := &customAgent{}

	testAgent, err := New(Config{
		Name: "test",
		BeforeAgentCallbacks: []BeforeAgentCallback{
			func(ctx Context) (*genai.Content, error) {
				return nil, nil
			},
		},
		Run: custom.Run,
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}

	endedBool := new(atomic.Bool)
	endedBool.Store(true)

	ctx := &invocationContext{
		Context:       t.Context(),
		agent:         testAgent,
		endInvocation: endedBool,
		session:       &mockSession{sessionID: "test-session"},
	}
	for _, err := range testAgent.Run(ctx) {
		if err != nil {
			t.Fatalf("unexpected error from the agent: %v", err)
		}
	}

	// Even though beforeAgentCallback returns nil, it stil doesn't call llm because
	// endInvocation is true.
	if custom.callCounter != 0 {
		t.Errorf("unexpected want_llm_calls, got: %v, want: %v", custom.callCounter, 0)
	}
}

func TestEndInvocation_EndsAfterMainCall(t *testing.T) {
	custom := &customAgent{endInvocation: true}

	testAgent, err := New(Config{
		Name: "test",
		AfterAgentCallbacks: []AfterAgentCallback{
			func(Context) (*genai.Content, error) {
				return genai.NewContentFromText("hello from after_agent_callback", genai.RoleModel), nil
			},
		},
		Run: custom.Run,
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}

	ctx := &invocationContext{
		Context:       t.Context(),
		agent:         testAgent,
		session:       &mockSession{sessionID: "test-session"},
		endInvocation: new(atomic.Bool),
	}
	var gotEvents []*session.Event
	for event, err := range testAgent.Run(ctx) {
		if err != nil {
			t.Fatalf("unexpected error from the agent: %v", err)
		}
		gotEvents = append(gotEvents, event)
	}

	if custom.callCounter != 1 {
		t.Errorf("unexpected want_llm_calls, got: %v, want: %v", custom.callCounter, 0)
	}
	// Even though AfterAgentCallbacks is present, it's not returned because EndInvocation is set to true
	wantEvent := &session.Event{
		Author: "test",
		LLMResponse: model.LLMResponse{
			Content: genai.NewContentFromText("hello", genai.RoleModel),
		},
	}
	if len(gotEvents) != 1 {
		t.Errorf("unexpected number of events, got: %v, want: %v", len(gotEvents), 1)
	}
	if diff := cmp.Diff(wantEvent, gotEvents[0], cmpopts.IgnoreFields(session.Event{}, "ID", "Timestamp", "InvocationID"),
		cmpopts.IgnoreFields(session.EventActions{}, "StateDelta")); diff != "" {
		t.Errorf("unexpected event, got: %v, want: %v, diff: %v", gotEvents[0], wantEvent, diff)
	}
}

// TODO: create test util allowing to create custom agents, agent trees for test etc.
type customAgent struct {
	callCounter   int
	endInvocation bool
}

func (a *customAgent) Run(ctx InvocationContext) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		a.callCounter++

		if a.endInvocation {
			ctx.EndInvocation()
		}

		yield(&session.Event{
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("hello", genai.RoleModel),
			},
		}, nil)
	}
}

type testKey struct{}

func TestWithContext(t *testing.T) {
	baseCtx := t.Context()
	inv := &invocationContext{
		Context:       baseCtx,
		invocationID:  "test",
		branch:        "branch",
		endInvocation: new(atomic.Bool),
	}

	key := testKey{}
	val := "val"
	got := inv.WithContext(context.WithValue(baseCtx, key, val))

	if got.Value(key) != val {
		t.Errorf("WithContext() did not update context")
	}
	atomicBoolComparer := cmp.Comparer(func(a, b *atomic.Bool) bool {
		if a == nil || b == nil {
			return a == b
		}
		return a.Load() == b.Load()
	})
	if diff := cmp.Diff(inv, got, cmp.AllowUnexported(invocationContext{}), cmpopts.IgnoreFields(invocationContext{}, "Context"), atomicBoolComparer); diff != "" {
		t.Errorf("WithContext() params mismatch (-want +got):\n%s", diff)
	}
}

type mockSession struct {
	session.Session
	sessionID string
}

func (m *mockSession) ID() string { return m.sessionID }

func TestFindAgent(t *testing.T) {
	t.Parallel()

	noOpRun := func(InvocationContext) iter.Seq2[*session.Event, error] {
		return func(func(*session.Event, error) bool) {}
	}

	createAgent := func(name string, subAgents ...Agent) Agent {
		t.Helper()
		a, err := New(Config{Name: name, Run: noOpRun, SubAgents: subAgents})
		if err != nil {
			t.Fatalf("failed to create agent %s: %v", name, err)
		}
		return a
	}

	// Setup hierarchy:
	// root -> child1
	// root -> child2 -> grandchild
	grandchild := createAgent("grandchild")
	child2 := createAgent("child2", grandchild)
	child1 := createAgent("child1")
	root := createAgent("root", child1, child2)

	tests := []struct {
		name      string
		agentName string
		want      Agent
	}{
		{
			name:      "Find self",
			agentName: "root",
			want:      root,
		},
		{
			name:      "Find direct child1",
			agentName: "child1",
			want:      child1,
		},
		{
			name:      "Find direct child2",
			agentName: "child2",
			want:      child2,
		},
		{
			name:      "Find nested grandchild",
			agentName: "grandchild",
			want:      grandchild,
		},
		{
			name:      "Find non-existent agent",
			agentName: "unknown",
			want:      nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := root.FindAgent(tt.agentName)
			if got != tt.want {
				t.Errorf("FindAgent(%q) = %v, want %v", tt.agentName, got, tt.want)
			}
		})
	}
}

func TestSubAgentCallbackEndInvocationPropagated(t *testing.T) {
	t.Parallel()

	subAgentCalled := false

	subAgent, err := New(Config{
		Name: "sub_agent",
		BeforeAgentCallbacks: []BeforeAgentCallback{
			func(ctx Context) (*genai.Content, error) {
				subAgentCalled = true
				return genai.NewContentFromText("sub agent ended", genai.RoleModel), nil
			},
		},
		Run: func(InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("failed to create sub_agent: %v", err)
	}

	parentRunCalled := false
	parentEndedAfterSubAgent := false

	parentAgent, err := New(Config{
		Name:      "parent_agent",
		SubAgents: []Agent{subAgent},
		Run: func(ctx InvocationContext) iter.Seq2[*session.Event, error] {
			parentRunCalled = true
			return func(yield func(*session.Event, error) bool) {
				for _, sa := range ctx.Agent().SubAgents() {
					for event, err := range sa.Run(ctx) {
						if !yield(event, err) {
							return
						}
					}
				}
				// Demonstrates that when sub-agent BeforeAgentCallback returns non-nil content
				// (which calls ctx.EndInvocation() on the sub-agent context), the endInvocation
				// flag propagates to the parent context via *atomic.Bool.
				parentEndedAfterSubAgent = ctx.Ended()
			}
		},
	})
	if err != nil {
		t.Fatalf("failed to create parent_agent: %v", err)
	}

	parentCtx := &invocationContext{
		Context:       t.Context(),
		agent:         parentAgent,
		session:       &mockSession{sessionID: "test-session"},
		endInvocation: new(atomic.Bool),
	}

	for _, err := range parentAgent.Run(parentCtx) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	if !parentRunCalled {
		t.Error("expected parent run to be called")
	}
	if !subAgentCalled {
		t.Error("expected sub_agent BeforeAgentCallback to be called")
	}
	if !parentEndedAfterSubAgent {
		t.Error("expected parentEndedAfterSubAgent to be true after sub-agent callback ended invocation")
	}
}
