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
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
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
				func(ctx CallbackContext) (*genai.Content, error) {
					return genai.NewContentFromText("hello from before_agent_callback", genai.RoleModel), nil
				},
			},
			wantEvents: []*session.Event{
				{
					Author: "test",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("hello from before_agent_callback", genai.RoleModel),
					},
				},
			},
		},
		{
			name: "no callback effect if callbacks return nil",
			beforeAgent: []BeforeAgentCallback{
				func(ctx CallbackContext) (*genai.Content, error) {
					return nil, nil
				},
			},
			afterAgent: []AfterAgentCallback{
				func(CallbackContext) (*genai.Content, error) {
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
				func(CallbackContext) (*genai.Content, error) {
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
				agent: testAgent,
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
				if diff := cmp.Diff(tt.wantEvents[i], gotEvent, cmpopts.IgnoreFields(session.Event{}, "ID", "Timestamp", "InvocationID"),
					cmpopts.IgnoreFields(session.EventActions{}, "StateDelta")); diff != "" {
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
			func(ctx CallbackContext) (*genai.Content, error) {
				return nil, nil
			},
		},
		Run: custom.Run,
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}

	ctx := &invocationContext{
		agent:         testAgent,
		endInvocation: true,
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
			func(CallbackContext) (*genai.Content, error) {
				return genai.NewContentFromText("hello from after_agent_callback", genai.RoleModel), nil
			},
		},
		Run: custom.Run,
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}

	ctx := &invocationContext{
		agent: testAgent,
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
