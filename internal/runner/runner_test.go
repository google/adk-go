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

package runner

import (
	"context"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/types"
	"google.golang.org/genai"
)

func TestRunAgent(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	tests := []struct {
		name                 string
		ictx                 *types.InvocationContext
		beforeAgentCallbacks []types.BeforeAgentCallback
		afterAgentCallbacks  []types.AfterAgentCallback
		wantLLMCalls         int
		wantEvents           []*types.Event
	}{
		{
			name: "before agent callback runs, no llm calls",
			beforeAgentCallbacks: []types.BeforeAgentCallback{
				func(ctx context.Context, callbackCtx *types.CallbackContext) *genai.Content {
					return genai.NewContentFromText("hello from before_agent_callback", genai.RoleModel)
				},
			},
			wantEvents: []*types.Event{
				{
					LLMResponse: &types.LLMResponse{
						Content: genai.NewContentFromText("hello from before_agent_callback", genai.RoleModel),
					},
				},
			},
		},
		{
			name: "no callback effect if callbacks return nil",
			beforeAgentCallbacks: []types.BeforeAgentCallback{
				func(ctx context.Context, callbackCtx *types.CallbackContext) *genai.Content {
					return nil
				},
			},
			afterAgentCallbacks: []types.AfterAgentCallback{
				func(ctx context.Context, callbackCtx *types.CallbackContext, content *genai.Content) *genai.Content {
					return nil
				},
			},
			wantLLMCalls: 1,
			wantEvents: []*types.Event{
				{
					LLMResponse: &types.LLMResponse{
						Content: genai.NewContentFromText("hello", genai.RoleModel),
					},
				},
			},
		},
		{
			name: "after agent callback replaces event content",
			afterAgentCallbacks: []types.AfterAgentCallback{
				func(ctx context.Context, callbackCtx *types.CallbackContext, content *genai.Content) *genai.Content {
					return genai.NewContentFromText("hello from after_agent_callback", genai.RoleModel)
				},
			},
			wantLLMCalls: 1,
			wantEvents: []*types.Event{
				{
					LLMResponse: &types.LLMResponse{
						Content: genai.NewContentFromText("hello from after_agent_callback", genai.RoleModel),
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			agent := &customAgent{
				spec: &types.AgentSpec{},
			}

			agent.spec.BeforeAgentCallbacks = tt.beforeAgentCallbacks
			agent.spec.AfterAgentCallbacks = tt.afterAgentCallbacks

			ctx, ictx := types.NewInvocationContext(ctx, agent, nil, nil, nil, nil)

			var gotEvents []*types.Event
			for event, err := range RunAgent(ctx, ictx, agent) {
				if err != nil {
					t.Fatalf("unexpected error from the agent: %v", err)
				}

				gotEvents = append(gotEvents, event)
			}

			if tt.wantLLMCalls != agent.callCounter {
				t.Errorf("unexpected want_llm_calls, got: %v, want: %v", agent.callCounter, tt.wantLLMCalls)
			}

			if len(tt.wantEvents) != len(gotEvents) {
				t.Errorf("unexpected event lengths, got: %v, want: %v", len(gotEvents), len(tt.wantEvents))
			}

			for i, gotEvent := range gotEvents {
				if diff := cmp.Diff(tt.wantEvents[i], gotEvent, cmpopts.IgnoreFields(types.Event{}, "ID", "Time", "InvocationID")); diff != "" {
					t.Errorf("diff in the events: got event[%d]: %v, want: %v, diff: %v", i, gotEvent, tt.wantEvents[i], diff)
				}
			}
		})
	}
}

// creates agentTree for tests and returns references to the agents
func agentTree(t *testing.T) agentTreeStruct {
	t.Helper()

	sub1 := must(agent.NewLLMAgent("no_transfer_agent", nil, agent.WithDisallowTransferToParent()))
	sub2 := must(agent.NewLLMAgent("allows_transfer_agent", nil))

	parent, err := agent.NewLLMAgent("root", nil, agent.WithSubAgents(sub1, sub2))
	if err != nil {
		t.Fatal(err)
	}

	return agentTreeStruct{
		root:                parent,
		noTransferAgent:     sub1,
		allowsTransferAgent: sub2,
	}
}

type agentTreeStruct struct {
	root, noTransferAgent, allowsTransferAgent types.Agent
}

func must[T types.Agent](a T, err error) T {
	if err != nil {
		panic(err)
	}
	return a
}

// TODO: create test util allowing to create custom agents, agent trees for test etc.
type customAgent struct {
	spec *types.AgentSpec

	callCounter int
}

func (a *customAgent) Spec() *types.AgentSpec { return a.spec }

func (a *customAgent) Run(context.Context, *types.InvocationContext) iter.Seq2[*types.Event, error] {
	return func(yield func(*types.Event, error) bool) {
		a.callCounter++

		yield(&types.Event{
			LLMResponse: &types.LLMResponse{
				Content: genai.NewContentFromText("hello", genai.RoleModel),
			},
		}, nil)
	}
}
