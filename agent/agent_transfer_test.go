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
	"slices"
	"strings"
	"testing"

	"github.com/google/adk-go"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

type mockAgent string

var _ adk.Agent = mockAgent("")

func (a mockAgent) Name() string        { return string(a) }
func (a mockAgent) Description() string { return "" }
func (a mockAgent) Run(ctx context.Context, invCtx *adk.InvocationContext) (adk.EventStream, error) {
	return func(yield func(*adk.Event, error) bool) {}, nil
}

func TestAgentTransferRequestProcessor(t *testing.T) {
	ctx := context.Background()
	tool := &transferToAgentTool{}

	if tool.Name() == "" || tool.Description() == "" || tool.FunctionDeclaration() == nil {
		t.Fatalf("unexpected transferToAgentTool: name=%q, desc=%q, decl=%v", tool.Name(), tool.Description(), tool)
	}

	check := func(t *testing.T, agent adk.Agent, wantParent string, wantAgents []string, unwantAgents []string) {
		invCtx := &adk.InvocationContext{Agent: agent}
		req := &adk.LLMRequest{}
		name := agent.Name()
		_ = name

		if err := agentTransferRequestProcessor(ctx, invCtx, req); err != nil {
			t.Fatalf("agentTransferRequestProcessor() = %v, want success", err)
		}

		// We don't expect transfer. Check agentTransferRequestProcessor was no-op.
		if wantParent == "" && len(wantAgents) == 0 {
			if diff := cmp.Diff(&adk.LLMRequest{}, req); diff != "" {
				t.Errorf("req was changed unexpectedly (-want, +got): %v", diff)
			}
			return
		}
		// We expect transfer. From here, it's true that either wantParent != "" or len(wantSubagents) > 0.

		// check tools dictionary.
		wantToolName := tool.Name()
		if gotTool, ok := req.Tools[wantToolName]; !ok || gotTool.Name() != wantToolName {
			t.Errorf("req.Tools does not include %v: req.Tools = %v", wantToolName, req.Tools)
		}

		// check instructions.
		instructions := textParts(req.GenerateConfig.SystemInstruction)
		if !slices.ContainsFunc(instructions, func(s string) bool {
			return strings.Contains(s, wantToolName) && strings.Contains(s, "You have a list of other agents to transfer to")
		}) {
			t.Errorf("instruction does not include agent transfer instruction, got: %s", strings.Join(instructions, "\n"))
		}
		if wantParent != "" && !slices.ContainsFunc(instructions, func(s string) bool {
			return strings.Contains(s, wantParent)
		}) {
			t.Errorf("instruction does not include parent agent, got: %s", strings.Join(instructions, "\n"))
		}
		if slices.Contains(instructions, agent.Name()) {
			t.Errorf("instruction should not suggest transfer to current agent, got: %s", strings.Join(instructions, "\n"))
		}
		if len(wantAgents) > 0 && !slices.ContainsFunc(instructions, func(s string) bool {
			return slices.ContainsFunc(wantAgents, func(sub string) bool {
				for _, subagent := range wantAgents {
					if !strings.Contains(s, subagent) {
						return false
					}
				}
				return true
			})
		}) {
			t.Errorf("instruction does not include subagents, got: %s", strings.Join(instructions, "\n"))
		}
		if len(unwantAgents) > 0 && slices.ContainsFunc(instructions, func(s string) bool {
			return slices.ContainsFunc(unwantAgents, func(unwanted string) bool {
				for _, unwanted := range unwantAgents {
					if strings.Contains(s, unwanted) {
						return true
					}
				}
				return false
			})
		}) {
			t.Errorf("instruction includes unwanted agents, got: %s", strings.Join(instructions, "\n"))
		}

		// check function declarations.
		wantToolDescription := tool.Description()
		functions := functionDecls(req.GenerateConfig)
		if !slices.ContainsFunc(functions, func(f *genai.FunctionDeclaration) bool {
			return f.Name == wantToolName && strings.Contains(f.Description, wantToolDescription) && f.ParametersJsonSchema == nil
		}) {
			t.Errorf("agentTransferRequestProcessor() did not append the function declaration, got: %v", stringify(functions))
		}
	}

	t.Run("SoloAgent", func(t *testing.T) {
		agent := &LLMAgent{AgentName: "Current"}
		check(t, agent, "", nil, []string{"Current"})
	})
	t.Run("NotLLMAgent", func(t *testing.T) {
		check(t, mockAgent("mockAgent"), "", nil, nil)
	})
	t.Run("LLMAgentParent", func(t *testing.T) {
		agent := &LLMAgent{AgentName: "Current"}
		parentAgent := &LLMAgent{AgentName: "Parent"}
		parentAgent.AddSubAgents(agent)
		check(t, agent, "Parent", nil, []string{"Current"})
	})
	t.Run("LLMAgentParentAndPeer", func(t *testing.T) {
		agent := &LLMAgent{AgentName: "Current"}
		peer := &LLMAgent{AgentName: "Peer"}
		parentAgent := &LLMAgent{AgentName: "Parent"}
		parentAgent.AddSubAgents(agent, peer)
		check(t, agent, "Parent", []string{"Peer"}, []string{"Current"})
	})
	t.Run("LLMAgentSubagents", func(t *testing.T) {
		agent := &LLMAgent{AgentName: "Current"}
		agent.AddSubAgents(mockAgent("Sub1"), &LLMAgent{AgentName: "Sub2"})
		check(t, agent, "", []string{"Sub1", "Sub2"}, []string{"Current"})
	})

	t.Run("AgentWithParentAndPeersAndSubagents", func(t *testing.T) {
		agent := &LLMAgent{AgentName: "Current"}
		agent.AddSubAgents(mockAgent("Sub1"), &LLMAgent{AgentName: "Sub2"})
		peer := mockAgent("Peer")
		parentAgent := &LLMAgent{AgentName: "Parent"}
		parentAgent.AddSubAgents(agent, peer)
		check(t, agent, "Parent", []string{"Peer", "Sub1", "Sub2"}, []string{"Current"})
	})

	t.Run("NonLLMAgentSubagents", func(t *testing.T) {
		agent := &LLMAgent{AgentName: "Current"}
		agent.AddSubAgents(mockAgent("Sub1"), mockAgent("Sub2"))
		check(t, agent, "", []string{"Sub1", "Sub2"}, []string{"Current"})
	})

	t.Run("AgentWithDisallowTransferToParent", func(t *testing.T) {
		agent := &LLMAgent{AgentName: "Current"}
		agent.AddSubAgents(&LLMAgent{AgentName: "Sub1"}, &LLMAgent{AgentName: "Sub2"})
		parentAgent := &LLMAgent{AgentName: "Parent"}
		parentAgent.AddSubAgents(agent)

		agent.DisallowTransferToParent = true
		check(t, agent, "", []string{"Sub1", "Sub2"}, []string{"Parent", "Current"})
	})

	t.Run("AgentWithDisallowTransferToPeers", func(t *testing.T) {
		agent := &LLMAgent{AgentName: "Current"}
		peer := &LLMAgent{AgentName: "Peer"}
		agent.AddSubAgents(mockAgent("Sub1"), &LLMAgent{AgentName: "Sub2"})
		parentAgent := &LLMAgent{AgentName: "Parent"}
		parentAgent.AddSubAgents(agent, peer)

		agent.DisallowTransferToPeers = true
		check(t, agent, "Parent", []string{"Sub1", "Sub2"}, []string{"Peer", "Current"})
	})

	t.Run("AgentWithDisallowTransferToParentAndPeers", func(t *testing.T) {
		agent := &LLMAgent{AgentName: "Current"}
		peer := &LLMAgent{AgentName: "Peer"}
		agent.AddSubAgents(mockAgent("Sub1"), &LLMAgent{AgentName: "Sub2"})
		parentAgent := &LLMAgent{AgentName: "Parent"}
		parentAgent.AddSubAgents(peer, agent)

		agent.DisallowTransferToPeers = true
		agent.DisallowTransferToParent = true
		check(t, agent, "", []string{"Sub1", "Sub2"}, []string{"Parent", "Peer", "Current"})
	})

	t.Run("AgentWithDisallowTransfer", func(t *testing.T) {
		agent := &LLMAgent{AgentName: "Current"}
		peer := &LLMAgent{AgentName: "Peer"}
		parentAgent := &LLMAgent{AgentName: "Parent"}
		parentAgent.AddSubAgents(peer, agent)

		agent.DisallowTransferToPeers = true
		agent.DisallowTransferToParent = true
		check(t, agent, "", nil, []string{"Parent", "Peer", "Current"})
	})
}

func TestTransferToAgentToolRun(t *testing.T) {
	t.Run("Success", func(t *testing.T) {
		tool := &transferToAgentTool{}
		tc := &adk.ToolContext{
			InvocationContext: &adk.InvocationContext{},
			EventActions:      &adk.EventActions{},
		}
		wantAgentName := "TestAgent"
		args := map[string]any{"agent_name": wantAgentName}
		ctx := t.Context()
		if _, err := tool.Run(ctx, tc, args); err != nil {
			t.Fatalf("Run(%v) failed: %v", args, err)
		}
		if got, want := tc.EventActions.TransferToAgent, wantAgentName; got != want {
			t.Errorf("Run(%v) did not set TransferToAgent, got %q, want %q", args, got, want)
		}
	})

	t.Run("InvalidArguments", func(t *testing.T) {
		testCases := []struct {
			name string
			args map[string]any
		}{
			{name: "NoAgentName", args: map[string]any{}},
			{name: "NilArg", args: nil},
			{name: "InvalidType", args: map[string]any{"agent_name": 123}},
			{name: "InvalidValue", args: map[string]any{"agent_name": ""}},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				tool := &transferToAgentTool{}
				toolCtx := &adk.ToolContext{
					InvocationContext: &adk.InvocationContext{},
					EventActions:      &adk.EventActions{},
				}

				ctx := t.Context()
				if got, err := tool.Run(ctx, toolCtx, tc.args); err == nil {
					t.Fatalf("Run(%v) = (%v, %v), want error", tc.args, got, err)
				}
			})
		}
	})
}
