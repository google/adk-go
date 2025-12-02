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

package llminternal_test

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/parallelagent"
	"google.golang.org/adk/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/internal/agent/parentmap"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/llminternal"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

func TestAgentTransferRequestProcessor(t *testing.T) {
	curTool := &llminternal.TransferToAgentTool{}
	llm := &struct{ model.LLM }{}

	if curTool.Name() == "" || curTool.Description() == "" || curTool.Declaration() == nil {
		t.Fatalf("unexpected TransferToAgentTool: name=%q, desc=%q, decl=%v", curTool.Name(), curTool.Description(), curTool)
	}

	check := func(t *testing.T, curAgent, root agent.Agent, wantParent string, wantAgents, unwantAgents []string) {
		req := &model.LLMRequest{}

		parents, err := parentmap.New(root)
		if err != nil {
			t.Fatal(err)
		}

		ctx := icontext.NewInvocationContext(parentmap.ToContext(t.Context(), parents), icontext.InvocationContextParams{
			Agent: curAgent,
		})

		if err := llminternal.AgentTransferRequestProcessor(ctx, req); err != nil {
			t.Fatalf("AgentTransferRequestProcessor() = %v, want success", err)
		}

		// We don't expect transfer. Check AgentTransferRequestProcessor was no-op.
		if wantParent == "" && len(wantAgents) == 0 {
			if diff := cmp.Diff(&model.LLMRequest{}, req); diff != "" {
				t.Errorf("req was changed unexpectedly (-want, +got): %v", diff)
			}
			return
		}
		// We expect transfer. From here, it's true that either wantParent != "" or len(wantSubagents) > 0.

		// check tools dictionary.
		wantToolName := curTool.Name()
		gotRawTool, ok := req.Tools[wantToolName]
		if !ok {
			t.Errorf("req.Tools does not include %v: req.Tools = %v", wantToolName, req.Tools)
		}
		gotTool, ok := gotRawTool.(tool.Tool)
		if !ok {
			t.Errorf("failed to type convert tool %v, got %T", wantToolName, gotRawTool)
		}

		if gotTool.Name() != wantToolName {
			t.Errorf("unexpected name for tool, got: %v, want: %v", gotTool.Name(), wantToolName)
		}

		// check instructions.
		instructions := utils.TextParts(req.Config.SystemInstruction)
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
		if slices.Contains(instructions, curAgent.Name()) {
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
		wantToolDescription := curTool.Description()
		functions := utils.FunctionDecls(req.Config)
		if !slices.ContainsFunc(functions, func(f *genai.FunctionDeclaration) bool {
			return f.Name == wantToolName && strings.Contains(f.Description, wantToolDescription) && f.ParametersJsonSchema == nil
		}) {
			t.Errorf("AgentTransferRequestProcessor() did not append the function declaration, got: %v", stringify(functions))
		}
	}

	t.Run("SoloAgent", func(t *testing.T) {
		agent := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Current",
			Model: llm,
		}))
		check(t, agent, agent, "", nil, []string{"Current"})
	})
	t.Run("NotLLMAgent", func(t *testing.T) {
		a := utils.Must(agent.New(agent.Config{
			Name: "mockAgent",
		}))
		check(t, a, a, "", nil, nil)
	})
	t.Run("LLMAgentParent", func(t *testing.T) {
		testAgent := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Current",
			Model: llm,
		}))
		root := utils.Must(llmagent.New(llmagent.Config{
			Name:      "Parent",
			Model:     llm,
			SubAgents: []agent.Agent{testAgent},
		}))
		check(t, testAgent, root, "Parent", nil, []string{"Current"})
	})
	t.Run("LLMAgentParentAndPeer", func(t *testing.T) {
		curAgent := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Current",
			Model: llm,
		}))
		peer := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Peer",
			Model: llm,
		}))
		root := utils.Must(llmagent.New(llmagent.Config{
			Name:      "Parent",
			Model:     llm,
			SubAgents: []agent.Agent{curAgent, peer},
		}))
		check(t, curAgent, root, "Parent", []string{"Peer"}, []string{"Current"})
	})
	t.Run("LLMAgentSubagents", func(t *testing.T) {
		agent := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Current",
			Model: llm,
			SubAgents: []agent.Agent{
				utils.Must(agent.New(agent.Config{
					Name: "Sub1",
				})),
				utils.Must(llmagent.New(llmagent.Config{
					Name:  "Sub2",
					Model: llm,
				})),
			},
		}))
		check(t, agent, agent, "", []string{"Sub1", "Sub2"}, []string{"Current"})
	})

	t.Run("AgentWithParentAndPeersAndSubagents", func(t *testing.T) {
		curAgent := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Current",
			Model: llm,
			SubAgents: []agent.Agent{
				utils.Must(agent.New(agent.Config{
					Name: "Sub1",
				})),
				utils.Must(llmagent.New(llmagent.Config{
					Name:  "Sub2",
					Model: llm,
				})),
			},
		}))
		peer := utils.Must(agent.New(agent.Config{
			Name: "Peer",
		}))
		root := utils.Must(llmagent.New(llmagent.Config{
			Name:      "Parent",
			Model:     llm,
			SubAgents: []agent.Agent{curAgent, peer},
		}))
		check(t, curAgent, root, "Parent", []string{"Peer", "Sub1", "Sub2"}, []string{"Current"})
	})

	t.Run("NonLLMAgentSubagents", func(t *testing.T) {
		agent := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Current",
			Model: llm,
			SubAgents: []agent.Agent{
				utils.Must(agent.New(agent.Config{
					Name: "Sub1",
				})),
				utils.Must(agent.New(agent.Config{
					Name: "Sub2",
				})),
			},
		}))
		check(t, agent, agent, "", []string{"Sub1", "Sub2"}, []string{"Current"})
	})

	t.Run("AgentWithDisallowTransferToParent", func(t *testing.T) {
		curAgent := utils.Must(llmagent.New(llmagent.Config{
			Name:                     "Current",
			Model:                    llm,
			DisallowTransferToParent: true,
			SubAgents: []agent.Agent{
				utils.Must(llmagent.New(llmagent.Config{
					Name:  "Sub1",
					Model: llm,
				})),
				utils.Must(llmagent.New(llmagent.Config{
					Name:  "Sub2",
					Model: llm,
				})),
			},
		}))
		root := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Parent",
			Model: llm,
			SubAgents: []agent.Agent{
				curAgent,
			},
		}))

		check(t, curAgent, root, "", []string{"Sub1", "Sub2"}, []string{"Parent", "Current"})
	})

	t.Run("AgentWithDisallowTransferToPeers", func(t *testing.T) {
		curAgent := utils.Must(llmagent.New(llmagent.Config{
			Name:                    "Current",
			Model:                   llm,
			DisallowTransferToPeers: true,
			SubAgents: []agent.Agent{
				utils.Must(agent.New(agent.Config{
					Name: "Sub1",
				})), utils.Must(llmagent.New(llmagent.Config{
					Name:  "Sub2",
					Model: llm,
				})),
			},
		}))
		peer := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Peer",
			Model: llm,
		}))
		root := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Parent",
			Model: llm,
			SubAgents: []agent.Agent{
				curAgent, peer,
			},
		}))
		check(t, curAgent, root, "Parent", []string{"Sub1", "Sub2"}, []string{"Peer", "Current"})
	})

	t.Run("AgentWithDisallowTransferToParentAndPeers", func(t *testing.T) {
		curAgent := utils.Must(llmagent.New(llmagent.Config{
			Name:                     "Current",
			Model:                    llm,
			DisallowTransferToParent: true,
			DisallowTransferToPeers:  true,
			SubAgents: []agent.Agent{
				utils.Must(agent.New(agent.Config{
					Name: "Sub1",
				})),
				utils.Must(llmagent.New(llmagent.Config{
					Name:  "Sub2",
					Model: llm,
				})),
			},
		}))
		peer := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Peer",
			Model: llm,
		}))
		root := utils.Must(llmagent.New(llmagent.Config{
			Name:      "Parent",
			Model:     llm,
			SubAgents: []agent.Agent{peer, curAgent},
		}))

		check(t, curAgent, root, "", []string{"Sub1", "Sub2"}, []string{"Parent", "Peer", "Current"})
	})

	t.Run("AgentWithDisallowTransfer", func(t *testing.T) {
		curAgent := utils.Must(llmagent.New(llmagent.Config{
			Name:                     "Current",
			Model:                    llm,
			DisallowTransferToParent: true,
			DisallowTransferToPeers:  true,
		}))
		peer := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Peer",
			Model: llm,
		}))
		root := utils.Must(llmagent.New(llmagent.Config{
			Name:      "Parent",
			Model:     llm,
			SubAgents: []agent.Agent{curAgent, peer},
		}))

		check(t, curAgent, root, "", nil, []string{"Parent", "Peer", "Current"})
	})

	t.Run("AgentWithParallelParent", func(t *testing.T) {
		curAgent := utils.Must(llmagent.New(llmagent.Config{
			Name:                     "Current",
			Model:                    llm,
			DisallowTransferToParent: false,
			DisallowTransferToPeers:  false,
		}))
		peer := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Peer",
			Model: llm,
		}))
		root := utils.Must(parallelagent.New(parallelagent.Config{
			AgentConfig: agent.Config{
				Name:      "Parent",
				SubAgents: []agent.Agent{curAgent, peer},
			},
		}))

		check(t, curAgent, root, "", nil, []string{"Parent", "Peer", "Current"})
	})

	t.Run("AgentWithSequentialParent", func(t *testing.T) {
		curAgent := utils.Must(llmagent.New(llmagent.Config{
			Name:                     "Current",
			Model:                    llm,
			DisallowTransferToParent: false,
			DisallowTransferToPeers:  false,
		}))
		peer := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Peer",
			Model: llm,
		}))
		root := utils.Must(sequentialagent.New(sequentialagent.Config{
			AgentConfig: agent.Config{
				Name:      "Parent",
				SubAgents: []agent.Agent{curAgent, peer},
			},
		}))

		check(t, curAgent, root, "", nil, []string{"Parent", "Peer", "Current"})
	})

	t.Run("AgentWithSequentialSubagent", func(t *testing.T) {
		seqSub := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Sub3",
			Model: llm,
		}))
		agent := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Current",
			Model: llm,
			SubAgents: []agent.Agent{
				utils.Must(sequentialagent.New(sequentialagent.Config{
					AgentConfig: agent.Config{
						Name:      "Sub1",
						SubAgents: []agent.Agent{seqSub},
					},
				})),
				utils.Must(llmagent.New(llmagent.Config{
					Name:  "Sub2",
					Model: llm,
				})),
			},
		}))
		check(t, agent, agent, "", []string{"Sub1", "Sub2"}, []string{"Current"})
	})
}

func TestAgentTransfer_ProcessRequest(t *testing.T) {
	// First Tool
	var req model.LLMRequest
	handler := func(ctx tool.Context, x int) (int, error) {
		return x, nil
	}
	identityTool, err := functiontool.New(functiontool.Config{
		Name:        "identity",
		Description: "returns the input value",
	}, handler)
	if err != nil {
		panic(err)
	}
	requestProcessor, ok := identityTool.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatal("identityTool does not implement itype.RequestProcessor")
	}
	if err := requestProcessor.ProcessRequest(nil, &req); err != nil {
		t.Fatalf("identityTool.ProcessRequest failed: %v", err)
	}
	// Second tool
	transferToAgentTool := &llminternal.TransferToAgentTool{}
	if err := transferToAgentTool.ProcessRequest(nil, &req); err != nil {
		t.Fatalf("transferToAgentTool.ProcessRequest failed: %v", err)
	}

	if len(req.Config.Tools) != 1 {
		t.Errorf("number of tools should be one, got: %d", len(req.Config.Tools))
	}
	if len(req.Config.Tools[0].FunctionDeclarations) != 2 {
		t.Errorf("number of function declarations should be two, got: %d", len(req.Config.Tools[0].FunctionDeclarations))
	}
}

func TestTransferToAgentToolRun(t *testing.T) {
	t.Run("Success", func(t *testing.T) {
		curTool := &llminternal.TransferToAgentTool{}

		invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
		ctx := toolinternal.NewToolContext(invCtx, "", &session.EventActions{})

		wantAgentName := "TestAgent"
		args := map[string]any{"agent_name": wantAgentName}
		if _, err := curTool.Run(ctx, args); err != nil {
			t.Fatalf("Run(%v) failed: %v", args, err)
		}
		if got, want := ctx.Actions().TransferToAgent, wantAgentName; got != want {
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
				curTool := &llminternal.TransferToAgentTool{}
				ctx := toolinternal.NewToolContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}), "", nil)
				if got, err := curTool.Run(ctx, tc.args); err == nil {
					t.Fatalf("Run(%v) = (%v, %v), want error", tc.args, got, err)
				}
			})
		}
	})
}

func stringify(v any) string {
	s, _ := json.Marshal(v)
	return string(s)
}
