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
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/exp/feature"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

func TestAgentTransferRequestProcessor(t *testing.T) {
	curTool := &llminternal.TransferToAgentTool{}
	llm := &struct{ model.LLM }{}

	if curTool.Name() == "" || curTool.Description() == "" || curTool.Declaration() == nil {
		t.Fatalf("unexpected TransferToAgentTool: name=%q, desc=%q, decl=%v", curTool.Name(), curTool.Description(), curTool)
	}

	check := func(t *testing.T, curAgent, root agent.Agent, wantParent string, wantAgents, unwantAgents []string) {
		t.Helper()
		req := &model.LLMRequest{}

		parents, err := parentmap.New(root)
		if err != nil {
			t.Fatal(err)
		}

		ctx := icontext.NewInvocationContext(parentmap.ToContext(t.Context(), parents), icontext.InvocationContextParams{
			Agent: curAgent,
		})

		for ev, err := range llminternal.AgentTransferRequestProcessor(ctx, req, &llminternal.Flow{}) {
			if ev != nil {
				t.Fatal("AgentTransferRequestProcessor generated an unexpected event")
			}
			if err != nil {
				t.Fatalf("AgentTransferRequestProcessor failed: %v", err)
			}
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
		if len(wantAgents) > 0 {
			transferTool, ok := gotTool.(*llminternal.TransferToAgentTool)
			if !ok {
				t.Fatalf("failed to type convert tool %v, got %T", wantToolName, gotTool)
			}
			declaration := transferTool.Declaration()
			gotEnums := jsonEnum(t, declaration)
			wantEnums := slices.Clone(wantAgents)
			// Add parent to the list of agents to transfer to, if not already present.
			if wantParent != "" && !slices.Contains(wantAgents, wantParent) {
				wantEnums = append(wantEnums, wantParent)
			}
			if diff := cmp.Diff(wantEnums, gotEnums, cmpopts.SortSlices(func(a, b string) bool { return a < b })); diff != "" {
				t.Fatalf("failed to set agent_name enums (-want, +got): %v", diff)
			}
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
			return f.Name == wantToolName && strings.Contains(f.Description, wantToolDescription) && f.ParametersJsonSchema != nil && f.Parameters == nil
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

	t.Run("TaskAndSingleTurnSubagentsExcludedAsTargets", func(t *testing.T) {
		taskSub := utils.Must(llmagent.New(llmagent.Config{
			Name:  "TaskSub",
			Model: llm,
			Mode:  llmagent.ModeTask,
		}))
		singleTurnSub := utils.Must(llmagent.New(llmagent.Config{
			Name:  "SingleTurnSub",
			Model: llm,
			Mode:  llmagent.ModeSingleTurn,
		}))
		chatSub := utils.Must(llmagent.New(llmagent.Config{
			Name:  "ChatSub",
			Model: llm,
		}))
		curAgent := utils.Must(llmagent.New(llmagent.Config{
			Name:      "Current",
			Model:     llm,
			SubAgents: []agent.Agent{taskSub, singleTurnSub, chatSub},
		}))
		check(t, curAgent, curAgent, "",
			[]string{"ChatSub"},
			[]string{"TaskSub", "SingleTurnSub", "Current"})
	})

	t.Run("TaskAndSingleTurnPeersExcludedAsTargets", func(t *testing.T) {
		curAgent := utils.Must(llmagent.New(llmagent.Config{
			Name:  "Current",
			Model: llm,
		}))
		taskPeer := utils.Must(llmagent.New(llmagent.Config{
			Name:  "TaskPeer",
			Model: llm,
			Mode:  llmagent.ModeTask,
		}))
		singleTurnPeer := utils.Must(llmagent.New(llmagent.Config{
			Name:  "SingleTurnPeer",
			Model: llm,
			Mode:  llmagent.ModeSingleTurn,
		}))
		chatPeer := utils.Must(llmagent.New(llmagent.Config{
			Name:  "ChatPeer",
			Model: llm,
		}))
		root := utils.Must(llmagent.New(llmagent.Config{
			Name:      "Parent",
			Model:     llm,
			SubAgents: []agent.Agent{curAgent, taskPeer, singleTurnPeer, chatPeer},
		}))
		check(t, curAgent, root, "Parent",
			[]string{"ChatPeer"},
			[]string{"TaskPeer", "SingleTurnPeer", "Current"})
	})
}

// TestAgentTransferRequestProcessor_TaskSingleTurnCurrentAgent verifies
// that LLM doesn't see transfer guidance for task / single_turn agents
// because they reach their callees via FC delegation (TaskAgentTool /
// SingleTurnTool), not via transfer.
func TestAgentTransferRequestProcessor_TaskSingleTurnCurrentAgent(t *testing.T) {
	llm := &struct{ model.LLM }{}

	for _, tc := range []struct {
		name string
		mode llmagent.Mode
	}{
		{"task", llmagent.ModeTask},
		{"single_turn", llmagent.ModeSingleTurn},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cur := utils.Must(llmagent.New(llmagent.Config{
				Name:  "Current",
				Model: llm,
				Mode:  tc.mode,
			}))
			root := utils.Must(llmagent.New(llmagent.Config{
				Name:      "Parent",
				Model:     llm,
				SubAgents: []agent.Agent{cur},
			}))

			req := &model.LLMRequest{}
			parents, err := parentmap.New(root)
			if err != nil {
				t.Fatal(err)
			}
			ctx := icontext.NewInvocationContext(
				parentmap.ToContext(t.Context(), parents),
				icontext.InvocationContextParams{Agent: cur},
			)

			for ev, err := range llminternal.AgentTransferRequestProcessor(ctx, req, &llminternal.Flow{}) {
				if ev != nil {
					t.Fatal("AgentTransferRequestProcessor generated an unexpected event")
				}
				if err != nil {
					t.Fatalf("AgentTransferRequestProcessor failed: %v", err)
				}
			}

			// No transfer-prose instructions should be appended.
			// The processor's only path to instruction text is
			// instructionsForTransferToAgent, which returns ""
			// for task/single_turn; AppendInstructions("") is a
			// no-op, leaving Config.SystemInstruction empty.
			if req.Config != nil {
				if got := utils.TextParts(req.Config.SystemInstruction); len(got) > 0 {
					for _, s := range got {
						if s != "" {
							t.Errorf("expected no transfer instructions for mode=%q, got: %q", tc.mode, s)
						}
					}
				}
			}
		})
	}
}

func TestAgentTransfer_ProcessRequest(t *testing.T) {
	// First Tool
	type Input struct {
		x int
	}
	var req model.LLMRequest
	handler := func(ctx agent.Context, input Input) (int, error) {
		return input.x, nil
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
	// newTool returns the transfer tool of an agent with a single sub-agent,
	// "TestAgent".
	newTool := func(t *testing.T) *llminternal.TransferToAgentTool {
		t.Helper()
		llm := &struct{ model.LLM }{}
		sub, err := llmagent.New(llmagent.Config{Name: "TestAgent", Model: llm})
		if err != nil {
			t.Fatal(err)
		}
		root, err := llmagent.New(llmagent.Config{Name: "Root", Model: llm, SubAgents: []agent.Agent{sub}})
		if err != nil {
			t.Fatal(err)
		}
		invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: root})
		curTool, err := llminternal.NewTransferToAgentTool(invCtx, root, nil, []agent.Agent{sub})
		if err != nil {
			t.Fatal(err)
		}
		return curTool
	}

	t.Run("Success", func(t *testing.T) {
		curTool := newTool(t)

		invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
		ctx := agent.NewToolContext(invCtx, "", &session.EventActions{}, nil)

		wantAgentName := "TestAgent"
		args := map[string]any{"agent_name": wantAgentName}
		if _, err := curTool.Run(ctx, args); err != nil {
			t.Fatalf("Run(%v) failed: %v", args, err)
		}
		if got, want := ctx.Actions().TransferToAgent, wantAgentName; got != want {
			t.Errorf("Run(%v) did not set TransferToAgent, got %q, want %q", args, got, want)
		}
	})

	t.Run("UnknownAgent", func(t *testing.T) {
		curTool := newTool(t)

		invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
		ctx := agent.NewToolContext(invCtx, "", &session.EventActions{}, nil)

		args := map[string]any{"agent_name": "nobody"}
		_, err := curTool.Run(ctx, args)
		if err == nil || !strings.Contains(err.Error(), "TestAgent") {
			t.Fatalf("Run(%v) error = %v, want an error naming the available agents", args, err)
		}
		if got := ctx.Actions().TransferToAgent; got != "" {
			t.Errorf("Run(%v) set TransferToAgent = %q, want it unset", args, got)
		}
	})

	t.Run("InvalidArguments", func(t *testing.T) {
		testCases := []struct {
			name    string
			args    map[string]any
			wantErr string
		}{
			{name: "NoAgentName", args: map[string]any{}, wantErr: "empty agent_name"},
			{name: "NilArg", args: nil, wantErr: "empty agent_name"},
			{name: "InvalidType", args: map[string]any{"agent_name": 123}, wantErr: "empty agent_name"},
			{name: "InvalidValue", args: map[string]any{"agent_name": ""}, wantErr: "empty agent_name"},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				curTool := newTool(t)
				ctx := agent.NewToolContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}), "", nil, nil)
				if got, err := curTool.Run(ctx, tc.args); err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("Run(%v) = (%v, %v), want error containing %q", tc.args, got, err, tc.wantErr)
				}
			})
		}
	})
}

// jsonEnum returns the agent_name enum of a JSON-form transfer_to_agent
// declaration, as strings.
func jsonEnum(t *testing.T, decl *genai.FunctionDeclaration) []string {
	t.Helper()
	schema, ok := decl.ParametersJsonSchema.(*jsonschema.Schema)
	if !ok {
		t.Fatalf("ParametersJsonSchema is %T, want *jsonschema.Schema", decl.ParametersJsonSchema)
	}
	var names []string
	for _, v := range schema.Properties["agent_name"].Enum {
		name, ok := v.(string)
		if !ok {
			t.Fatalf("agent_name enum value is %T, want string", v)
		}
		names = append(names, name)
	}
	return names
}

func stringify(v any) string {
	s, _ := json.Marshal(v)
	return string(s)
}

// transferInstruction returns the system instruction text of req.
func transferInstruction(req *model.LLMRequest) string {
	if req.Config == nil {
		return ""
	}
	return strings.Join(utils.TextParts(req.Config.SystemInstruction), "\n")
}

// Agent names and descriptions reach the transfer instruction exactly as
// given, as adk-python's f-strings write them.
// Fails when the template HTML-escapes its data (rows "special characters"
// and "special characters in the name"), when data is parsed as template
// text (row "template action"), or when an empty description drops its line.
func TestAgentTransferInstruction_KeepsNamesAndDescriptionsRaw(t *testing.T) {
	m := searchGuardModel{name: "unused"}
	for _, tc := range []struct {
		name        string
		subName     string
		description string
		want        []string
		wantAbsent  []string
	}{
		{
			name:        "special characters",
			subName:     "sub",
			description: `a<b>&'"c`,
			want:        []string{"Agent name: sub\nAgent description: a<b>&'\"c\n"},
		},
		{
			name:        "template action",
			subName:     "sub",
			description: "{{.AgentName}}",
			want:        []string{"Agent description: {{.AgentName}}\n"},
		},
		{
			name:    "empty description",
			subName: "sub",
			want:    []string{"Agent name: sub\nAgent description: \n\n"},
		},
		{
			name:       "special characters in the name",
			subName:    "r&d<1>",
			want:       []string{"Agent name: r&d<1>\n", "function are\n`r&d<1>`.\n"},
			wantAbsent: []string{"transfer to your parent agent"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sub := utils.Must(llmagent.New(llmagent.Config{Name: tc.subName, Description: tc.description, Model: m}))
			top := utils.Must(llmagent.New(llmagent.Config{Name: "top", Model: m, SubAgents: []agent.Agent{sub}}))

			got := transferInstruction(transferRequest(t, t.Context(), top))

			for _, want := range tc.want {
				if !strings.Contains(got, want) {
					t.Errorf("instruction does not contain %q; got:\n%s", want, got)
				}
			}
			for _, absent := range tc.wantAbsent {
				if strings.Contains(got, absent) {
					t.Errorf("instruction contains %q; got:\n%s", absent, got)
				}
			}
		})
	}
}

// The parent line of the transfer instruction names the parent only when the
// parent is a transfer target. A non-LLM parent is not one, wherever the
// parent comes from.
// Fails when the line is written whenever a parent exists, as adk-python
// writes it (_agent_transfer.py:226-229).
func TestAgentTransferInstruction_NonLLMParent(t *testing.T) {
	build := func(t *testing.T) (cur, seq agent.Agent) {
		m := searchGuardModel{name: "unused"}
		sub := utils.Must(llmagent.New(llmagent.Config{Name: "sub", Model: m}))
		cur = utils.Must(llmagent.New(llmagent.Config{Name: "cur", Model: m, SubAgents: []agent.Agent{sub}}))
		seq = utils.Must(sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "seq", SubAgents: []agent.Agent{cur}}}))
		return cur, seq
	}
	check := func(t *testing.T, req *model.LLMRequest) {
		t.Helper()
		if diff := cmp.Diff([]string{"sub"}, transferEnum(t, req)); diff != "" {
			t.Errorf("transfer_to_agent enum mismatch (-want +got):\n%s", diff)
		}
		got := transferInstruction(req)
		if want := "**NOTE**: the only available agents for `transfer_to_agent` function are\n`sub`."; !strings.Contains(got, want) {
			t.Errorf("instruction does not contain %q; got:\n%s", want, got)
		}
		for _, absent := range []string{"transfer to your parent agent", "seq"} {
			if strings.Contains(got, absent) {
				t.Errorf("instruction contains %q; got:\n%s", absent, got)
			}
		}
	}

	t.Run("ParentFromMap", func(t *testing.T) {
		cur, seq := build(t)
		check(t, transferRequest(t, withParents(t, t.Context(), seq), cur))
	})
	t.Run("ParentFromLink", func(t *testing.T) {
		cur, _ := build(t)
		check(t, transferRequest(t, t.Context(), cur))
	})
}

// An agent that can transfer to its LLM parent is told to fall back to it,
// as adk-python's parent line does.
// Fails when the parent line is never written.
func TestAgentTransferInstruction_LLMParentLine(t *testing.T) {
	m := searchGuardModel{name: "unused"}
	cur := utils.Must(llmagent.New(llmagent.Config{Name: "cur", Model: m}))
	utils.Must(llmagent.New(llmagent.Config{Name: "root", Model: m, SubAgents: []agent.Agent{cur}}))

	got := transferInstruction(transferRequest(t, t.Context(), cur))

	if want := "\nIf neither you nor the other agents are best for the question, transfer to your parent agent root.\n"; !strings.Contains(got, want) {
		t.Errorf("instruction does not contain %q; got:\n%s", want, got)
	}
}

// panickyDescriptionAgent is an agent whose Description panics, which makes
// the transfer instruction's template fail.
type panickyDescriptionAgent struct{ agent.Agent }

func (panickyDescriptionAgent) Description() string { panic("boom") }

// When the transfer instruction cannot be built, the processor yields the
// error and leaves the request as it was.
// Fails when the template error is ignored, or when the tool is added to the
// request before the instruction is built.
func TestAgentTransferRequestProcessor_TemplateErrorLeavesRequestUntouched(t *testing.T) {
	m := searchGuardModel{name: "unused"}
	sub := utils.Must(agent.New(agent.Config{Name: "bad"}))
	top := utils.Must(llmagent.New(llmagent.Config{Name: "top", Model: m, SubAgents: []agent.Agent{panickyDescriptionAgent{sub}}}))

	req, errs := runTransferProcessor(t, top, top, "unused")

	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "error calling Description: boom") {
		t.Errorf("processor errors = %v, want one containing %q", errs, "error calling Description: boom")
	}
	if diff := cmp.Diff(&model.LLMRequest{}, req); diff != "" {
		t.Errorf("request changed (-want +got):\n%s", diff)
	}
}

// transferBackendModel is a model double of a Google backend. The transfer
// processor reads only its backend.
type transferBackendModel struct {
	model.LLM
	backend genai.Backend
}

func (m transferBackendModel) GetGoogleLLMVariant() genai.Backend { return m.backend }

// transferDeclaration runs AgentTransferRequestProcessor for cur on a context
// derived from parent, with llm as the flow's model. It returns the
// transfer_to_agent declaration the request carries and the description of
// the tool the request holds.
func transferDeclaration(t *testing.T, parent context.Context, cur agent.Agent, llm model.LLM) (*genai.FunctionDeclaration, string) {
	t.Helper()
	ctx := icontext.NewInvocationContext(parent, icontext.InvocationContextParams{Agent: cur})
	req := &model.LLMRequest{}
	for _, err := range llminternal.AgentTransferRequestProcessor(ctx, req, &llminternal.Flow{Model: llm}) {
		if err != nil {
			t.Fatalf("AgentTransferRequestProcessor failed: %v", err)
		}
	}
	tt, ok := req.Tools["transfer_to_agent"].(*llminternal.TransferToAgentTool)
	if !ok {
		t.Fatalf("req.Tools[transfer_to_agent] is %T, want *llminternal.TransferToAgentTool", req.Tools["transfer_to_agent"])
	}
	decls := utils.FunctionDecls(req.Config)
	if len(decls) != 1 {
		t.Fatalf("request has %d function declarations, want 1", len(decls))
	}
	return decls[0], tt.Description()
}

// The transfer_to_agent declaration has Python's description and schema, in
// the JSON form by default and the typed form under the legacy switch, with a
// null response schema on Vertex AI only; the form follows each request.
// Fails when the description loses a line or keeps adk-go's old text, when a
// schema gains a description on agent_name or loses its root type, enum or
// required list, when the switch or the backend is ignored or read as fixed,
// or when both parameter fields are set.
func TestTransferToAgentTool_DeclarationFormsAndDescription(t *testing.T) {
	const wantDescription = "Transfer the query to another agent.\n\nUse this tool to hand off control to another agent that is more suitable to\nanswer the user's query according to the agent's description.\n\nArgs:\n  agent_name: the agent name to transfer to."
	jsonDecl := &genai.FunctionDeclaration{
		Name:        "transfer_to_agent",
		Description: wantDescription,
		ParametersJsonSchema: &jsonschema.Schema{
			Type: "object",
			Properties: map[string]*jsonschema.Schema{
				"agent_name": {Type: "string", Enum: []any{"Sub1", "Sub2"}},
			},
			Required: []string{"agent_name"},
		},
	}
	jsonVertexDecl := &genai.FunctionDeclaration{
		Name:        "transfer_to_agent",
		Description: wantDescription,
		ParametersJsonSchema: &jsonschema.Schema{
			Type: "object",
			Properties: map[string]*jsonschema.Schema{
				"agent_name": {Type: "string", Enum: []any{"Sub1", "Sub2"}},
			},
			Required: []string{"agent_name"},
		},
		ResponseJsonSchema: &jsonschema.Schema{Type: "null"},
	}
	typedDecl := &genai.FunctionDeclaration{
		Name:        "transfer_to_agent",
		Description: wantDescription,
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"agent_name": {Type: genai.TypeString, Enum: []string{"Sub1", "Sub2"}},
			},
			Required: []string{"agent_name"},
		},
	}
	typedVertexDecl := &genai.FunctionDeclaration{
		Name:        "transfer_to_agent",
		Description: wantDescription,
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"agent_name": {Type: genai.TypeString, Enum: []string{"Sub1", "Sub2"}},
			},
			Required: []string{"agent_name"},
		},
		Response: &genai.Schema{Type: genai.TypeNULL},
	}
	nonGoogle := &struct{ model.LLM }{}
	newAgent := func() agent.Agent {
		return utils.Must(llmagent.New(llmagent.Config{
			Name:  "Cur",
			Model: nonGoogle,
			SubAgents: []agent.Agent{
				utils.Must(llmagent.New(llmagent.Config{Name: "Sub1", Model: nonGoogle})),
				utils.Must(llmagent.New(llmagent.Config{Name: "Sub2", Model: nonGoogle})),
			},
		}))
	}

	tests := []struct {
		name   string
		parent context.Context
		llm    model.LLM
		want   *genai.FunctionDeclaration
	}{
		{
			name:   "default, non-Google model",
			parent: t.Context(),
			llm:    nonGoogle,
			want:   jsonDecl,
		},
		{
			name:   "switch on, non-Google model",
			parent: feature.WithLegacyFunctionDeclarationSchema(t.Context(), true),
			llm:    nonGoogle,
			want:   typedDecl,
		},
		{
			name:   "switch off, Gemini API",
			parent: feature.WithLegacyFunctionDeclarationSchema(t.Context(), false),
			llm:    transferBackendModel{backend: genai.BackendGeminiAPI},
			want:   jsonDecl,
		},
		{
			name:   "switch on, Gemini API",
			parent: feature.WithLegacyFunctionDeclarationSchema(t.Context(), true),
			llm:    transferBackendModel{backend: genai.BackendGeminiAPI},
			want:   typedDecl,
		},
		{
			name:   "switch off, Vertex AI",
			parent: feature.WithLegacyFunctionDeclarationSchema(t.Context(), false),
			llm:    transferBackendModel{backend: genai.BackendVertexAI},
			want:   jsonVertexDecl,
		},
		{
			name:   "switch on, Vertex AI",
			parent: feature.WithLegacyFunctionDeclarationSchema(t.Context(), true),
			llm:    transferBackendModel{backend: genai.BackendVertexAI},
			want:   typedVertexDecl,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, gotToolDescription := transferDeclaration(t, tc.parent, newAgent(), tc.llm)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("declaration mismatch (-want +got):\n%s", diff)
			}
			if gotToolDescription != wantDescription {
				t.Errorf("tool Description() = %q, want %q", gotToolDescription, wantDescription)
			}
		})
	}

	t.Run("one agent, three runs", func(t *testing.T) {
		cur := newAgent()
		gotDefault, _ := transferDeclaration(t, t.Context(), cur, nonGoogle)
		gotOn, _ := transferDeclaration(t, feature.WithLegacyFunctionDeclarationSchema(t.Context(), true), cur, nonGoogle)
		gotOff, _ := transferDeclaration(t, feature.WithLegacyFunctionDeclarationSchema(t.Context(), false), cur, nonGoogle)
		if diff := cmp.Diff(jsonDecl, gotDefault); diff != "" {
			t.Errorf("first run, no switch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(typedDecl, gotOn); diff != "" {
			t.Errorf("second run, switch on (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(jsonDecl, gotOff); diff != "" {
			t.Errorf("third run, switch off (-want +got):\n%s", diff)
		}
	})

	t.Run("zero value", func(t *testing.T) {
		zero := &llminternal.TransferToAgentTool{}
		want := &genai.FunctionDeclaration{
			Name:        "transfer_to_agent",
			Description: wantDescription,
			ParametersJsonSchema: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{
					"agent_name": {Type: "string"},
				},
				Required: []string{"agent_name"},
			},
		}
		if diff := cmp.Diff(want, zero.Declaration()); diff != "" {
			t.Errorf("zero value Declaration() mismatch (-want +got):\n%s", diff)
		}
		if got := zero.Description(); got != wantDescription {
			t.Errorf("zero value Description() = %q, want %q", got, wantDescription)
		}
	})
}
