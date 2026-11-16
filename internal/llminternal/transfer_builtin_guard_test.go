// Copyright 2026 Google LLC
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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/agent/parentmap"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/geminitool"
)

// searchGuardModel is a model double that reports a fixed name. The
// processor reads nothing else from it.
type searchGuardModel struct {
	model.LLM
	name string
}

func (m searchGuardModel) Name() string { return m.name }

// runTransferProcessor runs AgentTransferRequestProcessor for cur, with the
// parent map of root in the context and a Flow whose model is named
// modelName. It returns the request and the errors the processor yielded.
func runTransferProcessor(t *testing.T, cur, root agent.Agent, modelName string) (*model.LLMRequest, []error) {
	t.Helper()
	parents, err := parentmap.New(root)
	if err != nil {
		t.Fatal(err)
	}
	ctx := icontext.NewInvocationContext(parentmap.ToContext(t.Context(), parents), icontext.InvocationContextParams{Agent: cur})
	req := &model.LLMRequest{}
	var errs []error
	for ev, err := range llminternal.AgentTransferRequestProcessor(ctx, req, &llminternal.Flow{Model: searchGuardModel{name: modelName}}) {
		if ev != nil {
			t.Fatalf("AgentTransferRequestProcessor yielded an event: %v", ev)
		}
		if err != nil {
			errs = append(errs, err)
		}
	}
	return req, errs
}

// transferEnum returns the agent names the request's transfer_to_agent
// declaration offers, or nil when the request has no such tool.
func transferEnum(t *testing.T, req *model.LLMRequest) []string {
	t.Helper()
	raw, ok := req.Tools["transfer_to_agent"]
	if !ok {
		return nil
	}
	tt, ok := raw.(*llminternal.TransferToAgentTool)
	if !ok {
		t.Fatalf("req.Tools[transfer_to_agent] is %T, want *llminternal.TransferToAgentTool", raw)
	}
	return tt.Declaration().Parameters.Properties["agent_name"].Enum
}

// hasTransferInstruction reports whether any system instruction of req
// mentions transfer_to_agent.
func hasTransferInstruction(req *model.LLMRequest) bool {
	if req.Config == nil {
		return false
	}
	for _, s := range utils.TextParts(req.Config.SystemInstruction) {
		if strings.Contains(s, "transfer_to_agent") {
			return true
		}
	}
	return false
}

// newSearchWorker returns worker, holding tools and running in mode, as the
// only sub-agent of an LLM agent root.
func newSearchWorker(t *testing.T, mode llmagent.Mode, tools ...tool.Tool) (worker, root agent.Agent) {
	t.Helper()
	m := searchGuardModel{name: "unused"}
	worker = utils.Must(llmagent.New(llmagent.Config{Name: "worker", Model: m, Mode: mode, Tools: tools}))
	root = utils.Must(llmagent.New(llmagent.Config{Name: "root", Model: m, SubAgents: []agent.Agent{worker}}))
	return worker, root
}

// newSearchRoot returns root, holding tools, with one sub-agent sub.
func newSearchRoot(t *testing.T, tools ...tool.Tool) agent.Agent {
	t.Helper()
	m := searchGuardModel{name: "unused"}
	sub := utils.Must(llmagent.New(llmagent.Config{Name: "sub", Model: m}))
	return utils.Must(llmagent.New(llmagent.Config{Name: "root", Model: m, Tools: tools, SubAgents: []agent.Agent{sub}}))
}

// Fails when the guard ignores the model (the first case gets the tool) or
// always omits the tool (the second case loses it).
func TestAgentTransferRequestProcessor_EnterpriseWebSearchOnOlderModel(t *testing.T) {
	t.Run("gemini-2.0-flash", func(t *testing.T) {
		worker, root := newSearchWorker(t, llmagent.ModeUnset, geminitool.EnterpriseWebSearch{})

		req, errs := runTransferProcessor(t, worker, root, "gemini-2.0-flash")

		if len(errs) != 0 {
			t.Fatalf("processor errors = %v, want none", errs)
		}
		if got := transferEnum(t, req); got != nil {
			t.Errorf("transfer_to_agent offers %q, want no transfer_to_agent tool", got)
		}
		if hasTransferInstruction(req) {
			t.Errorf("system instruction mentions transfer_to_agent, want no transfer instruction")
		}
	})
	t.Run("gemini-3-pro-preview", func(t *testing.T) {
		worker, root := newSearchWorker(t, llmagent.ModeUnset, geminitool.EnterpriseWebSearch{})

		req, errs := runTransferProcessor(t, worker, root, "gemini-3-pro-preview")

		if len(errs) != 0 {
			t.Fatalf("processor errors = %v, want none", errs)
		}
		if diff := cmp.Diff([]string{"root"}, transferEnum(t, req)); diff != "" {
			t.Errorf("transfer_to_agent enum mismatch (-want +got):\n%s", diff)
		}
	})
}

// Fails when GoogleSearch is not recognized as an incompatible built-in
// search tool, in any mode of the worker.
func TestAgentTransferRequestProcessor_GoogleSearchWorkerOnOlderModel(t *testing.T) {
	for _, tc := range []struct {
		name string
		mode llmagent.Mode
	}{
		{"single_turn", llmagent.ModeSingleTurn},
		{"task", llmagent.ModeTask},
		{"chat", llmagent.ModeUnset},
	} {
		t.Run(tc.name, func(t *testing.T) {
			worker, root := newSearchWorker(t, tc.mode, geminitool.GoogleSearch{})

			req, errs := runTransferProcessor(t, worker, root, "gemini-2.5-flash")

			if len(errs) != 0 {
				t.Fatalf("processor errors = %v, want none", errs)
			}
			if got := transferEnum(t, req); got != nil {
				t.Errorf("transfer_to_agent offers %q, want no transfer_to_agent tool", got)
			}
		})
	}
}

// Fails when the guard counts every sub-agent of cur, not only the sub-agents
// that are transfer targets: a single_turn or task sub-agent is not one, so the
// request goes on without the tool instead of failing.
func TestAgentTransferRequestProcessor_GoogleSearchWithNonTargetSubAgentOnOlderModel(t *testing.T) {
	for _, tc := range []struct {
		name string
		mode llmagent.Mode
	}{
		{"single_turn", llmagent.ModeSingleTurn},
		{"task", llmagent.ModeTask},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := searchGuardModel{name: "unused"}
			helper := utils.Must(llmagent.New(llmagent.Config{Name: "helper", Model: m, Mode: tc.mode}))
			worker := utils.Must(llmagent.New(llmagent.Config{
				Name: "worker", Model: m, Tools: []tool.Tool{geminitool.GoogleSearch{}}, SubAgents: []agent.Agent{helper},
			}))
			root := utils.Must(llmagent.New(llmagent.Config{Name: "root", Model: m, SubAgents: []agent.Agent{worker}}))

			req, errs := runTransferProcessor(t, worker, root, "gemini-2.5-flash")

			if len(errs) != 0 {
				t.Fatalf("processor errors = %v, want none", errs)
			}
			if got := transferEnum(t, req); got != nil {
				t.Errorf("transfer_to_agent offers %q, want no transfer_to_agent tool", got)
			}
			if hasTransferInstruction(req) {
				t.Errorf("system instruction mentions transfer_to_agent, want no transfer instruction")
			}
		})
	}
}

// Fails when the sub-agent condition is dropped (the request goes on without
// the tool instead of failing) or when the error is not yielded.
func TestAgentTransferRequestProcessor_BuiltinSearchWithSubAgentTargetFails(t *testing.T) {
	for _, tc := range []struct {
		name    string
		tool    tool.Tool
		wantMsg string
	}{
		{"GoogleSearch", geminitool.GoogleSearch{}, "Agent 'root' has sub-agent transfer targets but is configured with GoogleSearchTool"},
		{"*GoogleSearch", &geminitool.GoogleSearch{}, "Agent 'root' has sub-agent transfer targets but is configured with GoogleSearchTool"},
		{"EnterpriseWebSearch", geminitool.EnterpriseWebSearch{}, "Agent 'root' has sub-agent transfer targets but is configured with EnterpriseWebSearchTool"},
		{"*EnterpriseWebSearch", &geminitool.EnterpriseWebSearch{}, "Agent 'root' has sub-agent transfer targets but is configured with EnterpriseWebSearchTool"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := newSearchRoot(t, tc.tool)

			req, errs := runTransferProcessor(t, root, root, "gemini-2.0-flash")

			if len(errs) != 1 {
				t.Fatalf("processor yielded %d errors (%v), want 1", len(errs), errs)
			}
			if got := errs[0].Error(); !strings.Contains(got, tc.wantMsg) {
				t.Errorf("error = %q, want it to contain %q", got, tc.wantMsg)
			}
			if len(req.Tools) != 0 {
				t.Errorf("req.Tools = %v, want empty", req.Tools)
			}
		})
	}
}

// Fails when the Gemini 3 test is wrong at the version boundary, ignores the
// path form of a model name, accepts a name that is not Gemini by Python's
// rule (a "projects/" path without a publisher is not), or reads a major
// version from a version Python cannot parse ("3.x").
func TestAgentTransferRequestProcessor_BuiltinSearchOnGemini3KeepsTransfer(t *testing.T) {
	for _, tc := range []struct {
		model    string
		wantEnum []string
		wantErr  bool
	}{
		{model: "gemini-3.1-pro-preview", wantEnum: []string{"sub"}},
		{model: "projects/p/locations/l/publishers/google/models/gemini-3-flash", wantEnum: []string{"sub"}},
		{model: "projects/p/locations/l/models/gemini-3-flash", wantErr: true},
		{model: "gemini-3.x-flash", wantErr: true},
		{model: "claude-sonnet", wantErr: true},
		{model: "", wantErr: true},
	} {
		t.Run(tc.model, func(t *testing.T) {
			root := newSearchRoot(t, geminitool.GoogleSearch{})

			req, errs := runTransferProcessor(t, root, root, tc.model)

			if gotErr := len(errs) != 0; gotErr != tc.wantErr {
				t.Fatalf("processor errors = %v, want error: %t", errs, tc.wantErr)
			}
			if diff := cmp.Diff(tc.wantEnum, transferEnum(t, req)); diff != "" {
				t.Errorf("transfer_to_agent enum mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// Fails when the guard fires on a tool other than Google Search and
// Enterprise Web Search.
func TestAgentTransferRequestProcessor_WithoutBuiltinSearchUnchanged(t *testing.T) {
	echo, err := functiontool.New(functiontool.Config{Name: "echo", Description: "echoes its input"},
		func(agent.Context, struct{}) (map[string]any, error) { return nil, nil })
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		tool tool.Tool
	}{
		{"function tool", echo},
		{"GoogleMaps", geminitool.GoogleMaps{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := newSearchRoot(t, tc.tool)

			req, errs := runTransferProcessor(t, root, root, "gemini-2.0-flash")

			if len(errs) != 0 {
				t.Fatalf("processor errors = %v, want none", errs)
			}
			if diff := cmp.Diff([]string{"sub"}, transferEnum(t, req)); diff != "" {
				t.Errorf("transfer_to_agent enum mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
