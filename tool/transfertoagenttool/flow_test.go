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

package transfertoagenttool_test

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/transfertoagenttool"
)

// transferCall is a model turn that calls transfer_to_agent with args.
func transferCall(args map[string]any) *genai.Content {
	return genai.NewContentFromFunctionCall("transfer_to_agent", args, genai.RoleModel)
}

// transferResponseEvent returns the stored event that holds the response of
// the transfer_to_agent call in session "s" of r.
func transferResponseEvent(t *testing.T, r *testutil.TestAgentRunner) (*session.Event, *genai.FunctionResponse) {
	t.Helper()
	resp, err := r.SessionService().Get(t.Context(), &session.GetRequest{AppName: "test_app", UserID: "test_user", SessionID: "s"})
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	for ev := range resp.Session.Events().All() {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil && p.FunctionResponse.Name == "transfer_to_agent" {
				return ev, p.FunctionResponse
			}
		}
	}
	t.Fatal("the session holds no transfer_to_agent function response")
	return nil, nil
}

func TestNew_ReasonReachesPersistedEvent(t *testing.T) {
	llm := &testutil.MockModel{Responses: []*genai.Content{
		transferCall(map[string]any{"agent_name": "ghost", "transfer_reason": "because"}),
	}}
	a, err := llmagent.New(llmagent.Config{
		Name:  "solo",
		Model: llm,
		Tools: []tool.Tool{transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{"ghost"}, IncludeTransferReason: true})},
	})
	if err != nil {
		t.Fatal(err)
	}
	r := testutil.NewTestAgentRunner(t, a)

	_, err = testutil.CollectEvents(r.Run(t, "s", "hi"))
	if err == nil || !strings.Contains(err.Error(), "failed to find agent: ghost") {
		t.Errorf("run error = %v, want the flow's failed to find agent: ghost", err)
	}
	ev, fr := transferResponseEvent(t, r)
	if ev.Actions.TransferToAgent != "ghost" || ev.Actions.TransferReason != "because" {
		t.Errorf("stored event (TransferToAgent, TransferReason) = (%q, %q), want (%q, %q)", ev.Actions.TransferToAgent, ev.Actions.TransferReason, "ghost", "because")
	}
	if want := map[string]any{"result": nil}; !reflect.DeepEqual(fr.Response, want) {
		t.Errorf("function response = %#v, want %#v", fr.Response, want)
	}
}

func TestNew_DuplicateRegistrationIsRefused(t *testing.T) {
	rp := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{"a"}}).(toolinternal.RequestProcessor)
	req := &model.LLMRequest{}
	if err := rp.ProcessRequest(nil, req); err != nil {
		t.Fatalf("first ProcessRequest() error = %v", err)
	}
	err := rp.ProcessRequest(nil, req)
	if want := `tools[0] duplicate tool: "transfer_to_agent"`; err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("second ProcessRequest() error = %v, want one containing %q", err, want)
	}
	if len(req.Tools) != 1 {
		t.Errorf("len(req.Tools) = %d, want 1", len(req.Tools))
	}
	if got := len(req.Config.Tools[0].FunctionDeclarations); got != 1 {
		t.Errorf("function declarations = %d, want 1", got)
	}
}

func TestNew_PublicToolOfSubAgentTransfersToRoot(t *testing.T) {
	childLLM := &testutil.MockModel{Responses: []*genai.Content{
		transferCall(map[string]any{"agent_name": "root"}),
	}}
	// child may transfer neither to its parent nor to peers, so the flow gives
	// it no transfer tool of its own and the public tool is its only one.
	child, err := llmagent.New(llmagent.Config{
		Name:                     "child",
		Model:                    childLLM,
		DisallowTransferToParent: true,
		DisallowTransferToPeers:  true,
		Tools:                    []tool.Tool{transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{"helper"}})},
	})
	if err != nil {
		t.Fatal(err)
	}
	rootLLM := &testutil.MockModel{Responses: []*genai.Content{
		transferCall(map[string]any{"agent_name": "child"}),
	}}
	root, err := llmagent.New(llmagent.Config{Name: "root", Model: rootLLM, SubAgents: []agent.Agent{child}})
	if err != nil {
		t.Fatal(err)
	}
	r := testutil.NewTestAgentRunner(t, root)

	_, err = testutil.CollectEvents(r.Run(t, "s", "hi"))
	// The tool accepts root, an agent of child's tree; the flow then refuses
	// the transfer because root is not one of child's transfer targets.
	if !errors.Is(err, agent.ErrInvalidTransferTarget) {
		t.Errorf("run error = %v, want the flow's refusal matching agent.ErrInvalidTransferTarget", err)
	}
	resp, getErr := r.SessionService().Get(t.Context(), &session.GetRequest{AppName: "test_app", UserID: "test_user", SessionID: "s"})
	if getErr != nil {
		t.Fatalf("Get() error = %v", getErr)
	}
	var childTransfer string
	for ev := range resp.Session.Events().All() {
		if ev.Author == "child" && ev.Actions.TransferToAgent != "" {
			childTransfer = ev.Actions.TransferToAgent
		}
	}
	if childTransfer != "root" {
		t.Errorf("child's function response event has TransferToAgent = %q, want %q", childTransfer, "root")
	}
}
