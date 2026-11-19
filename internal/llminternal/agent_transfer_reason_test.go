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
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestTransferToAgentTool_BuiltInNeverRecordsReason(t *testing.T) {
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
	builtIn, err := llminternal.NewTransferToAgentTool(invCtx, root, nil, []agent.Agent{sub})
	if err != nil {
		t.Fatal(err)
	}

	ctx := agent.NewToolContext(invCtx, "", &session.EventActions{}, nil)
	args := map[string]any{"agent_name": "TestAgent", "transfer_reason": "x"}
	if _, err := builtIn.Run(ctx, args); err != nil {
		t.Fatalf("Run(%v) error = %v", args, err)
	}
	if got := ctx.Actions().TransferToAgent; got != "TestAgent" {
		t.Errorf("TransferToAgent = %q, want %q", got, "TestAgent")
	}
	// The flow's tool never declares transfer_reason, as adk-python's
	// (flows/llm_flows/extensions/_agent_transfer.py), so it never records one.
	if got := ctx.Actions().TransferReason; got != "" {
		t.Errorf("TransferReason = %q, want it unset", got)
	}
}
