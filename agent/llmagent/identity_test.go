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

package llmagent_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/parallelagent"
	"google.golang.org/adk/v2/internal/testutil"
)

func TestIdentityWithoutTransfersPreservesSharedPrompt(t *testing.T) {
	t.Parallel()

	const instruction = "Answer the question concisely."
	var models []*testutil.MockModel
	var children []agent.Agent
	for _, name := range []string{"first_worker", "second_worker"} {
		m := &testutil.MockModel{
			Responses: []*genai.Content{genai.NewContentFromText("done", genai.RoleModel)},
		}
		child, err := llmagent.New(llmagent.Config{
			Name:                     name,
			Description:              "A worker with no agent transfers.",
			Model:                    m,
			Mode:                     llmagent.ModeChat,
			Instruction:              instruction,
			DisallowTransferToParent: true,
			DisallowTransferToPeers:  true,
		})
		if err != nil {
			t.Fatalf("llmagent.New() error: %v", err)
		}
		models = append(models, m)
		children = append(children, child)
	}
	parent, err := parallelagent.New(parallelagent.Config{
		AgentConfig: agent.Config{
			Name:      "parallel_workers",
			SubAgents: children,
		},
	})
	if err != nil {
		t.Fatalf("parallelagent.New() error: %v", err)
	}

	runner := testutil.NewTestAgentRunner(t, parent)
	texts, err := testutil.CollectTextParts(runner.Run(t, "identity_session", "Hello"))
	if err != nil {
		t.Fatalf("runner.Run() error: %v", err)
	}
	if len(texts) != len(children) {
		t.Fatalf("runner returned %d text responses, want %d", len(texts), len(children))
	}

	want := genai.NewContentFromText(instruction, genai.RoleUser)
	for i, m := range models {
		if len(m.Requests) != 1 {
			t.Fatalf("worker %d made %d model requests, want 1", i, len(m.Requests))
		}
		if m.Requests[0].Config == nil || !cmp.Equal(m.Requests[0].Config.SystemInstruction, want) {
			t.Errorf("worker %d system instruction differs from the shared instruction", i)
		}
	}
}
