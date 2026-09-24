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

package testcaseimpl

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
)

func newDelegation(t *testing.T) agent.Agent {
	t.Helper()
	delegate, err := llmagent.New(llmagent.Config{
		Name:        "some_delegate_agent",
		Description: "A delegate agent.",
		Model:       &testutil.MockModel{Responses: []*genai.Content{{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "delegate response"}}}}},
		Instruction: "you are helpful",
	})
	if err != nil {
		t.Fatalf("llmagent.New(delegate): %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name:        "some_delegating_agent",
		Description: "A delegating agent.",
		Model: &testutil.MockModel{Responses: []*genai.Content{
			{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "some_delegate_agent", Args: map[string]any{"request": "delegate this"}}}}},
			{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "text response"}}},
		}},
		Instruction: "you are helpful",
		Tools:       []tool.Tool{agenttool.New(delegate, nil)},
	})
	if err != nil {
		t.Fatalf("llmagent.New(root): %v", err)
	}
	return root
}
