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
	"errors"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/telemetry/functionaltest/scenarios/testcasedata/toolcall"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

func newToolCall(t *testing.T, s toolcall.Scenario) agent.Agent {
	t.Helper()
	// A MockModel with no responses left fails the call.
	var responses []*genai.Content
	if s.Failure != toolcall.InferenceError {
		responses = []*genai.Content{
			{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "some_tool", Args: map[string]any{"arg1": "val1"}}}}},
			{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "text response"}}},
		}
	}

	type Args struct {
		Arg1 string `json:"arg1"`
	}
	sampleTool, err := functiontool.New(functiontool.Config{
		Name:        "some_tool",
		Description: "A sample tool.",
	}, func(_ agent.Context, in Args) (string, error) {
		if s.Failure == toolcall.ToolError {
			return "", errors.New("this tool always fails")
		}
		return "processed " + in.Arg1, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:        "some_root_agent",
		Description: "A sample root agent.",
		Model:       &testutil.MockModel{Responses: responses},
		Instruction: "you are helpful",
		Tools:       []tool.Tool{sampleTool},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	return a
}
