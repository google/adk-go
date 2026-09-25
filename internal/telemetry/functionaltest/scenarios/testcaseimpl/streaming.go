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
)

// newStreaming has no tool call, unlike adk-python's: MockModel streams every
// call in StreamResponsesCount chunks, a tool call included.
func newStreaming(t *testing.T) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{
		Name:        "some_root_agent",
		Description: "A sample root agent.",
		Model: &testutil.MockModel{
			Responses: []*genai.Content{
				{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "text "}}},
				{Role: genai.RoleModel, Parts: []*genai.Part{{Text: "response"}}},
			},
			StreamResponsesCount: 2,
		},
		Instruction: "you are helpful",
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	return a
}
