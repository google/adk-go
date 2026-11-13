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

package agent_test

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
)

func TestBeforeModelCallbackMemoryErrorEndsRun(t *testing.T) {
	const want = "cannot add memory: memory service is not available"
	llm := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("reply", genai.RoleModel)}}
	a, err := llmagent.New(llmagent.Config{
		Name:  "test-agent",
		Model: llm,
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
			return nil, agent.AddMemory(ctx, []*memory.Entry{{ID: "m"}}, nil)
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	// The test runner configures no memory service.
	r := testutil.NewTestAgentRunner(t, a)

	var gotErr error
	for _, err := range r.Run(t, "sess-1", "hi") {
		if err != nil {
			gotErr = err
			break
		}
	}
	if gotErr == nil || gotErr.Error() != want {
		t.Errorf("run error = %v, want %q", gotErr, want)
	}
	if len(llm.Requests) != 0 {
		t.Errorf("model received %d requests, want 0", len(llm.Requests))
	}
}
