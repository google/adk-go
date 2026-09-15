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

package llmagent

import (
	"testing"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type (
	dummyArgs   struct{}
	dummyResult struct{}
)

func dummyHandler(ctx agent.Context, args dummyArgs) (dummyResult, error) {
	return dummyResult{}, nil
}

func TestNew_ClonesToolsSlice(t *testing.T) {
	fakeTool, err := functiontool.New(functiontool.Config{Name: "test_tool"}, dummyHandler)
	if err != nil {
		t.Fatalf("failed to create tool: %v", err)
	}

	// Create slice with spare capacity to test backing array aliasing.
	origTools := make([]tool.Tool, 1, 4)
	origTools[0] = fakeTool

	agentInterface, err := New(Config{
		Name:  "test_agent",
		Tools: origTools,
	})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	a, ok := agentInterface.(*llmAgent)
	if !ok {
		t.Fatalf("expected *llmAgent, got %T", agentInterface)
	}

	if len(a.State.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(a.State.Tools))
	}

	// Verify that the agent's Tools slice does not share the backing array with the caller.
	if &a.State.Tools[0] == &origTools[0] {
		t.Errorf("expected Tools slice to be cloned, but it shares the same backing array with caller")
	}

	// Appending to the caller's slice should not affect the agent's internal Tools slice.
	extraTool, err := functiontool.New(functiontool.Config{Name: "extra_tool"}, dummyHandler)
	if err != nil {
		t.Fatalf("failed to create extra tool: %v", err)
	}
	origTools = append(origTools, extraTool)

	if len(a.State.Tools) != 1 {
		t.Errorf("expected agent Tools length to remain 1 after mutating original slice, got %d", len(a.State.Tools))
	}
}
