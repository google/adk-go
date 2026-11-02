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

package llminternal

import (
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

type toolsProcessorMockTool struct {
	name string
}

func (m *toolsProcessorMockTool) Name() string        { return m.name }
func (m *toolsProcessorMockTool) Description() string { return "mock tool" }
func (m *toolsProcessorMockTool) IsLongRunning() bool { return false }
func (m *toolsProcessorMockTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: m.name}
}

func (m *toolsProcessorMockTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	return nil, nil
}

type dynamicTestToolset struct {
	calls []tool.Tool
	index int
}

func (d *dynamicTestToolset) Name() string { return "dynamic_toolset" }
func (d *dynamicTestToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	if d.index < len(d.calls) {
		t := d.calls[d.index]
		d.index++
		return []tool.Tool{t}, nil
	}
	return nil, nil
}

func TestToolProcessor_SpareCapacityNotCorrupted(t *testing.T) {
	baseTool := &toolsProcessorMockTool{name: "base_tool"}
	toolA := &toolsProcessorMockTool{name: "tool_a"}
	toolB := &toolsProcessorMockTool{name: "tool_b"}

	toolset := &dynamicTestToolset{
		calls: []tool.Tool{toolA, toolB},
	}

	// State.Tools built as make([]tool.Tool, 1, 2)
	storedTools := make([]tool.Tool, 1, 2)
	storedTools[0] = baseTool

	baseAgent := utils.Must(agent.New(agent.Config{Name: "test_agent"}))
	mockAgent := &mockLLMAgent{
		Agent: baseAgent,
		s: &State{
			Tools:    storedTools,
			Toolsets: []tool.Toolset{toolset},
		},
	}

	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent: mockAgent,
	})
	req := &model.LLMRequest{}

	// Run toolProcessor for flow 1
	flow1 := &Flow{}
	for _, err := range toolProcessor(ctx, req, flow1) {
		if err != nil {
			t.Fatalf("flow 1 toolProcessor error: %v", err)
		}
	}

	// Run toolProcessor for flow 2
	flow2 := &Flow{}
	for _, err := range toolProcessor(ctx, req, flow2) {
		if err != nil {
			t.Fatalf("flow 2 toolProcessor error: %v", err)
		}
	}

	// Check that the spare slot in storedTools is still nil
	storedWithCap := storedTools[:2]
	if storedWithCap[1] != nil {
		t.Errorf("storedTools spare slot was overwritten: expected nil, got %v", storedWithCap[1].Name())
	}

	// Check that flow 1 got its own tool
	if len(flow1.Tools) != 2 {
		t.Fatalf("expected flow1 to have 2 tools, got %d", len(flow1.Tools))
	}
	if flow1.Tools[0].Name() != "base_tool" || flow1.Tools[1].Name() != "tool_a" {
		t.Errorf("unexpected flow1 tools: %v, %v", flow1.Tools[0].Name(), flow1.Tools[1].Name())
	}

	// Check that flow 2 got its own tool
	if len(flow2.Tools) != 2 {
		t.Fatalf("expected flow2 to have 2 tools, got %d", len(flow2.Tools))
	}
	if flow2.Tools[0].Name() != "base_tool" || flow2.Tools[1].Name() != "tool_b" {
		t.Errorf("unexpected flow2 tools: %v, %v", flow2.Tools[0].Name(), flow2.Tools[1].Name())
	}
}
