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

package configurable

import (
	"context"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

type mockSession struct{}

func (s *mockSession) ID() string                { return "mock-sess-id" }
func (s *mockSession) AppName() string           { return "mock-app" }
func (s *mockSession) UserID() string            { return "mock-user" }
func (s *mockSession) State() session.State      { return &mockState{} }
func (s *mockSession) Events() session.Events    { return nil }
func (s *mockSession) LastUpdateTime() time.Time { return time.Now() }

type mockState struct{}

func (s *mockState) Get(key string) (any, error)   { return nil, session.ErrStateKeyNotExist }
func (s *mockState) Set(key string, val any) error { return nil }
func (s *mockState) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {}
}

// upperFn converts input to uppercase and sets it as a route if it is "ALPHA" or "BETA"
func upperFn(ctx agent.Context, input any) (any, error) {
	var s string
	if input != nil {
		if val, ok := input.(string); ok {
			s = val
		} else {
			s = fmt.Sprint(input)
		}
	}
	val := stringsToUpper(s)
	if val == "ALPHA" || val == "BETA" {
		ev := session.NewEvent(ctx.InvocationID())
		ev.Output = val
		ev.Routes = []string{val}
		return ev, nil
	}
	return val, nil
}

func stringsToUpper(s string) string {
	var res []rune
	for _, r := range s {
		if r >= 'a' && r <= 'z' {
			res = append(res, r-32)
		} else {
			res = append(res, r)
		}
	}
	return string(res)
}

// suffixFn appends " done" to input
func suffixFn(ctx agent.Context, input string) (string, error) {
	return input + " done", nil
}

func init() {
	RegisterNodeFunction("upper_fn", upperFn)
	RegisterNodeFunction("suffix_fn", suffixFn)
}

type MockInvocationContext struct {
	context.Context
	sess        session.Session
	userContent *genai.Content
}

func (m *MockInvocationContext) Session() session.Session        { return m.sess }
func (m *MockInvocationContext) InvocationID() string            { return "test-inv-id" }
func (m *MockInvocationContext) UserContent() *genai.Content     { return m.userContent }
func (m *MockInvocationContext) ResumedInput(string) (any, bool) { return nil, false }
func (m *MockInvocationContext) Agent() agent.Agent              { return nil }
func (m *MockInvocationContext) Artifacts() agent.Artifacts      { return nil }
func (m *MockInvocationContext) Memory() agent.Memory            { return nil }
func (m *MockInvocationContext) Branch() string                  { return "" }
func (m *MockInvocationContext) RunConfig() *agent.RunConfig     { return nil }
func (m *MockInvocationContext) Ended() bool                     { return false }
func (m *MockInvocationContext) IsolationScope() string          { return "" }
func (m *MockInvocationContext) EndInvocation()                  {}
func (m *MockInvocationContext) WithContext(ctx context.Context) agent.InvocationContext {
	cp := *m
	cp.Context = ctx
	return &cp
}

func TestLoadWorkflowYAML(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "workflow.yaml")

	yamlContent := `
name: my_wf
agent_class: Workflow
edges:
  - - START
    - upper_fn
    - suffix_fn
`
	if err := os.WriteFile(configPath, []byte(yamlContent), 0o644); err != nil {
		t.Fatalf("failed to write temp workflow: %v", err)
	}

	ctx := context.Background()
	ag, err := FromConfig(ctx, configPath)
	if err != nil {
		t.Fatalf("FromConfig failed: %v", err)
	}

	if ag.Name() != "my_wf" {
		t.Errorf("expected workflow name 'my_wf', got %s", ag.Name())
	}

	mockCtx := &MockInvocationContext{
		Context: ctx,
		sess:    &mockSession{},
		userContent: &genai.Content{
			Parts: []*genai.Part{{Text: "hello"}},
		},
	}

	events := ag.Run(mockCtx)
	var outputs []any
	for ev, err := range events {
		if err != nil {
			t.Fatalf("run failed: %v", err)
		}
		if ev.Output != nil {
			outputs = append(outputs, ev.Output)
		}
	}

	if len(outputs) != 2 {
		t.Fatalf("expected 2 outputs, got %d: %+v", len(outputs), outputs)
	}

	if outputs[len(outputs)-1] != "HELLO done" {
		t.Errorf("expected final output 'HELLO done', got %v", outputs[len(outputs)-1])
	}
}

func TestLoadWorkflowWithRoutingYAML(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "routing_workflow.yaml")

	// Test conditional string route matching
	yamlContent := `
name: routing_wf
agent_class: Workflow
edges:
  - - START
    - upper_fn
    - default: suffix_fn
`
	if err := os.WriteFile(configPath, []byte(yamlContent), 0o644); err != nil {
		t.Fatalf("failed to write temp routing workflow: %v", err)
	}

	ctx := context.Background()
	ag, err := FromConfig(ctx, configPath)
	if err != nil {
		t.Fatalf("FromConfig failed: %v", err)
	}

	mockCtx := &MockInvocationContext{
		Context: ctx,
		sess:    &mockSession{},
		userContent: &genai.Content{
			Parts: []*genai.Part{{Text: "routing"}},
		},
	}

	events := ag.Run(mockCtx)
	var outputs []any
	for ev, err := range events {
		if err != nil {
			t.Fatalf("run failed: %v", err)
		}
		if ev.Output != nil {
			outputs = append(outputs, ev.Output)
		}
	}

	if len(outputs) != 2 {
		t.Fatalf("expected 2 outputs, got %d", len(outputs))
	}

	if outputs[1] != "ROUTING done" {
		t.Errorf("expected output 'ROUTING done', got %v", outputs[1])
	}
}

func alphaFn(ctx agent.Context, input string) (string, error) {
	return "alpha: " + input, nil
}

func betaFn(ctx agent.Context, input string) (string, error) {
	return "beta: " + input, nil
}

func init() {
	RegisterNodeFunction("alpha_fn", alphaFn)
	RegisterNodeFunction("beta_fn", betaFn)
}

func TestLoadComplexWorkflowWithSubAgentsYAML(t *testing.T) {
	tempDir := t.TempDir()

	// 1. Write the sub-agent YAML config
	subAgentYAML := `
name: agent_alpha
agent_class: FunctionNode
func_code: alpha_fn
`
	subAgentPath := filepath.Join(tempDir, "agent_alpha.yaml")
	if err := os.WriteFile(subAgentPath, []byte(subAgentYAML), 0o644); err != nil {
		t.Fatalf("failed to write sub-agent yaml: %v", err)
	}

	// 2. Write the main Workflow YAML config referencing the sub-agent and beta function
	workflowYAML := `
name: complex_wf
agent_class: Workflow
edges:
  - - START
    - upper_fn
    - ALPHA: agent_alpha.yaml
      BETA: beta_fn
`
	workflowPath := filepath.Join(tempDir, "complex_workflow.yaml")
	if err := os.WriteFile(workflowPath, []byte(workflowYAML), 0o644); err != nil {
		t.Fatalf("failed to write workflow yaml: %v", err)
	}

	ctx := context.Background()
	ag, err := FromConfig(ctx, workflowPath)
	if err != nil {
		t.Fatalf("FromConfig failed for complex workflow: %v", err)
	}

	if ag.Name() != "complex_wf" {
		t.Errorf("expected complex workflow name 'complex_wf', got %s", ag.Name())
	}

	// Verify that sub-agents are correctly loaded and registered in the Agent hierarchy
	subs := ag.SubAgents()
	if len(subs) != 0 {
		t.Fatalf("expected 0 sub-agents in hierarchy for pure function nodes, got %d", len(subs))
	}

	// Test Branch 1 (ALPHA route)
	{
		mockCtx := &MockInvocationContext{
			Context: ctx,
			sess:    &mockSession{},
			userContent: &genai.Content{
				Parts: []*genai.Part{{Text: "alpha"}},
			},
		}

		events := ag.Run(mockCtx)
		var outputs []any
		for ev, err := range events {
			if err != nil {
				t.Fatalf("run failed for ALPHA branch: %v", err)
			}
			if ev.Output != nil {
				outputs = append(outputs, ev.Output)
			}
		}

		// START -> upper_fn (produces "ALPHA") -> ALPHA route -> agent_alpha (produces "alpha: ALPHA")
		if len(outputs) != 2 {
			t.Fatalf("expected 2 outputs for ALPHA branch, got %d: %+v", len(outputs), outputs)
		}

		if outputs[0] != "ALPHA" {
			t.Errorf("expected first output 'ALPHA', got %v", outputs[0])
		}

		if outputs[1] != "alpha: ALPHA" {
			t.Errorf("expected final output 'alpha: ALPHA', got %v", outputs[1])
		}
	}

	// Test Branch 2 (BETA route)
	{
		mockCtx := &MockInvocationContext{
			Context: ctx,
			sess:    &mockSession{},
			userContent: &genai.Content{
				Parts: []*genai.Part{{Text: "beta"}},
			},
		}

		events := ag.Run(mockCtx)
		var outputs []any
		for ev, err := range events {
			if err != nil {
				t.Fatalf("run failed for BETA branch: %v", err)
			}
			if ev.Output != nil {
				outputs = append(outputs, ev.Output)
			}
		}

		// START -> upper_fn (produces "BETA") -> BETA route -> beta_fn (produces "beta: BETA")
		if len(outputs) != 2 {
			t.Fatalf("expected 2 outputs for BETA branch, got %d: %+v", len(outputs), outputs)
		}

		if outputs[0] != "BETA" {
			t.Errorf("expected first output 'BETA', got %v", outputs[0])
		}

		if outputs[1] != "beta: BETA" {
			t.Errorf("expected final output 'beta: BETA', got %v", outputs[1])
		}
	}
}

func TestLoadComplexWorkflowWithActualSubAgentsYAML(t *testing.T) {
	tempDir := t.TempDir()

	// 1. Write the actual sub-agent YAML config (using LoopAgent, which is standard agent)
	subAgentYAML := `
name: agent_loop
agent_class: LoopAgent
max_iterations: 5
`
	subAgentPath := filepath.Join(tempDir, "agent_loop.yaml")
	if err := os.WriteFile(subAgentPath, []byte(subAgentYAML), 0o644); err != nil {
		t.Fatalf("failed to write sub-agent yaml: %v", err)
	}

	// 2. Write the main Workflow YAML config referencing the sub-agent
	workflowYAML := `
name: loop_wf
agent_class: Workflow
edges:
  - - START
    - agent_loop.yaml
`
	workflowPath := filepath.Join(tempDir, "loop_workflow.yaml")
	if err := os.WriteFile(workflowPath, []byte(workflowYAML), 0o644); err != nil {
		t.Fatalf("failed to write workflow yaml: %v", err)
	}

	ctx := context.Background()
	ag, err := FromConfig(ctx, workflowPath)
	if err != nil {
		t.Fatalf("FromConfig failed for workflow with sub-agent: %v", err)
	}

	if ag.Name() != "loop_wf" {
		t.Errorf("expected workflow name 'loop_wf', got %s", ag.Name())
	}

	// Verify that sub-agents are correctly loaded and registered in the Agent hierarchy
	subs := ag.SubAgents()
	if len(subs) != 1 {
		t.Fatalf("expected 1 sub-agent in hierarchy, got %d", len(subs))
	}
	if subs[0].Name() != "agent_loop" {
		t.Errorf("expected sub-agent name 'agent_loop', got %s", subs[0].Name())
	}
}

type testTool struct{}

func (t *testTool) Name() string        { return "test_tool" }
func (t *testTool) Description() string { return "A simple test tool" }
func (t *testTool) IsLongRunning() bool { return false }
func (t *testTool) Run(ctx agent.ToolContext, args any) (map[string]any, error) {
	return map[string]any{"result": "tool_output"}, nil
}

func init() {
	err := RegisterToolFactory("test_tool", func(_ context.Context, _ map[string]any) (tool.Tool, error) {
		return &testTool{}, nil
	})
	if err != nil {
		panic(err)
	}
}

func TestLoadWorkflowWithJoinNodeYAML(t *testing.T) {
	tempDir := t.TempDir()

	// 1. Write the join-node YAML config
	joinYAML := `
name: aggregate_join
agent_class: JoinNode
`
	joinPath := filepath.Join(tempDir, "aggregate_join.yaml")
	if err := os.WriteFile(joinPath, []byte(joinYAML), 0o644); err != nil {
		t.Fatalf("failed to write join node yaml: %v", err)
	}

	// 2. Write the main Workflow YAML config referencing the join-node
	workflowYAML := `
name: join_wf
agent_class: Workflow
edges:
  - - START
    - alpha_fn
    - aggregate_join.yaml
  - - START
    - beta_fn
    - aggregate_join.yaml
`
	workflowPath := filepath.Join(tempDir, "join_workflow.yaml")
	if err := os.WriteFile(workflowPath, []byte(workflowYAML), 0o644); err != nil {
		t.Fatalf("failed to write workflow yaml: %v", err)
	}

	ctx := context.Background()
	ag, err := FromConfig(ctx, workflowPath)
	if err != nil {
		t.Fatalf("FromConfig failed for join workflow: %v", err)
	}

	if ag.Name() != "join_wf" {
		t.Errorf("expected workflow name 'join_wf', got %s", ag.Name())
	}

	mockCtx := &MockInvocationContext{
		Context: ctx,
		sess:    &mockSession{},
		userContent: &genai.Content{
			Parts: []*genai.Part{{Text: "hello"}},
		},
	}

	events := ag.Run(mockCtx)
	var outputs []any
	for ev, err := range events {
		if err != nil {
			t.Fatalf("run failed: %v", err)
		}
		if ev.Output != nil {
			outputs = append(outputs, ev.Output)
		}
	}

	// Expect outputs from alpha_fn, beta_fn, and aggregate_join (the aggregated map)
	if len(outputs) != 3 {
		t.Fatalf("expected 3 outputs, got %d: %+v", len(outputs), outputs)
	}

	// Find the aggregated map from JoinNode
	var aggregatedMap map[string]any
	for _, out := range outputs {
		if m, ok := out.(map[string]any); ok {
			aggregatedMap = m
			break
		}
	}

	if aggregatedMap == nil {
		t.Fatalf("JoinNode aggregated output not found in outputs: %+v", outputs)
	}

	if val, ok := aggregatedMap["alpha_fn"].(string); !ok || val != "alpha: hello" {
		t.Errorf("expected alpha_fn output 'alpha: hello', got %v", aggregatedMap["alpha_fn"])
	}

	if val, ok := aggregatedMap["beta_fn"].(string); !ok || val != "beta: hello" {
		t.Errorf("expected beta_fn output 'beta: hello', got %v", aggregatedMap["beta_fn"])
	}
}

func TestLoadWorkflowWithToolNodeYAML(t *testing.T) {
	tempDir := t.TempDir()

	// 1. Write the tool-node YAML config
	toolNodeYAML := `
name: my_tool_node
agent_class: ToolNode
tool_code: test_tool
`
	toolNodePath := filepath.Join(tempDir, "my_tool_node.yaml")
	if err := os.WriteFile(toolNodePath, []byte(toolNodeYAML), 0o644); err != nil {
		t.Fatalf("failed to write tool node yaml: %v", err)
	}

	// 2. Write the main Workflow YAML config referencing the tool-node
	workflowYAML := `
name: tool_wf
agent_class: Workflow
edges:
  - - START
    - my_tool_node.yaml
`
	workflowPath := filepath.Join(tempDir, "tool_workflow.yaml")
	if err := os.WriteFile(workflowPath, []byte(workflowYAML), 0o644); err != nil {
		t.Fatalf("failed to write workflow yaml: %v", err)
	}

	ctx := context.Background()
	ag, err := FromConfig(ctx, workflowPath)
	if err != nil {
		t.Fatalf("FromConfig failed for tool workflow: %v", err)
	}

	mockCtx := &MockInvocationContext{
		Context: ctx,
		sess:    &mockSession{},
		userContent: &genai.Content{
			Parts: []*genai.Part{{Text: "{}"}},
		},
	}

	events := ag.Run(mockCtx)
	var outputs []any
	for ev, err := range events {
		if err != nil {
			t.Fatalf("run failed: %v", err)
		}
		if ev.Output != nil {
			outputs = append(outputs, ev.Output)
		}
	}

	if len(outputs) != 1 {
		t.Fatalf("expected 1 output, got %d: %+v", len(outputs), outputs)
	}

	toolOut, ok := outputs[0].(map[string]any)
	if !ok {
		t.Fatalf("expected tool output to be a map, got %T: %v", outputs[0], outputs[0])
	}

	if val, ok := toolOut["result"].(string); !ok || val != "tool_output" {
		t.Errorf("expected tool output result 'tool_output', got %v", toolOut["result"])
	}
}
