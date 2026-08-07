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
	"slices"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"
	"gopkg.in/yaml.v3"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/workflow"
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
		ev := session.NewEvent(ctx, ctx.InvocationID())
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
	agent       agent.Agent
	branch      string
}

// WithICDelta implements [agent.InvocationContext].
func (m *MockInvocationContext) WithICDelta(d *agent.InvocationContextDelta) agent.InvocationContext {
	if d == nil {
		return m
	}
	res := *m
	if d.UserContent != nil {
		res.userContent = *d.UserContent
	}
	if d.Agent != nil {
		res.agent = *d.Agent
	}
	if d.Context != nil {
		res.Context = *d.Context
	}
	if d.Branch != nil {
		res.branch = *d.Branch
	}

	return &res
}

func (m *MockInvocationContext) Session() session.Session        { return m.sess }
func (m *MockInvocationContext) InvocationID() string            { return "test-inv-id" }
func (m *MockInvocationContext) UserContent() *genai.Content     { return m.userContent }
func (m *MockInvocationContext) ResumedInput(string) (any, bool) { return nil, false }
func (m *MockInvocationContext) Agent() agent.Agent              { return m.agent }
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

	ctx := t.Context()
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

	ctx := t.Context()
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

	ctx := t.Context()
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

	ctx := t.Context()
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
func (t *testTool) Run(ctx agent.Context, args any) (map[string]any, error) {
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

	ctx := t.Context()
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

	ctx := t.Context()
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

// handlerFn stands in for a route target; the edge-order tests only parse the
// graph, they never run it.
func handlerFn(ctx agent.Context, input string) (string, error) {
	return input, nil
}

func init() {
	for i := 1; i <= 9; i++ {
		RegisterNodeFunction(fmt.Sprintf("h%d", i), handlerFn)
	}
}

func TestParseEdges_PreservesRouteDeclarationOrder(t *testing.T) {
	// Routes are declared in neither alphabetical nor handler order, so this
	// distinguishes "declaration order" from both a map range and a sort. Enough
	// of them to spill past Go's single-bucket map layout, where iteration is
	// only a random rotation and would too often match by luck.
	const config = `
edges:
  - - START
    - upper_fn
    - ZETA: h1
      ALPHA: h2
      MIKE: h3
      OSCAR: h4
      BRAVO: h5
      YANKEE: h6
      CHARLIE: h7
      TANGO: h8
      default: h9
`
	var cfg struct {
		Edges []yaml.Node `yaml:"edges"`
	}
	if err := yaml.Unmarshal([]byte(config), &cfg); err != nil {
		t.Fatalf("yaml.Unmarshal() error = %v", err)
	}

	edges, err := parseEdges(t.Context(), "", cfg.Edges)
	if err != nil {
		t.Fatalf("parseEdges() error = %v", err)
	}

	got := make([]string, len(edges))
	for i, e := range edges {
		got[i] = fmt.Sprintf("%s->%s(%s)", e.From.Name(), e.To.Name(), routeLabel(e.Route))
	}
	want := []string{
		"START->upper_fn(none)",
		"upper_fn->h1(ZETA)",
		"upper_fn->h2(ALPHA)",
		"upper_fn->h3(MIKE)",
		"upper_fn->h4(OSCAR)",
		"upper_fn->h5(BRAVO)",
		"upper_fn->h6(YANKEE)",
		"upper_fn->h7(CHARLIE)",
		"upper_fn->h8(TANGO)",
		"upper_fn->h9(<default>)",
	}
	if !slices.Equal(got, want) {
		t.Errorf("parseEdges() =\n\t%v\nwant\n\t%v", got, want)
	}
}

// parseConfigEdges parses `edges:` out of a YAML document and runs parseEdges,
// returning the edges rendered as "from->to(route)" strings.
func parseConfigEdges(t *testing.T, config string) ([]string, error) {
	t.Helper()
	var cfg struct {
		Edges []yaml.Node `yaml:"edges"`
	}
	if err := yaml.Unmarshal([]byte(config), &cfg); err != nil {
		t.Fatalf("yaml.Unmarshal() error = %v", err)
	}
	edges, err := parseEdges(t.Context(), "", cfg.Edges)
	if err != nil {
		return nil, err
	}
	got := make([]string, len(edges))
	for i, e := range edges {
		got[i] = fmt.Sprintf("%s->%s(%s)", e.From.Name(), e.To.Name(), routeLabel(e.Route))
	}
	return got, nil
}

// Merge keys let one route map reuse another; they must expand in place, with an
// explicitly written route beating a merged one of the same name.
func TestParseEdges_MergeKeys(t *testing.T) {
	tests := []struct {
		name   string
		config string
		want   []string
	}{
		{
			name: "expands in place",
			config: `
shared: &shared
  ZETA: h1
  ALPHA: h2
edges:
  - - START
    - upper_fn
    - <<: *shared
      MIKE: h3
`,
			want: []string{"START->upper_fn(none)", "upper_fn->h1(ZETA)", "upper_fn->h2(ALPHA)", "upper_fn->h3(MIKE)"},
		},
		{
			name: "explicit route beats merged",
			config: `
shared: &shared
  ZETA: h1
edges:
  - - START
    - upper_fn
    - ZETA: h2
      <<: *shared
`,
			want: []string{"START->upper_fn(none)", "upper_fn->h2(ZETA)"},
		},
		{
			name: "sequence of mappings, earlier wins",
			config: `
a: &a
  ZETA: h1
b: &b
  ZETA: h2
  ALPHA: h3
edges:
  - - START
    - upper_fn
    - <<: [*a, *b]
`,
			want: []string{"START->upper_fn(none)", "upper_fn->h1(ZETA)", "upper_fn->h3(ALPHA)"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseConfigEdges(t, tc.config)
			if err != nil {
				t.Fatalf("parseEdges() error = %v", err)
			}
			if !slices.Equal(got, tc.want) {
				t.Errorf("parseEdges() =\n\t%v\nwant\n\t%v", got, tc.want)
			}
		})
	}
}

// Keys and values must go through Decode, not Node.Value, or aliases stop
// resolving.
func TestParseEdges_ResolvesAliases(t *testing.T) {
	const config = `
routeName: &route ZETA
target: &target h1
edges:
  - - START
    - upper_fn
    - *route : *target
`
	got, err := parseConfigEdges(t, config)
	if err != nil {
		t.Fatalf("parseEdges() error = %v", err)
	}
	want := []string{"START->upper_fn(none)", "upper_fn->h1(ZETA)"}
	if !slices.Equal(got, want) {
		t.Errorf("parseEdges() = %v, want %v", got, want)
	}
}

func TestParseEdges_DefaultRoute(t *testing.T) {
	tests := []struct {
		name   string
		config string
		want   []string
	}{
		{
			name:   "case insensitive",
			config: "edges:\n  - - START\n    - upper_fn\n    - DEFAULT: h1\n",
			want:   []string{"START->upper_fn(none)", "upper_fn->h1(<default>)"},
		},
		{
			name:   "keeps its declared position when first",
			config: "edges:\n  - - START\n    - upper_fn\n    - default: h1\n      ZETA: h2\n",
			want:   []string{"START->upper_fn(none)", "upper_fn->h1(<default>)", "upper_fn->h2(ZETA)"},
		},
		{
			name:   "keeps its declared position when in the middle",
			config: "edges:\n  - - START\n    - upper_fn\n    - ZETA: h2\n      Default: h1\n      ALPHA: h3\n",
			want:   []string{"START->upper_fn(none)", "upper_fn->h2(ZETA)", "upper_fn->h1(<default>)", "upper_fn->h3(ALPHA)"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseConfigEdges(t, tc.config)
			if err != nil {
				t.Fatalf("parseEdges() error = %v", err)
			}
			if !slices.Equal(got, tc.want) {
				t.Errorf("parseEdges() = %v, want %v", got, tc.want)
			}
		})
	}
}

// An empty route map still selects the routing branch, which tolerates a
// single-node chain.
func TestParseEdges_EmptyRouteMap(t *testing.T) {
	got, err := parseConfigEdges(t, "edges:\n  - - upper_fn\n    - {}\n")
	if err != nil {
		t.Fatalf("parseEdges() error = %v", err)
	}
	if len(got) != 0 {
		t.Errorf("parseEdges() = %v, want no edges", got)
	}
}

func TestParseEdges_RouteMapErrors(t *testing.T) {
	tests := []struct {
		name, config, wantErr string
	}{
		{
			name:    "duplicate route reports both lines",
			config:  "edges:\n  - - START\n    - upper_fn\n    - ALPHA: h1\n      ALPHA: h2\n",
			wantErr: `line 5: route "ALPHA" is already defined at line 4`,
		},
		{
			name:    "non-scalar route",
			config:  "edges:\n  - - START\n    - upper_fn\n    - ? [a, b]\n      : h1\n",
			wantErr: "route must be a scalar",
		},
		{
			name:    "non-scalar target",
			config:  "edges:\n  - - START\n    - upper_fn\n    - ALPHA: {x: y}\n",
			wantErr: `target of route "ALPHA" must be a scalar node reference`,
		},
		{
			name:    "null route",
			config:  "edges:\n  - - START\n    - upper_fn\n    - ~: h1\n",
			wantErr: "route must not be null",
		},
		{
			name:    "merge key with a scalar value",
			config:  "edges:\n  - - START\n    - upper_fn\n    - <<: h1\n",
			wantErr: "merge key requires a mapping or a sequence of mappings",
		},
		{
			// Without the cycle guard this recurses until the stack overflows,
			// which is fatal and cannot be recovered.
			name:    "self-referential merge key",
			config:  "m: &m\n  <<: *m\nedges:\n  - - START\n    - upper_fn\n    - <<: *m\n",
			wantErr: "already being merged",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseConfigEdges(t, tc.config)
			if err == nil {
				t.Fatalf("parseEdges() succeeded, want error containing %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("parseEdges() error = %v, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

// The odd-Content guard is unreachable through the YAML parser, which always
// emits key/value pairs, so it takes a hand-built node.
func TestDecodeRouteMap_OddContent(t *testing.T) {
	n := &yaml.Node{
		Kind:    yaml.MappingNode,
		Content: []*yaml.Node{{Kind: yaml.ScalarNode, Value: "ALPHA"}},
	}
	_, err := decodeRouteMap(n)
	if err == nil {
		t.Fatal("decodeRouteMap() succeeded on an odd Content length, want error")
	}
	if want := "odd number of nodes (1)"; !strings.Contains(err.Error(), want) {
		t.Errorf("decodeRouteMap() error = %v, want it to contain %q", err, want)
	}
}

func routeLabel(r workflow.Route) string {
	// Checked before the type switch so it cannot be confused with the literal
	// StringRoute("default").
	if r == workflow.Default {
		return "<default>"
	}
	switch v := r.(type) {
	case nil:
		return "none"
	case workflow.StringRoute:
		return string(v)
	}
	return fmt.Sprintf("%T", r)
}
