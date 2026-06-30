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

package workflow

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"sync"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// defaultNodeConfig is the explicit "use defaults" NodeConfig used
// across this package's tests where no per-node knobs are exercised.
var defaultNodeConfig = NodeConfig{}

// MockInvocationContext is a minimal implementation of
// agent.Context for testing. It embeds a real
// context.Context so child cancellation works.
type MockInvocationContext struct {
	context.Context
	sess           session.Session
	userContent    *genai.Content
	isolationScope string
	branch         string
}

// WithICDelta implements [agent.InvocationContext].
func (m *MockInvocationContext) WithICDelta(d *agent.InvocationContextDelta) agent.InvocationContext {
	if d == nil {
		return m
	}
	res := *m
	if d.IsolationScope != nil {
		res.isolationScope = *d.IsolationScope
	}
	if d.Branch != nil {
		res.branch = *d.Branch
	}
	return &res
}

// newMockCtx returns a fresh MockInvocationContext backed by
// t.Context(), which is automatically cancelled when the test ends
// — preventing leaked scheduler goroutines from outliving the test.
func newMockCtx(t *testing.T) *MockInvocationContext {
	t.Helper()
	return &MockInvocationContext{Context: t.Context()}
}

// newSeededMockCtx returns a mockCtx pre-loaded with a "seed" user
// content part — the standard fixture for scheduler tests that need
// an initial input flowing into Start.
func newSeededMockCtx(t *testing.T) *MockInvocationContext {
	t.Helper()
	ctx := newMockCtx(t)
	ctx.userContent = &genai.Content{Parts: []*genai.Part{{Text: "seed"}}}
	return ctx
}

// mustNew builds an anonymous workflow from edges and fails the
// test if construction errors. The returned workflow is ready to
// Run; persistence is disabled.
func mustNew(t *testing.T, edges []Edge) *Workflow {
	t.Helper()
	w, err := New("", edges)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return w
}

func (m *MockInvocationContext) Session() session.Session        { return m.sess }
func (m *MockInvocationContext) InvocationID() string            { return "test-invocation-id" }
func (m *MockInvocationContext) UserContent() *genai.Content     { return m.userContent }
func (m *MockInvocationContext) ResumedInput(string) (any, bool) { return nil, false }
func (m *MockInvocationContext) Agent() agent.Agent              { return nil }
func (m *MockInvocationContext) Artifacts() agent.Artifacts      { return nil }
func (m *MockInvocationContext) Memory() agent.Memory            { return nil }
func (m *MockInvocationContext) Branch() string                  { return m.branch }
func (m *MockInvocationContext) IsolationScope() string          { return m.isolationScope }
func (m *MockInvocationContext) RunConfig() *agent.RunConfig     { return nil }
func (m *MockInvocationContext) Ended() bool                     { return false }
func (m *MockInvocationContext) EndInvocation()                  {}

func (m *MockInvocationContext) WithContext(ctx context.Context) agent.InvocationContext {
	cp := *m
	cp.Context = ctx
	return &cp
}

func TestFunctionNode(t *testing.T) {
	upperFn := func(ctx agent.Context, input string) (string, error) {
		return strings.ToUpper(input), nil
	}

	node := NewFunctionNode("upper", upperFn, defaultNodeConfig)

	// Create a mock context
	mockCtx := newMockCtx(t)
	exCtx := agent.NewContext(mockCtx)

	// Run the node
	events := node.Run(exCtx, "hello")

	count := 0
	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		count++

		if ev.Output != "HELLO" {
			t.Errorf("expected Output 'HELLO', got %v", ev.Output)
		}
	}

	if count != 1 {
		t.Errorf("expected 1 event, got %d", count)
	}
}

func TestSequentialWorkflow(t *testing.T) {
	upperFn := func(ctx agent.Context, input any) (string, error) {
		s, ok := input.(string)
		if !ok {
			return "", fmt.Errorf("expected string input")
		}
		return strings.ToUpper(s), nil
	}

	suffixFn := func(ctx agent.Context, input string) (string, error) {
		return input + " done", nil
	}

	nodeA := NewFunctionNode("upper", upperFn, defaultNodeConfig)
	nodeB := NewFunctionNode("suffix", suffixFn, defaultNodeConfig)

	edges := Chain(Start, nodeA, nodeB)

	w := mustNew(t, edges)

	mockCtx := newMockCtx(t)
	mockCtx.userContent = &genai.Content{
		Parts: []*genai.Part{{Text: "hello"}},
	}

	events := w.Run(mockCtx)

	var lastOutput any
	count := 0
	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		count++

		if ev.Output != nil {
			lastOutput = ev.Output
		}
	}

	if count != 2 {
		t.Errorf("expected 2 events, got %d", count)
	}

	if lastOutput != "HELLO done" {
		t.Errorf("expected last output 'HELLO done', got %v", lastOutput)
	}
}

func TestStringRoute(t *testing.T) {
	event := &session.Event{
		Routes: []string{"hello", "42", "true"},
	}

	if !StringRoute("hello").Matches(event) {
		t.Errorf("StringRoute should match")
	}
	if StringRoute("world").Matches(event) {
		t.Errorf("StringRoute should not match")
	}
}

func TestIntRoute(t *testing.T) {
	event := &session.Event{
		Routes: []string{"hello", "42", "true"},
	}

	if !IntRoute(42).Matches(event) {
		t.Errorf("IntRoute should match")
	}
	if IntRoute(10).Matches(event) {
		t.Errorf("IntRoute should not match")
	}
}

func TestBoolRoute(t *testing.T) {
	event := &session.Event{
		Routes: []string{"hello", "42", "true"},
	}

	if !BoolRoute(true).Matches(event) {
		t.Errorf("BoolRoute should match")
	}
	if BoolRoute(false).Matches(event) {
		t.Errorf("BoolRoute should not match")
	}
}

func TestMultiRouteString(t *testing.T) {
	event := &session.Event{
		Routes: []string{"hello", "42", "true"},
	}

	strMulti := MultiRoute[string]{"world", "hello"}
	if !strMulti.Matches(event) {
		t.Errorf("MultiRoute[string] should match")
	}
	strMultiNoMatch := MultiRoute[string]{"world", "golang"}
	if strMultiNoMatch.Matches(event) {
		t.Errorf("MultiRoute[string] should not match")
	}
}

func TestMultiRouteInt(t *testing.T) {
	event := &session.Event{
		Routes: []string{"hello", "42", "true"},
	}

	intMulti := MultiRoute[int]{10, 42}
	if !intMulti.Matches(event) {
		t.Errorf("MultiRoute[int] should match")
	}
	intMultiNoMatch := MultiRoute[int]{10, 20}
	if intMultiNoMatch.Matches(event) {
		t.Errorf("MultiRoute[int] should not match")
	}
}

type CustomRouteNode struct {
	BaseNode
	route []string
	onRun func()
}

func (n *CustomRouteNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	if n.onRun != nil {
		n.onRun()
	}
	return func(yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx, ctx.InvocationID())
		ev.Routes = n.route
		yield(ev, nil)
	}
}

func TestWorkflowRouting(t *testing.T) {
	// testTracker collects the names of nodes that ran. The
	// scheduler runs sibling nodes concurrently, so appends must be
	// serialised.
	type testTracker struct {
		mu       sync.Mutex
		executed []string
	}
	record := func(tracker *testTracker, name string) {
		tracker.mu.Lock()
		tracker.executed = append(tracker.executed, name)
		tracker.mu.Unlock()
	}

	type testCase struct {
		name         string
		startRoutes  []string
		edges        func(nodeStart, nodeA, nodeB, nodeC, nodeD *CustomRouteNode) []Edge
		expectedExec []string
	}

	// A, B, and D are leaf nodes whose execution is recorded but which
	// do not emit Event.Output: this test asserts on which nodes ran
	// (tracker.executed), not on their output.
	createNodes := func() (*CustomRouteNode, *CustomRouteNode, *CustomRouteNode, *CustomRouteNode, *CustomRouteNode, *testTracker) {
		tracker := &testTracker{}
		nodeX := &CustomRouteNode{
			BaseNode: NewBaseNode("X", "", defaultNodeConfig),
		}
		nodeA := &CustomRouteNode{
			BaseNode: NewBaseNode("A", "", defaultNodeConfig),
			onRun:    func() { record(tracker, "A") },
		}
		nodeB := &CustomRouteNode{
			BaseNode: NewBaseNode("B", "", defaultNodeConfig),
			onRun:    func() { record(tracker, "B") },
		}
		nodeC := &CustomRouteNode{
			BaseNode: NewBaseNode("C", "", defaultNodeConfig),
			route:    []string{"branchD"},
			onRun: func() {
				record(tracker, "C")
			},
		}
		nodeD := &CustomRouteNode{
			BaseNode: NewBaseNode("D", "", defaultNodeConfig),
			onRun:    func() { record(tracker, "D") },
		}
		return nodeX, nodeA, nodeB, nodeC, nodeD, tracker
	}

	tests := []testCase{
		{
			name:        "all edges don't have routing",
			startRoutes: []string{"branchA", "branchB"},
			edges: func(x, a, b, c, d *CustomRouteNode) []Edge {
				return []Edge{
					{From: Start, To: x},
					{From: x, To: a},
					{From: x, To: b},
					{From: x, To: c},
					{From: c, To: d},
				}
			},
			expectedExec: []string{"A", "B", "C", "D"},
		},
		{
			name:        "only one edge has correct routing and the rest have no routing",
			startRoutes: []string{"branchA"},
			edges: func(x, a, b, c, d *CustomRouteNode) []Edge {
				return []Edge{
					{From: Start, To: x},
					{From: x, To: a, Route: StringRoute("branchA")},
					{From: x, To: b},
					{From: x, To: c},
					{From: c, To: d},
				}
			},
			expectedExec: []string{"A", "B", "C", "D"},
		},
		{
			name:        "one edge has no routing and the rest have a correct routing",
			startRoutes: []string{"branchA", "branchB"},
			edges: func(x, a, b, c, d *CustomRouteNode) []Edge {
				return []Edge{
					{From: Start, To: x},
					{From: x, To: a, Route: StringRoute("branchA")},
					{From: x, To: b, Route: StringRoute("branchB")},
					{From: x, To: c},
					{From: c, To: d, Route: StringRoute("branchD")},
				}
			},
			expectedExec: []string{"A", "B", "C", "D"},
		},
		{
			name:        "any edge has incorrect routing",
			startRoutes: []string{"invalid"},
			edges: func(x, a, b, c, d *CustomRouteNode) []Edge {
				return []Edge{
					{From: Start, To: x},
					{From: x, To: a, Route: StringRoute("branchA")},
					{From: x, To: b, Route: StringRoute("branchB")},
					{From: x, To: c, Route: StringRoute("branchC")},
					{From: c, To: d},
				}
			},
		},
		{
			name:        "fallback to default route when no concrete route matches",
			startRoutes: []string{"unmatched"},
			edges: func(x, a, b, c, d *CustomRouteNode) []Edge {
				return []Edge{
					{From: Start, To: x},
					{From: x, To: a, Route: StringRoute("branchA")},
					{From: x, To: b, Route: (Default)},
				}
			},
			expectedExec: []string{"B"},
		},
		{
			name:        "default route is suppressed by concrete route match",
			startRoutes: []string{"branchA"},
			edges: func(x, a, b, c, d *CustomRouteNode) []Edge {
				return []Edge{
					{From: Start, To: x},
					{From: x, To: a, Route: StringRoute("branchA")},
					{From: x, To: b, Route: Default},
				}
			},
			expectedExec: []string{"A"},
		},
		{
			name:        "unconditional edge does not suppress default route",
			startRoutes: []string{"unmatched"},
			edges: func(x, a, b, c, d *CustomRouteNode) []Edge {
				return []Edge{
					{From: Start, To: x},
					{From: x, To: a},
					{From: x, To: b, Route: (Default)},
				}
			},
			expectedExec: []string{"A", "B"},
		},
		{
			name:        "correct MultiRoute",
			startRoutes: []string{"branchA"},
			edges: func(x, a, b, c, d *CustomRouteNode) []Edge {
				return []Edge{
					{From: Start, To: x},
					{From: x, To: a, Route: MultiRoute[string]{"branchX", "branchA"}},
					{From: x, To: b},
					{From: x, To: c},
					{From: c, To: d},
				}
			},
			expectedExec: []string{"A", "B", "C", "D"},
		},
		{
			name:        "no MultiRoute matches event routes",
			startRoutes: []string{"invalid"},
			edges: func(x, a, b, c, d *CustomRouteNode) []Edge {
				return []Edge{
					{From: Start, To: x},
					{From: x, To: a, Route: MultiRoute[string]{"branchX", "branchY"}},
					{From: x, To: b, Route: MultiRoute[string]{"branchZ"}},
				}
			},
			expectedExec: nil,
		},
		{
			name:        "MultiRoute with multiple matching routes",
			startRoutes: []string{"branchA", "branchB"},
			edges: func(x, a, b, c, d *CustomRouteNode) []Edge {
				return []Edge{
					{From: Start, To: x},
					{From: x, To: a, Route: MultiRoute[string]{"branchA", "branchB"}},
				}
			},
			expectedExec: []string{"A"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			start, a, b, c, d, tracker := createNodes()
			start.route = tc.startRoutes
			edges := tc.edges(start, a, b, c, d)

			w := mustNew(t, edges)
			mockCtx := newMockCtx(t)

			var err error
			for _, testErr := range w.Run(mockCtx) {
				if testErr != nil {
					err = testErr
					break
				}
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if len(tracker.executed) != len(tc.expectedExec) {
				t.Errorf("expected %v executed, got %v", tc.expectedExec, tracker.executed)
			}
		})
	}
}

func TestWorkflow_StateSchemaConsistency(t *testing.T) {
	schemaRaw := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"Foo": {Type: "string"},
		},
	}
	schema, err := schemaRaw.Resolve(nil)
	if err != nil {
		t.Fatalf("failed to resolve schema: %v", err)
	}

	validNode, err := NewFunctionNodeFromState("valid_node", dummyFnValid, NodeConfig{})
	if err != nil {
		t.Fatalf("NewFunctionNodeFromState: %v", err)
	}

	invalidNode, err := NewFunctionNodeFromState("invalid_node", dummyFnInvalid, NodeConfig{})
	if err != nil {
		t.Fatalf("NewFunctionNodeFromState: %v", err)
	}

	agentNode := newDummyNode("agentNode")

	tests := []struct {
		name    string
		nodes   []Node
		schema  *jsonschema.Resolved
		wantErr string
	}{
		{
			name:   "valid schema matches exactly",
			nodes:  []Node{validNode},
			schema: schema,
		},
		{
			name:    "typo in tag causes error",
			nodes:   []Node{invalidNode},
			schema:  schema,
			wantErr: `node "invalid_node" references state field "foo" which is not declared in StateSchema`,
		},
		{
			name:   "nil schema skips validation",
			nodes:  []Node{invalidNode},
			schema: nil,
		},
		{
			name:   "non state params aware nodes are ignored",
			nodes:  []Node{agentNode},
			schema: schema,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			edges, _ := fanOutToJoin(tc.nodes)
			_, err := New("wf", edges, WithStateSchema(tc.schema))
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("expected error to contain %q, got: %v", tc.wantErr, err)
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

type validParams struct {
	Foo       string `state:"Foo"`
	NodeInput string `state:"node_input"`
}

type invalidParams struct {
	Foo string `state:"foo"` // Typo: case mismatch
}

func dummyFnValid(ctx agent.InvocationContext, p validParams) (string, error) {
	return "ok", nil
}

func dummyFnInvalid(ctx agent.InvocationContext, p invalidParams) (string, error) {
	return "ok", nil
}

func TestEndToEndInputValidationFlow(t *testing.T) {
	type InitInput struct {
		UserQuery string `json:"user_query"`
	}
	type ParsedQuery struct {
		Intent string `json:"intent"`
	}

	schemaRaw, _ := jsonschema.For[InitInput](nil)
	fnNode, _ := NewFunctionNodeWithSchema("parser", func(ctx agent.Context, input InitInput) (ParsedQuery, error) {
		return ParsedQuery{Intent: input.UserQuery + "_intent"}, nil
	}, schemaRaw, nil, defaultNodeConfig)

	joinSchema, _ := jsonschema.For[ParsedQuery](nil)
	joinSchemaResolved, _ := joinSchema.Resolve(nil)
	joinNode := NewJoinNodeWithSchema("join", joinSchemaResolved)

	type AgentInput struct {
		Parser struct {
			Intent string `json:"intent"`
		} `json:"parser"`
	}

	var receivedInput string
	dummyAgent, _ := agent.New(agent.Config{
		Name: "e2e_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				if ctx.UserContent() != nil && len(ctx.UserContent().Parts) > 0 {
					receivedInput = ctx.UserContent().Parts[0].Text
				}
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Output = "ok"
				yield(ev, nil)
			}
		},
	})

	agentNode, _ := NewAgentNodeTyped[AgentInput, string](dummyAgent, defaultNodeConfig)

	wf := mustNew(t, []Edge{
		{From: Start, To: fnNode},
		{From: fnNode, To: joinNode},
		{From: joinNode, To: agentNode},
	})

	mockCtx := newMockCtx(t)
	mockCtx.sess = &mockSession{id: "test"}
	mockCtx.userContent = &genai.Content{
		Parts: []*genai.Part{{Text: `{"user_query": "hello"}`}},
	}

	for ev, err := range wf.Run(mockCtx) {
		if err != nil {
			t.Fatalf("expected successful end-to-end run, got error: %v", err)
		}
		_ = ev
	}

	if !strings.Contains(receivedInput, "hello_intent") {
		t.Errorf("expected json payload at the end to contain 'hello_intent', got %q", receivedInput)
	}
}
