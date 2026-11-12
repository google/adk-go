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
	"errors"
	"fmt"
	"iter"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
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
// t.Context(), which is automatically canceled when the test ends
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
	// serialized.
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
					{From: x, To: b, Route: Default},
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
					{From: x, To: b, Route: Default},
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

// A *genai.Part element can be nil
// userInput must skip those rather than dereference them.
func TestWorkflowRunSkipsNilUserContentParts(t *testing.T) {
	hi := &genai.Part{Text: "hi"}

	for _, tc := range []struct {
		name  string
		parts []*genai.Part
		want  any
	}{
		{"leading", []*genai.Part{nil, hi}, "hi"},
		{"middle", []*genai.Part{hi, nil, hi}, "hihi"},
		{"trailing", []*genai.Part{hi, nil}, "hi"},
		{"only", []*genai.Part{nil}, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			echo := NewFunctionNode("echo", func(ctx agent.Context, input any) (any, error) {
				return input, nil
			}, defaultNodeConfig)
			w := mustNew(t, Chain(Start, echo))

			mockCtx := newMockCtx(t)
			mockCtx.userContent = &genai.Content{Parts: tc.parts}

			var got any
			for ev, err := range w.Run(mockCtx) {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if ev.Output != nil {
					got = ev.Output
				}
			}

			if got != tc.want {
				t.Errorf("seed input = %v, want %v", got, tc.want)
			}
		})
	}
}

// A workflow with no edges builds, and running it ends without an
// error or any event, matching adk-python's Workflow(edges=[]).
func TestWorkflowWithNoEdgesRunsToCompletion(t *testing.T) {
	for _, tc := range []struct {
		name  string
		edges []Edge
	}{
		{"nil", nil},
		{"empty", []Edge{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w, err := New("wf", tc.edges)
			if err != nil {
				t.Fatalf("New() error = %v, want nil", err)
			}

			var events []*session.Event
			for ev, err := range w.Run(newSeededMockCtx(t)) {
				if err != nil {
					t.Fatalf("Run() error = %v", err)
				}
				events = append(events, ev)
			}

			if len(events) != 0 {
				t.Errorf("Run() yielded %d events, want none: %+v", len(events), events)
			}
		})
	}
}

// inputLog records, per node name, every input a node ran with. Nodes
// of one workflow run on their own goroutines.
type inputLog struct {
	mu     sync.Mutex
	inputs map[string][]any
}

func (l *inputLog) record(name string, in any) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.inputs == nil {
		l.inputs = map[string][]any{}
	}
	l.inputs[name] = append(l.inputs[name], in)
}

func (l *inputLog) recorded() map[string][]any {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make(map[string][]any, len(l.inputs))
	for name, in := range l.inputs {
		out[name] = append([]any(nil), in...)
	}
	return out
}

// newLoggingNode returns a custom BaseNode embedder that records its
// input in log and emits nothing.
func newLoggingNode(name string, cfg NodeConfig, log *inputLog) Node {
	return &hitlNode{
		BaseNode: NewBaseNode(name, "", cfg),
		run: func(_ agent.Context, input any, _ func(*session.Event, error) bool) {
			log.record(name, input)
		},
	}
}

func TestWorkflow_StartSeed_ContentOnlyForContentTypedSuccessors(t *testing.T) {
	log := &inputLog{}
	type stateParams struct {
		NodeInput *genai.Content
	}
	fromState, err := NewFunctionNodeFromState("from_state", func(_ agent.InvocationContext, p stateParams) (any, error) {
		log.record("from_state", p.NodeInput)
		return nil, nil
	}, NodeConfig{})
	if err != nil {
		t.Fatalf("NewFunctionNodeFromState: %v", err)
	}
	successors := []Node{
		NewFunctionNode("function", func(_ agent.Context, in *genai.Content) (any, error) {
			log.record("function", in)
			return nil, nil
		}, NodeConfig{}),
		NewFunctionNode("function_value", func(_ agent.Context, in genai.Content) (any, error) {
			log.record("function_value", in)
			return nil, nil
		}, NodeConfig{}),
		NewEmittingFunctionNode("emitting", func(_ agent.Context, in *genai.Content, _ func(*session.Event) error) (any, error) {
			log.record("emitting", in)
			return nil, nil
		}, NodeConfig{}),
		NewDynamicNode("dynamic", func(_ agent.Context, in *genai.Content, _ func(*session.Event) error) (any, error) {
			log.record("dynamic", in)
			return nil, nil
		}, NodeConfig{}),
		fromState,
		newLoggingNode("custom_content", NodeConfig{ContentInput: true}, log),
		NewFunctionNode("function_any", func(_ agent.Context, in any) (any, error) {
			log.record("function_any", in)
			return nil, nil
		}, NodeConfig{}),
		newLoggingNode("custom_text", NodeConfig{}, log),
	}
	var edges []Edge
	for _, n := range successors {
		edges = append(edges, Edge{From: Start, To: n})
	}
	w := mustNew(t, edges)
	mockCtx := newMockCtx(t)
	mockCtx.sess = fakeSession{events: sliceEvents{}}
	mockCtx.userContent = &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "hello"}, nil, {Text: " world"}}}

	drain(t, w.Run(mockCtx))

	content := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "hello"}, nil, {Text: " world"}}}
	want := map[string][]any{
		"function":       {content},
		"function_value": {*content},
		"emitting":       {content},
		"dynamic":        {content},
		"from_state":     {content},
		"custom_content": {content},
		"function_any":   {"hello world"},
		"custom_text":    {"hello world"},
	}
	if diff := cmp.Diff(want, log.recorded()); diff != "" {
		t.Errorf("inputs from Start (-want +got):\n%s", diff)
	}
}

// A JoinNode after Start receives the message text as START's entry, as
// adk-python's join barrier receives START's node input.
func TestWorkflow_StartSeed_JoinNodeReceivesText(t *testing.T) {
	join := NewJoinNode("join")
	w := mustNew(t, []Edge{{From: Start, To: join}})
	mockCtx := newMockCtx(t)
	mockCtx.userContent = &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "hello"}}}

	events := drain(t, w.Run(mockCtx))

	if len(events) != 1 {
		t.Fatalf("Run yielded %d events, want 1", len(events))
	}
	if diff := cmp.Diff(map[string]any{"START": "hello"}, events[0].Output); diff != "" {
		t.Errorf("join output (-want +got):\n%s", diff)
	}
}

func TestWorkflow_NodeRootWrapper_RootSeed(t *testing.T) {
	t.Run("node_root_receives_content", func(t *testing.T) {
		log := &inputLog{}
		root := newLoggingNode("root", NodeConfig{}, log)
		w, err := New("", []Edge{{From: Start, To: root}}, WithNodeRootWrapper())
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		mockCtx := newMockCtx(t)
		mockCtx.userContent = &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}

		drain(t, w.Run(mockCtx))

		want := map[string][]any{"root": {&genai.Content{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}}}
		if diff := cmp.Diff(want, log.recorded()); diff != "" {
			t.Errorf("node root input (-want +got):\n%s", diff)
		}
	})

	t.Run("node_root_without_user_content_receives_nil", func(t *testing.T) {
		log := &inputLog{}
		root := newLoggingNode("root", NodeConfig{}, log)
		w, err := New("", []Edge{{From: Start, To: root}}, WithNodeRootWrapper())
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		drain(t, w.Run(newMockCtx(t)))

		if diff := cmp.Diff(map[string][]any{"root": {nil}}, log.recorded()); diff != "" {
			t.Errorf("node root input (-want +got):\n%s", diff)
		}
	})

	t.Run("workflow_node_root_resolves_seed_per_inner_successor", func(t *testing.T) {
		log := &inputLog{}
		anyNode := NewFunctionNode("any_node", func(_ agent.Context, in any) (string, error) {
			log.record("any_node", in)
			return "a", nil
		}, NodeConfig{})
		contentNode := NewFunctionNode("content_node", func(_ agent.Context, in *genai.Content) (string, error) {
			log.record("content_node", in)
			return "c", nil
		}, NodeConfig{})
		join := NewJoinNode("join")
		root, err := NewWorkflowNode("wf", []Edge{
			{From: Start, To: anyNode},
			{From: Start, To: contentNode},
			{From: anyNode, To: join},
			{From: contentNode, To: join},
		})
		if err != nil {
			t.Fatalf("NewWorkflowNode: %v", err)
		}
		w, err := New("", []Edge{{From: Start, To: root}}, WithNodeRootWrapper())
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		mockCtx := newMockCtx(t)
		mockCtx.userContent = &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}

		drain(t, w.Run(mockCtx))

		want := map[string][]any{
			"any_node":     {"hi"},
			"content_node": {&genai.Content{Role: "user", Parts: []*genai.Part{{Text: "hi"}}}},
		}
		if diff := cmp.Diff(want, log.recorded()); diff != "" {
			t.Errorf("inner successor inputs (-want +got):\n%s", diff)
		}
	})
}

func TestWorkflow_NodeRootWrapper_NoBookkeepingEvents(t *testing.T) {
	t.Run("no_output", func(t *testing.T) {
		root := newHitlNode("root", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Content = genai.NewContentFromText("progress", genai.RoleModel)
			yield(ev, nil)
		})
		w, err := New("", []Edge{{From: Start, To: root}}, WithNodeRootWrapper())
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		events := drain(t, w.Run(newSeededMockCtx(t)))

		if len(events) != 1 {
			t.Fatalf("Run yielded %d events, want only the node's own event", len(events))
		}
		for key := range events[0].CustomMetadata {
			if strings.HasPrefix(key, workflowBasePathPrefix) {
				t.Errorf("event custom metadata has key %q", key)
			}
		}
	})

	// The option, not the empty name, turns the bookkeeping off.
	t.Run("unmarked_anonymous_workflow_keeps_bookkeeping", func(t *testing.T) {
		root := newHitlNode("root", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
			ev := session.NewEvent(ctx, ctx.InvocationID())
			ev.Content = genai.NewContentFromText("progress", genai.RoleModel)
			yield(ev, nil)
		})
		w := mustNew(t, []Edge{{From: Start, To: root}})

		events := drain(t, w.Run(newSeededMockCtx(t)))

		if len(events) != 2 {
			t.Fatalf("Run yielded %d events, want the node's event and a completion record", len(events))
		}
		if _, ok := events[0].CustomMetadata[workflowBasePathPrefix]; !ok {
			t.Errorf("node event custom metadata = %v, want key %q", events[0].CustomMetadata, workflowBasePathPrefix)
		}
	})

	t.Run("failure", func(t *testing.T) {
		boom := errors.New("boom")
		root := newHitlNode("root", func(_ agent.Context, _ any, yield func(*session.Event, error) bool) {
			yield(nil, boom)
		})
		w, err := New("", []Edge{{From: Start, To: root}}, WithNodeRootWrapper())
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		var events []*session.Event
		var runErr error
		for ev, err := range w.Run(newSeededMockCtx(t)) {
			if err != nil {
				runErr = err
				continue
			}
			events = append(events, ev)
		}

		if !errors.Is(runErr, boom) {
			t.Errorf("Run error = %v, want %v", runErr, boom)
		}
		if len(events) != 0 {
			t.Errorf("Run yielded %d events, want none; first has custom metadata %v", len(events), events[0].CustomMetadata)
		}
	})
}

func TestWorkflow_NodeRootWrapper_ResumeReentersRootWithOriginalContent(t *testing.T) {
	const invocationID = "test-invocation-id" // MockInvocationContext.InvocationID
	newRoot := func(log *inputLog) Node {
		return newHitlNode("root", func(ctx agent.Context, input any, yield func(*session.Event, error) bool) {
			log.record("root", input)
			if _, ok := ctx.ResumedInput("approval"); ok {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Output = "resumed"
				yield(ev, nil)
				return
			}
			yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "approval"}), nil)
		})
	}
	newAnswer := func() *genai.Content {
		c := genai.NewContentFromFunctionResponse(WorkflowInputFunctionCallName, map[string]any{"result": "approved"}, genai.RoleUser)
		c.Parts[0].FunctionResponse.ID = "approval"
		return c
	}
	// runTurn1 runs the first turn and returns the history the runner
	// would have stored: the user message, then the node root's events.
	runTurn1 := func(t *testing.T, w *Workflow) sliceEvents {
		t.Helper()
		userEvent := &session.Event{Author: "user", InvocationID: invocationID}
		userEvent.Content = &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "initial prompt"}}}
		turn1 := newMockCtx(t)
		turn1.userContent = userEvent.Content
		return append(sliceEvents{userEvent}, drain(t, w.Run(turn1))...)
	}
	resumeOutputs := func(t *testing.T, w *Workflow, history sliceEvents) []any {
		t.Helper()
		state, err := w.ReconstructRunState(fakeSession{events: history}, invocationID)
		if err != nil {
			t.Fatalf("ReconstructRunState: %v", err)
		}
		turn2 := newMockCtx(t)
		turn2.userContent = newAnswer()
		var outputs []any
		for _, ev := range drain(t, w.Resume(agent.Promote(turn2), state, map[string]any{"approval": map[string]any{"result": "approved"}})) {
			if ev.Output != nil {
				outputs = append(outputs, ev.Output)
			}
		}
		return outputs
	}

	t.Run("answer_in_history", func(t *testing.T) {
		log := &inputLog{}
		w, err := New("", []Edge{{From: Start, To: newRoot(log)}}, WithNodeRootWrapper())
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		history := runTurn1(t, w)
		answerEvent := &session.Event{Author: "user", InvocationID: invocationID}
		answerEvent.Content = newAnswer()
		history = append(history, answerEvent)

		outputs := resumeOutputs(t, w, history)

		original := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "initial prompt"}}}
		if diff := cmp.Diff(map[string][]any{"root": {original, original}}, log.recorded()); diff != "" {
			t.Errorf("node root inputs over both turns (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]any{"resumed"}, outputs); diff != "" {
			t.Errorf("resume outputs (-want +got):\n%s", diff)
		}
	})

	// As adk-python's Runner._find_user_message_for_invocation, the seed
	// skips user events without parts and function-response turns.
	t.Run("seed_skips_partless_user_event", func(t *testing.T) {
		log := &inputLog{}
		w, err := New("", []Edge{{From: Start, To: newRoot(log)}}, WithNodeRootWrapper())
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		partless := &session.Event{Author: "user", InvocationID: invocationID}
		partless.Content = &genai.Content{Role: "user"}
		answerEvent := &session.Event{Author: "user", InvocationID: invocationID}
		answerEvent.Content = newAnswer()
		history := append(sliceEvents{partless}, runTurn1(t, w)...)
		history = append(history, answerEvent)

		resumeOutputs(t, w, history)

		original := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "initial prompt"}}}
		if diff := cmp.Diff(map[string][]any{"root": {original, original}}, log.recorded()); diff != "" {
			t.Errorf("node root inputs over both turns (-want +got):\n%s", diff)
		}
	})

	t.Run("seed_skips_function_response_turn", func(t *testing.T) {
		log := &inputLog{}
		w, err := New("", []Edge{{From: Start, To: newRoot(log)}}, WithNodeRootWrapper())
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		answerEvent := &session.Event{Author: "user", InvocationID: invocationID}
		answerEvent.Content = newAnswer()
		// History holds no prompt for this invocation, only the answer.
		history := append(runTurn1(t, w)[1:], answerEvent)

		resumeOutputs(t, w, history)

		original := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "initial prompt"}}}
		if diff := cmp.Diff(map[string][]any{"root": {original, nil}}, log.recorded()); diff != "" {
			t.Errorf("node root inputs over both turns (-want +got):\n%s", diff)
		}
	})

	// Workflow.Resume also accepts an answer that history does not hold
	// yet; the node root is re-run rather than handed off.
	t.Run("answer_only_in_responses", func(t *testing.T) {
		w, err := New("", []Edge{{From: Start, To: newRoot(&inputLog{})}}, WithNodeRootWrapper())
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		history := runTurn1(t, w)

		outputs := resumeOutputs(t, w, history)

		if diff := cmp.Diff([]any{"resumed"}, outputs); diff != "" {
			t.Errorf("resume outputs (-want +got):\n%s", diff)
		}
	})
}

func TestWorkflow_NodeRootWrapper_RequiresSingleEdge(t *testing.T) {
	a := NewFunctionNode("a", func(_ agent.Context, in any) (any, error) { return in, nil }, NodeConfig{})
	b := NewFunctionNode("b", func(_ agent.Context, in any) (any, error) { return in, nil }, NodeConfig{})

	w, err := New("", Chain(Start, a, b), WithNodeRootWrapper())

	if err == nil || w != nil {
		t.Errorf("New with two edges = (%v, %v), want (nil, error)", w, err)
	}
}
