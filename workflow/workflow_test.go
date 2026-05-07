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
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

var defaultNodeConfig = NodeConfig{}

// MockInvocationContext is a minimal implementation of agent.InvocationContext for testing.
type MockInvocationContext struct {
	agent.InvocationContext
	sess        session.Session
	userContent *genai.Content
}

func (m *MockInvocationContext) Session() session.Session    { return m.sess }
func (m *MockInvocationContext) InvocationID() string        { return "test-invocation-id" }
func (m *MockInvocationContext) UserContent() *genai.Content { return m.userContent }
func (m *MockInvocationContext) Artifacts() agent.Artifacts  { return nil }
func (m *MockInvocationContext) Memory() agent.Memory        { return nil }
func (m *MockInvocationContext) Agent() agent.Agent          { return nil }
func (m *MockInvocationContext) Branch() string              { return "" }
func (m *MockInvocationContext) RunConfig() *agent.RunConfig { return nil }
func (m *MockInvocationContext) EndInvocation()              {}
func (m *MockInvocationContext) Ended() bool                 { return false }
func (m *MockInvocationContext) WithContext(ctx context.Context) agent.InvocationContext {
	return m
}

func TestFunctionNode(t *testing.T) {
	upperFn := func(ctx agent.InvocationContext, input string) (string, error) {
		return strings.ToUpper(input), nil
	}

	node := NewFunctionNode("upper", upperFn, defaultNodeConfig)

	// Create a mock context
	mockCtx := &MockInvocationContext{sess: nil}

	// Run the node
	events := node.Run(mockCtx, "hello")

	count := 0
	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		count++

		output, ok := ev.Actions.StateDelta["output"]
		if !ok {
			t.Errorf("expected output in state delta")
		}
		if output != "HELLO" {
			t.Errorf("expected output 'HELLO', got %v", output)
		}
	}

	if count != 1 {
		t.Errorf("expected 1 event, got %d", count)
	}
}

func TestSequentialWorkflow(t *testing.T) {
	upperFn := func(ctx agent.InvocationContext, input any) (string, error) {
		s, ok := input.(string)
		if !ok {
			return "", fmt.Errorf("expected string input")
		}
		return strings.ToUpper(s), nil
	}

	suffixFn := func(ctx agent.InvocationContext, input string) (string, error) {
		return input + " done", nil
	}

	nodeA := NewFunctionNode("upper", upperFn, defaultNodeConfig)
	nodeB := NewFunctionNode("suffix", suffixFn, defaultNodeConfig)

	edges := Chain(Start, nodeA, nodeB)
	
	w, err := New(edges)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	mockCtx := &MockInvocationContext{
		sess: nil,
		userContent: &genai.Content{
			Parts: []*genai.Part{{Text: "hello"}},
		},
	}

	events := w.Run(mockCtx)

	var lastOutput any
	count := 0
	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		count++

		if ev.Actions.StateDelta != nil {
			if out, ok := ev.Actions.StateDelta["output"]; ok {
				lastOutput = out
			}
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
	baseNode
	route []string
	onRun func()
}

func (n *CustomRouteNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	if n.onRun != nil {
		n.onRun()
	}
	return func(yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx.InvocationID())
		ev.Routes = n.route
		yield(ev, nil)
	}
}

func TestWorkflowRouting(t *testing.T) {
	type testTracker struct {
		executed []string
	}

	type testCase struct {
		name           string
		startRoutes    []string
		edges          func(nodeStart *CustomRouteNode, nodeA, nodeB *FunctionNode, nodeC *CustomRouteNode, nodeD *FunctionNode) []Edge
		expectedExec   []string
	}

	createNodes := func() (*CustomRouteNode, *FunctionNode, *FunctionNode, *CustomRouteNode, *FunctionNode, *testTracker) {
		tracker := &testTracker{}
		nodeX := &CustomRouteNode{
			baseNode: baseNode{name: "X"},
		}
		nodeA := NewFunctionNode("A", func(ctx agent.InvocationContext, input any) (string, error) {
			tracker.executed = append(tracker.executed, "A")
			return "pathA", nil
		}, defaultNodeConfig)
		nodeB := NewFunctionNode("B", func(ctx agent.InvocationContext, input any) (string, error) {
			tracker.executed = append(tracker.executed, "B")
			return "pathB", nil
		}, defaultNodeConfig)
		nodeC := &CustomRouteNode{
			baseNode: baseNode{name: "C"},
			route:    []string{"branchD"},
			onRun: func() {
				tracker.executed = append(tracker.executed, "C")
			},
		}
		nodeD := NewFunctionNode("D", func(ctx agent.InvocationContext, input any) (string, error) {
			tracker.executed = append(tracker.executed, "D")
			return "pathD", nil
		}, defaultNodeConfig)
		return nodeX, nodeA, nodeB, nodeC, nodeD, tracker
	}

	tests := []testCase{
		{
			name:        "all edges don't have routing",
			startRoutes: []string{"branchA", "branchB"},
			edges: func(x *CustomRouteNode, a, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
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
			edges: func(x *CustomRouteNode, a, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
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
			edges: func(x *CustomRouteNode, a, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
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
			edges: func(x *CustomRouteNode, a, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
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
			edges: func(x *CustomRouteNode, a *FunctionNode, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
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
			edges: func(x *CustomRouteNode, a *FunctionNode, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
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
			edges: func(x *CustomRouteNode, a *FunctionNode, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
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
			edges: func(x *CustomRouteNode, a, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
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
			edges: func(x *CustomRouteNode, a, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
				return []Edge{
					{From: Start, To: x},
					{From: x, To: a, Route: MultiRoute[string]{"branchX", "branchY"}},
					{From: x, To: b, Route: MultiRoute[string]{"branchZ"}},
				}
			},
			expectedExec: nil,
		},
		{
			name:        "duplicate edges to same node",
			startRoutes: []string{"branchA"},
			edges: func(x *CustomRouteNode, a, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
				return []Edge{
					{From: Start, To: x},
					{From: x, To: a},
					{From: x, To: a, Route: StringRoute("branchA")},
					{From: x, To: b},
					{From: x, To: c},
					{From: c, To: d},
				}
			},
			expectedExec: []string{"A", "B", "C", "D"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			start, a, b, c, d, tracker := createNodes()
			start.route = tc.startRoutes
			edges := tc.edges(start, a, b, c, d)

			w, err := New(edges)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			mockCtx := &MockInvocationContext{sess: nil}

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
