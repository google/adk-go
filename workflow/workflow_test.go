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
	"fmt"
	"iter"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// MockInvocationContext is a minimal implementation of agent.InvocationContext for testing.
type MockInvocationContext struct {
	agent.InvocationContext
	sess        session.Session
	userContent *genai.Content
}

func (m *MockInvocationContext) Session() session.Session {
	return m.sess
}

func (m *MockInvocationContext) InvocationID() string {
	return "test-invocation-id"
}

func (m *MockInvocationContext) UserContent() *genai.Content {
	return m.userContent
}

func TestFunctionNode(t *testing.T) {
	upperFn := func(ctx agent.InvocationContext, input string) (string, error) {
		return strings.ToUpper(input), nil
	}

	node := NewFunctionNode("upper", upperFn)

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

	nodeA := NewFunctionNode("upper", upperFn)
	nodeB := NewFunctionNode("suffix", suffixFn)

	edges := Chain(START, nodeA, nodeB)

	w := New(edges)

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

func TestRoutes(t *testing.T) {
	event := &session.Event{
		Route: []string{"hello", "42", "true"},
	}

	if !StringRoute("hello").Matches(event) {
		t.Errorf("StringRoute should match")
	}
	if StringRoute("world").Matches(event) {
		t.Errorf("StringRoute should not match")
	}

	if !IntRoute(42).Matches(event) {
		t.Errorf("IntRoute should match")
	}
	if IntRoute(10).Matches(event) {
		t.Errorf("IntRoute should not match")
	}

	if !BoolRoute(true).Matches(event) {
		t.Errorf("BoolRoute should match")
	}
	if BoolRoute(false).Matches(event) {
		t.Errorf("BoolRoute should not match")
	}
}

type CustomRouteNode struct {
	BaseNode
	route []string
	onRun func()
}

func (n *CustomRouteNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	if n.onRun != nil {
		n.onRun()
	}
	return func(yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx.InvocationID())
		ev.Route = n.route
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
		edges          func(START Node, nodeStart *CustomRouteNode, nodeA, nodeB *FunctionNode, nodeC *CustomRouteNode, nodeD *FunctionNode) []Edge
		expectedExec   []string
		expectErrorMsg string
	}

	createNodes := func() (*CustomRouteNode, *FunctionNode, *FunctionNode, *CustomRouteNode, *FunctionNode, *testTracker) {
		tracker := &testTracker{}
		nodeStart := &CustomRouteNode{
			BaseNode: BaseNode{name: "START_NODE"},
		}
		nodeA := NewFunctionNode("A", func(ctx agent.InvocationContext, input any) (string, error) {
			tracker.executed = append(tracker.executed, "A")
			return "pathA", nil
		})
		nodeB := NewFunctionNode("B", func(ctx agent.InvocationContext, input any) (string, error) {
			tracker.executed = append(tracker.executed, "B")
			return "pathB", nil
		})
		nodeC := &CustomRouteNode{
			BaseNode: BaseNode{name: "C"},
			route:    []string{"branchD"},
			onRun: func() {
				tracker.executed = append(tracker.executed, "C")
			},
		}
		nodeD := NewFunctionNode("D", func(ctx agent.InvocationContext, input any) (string, error) {
			tracker.executed = append(tracker.executed, "D")
			return "pathD", nil
		})
		return nodeStart, nodeA, nodeB, nodeC, nodeD, tracker
	}

	tests := []testCase{
		{
			name:        "all edges don't have routing",
			startRoutes: []string{"branchA", "branchB"},
			edges: func(START Node, start *CustomRouteNode, a *FunctionNode, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
				return []Edge{
					{From: START, To: start},
					{From: start, To: a},
					{From: start, To: b},
					{From: start, To: c},
					{From: c, To: d},
				}
			},
			expectedExec: []string{"A", "B", "C", "D"},
		},
		{
			name:        "only one edge has correct routing and the rest have no routing",
			startRoutes: []string{"branchA"},
			edges: func(START Node, start *CustomRouteNode, a *FunctionNode, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
				return []Edge{
					{From: START, To: start},
					{From: start, To: a, Route: StringRoute("branchA")},
					{From: start, To: b},
					{From: start, To: c},
					{From: c, To: d},
				}
			},
			expectedExec: []string{"A", "B", "C", "D"},
		},
		{
			name:        "one edge has no routing and the rest have a correct routing",
			startRoutes: []string{"branchA", "branchB"},
			edges: func(START Node, start *CustomRouteNode, a *FunctionNode, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
				return []Edge{
					{From: START, To: start},
					{From: start, To: a, Route: StringRoute("branchA")},
					{From: start, To: b, Route: StringRoute("branchB")},
					{From: start, To: c},
					{From: c, To: d, Route: StringRoute("branchD")},
				}
			},
			expectedExec: []string{"A", "B", "C", "D"},
		},
		{
			name:        "any edge has incorrect routing",
			startRoutes: []string{"invalid"},
			edges: func(START Node, start *CustomRouteNode, a *FunctionNode, b *FunctionNode, c *CustomRouteNode, d *FunctionNode) []Edge {
				return []Edge{
					{From: START, To: start},
					{From: start, To: a, Route: StringRoute("branchA")},
					{From: start, To: b, Route: StringRoute("branchB")},
					{From: start, To: c, Route: StringRoute("branchC")},
					{From: c, To: d},
				}
			},
			expectErrorMsg: "produces route tags that do not match any valid outgoing edge",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			start, a, b, c, d, tracker := createNodes()
			start.route = tc.startRoutes
			edges := tc.edges(START, start, a, b, c, d)

			w := New(edges)
			mockCtx := &MockInvocationContext{sess: nil}

			var err error
			for _, testErr := range w.Run(mockCtx) {
				if testErr != nil {
					err = testErr
					break
				}
			}

			if tc.expectErrorMsg != "" {
				if err == nil {
					t.Errorf("expected error matching %q, got none", tc.expectErrorMsg)
				} else if !strings.Contains(err.Error(), tc.expectErrorMsg) {
					t.Errorf("expected error containing %q, got %v", tc.expectErrorMsg, err)
				}
				return
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
