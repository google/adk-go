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

	edges := Chain(Start, nodeA, nodeB)

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
