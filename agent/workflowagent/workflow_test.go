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

package workflowagent

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// MockSession is a minimal implementation of session.Session for testing.
type MockSession struct {
	session.Session
}

func (m MockSession) ID() string {
	return "test-session-id"
}

// MockInvocationContext is a minimal implementation of agent.InvocationContext for testing.
type MockInvocationContext struct {
	context.Context
	sess        session.Session
	userContent *genai.Content
	myAgent     agent.Agent
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

func (m *MockInvocationContext) Agent() agent.Agent {
	return m.myAgent
}

func (m *MockInvocationContext) WithContext(ctx context.Context) agent.InvocationContext {
	return &MockInvocationContext{
		Context:     ctx,
		sess:        m.sess,
		userContent: m.userContent,
		myAgent:     m.myAgent,
	}
}

func (m *MockInvocationContext) Deadline() (deadline time.Time, ok bool) {
	return m.Context.Deadline()
}

func (m *MockInvocationContext) Done() <-chan struct{} {
	return m.Context.Done()
}

func (m *MockInvocationContext) Err() error {
	return m.Context.Err()
}

func (m *MockInvocationContext) Value(key any) any {
	return m.Context.Value(key)
}

func (m *MockInvocationContext) Artifacts() agent.Artifacts { return nil }
func (m *MockInvocationContext) Memory() agent.Memory       { return nil }
func (m *MockInvocationContext) Branch() string             { return "" }
func (m *MockInvocationContext) RunConfig() *agent.RunConfig { return nil }
func (m *MockInvocationContext) Ended() bool                { return false }
func (m *MockInvocationContext) EndInvocation()             {}

func TestWorkflowAgent(t *testing.T) {
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

	nodeA := workflow.NewFunctionNode("upper", upperFn)
	nodeB := workflow.NewFunctionNode("suffix", suffixFn)

	edges := workflow.Chain(workflow.START, nodeA, nodeB)

	myWorkflow, err := New(Config{
		AgentConfig: agent.Config{
			Name: "test_workflow",
		},
		Edges: edges,
	})
	if err != nil {
		t.Fatalf("failed to create workflow agent: %v", err)
	}

	mockCtx := &MockInvocationContext{
		Context: context.TODO(),
		sess:    MockSession{},
		userContent: &genai.Content{
			Parts: []*genai.Part{{Text: "hello"}},
		},
		myAgent: myWorkflow,
	}

	events := myWorkflow.Run(mockCtx)

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
