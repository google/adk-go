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

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

var defaultNodeConfig = workflow.NodeConfig{}

// MockSession is a minimal implementation of session.Session for testing.
type MockSession struct {
	session.Session
}

func (m MockSession) ID() string {
	return "test-session-id"
}

// State returns nil; the workflow persistence helpers handle a nil
// session.State by treating the session as non-persisting.
func (m MockSession) State() session.State {
	return nil
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

func (m *MockInvocationContext) Artifacts() agent.Artifacts      { return nil }
func (m *MockInvocationContext) Memory() agent.Memory            { return nil }
func (m *MockInvocationContext) Branch() string                  { return "" }
func (m *MockInvocationContext) RunConfig() *agent.RunConfig     { return nil }
func (m *MockInvocationContext) Ended() bool                     { return false }
func (m *MockInvocationContext) EndInvocation()                  {}
func (m *MockInvocationContext) TriggeredBy() string             { return "" }
func (m *MockInvocationContext) ResumedInput(string) (any, bool) { return nil, false }

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

	nodeA := workflow.NewFunctionNode("upper", upperFn, defaultNodeConfig)
	nodeB := workflow.NewFunctionNode("suffix", suffixFn, defaultNodeConfig)

	edges := workflow.Chain(workflow.Start, nodeA, nodeB)

	myWorkflow, err := New(Config{
		Name:  "test_workflow",
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
	nodeEvents := 0
	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if ev.Actions.StateDelta != nil {
			if out, ok := ev.Actions.StateDelta["output"]; ok {
				lastOutput = out
				nodeEvents++
			}
		}
	}

	// One output event per FunctionNode (upper, suffix); Start is
	// a no-op sentinel and emits nothing.
	if nodeEvents != 2 {
		t.Errorf("expected 2 node-output events, got %d", nodeEvents)
	}

	if lastOutput != "HELLO done" {
		t.Errorf("expected last output 'HELLO done', got %v", lastOutput)
	}
}

func TestDecodeWorkflowInputResponse(t *testing.T) {
	tests := []struct {
		name string
		fr   *genai.FunctionResponse
		want any
	}{
		{
			name: "ResponseShape_JSONObject",
			fr: &genai.FunctionResponse{
				Response: map[string]any{"response": `{"approved":true}`},
			},
			want: map[string]any{"approved": true},
		},
		{
			name: "ResponseShape_JSONScalar",
			fr: &genai.FunctionResponse{
				Response: map[string]any{"response": `42`},
			},
			want: float64(42), // json.Unmarshal decodes JSON numbers to float64
		},
		{
			name: "ResponseShape_InvalidJSONFallsBackToRawString",
			fr: &genai.FunctionResponse{
				Response: map[string]any{"response": "not valid json"},
			},
			want: "not valid json",
		},
		{
			name: "ResponseShape_NonStringValueReturnedVerbatim",
			fr: &genai.FunctionResponse{
				Response: map[string]any{"response": map[string]any{"already": "decoded"}},
			},
			want: map[string]any{"already": "decoded"},
		},
		{
			name: "PayloadShape_StringPayload",
			fr: &genai.FunctionResponse{
				Response: map[string]any{"payload": "yes"},
			},
			want: "yes",
		},
		{
			name: "PayloadShape_StructuredPayload",
			fr: &genai.FunctionResponse{
				Response: map[string]any{"payload": map[string]any{"k": "v"}},
			},
			want: map[string]any{"k": "v"},
		},
		{
			name: "PriorityOrder_ResponseWinsOverPayload",
			fr: &genai.FunctionResponse{
				Response: map[string]any{
					"response": `"from-response"`,
					"payload":  "from-payload",
				},
			},
			want: "from-response",
		},
		{
			name: "Fallback_NeitherKey_ReturnsRawMap",
			fr: &genai.FunctionResponse{
				Response: map[string]any{"custom": "shape"},
			},
			want: map[string]any{"custom": "shape"},
		},
		{
			name: "Fallback_EmptyResponseMap",
			fr:   &genai.FunctionResponse{Response: map[string]any{}},
			want: map[string]any{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := decodeWorkflowInputResponse(tc.fr)
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("decodeWorkflowInputResponse(%+v) mismatch (-want +got):\n%s", tc.fr.Response, diff)
			}
		})
	}
}
