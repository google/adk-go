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
	"time"

	"google.golang.org/genai"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

type mockWorkflowState struct {
	data map[string]any
}

func (m *mockWorkflowState) Get(k string) (any, error) {
	v, ok := m.data[k]
	if !ok {
		return nil, session.ErrStateKeyNotExist
	}
	return v, nil
}

func (m *mockWorkflowState) Set(k string, v any) error {
	if m.data == nil {
		m.data = make(map[string]any)
	}
	m.data[k] = v
	return nil
}

func (m *mockWorkflowState) All() iter.Seq2[string, any] { return nil }

type mockWorkflowSession struct {
	state session.State
}

func (m *mockWorkflowSession) ID() string              { return "test-session" }
func (m *mockWorkflowSession) AppName() string         { return "test-app" }
func (m *mockWorkflowSession) UserID() string          { return "test-user" }
func (m *mockWorkflowSession) State() session.State   { return m.state }
func (m *mockWorkflowSession) Events() session.Events { return nil }
func (m *mockWorkflowSession) LastUpdateTime() time.Time { return time.Now() }

func TestNestedWorkflow(t *testing.T) {
	// Create inner workflow edges
	innerFn := func(ctx agent.InvocationContext, input string) (string, error) {
		return input + "-inner", nil
	}
	innerNode := NewFunctionNode("inner_node", innerFn, defaultNodeConfig)
	innerEdges := Chain(Start, innerNode)

	// Create outer workflow with WorkflowNode
	wfNode, err := NewWorkflowNode("nested_step", innerEdges)
	if err != nil {
		t.Fatalf("failed to create workflow node: %v", err)
	}

	outerFn := func(ctx agent.InvocationContext, input string) (string, error) {
		return input + "-outer", nil
	}
	outerNode := NewFunctionNode("outer_node", outerFn, defaultNodeConfig)

	outerEdges := Chain(Start, wfNode, outerNode)
	outerWf := mustNew(t, outerEdges)

	// Run outer workflow
	mockCtx := newMockCtx(t)
	mockCtx.userContent = &genai.Content{
		Parts: []*genai.Part{{Text: "input"}},
	}

	events := outerWf.Run(mockCtx)

	var lastOutput any
	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if ev.Output != nil {
			lastOutput = ev.Output
		}
	}

	if lastOutput != "input-inner-outer" {
		t.Errorf("expected last output 'input-inner-outer', got %v", lastOutput)
	}
}

func TestNestedWorkflow_MultipleOutputs(t *testing.T) {
	// Create inner workflow edges with two nodes producing outputs
	innerFn1 := func(ctx agent.InvocationContext, input string) (string, error) {
		return input + "-inner1", nil
	}
	innerFn2 := func(ctx agent.InvocationContext, input string) (string, error) {
		return input + "-inner2", nil
	}
	innerNode1 := NewFunctionNode("inner_node1", innerFn1, defaultNodeConfig)
	innerNode2 := NewFunctionNode("inner_node2", innerFn2, defaultNodeConfig)
	innerEdges := Chain(Start, innerNode1, innerNode2)

	// Create outer workflow with WorkflowNode
	wfNode, err := NewWorkflowNode("nested_step", innerEdges)
	if err != nil {
		t.Fatalf("failed to create workflow node: %v", err)
	}

	outerEdges := Chain(Start, wfNode)
	outerWf := mustNew(t, outerEdges)

	// Run outer workflow
	mockCtx := newMockCtx(t)
	mockCtx.userContent = &genai.Content{
		Parts: []*genai.Part{{Text: "input"}},
	}

	events := outerWf.Run(mockCtx)

	var lastOutput any
	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if ev.Output != nil {
			lastOutput = ev.Output
		}
	}

	if lastOutput != "input-inner1-inner2" {
		t.Errorf("expected last output 'input-inner1-inner2', got %v", lastOutput)
	}
}

func TestNestedWorkflowUpdatesStateOuterReads(t *testing.T) {
	// Create inner workflow edges
	nestedStateUpdater := func(ctx agent.InvocationContext, input string) (string, error) {
		err := ctx.Session().State().Set("my_key", "my_value")
		if err != nil {
			return "", err
		}
		return "nested agent finished", nil
	}
	nestedNode := NewFunctionNode("nested_state_updater", nestedStateUpdater, defaultNodeConfig)
	nestedEdges := Chain(Start, nestedNode)
	
	wfNode, err := NewWorkflowNode("nested_agent", nestedEdges)
	if err != nil {
		t.Fatalf("failed to create workflow node: %v", err)
	}

	// Create outer workflow
	outerStateReader := func(ctx agent.InvocationContext, input string) (string, error) {
		val, err := ctx.Session().State().Get("my_key")
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("Nested agent output: %s, state value: %v", input, val), nil
	}
	outerNode := NewFunctionNode("outer_state_reader", outerStateReader, defaultNodeConfig)

	outerEdges := Chain(Start, wfNode, outerNode)
	outerWf := mustNew(t, outerEdges)

	// Run outer workflow
	mockCtx := newMockCtx(t)
	mockCtx.userContent = &genai.Content{
		Parts: []*genai.Part{{Text: "input"}},
	}
	
	mState := &mockWorkflowState{data: make(map[string]any)}
	mSess := &mockWorkflowSession{state: mState}
	mockCtx.sess = mSess

	events := outerWf.Run(mockCtx)

	var lastOutput any
	for ev, err := range events {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if ev.Output != nil {
			lastOutput = ev.Output
		}
	}

	expected := "Nested agent output: nested agent finished, state value: my_value"
	if lastOutput != expected {
		t.Errorf("expected last output %q, got %q", expected, lastOutput)
	}
}

func TestNestedWorkflow_Cancellation(t *testing.T) {
	// Arrange: Create inner workflow with a node that waits
	// and the outer workflow that should be cancelled.
	ch := make(chan struct{})
	started := make(chan struct{})
	waitingFn := func(ctx agent.InvocationContext, input string) (string, error) {
		close(started)
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-ch:
			return "done", nil
		}
	}
	waitingNode := NewFunctionNode("waiting_node", waitingFn, defaultNodeConfig)
	innerEdges := Chain(Start, waitingNode)
	
	wfNode, err := NewWorkflowNode("nested_agent", innerEdges)
	if err != nil {
		t.Fatalf("failed to create workflow node: %v", err)
	}
	outerEdges := Chain(Start, wfNode)
	outerWf := mustNew(t, outerEdges)


	// Act: run the outer workflow with a cancellable context and cancel it.
	baseCtx, cancel := context.WithCancel(t.Context())
	mockCtx := &MockInvocationContext{Context: baseCtx}
	
	var runErr error
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for _, err := range outerWf.Run(mockCtx) {
			if err != nil {
				runErr = err
			}
		}
	}()
	// Wait for the node to start running deterministically
	<-started
	cancel()
	wg.Wait()

	// Assert: run finished without error (cancellation is handled gracefully) and context was cancelled.
	if runErr != nil {
		t.Errorf("expected no error on cancellation, got %v", runErr)
	}
	if !errors.Is(baseCtx.Err(), context.Canceled) {
		t.Errorf("expected baseCtx.Err() to be context.Canceled, got %v", baseCtx.Err())
	}
}

func TestNestedWorkflow_ErrorPropagation(t *testing.T) {
	// Create inner workflow with a node that fails
	failingFn := func(ctx agent.InvocationContext, input string) (string, error) {
		return "", errors.New("intentional failure")
	}
	failingNode := NewFunctionNode("failing_node", failingFn, defaultNodeConfig)
	innerEdges := Chain(Start, failingNode)
	
	wfNode, err := NewWorkflowNode("nested_agent", innerEdges)
	if err != nil {
		t.Fatalf("failed to create workflow node: %v", err)
	}

	outerEdges := Chain(Start, wfNode)
	outerWf := mustNew(t, outerEdges)

	// Run outer workflow
	mockCtx := newMockCtx(t)
	
	var runErr error
	for _, err := range outerWf.Run(mockCtx) {
		if err != nil {
			runErr = err
		}
	}

	if runErr == nil {
		t.Fatalf("expected error, got nil")
	}
	if !strings.Contains(runErr.Error(), "intentional failure") {
		t.Errorf("expected error containing 'intentional failure', got %v", runErr)
	}
}
