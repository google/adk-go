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
	"errors"
	"iter"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// agentNodeMockSession satisfies the small subset of
// session.Session that agent.New's wrapper touches when its Run
// is invoked: ID() is read by telemetry's StartInvokeAgentSpan.
// The shared MockInvocationContext in workflow_test.go has a nil
// Session() by default, which is fine for tests that drive only
// FunctionNode/BaseNode bodies but panics the moment we route an
// activation through agent.New.
type agentNodeMockSession struct {
	session.Session
}

func (agentNodeMockSession) ID() string { return "test-session" }

// withMockSession returns a copy of ctx whose Session() yields a
// minimal stub. Wrapping is enough for agent.New's hot path
// because nothing in the wrapped agent's Run body looks any
// further into the session.
func withMockSession(ctx *MockInvocationContext) *MockInvocationContext {
	out := *ctx
	out.sess = agentNodeMockSession{}
	return &out
}

// newTestAgent builds a minimal agent.Agent whose Run yields the
// supplied events in order. Errors carried on those tuples
// surface through the iterator unchanged.
func newTestAgent(t *testing.T, name string, events []agentEmit) agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		Name:        name,
		Description: "test agent for AgentNode wrapping",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for _, e := range events {
					if !yield(e.event, e.err) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	return a
}

// agentEmit pairs an event with an error so test fixtures can
// describe both happy and unhappy emissions in a single slice.
type agentEmit struct {
	event *session.Event
	err   error
}

// makeOutputEvent builds an event carrying value under
// StateDelta["output"], the magic key the workflow scheduler
// reads to decide what input to hand the next node.
func makeOutputEvent(invocationID string, value any) *session.Event {
	ev := session.NewEvent(invocationID)
	ev.Actions.StateDelta["output"] = value
	return ev
}

// TestNewAgentNode_ForwardsAllEventsToWorkflow verifies that the
// Run wrapper hands every event the wrapped agent yields straight
// through to the workflow's iterator, in order, including any
// error carried alongside an event.
func TestNewAgentNode_ForwardsAllEventsToWorkflow(t *testing.T) {
	mockCtx := withMockSession(newMockCtx(t))
	want := []agentEmit{
		{event: makeOutputEvent(mockCtx.InvocationID(), "first"), err: nil},
		{event: makeOutputEvent(mockCtx.InvocationID(), "second"), err: nil},
	}
	a := newTestAgent(t, "fwd_agent", want)
	node := NewAgentNode(a, NodeConfig{})

	got := drain(t, node.Run(mockCtx, nil))
	if len(got) != len(want) {
		t.Fatalf("event count = %d, want %d", len(got), len(want))
	}
	for i, ev := range got {
		if ev != want[i].event {
			t.Errorf("event[%d] = %p, want %p (events must be forwarded by reference)", i, ev, want[i].event)
		}
	}
}

// TestNewAgentNode_PropagatesError verifies that an error the
// wrapped agent emits surfaces through the wrapper unchanged.
func TestNewAgentNode_PropagatesError(t *testing.T) {
	mockCtx := withMockSession(newMockCtx(t))
	wantErr := errors.New("agent failed")
	a := newTestAgent(t, "err_agent", []agentEmit{{event: nil, err: wantErr}})
	node := NewAgentNode(a, NodeConfig{})

	gotErr := drainErr(t, node.Run(mockCtx, nil))
	if !errors.Is(gotErr, wantErr) {
		t.Errorf("error = %v, want %v", gotErr, wantErr)
	}
}

// TestNewAgentNode_StopsOnYieldFalse verifies that the wrapper
// honours a downstream cancel: when the consumer returns false
// from yield, the wrapped agent's iterator stops being driven.
// Without this, a cancelled workflow would keep pulling events
// from the agent.
func TestNewAgentNode_StopsOnYieldFalse(t *testing.T) {
	mockCtx := withMockSession(newMockCtx(t))
	var pulled atomic.Int32

	a, err := agent.New(agent.Config{
		Name: "long_agent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				for i := 0; i < 5; i++ {
					pulled.Add(1)
					if !yield(makeOutputEvent(ctx.InvocationID(), i), nil) {
						return
					}
				}
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	node := NewAgentNode(a, NodeConfig{})

	for ev, err := range node.Run(mockCtx, nil) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_ = ev
		break // cancel after the first event
	}

	if got := pulled.Load(); got != 1 {
		t.Errorf("agent yielded %d events, want exactly 1 before cancel", got)
	}
}

// TestNewAgentNode_InheritsNameAndDescriptionFromAgent pins the
// contract that the node identifies itself with the wrapped
// agent's name and description, so the scheduler's per-name
// bookkeeping (RunState.Nodes, runsByName) and any per-name UI
// label match the agent the user actually configured.
func TestNewAgentNode_InheritsNameAndDescriptionFromAgent(t *testing.T) {
	a := newTestAgent(t, "configured_name", nil)
	node := NewAgentNode(a, NodeConfig{})

	if got, want := node.Name(), "configured_name"; got != want {
		t.Errorf("Name() = %q, want %q", got, want)
	}
	if got, want := node.Description(), "test agent for AgentNode wrapping"; got != want {
		t.Errorf("Description() = %q, want %q", got, want)
	}
}

// TestNewAgentNode_E2E_AgentInsideWorkflow exercises the full
// integration: build a Workflow with START → AgentNode →
// FunctionNode and verify (a) the agent runs, (b) its output
// (carried via StateDelta["output"]) is forwarded as the next
// node's input. This is the contract that makes AgentNode
// composable with the rest of the engine.
func TestNewAgentNode_E2E_AgentInsideWorkflow(t *testing.T) {
	mockCtx := withMockSession(newSeededMockCtx(t))

	// The wrapped agent emits one event with output="from agent",
	// imitating an LLMAgent configured with OutputKey="output".
	a := newTestAgent(t, "producer_agent", []agentEmit{
		{event: makeOutputEvent(mockCtx.InvocationID(), "from agent"), err: nil},
	})
	agentNode := NewAgentNode(a, NodeConfig{})

	// Successor reads the agent's output as its typed input.
	var consumed atomic.Value
	consumer := NewFunctionNode(
		"consumer",
		func(_ agent.InvocationContext, in string) (any, error) {
			consumed.Store(in)
			return nil, nil
		},
		NodeConfig{},
	)

	w := mustNew(t, []Edge{
		{From: Start, To: agentNode},
		{From: agentNode, To: consumer},
	})

	drain(t, w.Run(mockCtx))

	if got := consumed.Load(); got != "from agent" {
		t.Errorf("consumer input = %v, want %q (AgentNode must forward output via StateDelta)", got, "from agent")
	}
}

// makeOutputEvent's signature uses *session.Event so we want a
// compile-time assertion that it builds something with the
// embedded Content / Actions structure the scheduler relies on.
// Genai package gets imported and used here so the build catches
// an accidental drift in the Event shape.
var _ = genai.NewContentFromText
