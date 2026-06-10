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
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

func TestNewDynamicNode_DefaultsRerunOnResume(t *testing.T) {
	fn := func(NodeContext, string, func(*session.Event) error) (string, error) {
		return "", nil
	}
	n := NewDynamicNode[string, string]("d", fn, NodeConfig{})
	rr := n.Config().RerunOnResume
	if rr == nil || !*rr {
		t.Errorf("RerunOnResume = %v, want &true (default)", rr)
	}
}

func TestNewDynamicNode_RespectsExplicitFalse(t *testing.T) {
	f := false
	fn := func(NodeContext, string, func(*session.Event) error) (string, error) {
		return "", nil
	}
	n := NewDynamicNode[string, string]("d", fn, NodeConfig{RerunOnResume: &f})
	if rr := n.Config().RerunOnResume; rr == nil || *rr {
		t.Errorf("RerunOnResume = %v, want &false (explicit override)", rr)
	}
}

func TestDynamicNode_Sequential_RunNodeChain(t *testing.T) {
	stepA := newStubNode("stepA", "from A")
	stepB := newStubNode("stepB", "from B")

	orchestrator := NewDynamicNode[string, string]("orch",
		func(ctx NodeContext, _ string, _ func(*session.Event) error) (string, error) {
			outA, err := RunNode[string](ctx, stepA, "ignored")
			if err != nil {
				return "", err
			}
			outB, err := RunNode[string](ctx, stepB, outA)
			if err != nil {
				return "", err
			}
			return outA + " | " + outB, nil
		},
		NodeConfig{},
	)

	events := drainDynamic(t, orchestrator, "input")
	last := events[len(events)-1]
	if last.Output != "from A | from B" {
		t.Errorf("terminal Output = %v, want %q", last.Output, "from A | from B")
	}
}

func TestDynamicNode_TypedInput(t *testing.T) {
	type Req struct{ Name string }

	var observed Req
	orchestrator := NewDynamicNode[Req, string]("orch",
		func(_ NodeContext, in Req, _ func(*session.Event) error) (string, error) {
			observed = in
			return "ok", nil
		},
		NodeConfig{},
	)

	drainDynamic(t, orchestrator, Req{Name: "alice"})
	if observed.Name != "alice" {
		t.Errorf("observed Req.Name = %q, want %q", observed.Name, "alice")
	}
}

func TestDynamicNode_TypedInput_JSONFallback(t *testing.T) {
	// Upstream produces map[string]any (e.g. a tool node). Constructor
	// coerces to the typed struct via typeutil JSON roundtrip.
	type Req struct {
		Name string `json:"name"`
		N    int    `json:"n"`
	}

	var observed Req
	orchestrator := NewDynamicNode[Req, string]("orch",
		func(_ NodeContext, in Req, _ func(*session.Event) error) (string, error) {
			observed = in
			return "ok", nil
		},
		NodeConfig{},
	)

	drainDynamic(t, orchestrator, map[string]any{"name": "bob", "n": 7})
	if observed.Name != "bob" || observed.N != 7 {
		t.Errorf("observed = %+v, want {Name:bob N:7}", observed)
	}
}

func TestDynamicNode_EmitMidBody(t *testing.T) {
	orchestrator := NewDynamicNode[string, string]("orch",
		func(_ NodeContext, _ string, emit func(*session.Event) error) (string, error) {
			if err := emit(&session.Event{Actions: session.EventActions{
				StateDelta: map[string]any{"progress": "halfway"},
			}}); err != nil {
				return "", err
			}
			return "done", nil
		},
		NodeConfig{},
	)

	events := drainDynamic(t, orchestrator, "")
	if len(events) < 2 {
		t.Fatalf("got %d events, want >= 2 (mid-body emit + terminal)", len(events))
	}
	if got, want := events[0].Actions.StateDelta["progress"], "halfway"; got != want {
		t.Errorf("first event StateDelta progress = %v, want %q", got, want)
	}
	if events[len(events)-1].Output != "done" {
		t.Errorf("terminal Output = %v, want \"done\"", events[len(events)-1].Output)
	}
}

func TestDynamicNode_HITL_SwallowsInterrupt(t *testing.T) {
	// Pause already reached the engine via the forwarded
	// RequestedInput event; Run must not also yield the sentinel.
	asker := newRequestInputNode("asker", "approve?")
	orchestrator := NewDynamicNode[string, string]("orch",
		func(ctx NodeContext, _ string, _ func(*session.Event) error) (string, error) {
			_, err := RunNode[string](ctx, asker, nil)
			return "", err
		},
		NodeConfig{},
	)

	events, runErr := drainDynamicWithErr(t, orchestrator, "")
	if runErr != nil {
		t.Errorf("Run yielded error %v; want nil (HITL swallowed)", runErr)
	}
	if !hasRequestedInput(events) {
		t.Errorf("expected RequestedInput event in stream, got %+v", events)
	}
}

func TestDynamicNode_ChildFailure_PropagatesError(t *testing.T) {
	failer := newFailingNode("failer", errors.New("boom"))
	orchestrator := NewDynamicNode[string, string]("orch",
		func(ctx NodeContext, _ string, _ func(*session.Event) error) (string, error) {
			_, err := RunNode[string](ctx, failer, nil)
			return "", err
		},
		NodeConfig{},
	)

	_, runErr := drainDynamicWithErr(t, orchestrator, "")
	if !errors.Is(runErr, ErrNodeFailed) {
		t.Errorf("Run error = %v, want errors.Is ErrNodeFailed", runErr)
	}
}

func TestDynamicNode_TerminalOutputEvent(t *testing.T) {
	orchestrator := NewDynamicNode[string, int]("orch",
		func(NodeContext, string, func(*session.Event) error) (int, error) {
			return 42, nil
		},
		NodeConfig{},
	)
	events := drainDynamic(t, orchestrator, "")
	last := events[len(events)-1]
	if last.Output != 42 {
		t.Errorf("Output = %v, want 42", last.Output)
	}
}

// TestDynamicNode_Integration_ChildAndParentOutputs verifies that
// when a dynamic orchestrator calls a child via RunNode, both the
// child's terminal output event and the parent's own terminal output
// reach the workflow stream without the top-level scheduler rejecting
// the pair as "multiple outputs per activation".
func TestDynamicNode_Integration_ChildAndParentOutputs(t *testing.T) {
	helloNode := NewFunctionNode("hello_node",
		func(_ agent.InvocationContext, _ string) (string, error) {
			return "Hello World", nil
		},
		NodeConfig{},
	)
	orch := NewDynamicNode[string, string]("my_workflow",
		func(ctx NodeContext, _ string, _ func(*session.Event) error) (string, error) {
			return RunNode[string](ctx, helloNode, "hello")
		},
		NodeConfig{},
	)

	w, err := New("root", Chain(Start, orch))
	if err != nil {
		t.Fatalf("workflow.New: %v", err)
	}

	var outputs []any
	for ev, err := range w.Run(newMockCtx(t)) {
		if err != nil {
			t.Fatalf("workflow.Run error: %v", err)
		}
		if ev != nil && ev.Output != nil {
			outputs = append(outputs, ev.Output)
		}
	}
	if len(outputs) != 2 {
		t.Fatalf("got %d output events, want 2 (child + parent terminal); outputs=%v", len(outputs), outputs)
	}
	for _, out := range outputs {
		if out != "Hello World" {
			t.Errorf("output = %v, want %q", out, "Hello World")
		}
	}
}

func TestNewDynamicNodeWithSchema_NilSchemasOK(t *testing.T) {
	fn := func(NodeContext, string, func(*session.Event) error) (string, error) { return "", nil }
	if _, err := NewDynamicNodeWithSchema[string, string]("d", fn, nil, nil, NodeConfig{}); err != nil {
		t.Errorf("nil schemas should construct cleanly, got %v", err)
	}
}

// --- test helpers ---

func drainDynamic(t *testing.T, n Node, input any) []*session.Event {
	t.Helper()
	events, err := drainDynamicWithErr(t, n, input)
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	return events
}

func drainDynamicWithErr(t *testing.T, n Node, input any) ([]*session.Event, error) {
	t.Helper()
	parent := newNodeContext(newMockCtx(t), nil)
	var events []*session.Event
	for ev, err := range n.Run(parent, input) {
		if err != nil {
			return events, err
		}
		if ev != nil {
			events = append(events, ev)
		}
	}
	return events, nil
}

func hasRequestedInput(events []*session.Event) bool {
	for _, ev := range events {
		if ev.RequestedInput != nil {
			return true
		}
	}
	return false
}

type failingNode struct {
	BaseNode
	err error
}

func newFailingNode(name string, err error) *failingNode {
	return &failingNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{}),
		err:      err,
	}
}

func (n *failingNode) Run(agent.InvocationContext, any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		yield(nil, n.err)
	}
}

// requestInputNode emits one RequestedInput event and exits cleanly.
type requestInputNode struct {
	BaseNode
	message string
}

func newRequestInputNode(name, msg string) *requestInputNode {
	return &requestInputNode{
		BaseNode: NewBaseNode(name, "", NodeConfig{}),
		message:  msg,
	}
}

func (n *requestInputNode) Run(agent.InvocationContext, any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		yield(&session.Event{
			RequestedInput: &session.RequestInput{
				InterruptID: "iid-1",
				Message:     n.message,
			},
		}, nil)
	}
}
