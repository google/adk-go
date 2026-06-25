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
	"iter"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// hitlNode is a small custom Node used by the HITL pause tests. The
// Run callback is supplied per test so each scenario can shape its
// own emission pattern (single request, multiple requests, request
// then error, etc.).
type hitlNode struct {
	BaseNode
	run func(ctx agent.Context, input any, yield func(*session.Event, error) bool)
}

func newHitlNode(name string, run func(ctx agent.Context, input any, yield func(*session.Event, error) bool)) *hitlNode {
	return &hitlNode{
		BaseNode: NewBaseNode(name, "", defaultNodeConfig),
		run:      run,
	}
}

func (n *hitlNode) Run(_ context.Context, ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.run(ctx, input, yield)
	}
}

// findRequestedInputEvent returns the first yielded event whose
// RequestedInput field is non-nil, plus how many such events were
// observed in total. Useful for asserting that the engine forwarded
// the HITL prompt to the caller.
func findRequestedInputEvent(events []*session.Event) (req *session.Event, count int) {
	for _, ev := range events {
		if ev != nil && ev.RequestedInput != nil {
			if req == nil {
				req = ev
			}
			count++
		}
	}
	return req, count
}

// TestScheduler_HitlNode_PausesAndForwardsRequest verifies the
// happy-path single-waiting-node scenario: a node yields one
// RequestInput event and exits; the engine forwards the event
// downstream, does not schedule the successor, and the workflow
// terminates cleanly with the asker parked.
func TestScheduler_HitlNode_PausesAndForwardsRequest(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	var downstreamRan atomic.Bool
	asker := newHitlNode("asker", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		yield(NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID: "human_review",
			Message:     "Please approve",
			Payload:     "the draft",
		}), nil)
	})
	downstream := newHitlNode("downstream", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		downstreamRan.Store(true)
	})

	w := mustNew(t, []Edge{
		{From: Start, To: asker},
		{From: asker, To: downstream},
	})

	events := drain(t, w.Run(t.Context(), mockCtx))

	if downstreamRan.Load() {
		t.Error("downstream node ran; HITL pause must suppress successor scheduling")
	}

	req, count := findRequestedInputEvent(events)
	if count != 1 {
		t.Fatalf("expected exactly 1 event with RequestedInput, got %d (events=%d total)", count, len(events))
	}
	if got, want := req.RequestedInput.InterruptID, "human_review"; got != want {
		t.Errorf("InterruptID = %q, want %q", got, want)
	}
	if got, want := req.RequestedInput.Message, "Please approve"; got != want {
		t.Errorf("Message = %q, want %q", got, want)
	}
	if got, want := req.RequestedInput.Payload, "the draft"; got != want {
		t.Errorf("Payload = %v, want %q", got, want)
	}
}

// TestScheduler_HitlNode_AutoGeneratesInterruptID verifies the
// scheduler-side contract that an empty InterruptID supplied by the
// node author becomes a non-empty UUID by the time the request
// reaches the consumer.
func TestScheduler_HitlNode_AutoGeneratesInterruptID(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	asker := newHitlNode("asker", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		// InterruptID intentionally left empty.
		yield(NewRequestInputEvent(ctx, session.RequestInput{
			Message: "approve?",
		}), nil)
	})
	w := mustNew(t, []Edge{{From: Start, To: asker}})

	events := drain(t, w.Run(t.Context(), mockCtx))
	req, count := findRequestedInputEvent(events)
	if count != 1 {
		t.Fatalf("expected exactly 1 RequestedInput event, got %d", count)
	}
	if req.RequestedInput.InterruptID == "" {
		t.Error("InterruptID is empty; engine must auto-generate when caller omits it")
	}
}

// TestScheduler_HitlNode_PreservesExplicitInterruptID verifies
// that an explicit InterruptID supplied by the node author flows
// through to the consumer unchanged.
func TestScheduler_HitlNode_PreservesExplicitInterruptID(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	asker := newHitlNode("asker", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		yield(NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID: "stable-id-42",
			Message:     "approve?",
		}), nil)
	})
	w := mustNew(t, []Edge{{From: Start, To: asker}})

	events := drain(t, w.Run(t.Context(), mockCtx))
	req, _ := findRequestedInputEvent(events)
	if req == nil {
		t.Fatal("no RequestedInput event found")
	}
	if got, want := req.RequestedInput.InterruptID, "stable-id-42"; got != want {
		t.Errorf("InterruptID = %q, want %q (engine must preserve caller-supplied IDs)", got, want)
	}
}

// TestScheduler_HitlNode_MultipleRequestsPark verifies a node may
// raise more than one interrupt in a single activation: both are
// recorded on NodeState.Interrupts and the node parks NodeWaiting
// (matching adk-python, which accumulates a set of interrupt IDs).
func TestScheduler_HitlNode_MultipleRequestsPark(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	asker := newHitlNode("asker", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "first"}), nil)
		yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "second"}), nil)
	})

	w := mustNew(t, []Edge{{From: Start, To: asker}})

	// drain fails the test on any error; multiple interrupts must
	// park cleanly rather than surface an error. Both pause events
	// carry their interrupt on LongRunningToolIDs — the signal the
	// scheduler accumulates and history rehydration reads back.
	events := drain(t, w.Run(t.Context(), mockCtx))
	got := map[string]bool{}
	for _, ev := range events {
		for _, id := range ev.LongRunningToolIDs {
			got[id] = true
		}
	}
	if !got["first"] || !got["second"] {
		t.Errorf("long-running interrupts = %v, want both first and second", got)
	}
}

// TestScheduler_HitlNode_ErrorAfterRequestFails verifies the
// precedence ordering inside handleCompletion: a node that
// recorded a request and then returned an error surfaces as
// failed, not as waiting. The waiting branch runs only on a
// clean activation.
func TestScheduler_HitlNode_ErrorAfterRequestFails(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	wantErr := errors.New("downstream of request")
	asker := newHitlNode("asker", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "ignored"}), nil)
		yield(nil, wantErr)
	})
	w := mustNew(t, []Edge{{From: Start, To: asker}})

	gotErr := drainErr(t, w.Run(t.Context(), mockCtx))
	if !errors.Is(gotErr, wantErr) {
		t.Errorf("Run error = %v, want %v (failure must take precedence over the recorded request)", gotErr, wantErr)
	}
}

// TestScheduler_HitlNode_ConcurrentBranches_PausesOnlyWhenAllNonRunning
// verifies that a workflow with two parallel branches — one that
// requests input, one that completes normally — pauses cleanly:
// the non-HITL branch's downstream still runs, the HITL branch's
// downstream does not. Termination happens once the last live node
// has either completed or moved into NodeWaiting.
func TestScheduler_HitlNode_ConcurrentBranches_PausesOnlyWhenAllNonRunning(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	var hitlDownstreamRan atomic.Bool
	var plainDownstreamRan atomic.Bool

	hitlAsker := newHitlNode("hitl_asker", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "ask"}), nil)
	})
	hitlDownstream := newHitlNode("hitl_downstream", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		hitlDownstreamRan.Store(true)
	})
	plainNode := newHitlNode("plain", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx.InvocationID())
		ev.Output = "done"
		yield(ev, nil)
	})
	plainDownstream := newHitlNode("plain_downstream", func(ctx agent.Context, _ any, yield func(*session.Event, error) bool) {
		plainDownstreamRan.Store(true)
	})

	w := mustNew(t, []Edge{
		{From: Start, To: hitlAsker},
		{From: Start, To: plainNode},
		{From: hitlAsker, To: hitlDownstream},
		{From: plainNode, To: plainDownstream},
	})

	events := drain(t, w.Run(t.Context(), mockCtx))

	if hitlDownstreamRan.Load() {
		t.Error("hitl_downstream ran; HITL branch must not schedule successors before resume")
	}
	if !plainDownstreamRan.Load() {
		t.Error("plain_downstream did not run; non-HITL branch must complete normally even while a sibling pauses")
	}
	if _, count := findRequestedInputEvent(events); count != 1 {
		t.Errorf("expected exactly 1 RequestedInput event in the stream, got %d", count)
	}
}

func TestResumeOrRequestInput(t *testing.T) {
	t.Run("first pass emits request and pauses", func(t *testing.T) {
		ctx := agent.NewNodeContext(t.Context(), newMockCtx(t), nil)

		var emitted []*session.Event
		emit := func(ev *session.Event) error {
			emitted = append(emitted, ev)
			return nil
		}

		reply, err := ResumeOrRequestInput(ctx, emit, session.RequestInput{
			InterruptID: "ask_name",
			Message:     "What's your name?",
		})
		if !errors.Is(err, ErrNodeInterrupted) {
			t.Fatalf("err = %v, want ErrNodeInterrupted", err)
		}
		if reply != nil {
			t.Errorf("reply = %v, want nil on first pass", reply)
		}
		if len(emitted) != 1 || emitted[0].RequestedInput == nil {
			t.Fatalf("expected one RequestInput event, got %v", emitted)
		}
		if got := emitted[0].RequestedInput.InterruptID; got != "ask_name" {
			t.Errorf("InterruptID = %q, want %q", got, "ask_name")
		}
	})

	t.Run("resume returns reply without emitting", func(t *testing.T) {
		ctx := agent.NewNodeContext(t.Context(), newMockCtx(t), map[string]any{"ask_name": "Alice"})

		emitted := false
		emit := func(*session.Event) error {
			emitted = true
			return nil
		}

		reply, err := ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "ask_name"})
		if err != nil {
			t.Fatalf("err = %v, want nil on resume", err)
		}
		if reply != "Alice" {
			t.Errorf("reply = %v, want %q", reply, "Alice")
		}
		if emitted {
			t.Error("emit called on resume; the request must not be re-sent")
		}
	})

	t.Run("emit failure is returned instead of ErrNodeInterrupted", func(t *testing.T) {
		ctx := agent.NewNodeContext(t.Context(), newMockCtx(t), nil)

		wantErr := errors.New("emit failed")
		emit := func(*session.Event) error { return wantErr }

		_, err := ResumeOrRequestInput(ctx, emit, session.RequestInput{InterruptID: "ask_name"})
		if !errors.Is(err, wantErr) {
			t.Fatalf("err = %v, want %v", err, wantErr)
		}
		if errors.Is(err, ErrNodeInterrupted) {
			t.Error("emit failure must not be reported as ErrNodeInterrupted")
		}
	})
}

// TestScheduler_WaitForOutputPause_SuspendsWorkflow drives the
// WaitForOutput pause through a real workflow (Start -> orch ->
// downstream). orch runs a WaitForOutput child that yields no output,
// so RunNode parks the parent. A genuine pause must suspend the
// workflow: orch lands NodeWaiting and downstream must NOT run.
func TestScheduler_WaitForOutputPause_SuspendsWorkflow(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	child := newWaitForOutputNode("waiter")
	orch := NewDynamicNode[any, any](
		"orch",
		func(ctx context.Context, invCleanCtx NodeContext, _ any, _ func(*session.Event) error) (any, error) {
			return RunNode[any](ctx, invCleanCtx, child, nil)
		},
		NodeConfig{},
	)
	var downstreamRan atomic.Bool
	downstream := newHitlNode("downstream", func(_ agent.Context, _ any, _ func(*session.Event, error) bool) {
		downstreamRan.Store(true)
	})

	w := mustNew(t, []Edge{
		{From: Start, To: orch},
		{From: orch, To: downstream},
	})

	drain(t, w.Run(t.Context(), mockCtx))

	if downstreamRan.Load() {
		t.Error("downstream ran; a WaitForOutput pause must suspend the workflow, not complete the parent")
	}
}
