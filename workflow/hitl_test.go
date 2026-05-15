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

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// hitlNode is a small custom Node used by the HITL pause tests. The
// Run callback is supplied per test so each scenario can shape its
// own emission pattern (single request, multiple requests, request
// then error, etc.).
type hitlNode struct {
	BaseNode
	run func(ctx agent.InvocationContext, input any, yield func(*session.Event, error) bool)
}

func newHitlNode(name string, run func(ctx agent.InvocationContext, input any, yield func(*session.Event, error) bool)) *hitlNode {
	return &hitlNode{
		BaseNode: NewBaseNode(name, "", defaultNodeConfig),
		run:      run,
	}
}

func (n *hitlNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
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
	asker := newHitlNode("asker", func(ctx agent.InvocationContext, _ any, yield func(*session.Event, error) bool) {
		yield(NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID: "human_review",
			Message:     "Please approve",
			Payload:     "the draft",
		}), nil)
	})
	downstream := newHitlNode("downstream", func(ctx agent.InvocationContext, _ any, yield func(*session.Event, error) bool) {
		downstreamRan.Store(true)
	})

	w := mustNew(t, []Edge{
		{From: Start, To: asker},
		{From: asker, To: downstream},
	})

	events := drain(t, w.Run(mockCtx))

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

	asker := newHitlNode("asker", func(ctx agent.InvocationContext, _ any, yield func(*session.Event, error) bool) {
		// InterruptID intentionally left empty.
		yield(NewRequestInputEvent(ctx, session.RequestInput{
			Message: "approve?",
		}), nil)
	})
	w := mustNew(t, []Edge{{From: Start, To: asker}})

	events := drain(t, w.Run(mockCtx))
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

	asker := newHitlNode("asker", func(ctx agent.InvocationContext, _ any, yield func(*session.Event, error) bool) {
		yield(NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID: "stable-id-42",
			Message:     "approve?",
		}), nil)
	})
	w := mustNew(t, []Edge{{From: Start, To: asker}})

	events := drain(t, w.Run(mockCtx))
	req, _ := findRequestedInputEvent(events)
	if req == nil {
		t.Fatal("no RequestedInput event found")
	}
	if got, want := req.RequestedInput.InterruptID, "stable-id-42"; got != want {
		t.Errorf("InterruptID = %q, want %q (engine must preserve caller-supplied IDs)", got, want)
	}
}

// TestScheduler_HitlNode_MultipleRequestsFails verifies the
// single-request-per-activation invariant: a node yielding two
// RequestedInput events surfaces ErrMultipleInputRequests at
// completion and is treated as failed, so it does not silently
// park in NodeWaiting.
func TestScheduler_HitlNode_MultipleRequestsFails(t *testing.T) {
	mockCtx := newSeededMockCtx(t)

	asker := newHitlNode("asker", func(ctx agent.InvocationContext, _ any, yield func(*session.Event, error) bool) {
		yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "first"}), nil)
		yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "second"}), nil)
	})

	w := mustNew(t, []Edge{{From: Start, To: asker}})

	gotErr := drainErr(t, w.Run(mockCtx))
	if !errors.Is(gotErr, ErrMultipleInputRequests) {
		t.Errorf("Run error = %v, want ErrMultipleInputRequests (node must fail, not park)", gotErr)
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
	asker := newHitlNode("asker", func(ctx agent.InvocationContext, _ any, yield func(*session.Event, error) bool) {
		yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "ignored"}), nil)
		yield(nil, wantErr)
	})
	w := mustNew(t, []Edge{{From: Start, To: asker}})

	gotErr := drainErr(t, w.Run(mockCtx))
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

	hitlAsker := newHitlNode("hitl_asker", func(ctx agent.InvocationContext, _ any, yield func(*session.Event, error) bool) {
		yield(NewRequestInputEvent(ctx, session.RequestInput{InterruptID: "ask"}), nil)
	})
	hitlDownstream := newHitlNode("hitl_downstream", func(ctx agent.InvocationContext, _ any, yield func(*session.Event, error) bool) {
		hitlDownstreamRan.Store(true)
	})
	plainNode := newHitlNode("plain", func(ctx agent.InvocationContext, _ any, yield func(*session.Event, error) bool) {
		ev := session.NewEvent(ctx.InvocationID())
		ev.Actions.StateDelta["output"] = "done"
		yield(ev, nil)
	})
	plainDownstream := newHitlNode("plain_downstream", func(ctx agent.InvocationContext, _ any, yield func(*session.Event, error) bool) {
		plainDownstreamRan.Store(true)
	})

	w := mustNew(t, []Edge{
		{From: Start, To: hitlAsker},
		{From: Start, To: plainNode},
		{From: hitlAsker, To: hitlDownstream},
		{From: plainNode, To: plainDownstream},
	})

	events := drain(t, w.Run(mockCtx))

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
