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
	"iter"
	"sync"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
	"google.golang.org/adk/workflow"
)

// reentryNode is a custom Node with NodeConfig.RerunOnResume = true.
// On the first activation it issues a RequestInput and exits; on the
// re-entry activation it observes the user's response via
// ctx.ResumedInput and emits an output, optionally calling a hook
// supplied by the test.
type reentryNode struct {
	workflow.BaseNode
	interruptID string
	onResume    func(response any, originalInput any) (output any)
}

func newReentryNode(name, interruptID string, onResume func(response, originalInput any) any) *reentryNode {
	t := true
	return &reentryNode{
		BaseNode:    workflow.NewBaseNode(name, "", workflow.NodeConfig{RerunOnResume: &t}),
		interruptID: interruptID,
		onResume:    onResume,
	}
}

func (n *reentryNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		if response, ok := ctx.ResumedInput(n.interruptID); ok {
			out := n.onResume(response, input)
			ev := session.NewEvent(ctx.InvocationID())
			ev.Actions.StateDelta["output"] = out
			yield(ev, nil)
			return
		}
		// First activation: ask.
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID: n.interruptID,
			Message:     "decide",
		}), nil)
	}
}

// TestWorkflowAgent_ReEntry_ResumesAtSameNode verifies the canonical
// re-entry round-trip: the asker is re-activated on resume, sees
// the response via ResumedInput, and produces an output that flows
// to its successor.
func TestWorkflowAgent_ReEntry_ResumesAtSameNode(t *testing.T) {
	var resumeActivations atomic.Int32
	var capturedResponse atomic.Value

	asker := newReentryNode("asker", "decide", func(response, _ any) any {
		resumeActivations.Add(1)
		capturedResponse.Store(response)
		return "decision: " + response.(string)
	})

	var handlerInput atomic.Value
	handler := newStringHandlerNode("handler", &handlerInput)

	a := makeAgent(t, workflow.Chain(workflow.Start, asker, handler))
	sess := newFakeSession()

	// Turn 1: fresh; pauses with RequestedInput.
	turn1 := runFreshTurn(t, sess, a, "draft")
	if got := findRequest(turn1); got != "decide" {
		t.Fatalf("turn 1 RequestedInput = %q, want %q", got, "decide")
	}
	if resumeActivations.Load() != 0 {
		t.Errorf("re-entry happened on turn 1; resumeActivations = %d", resumeActivations.Load())
	}

	// Turn 2: resume; asker re-runs with response visible via
	// ResumedInput, then handler runs.
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("decide", "approve"))), nil)
	if got := resumeActivations.Load(); got != 1 {
		t.Errorf("resume activations = %d, want 1", got)
	}
	if got := capturedResponse.Load(); got != "approve" {
		t.Errorf("ResumedInput response = %v, want %q", got, "approve")
	}
	if got, want := handlerInput.Load(), "decision: approve"; got != want {
		t.Errorf("handler input = %v, want %q", got, want)
	}
}

// TestWorkflowAgent_ReEntry_PreservesOriginalInput verifies that on
// re-entry the asker receives the SAME input value it saw on its
// first activation, not the user's response. The response is
// delivered separately via ResumedInput.
func TestWorkflowAgent_ReEntry_PreservesOriginalInput(t *testing.T) {
	var seenInputs sync.Map // attempt index -> input
	var attempts atomic.Int32

	// observeInput captures input on every activation (first or
	// re-entry); the body uses ResumedInput to decide whether to
	// emit the request or the final output.
	observeInput := func(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			seenInputs.Store(attempts.Load(), input)
			attempts.Add(1)
			if _, ok := ctx.ResumedInput("decide"); ok {
				ev := session.NewEvent(ctx.InvocationID())
				ev.Actions.StateDelta["output"] = "ack"
				yield(ev, nil)
				return
			}
			yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
				InterruptID: "decide",
				Message:     "?",
			}), nil)
		}
	}

	asker := &observingReentryNode{
		BaseNode: workflow.NewBaseNode("asker", "", workflow.NodeConfig{RerunOnResume: ptrTrue()}),
		runFn:    observeInput,
	}

	a := makeAgent(t, workflow.Chain(workflow.Start, asker))
	sess := newFakeSession()

	// Turn 1: fresh, asker sees "draft" as input.
	runFreshTurn(t, sess, a, "draft")

	// Turn 2: resume with response "approve". Asker re-runs and
	// must see "draft" again as input (not "approve").
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("decide", "approve"))), nil)

	first, _ := seenInputs.Load(int32(0))
	second, _ := seenInputs.Load(int32(1))
	if first != "draft" {
		t.Errorf("first activation input = %v, want %q", first, "draft")
	}
	if second != "draft" {
		t.Errorf("re-entry activation input = %v, want %q (must preserve original input, not the response)", second, "draft")
	}
}

// observingReentryNode is a Node whose Run delegates to a function-
// typed field, so individual tests can supply their own Run body
// without subclassing.
type observingReentryNode struct {
	workflow.BaseNode
	runFn func(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error]
}

func (n *observingReentryNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return n.runFn(ctx, input)
}

// TestWorkflowAgent_ReEntry_NoSuccessorBeforeOutput verifies that a
// re-entry asker's downstream successor only fires once the
// re-entry activation actually emits an output event. (Compare to
// handoff mode, where the successor would fire immediately on
// resume even without a re-entry activation.)
func TestWorkflowAgent_ReEntry_NoSuccessorBeforeOutput(t *testing.T) {
	var handlerCount atomic.Int32

	asker := newReentryNode("asker", "decide", func(response, _ any) any {
		return "ack"
	})
	handler := newCountingHandlerNode("handler", &handlerCount)

	a := makeAgent(t, workflow.Chain(workflow.Start, asker, handler))
	sess := newFakeSession()

	// Pause turn.
	runFreshTurn(t, sess, a, "x")
	if handlerCount.Load() != 0 {
		t.Errorf("handler ran during pause turn; count = %d", handlerCount.Load())
	}

	// Resume turn — asker re-runs and emits "ack"; handler runs
	// once with that input.
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("decide", "yes"))), nil)
	if got := handlerCount.Load(); got != 1 {
		t.Errorf("handler runs after re-entry = %d, want 1", got)
	}
}

// TestWorkflowAgent_ReEntry_DefaultModeIsHandoff confirms that a
// node without RerunOnResume = true uses handoff mode on resume:
// the response flows to the successor as its input and the asker
// does not re-run.
func TestWorkflowAgent_ReEntry_DefaultModeIsHandoff(t *testing.T) {
	var askerActivations atomic.Int32

	asker := newHitlNode("asker", func(ctx agent.InvocationContext, _ any, yield func(*session.Event, error) bool) {
		askerActivations.Add(1)
		yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
			InterruptID: "decide",
			Message:     "?",
		}), nil)
	})
	var handlerInput atomic.Value
	handler := newStringHandlerNode("handler", &handlerInput)

	a := makeAgent(t, workflow.Chain(workflow.Start, asker, handler))
	sess := newFakeSession()

	runFreshTurn(t, sess, a, "x")
	drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage("decide", "approve"))), nil)

	if got := askerActivations.Load(); got != 1 {
		t.Errorf("asker activations = %d, want 1 (handoff mode must NOT re-activate the asker)", got)
	}
	if got := handlerInput.Load(); got != "approve" {
		t.Errorf("handler input = %v, want %q (handoff mode delivers response as next-node input)", got, "approve")
	}
}

func ptrTrue() *bool {
	t := true
	return &t
}
