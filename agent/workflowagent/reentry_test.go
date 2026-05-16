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

// TestWorkflowAgent_ReEntry_ResumesAtSameNode verifies the canonical
// re-entry round-trip: the asker is re-activated on resume, sees
// the response via ResumedInput, and produces an output that flows
// to its successor.
func TestWorkflowAgent_ReEntry_ResumesAtSameNode(t *testing.T) {
	asker, obs := askerForSequence("asker", []string{"decide"}, "decision")

	var handlerInput atomic.Value
	handler := newStringHandlerNode("handler", &handlerInput)

	a := makeAgent(t, workflow.Chain(workflow.Start, asker, handler))
	sess := newFakeSession()

	// Turn 1: fresh; asker pauses with RequestedInput "decide".
	turn1 := runFreshTurn(t, sess, a, "draft")
	if got := findRequest(turn1); got != "decide" {
		t.Fatalf("turn 1 RequestedInput = %q, want %q", got, "decide")
	}

	// Turn 2: resume; asker re-runs, emits output, handler runs.
	resumeAndExpect(t, sess, a, "decide", "approve", "")

	if got, want := obs.count(), 2; got != want {
		t.Fatalf("activations = %d, want %d (one initial + one re-entry)", got, want)
	}
	if act, _ := obs.at(1); act.resumed["decide"] != "approve" {
		t.Errorf("re-entry resumed[\"decide\"] = %v, want %q", act.resumed["decide"], "approve")
	}
	if got, want := handlerInput.Load(), "decision"; got != want {
		t.Errorf("handler input = %v, want %q", got, want)
	}
}

// TestWorkflowAgent_ReEntry_PreservesOriginalInput verifies that on
// re-entry the asker receives the SAME input value it saw on its
// first activation, not the user's response. The response is
// delivered separately via ResumedInput.
func TestWorkflowAgent_ReEntry_PreservesOriginalInput(t *testing.T) {
	asker, obs := askerForSequence("asker", []string{"decide"}, "ack")

	a := makeAgent(t, workflow.Chain(workflow.Start, asker))
	sess := newFakeSession()

	// Turn 1: fresh, asker sees "draft" as input.
	runFreshTurn(t, sess, a, "draft")

	// Turn 2: resume with response "approve". Asker re-runs and
	// must see "draft" again as input (not "approve").
	resumeAndExpect(t, sess, a, "decide", "approve", "")

	if got, want := obs.count(), 2; got != want {
		t.Fatalf("activations = %d, want %d", got, want)
	}
	first, _ := obs.at(0)
	second, _ := obs.at(1)
	if first.input != "draft" {
		t.Errorf("initial activation input = %v, want %q", first.input, "draft")
	}
	if second.input != "draft" {
		t.Errorf("re-entry activation input = %v, want %q (must preserve original input, not the response)", second.input, "draft")
	}
}

// TestWorkflowAgent_ReEntry_NoSuccessorBeforeOutput verifies that a
// re-entry asker's downstream successor only fires once the
// re-entry activation actually emits an output event. (Compare to
// handoff mode, where the successor would fire immediately on
// resume even without a re-entry activation.)
func TestWorkflowAgent_ReEntry_NoSuccessorBeforeOutput(t *testing.T) {
	var handlerCount atomic.Int32

	asker, _ := askerForSequence("asker", []string{"decide"}, "ack")
	handler := newCountingHandlerNode("handler", &handlerCount)

	a := makeAgent(t, workflow.Chain(workflow.Start, asker, handler))
	sess := newFakeSession()

	// Pause turn: handler must not have fired.
	runFreshTurn(t, sess, a, "x")
	if got := handlerCount.Load(); got != 0 {
		t.Errorf("handler ran during pause turn; count = %d", got)
	}

	// Resume turn: asker re-runs and emits output; handler runs once.
	resumeAndExpect(t, sess, a, "decide", "yes", "")
	if got := handlerCount.Load(); got != 1 {
		t.Errorf("handler runs after re-entry = %d, want 1", got)
	}
}

// TestWorkflowAgent_ReEntry_DefaultModeIsHandoff confirms that
// without RerunOnResume = true the engine still uses handoff mode
// (the response flows to the successor as input; the asker does
// not re-run). Pins the default-mode contract from PR #2 against
// regression in this PR.
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

// --- test helpers below ---------------------------------------------------

// reentryNode is a custom Node with NodeConfig.RerunOnResume = true
// whose Run body is supplied by the test via runFn. Each test
// decides what to do on the first activation (typically yield a
// RequestInput) and on re-entry (read ctx.ResumedInput, emit an
// output).
type reentryNode struct {
	workflow.BaseNode
	runFn func(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error]
}

func newReentryNode(name string, runFn func(agent.InvocationContext, any) iter.Seq2[*session.Event, error]) *reentryNode {
	return &reentryNode{
		BaseNode: workflow.NewBaseNode(name, "", workflow.NodeConfig{RerunOnResume: ptrTrue()}),
		runFn:    runFn,
	}
}

func (n *reentryNode) Run(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
	return n.runFn(ctx, input)
}

// reentryActivation records what a re-entry asker observed during
// a single Run call: the input it received and the snapshot of
// resume-input payloads visible to it via ctx.ResumedInput.
type reentryActivation struct {
	input   any
	resumed map[string]any
}

// activations is a concurrent-safe ordered log of reentryActivation
// entries, used by tests to assert what an asker observed on each
// of its successive activations (initial run + N re-entries).
type activations struct {
	mu   sync.Mutex
	list []reentryActivation
}

func (a *activations) record(input any, resumed map[string]any) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.list = append(a.list, reentryActivation{input: input, resumed: resumed})
}

func (a *activations) at(i int) (reentryActivation, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if i < 0 || i >= len(a.list) {
		return reentryActivation{}, false
	}
	return a.list[i], true
}

func (a *activations) count() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.list)
}

// snapshotResumed reads ctx.ResumedInput for each of ids and
// returns the subset that the scheduler currently exposes. Missing
// IDs are omitted from the map so test assertions can compare
// against literal map[string]any{"id": value} expectations.
func snapshotResumed(ctx agent.InvocationContext, ids ...string) map[string]any {
	out := map[string]any{}
	for _, id := range ids {
		if v, ok := ctx.ResumedInput(id); ok {
			out[id] = v
		}
	}
	return out
}

// askerForSequence builds a reentryNode that walks through ids,
// pausing on each with a RequestInput. On every activation it
// records what input and resumed map it saw; once every id has a
// response it emits a final output event carrying finalOutput.
//
// Use it for tests that exercise multi-step ask-then-observe
// patterns without the per-test ceremony of branching on which
// question is next or maintaining a per-test activation counter.
func askerForSequence(name string, ids []string, finalOutput any) (*reentryNode, *activations) {
	obs := &activations{}
	node := newReentryNode(name, func(ctx agent.InvocationContext, input any) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			resumed := snapshotResumed(ctx, ids...)
			obs.record(input, resumed)

			// Pause on the first id we haven't seen a response for.
			for _, id := range ids {
				if _, answered := resumed[id]; !answered {
					yield(workflow.NewRequestInputEvent(ctx, session.RequestInput{
						InterruptID: id,
						Message:     id,
					}), nil)
					return
				}
			}

			// All answered: emit the terminal output.
			ev := session.NewEvent(ctx.InvocationID())
			ev.Actions.StateDelta["output"] = finalOutput
			yield(ev, nil)
		}
	})
	return node, obs
}

// resumeAndExpect performs one resume turn replying to replyID
// with replyValue, then asserts the next pause (if any) carries
// the expected InterruptID. Pass wantNextRequest = "" to assert
// that the workflow did not pause again (i.e. the resume turn
// completed without yielding another RequestInput).
func resumeAndExpect(t *testing.T, sess *fakeSession, a agent.Agent, replyID string, replyValue any, wantNextRequest string) []*session.Event {
	t.Helper()
	events := drainAgent(t, sess, a.Run(newMockCtx(sess, a, resumeMessage(replyID, replyValue))), nil)
	if got := findRequest(events); got != wantNextRequest {
		t.Fatalf("after resume %q=%v: RequestedInput = %q, want %q", replyID, replyValue, got, wantNextRequest)
	}
	return events
}

func ptrTrue() *bool {
	t := true
	return &t
}
