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

// FINDING L4 — Workflow.Resume only folds a fresh this-turn response for
// NodeWaiting nodes, not for partial re-entry nodes rehydrated to
// NodePending.
//
// BUG: In Workflow.Resume (resume.go), the "fold a fresh response not yet in
// history" path (the freshMatched block) is gated on
// `if ns.Status == NodeWaiting`. A re-entry node that already had an EARLIER
// interrupt resolved in history but still has a LATER interrupt open is
// rehydrated by inferNodeState (persistence.go) to NodePending with
// ResumedInputs={earlier:...} and Interrupts=[later] (the "partial resume"
// branch). A direct Workflow.Resume caller that supplies a response for the
// still-open `later` interrupt — a response that is NOT already in session
// history — never has it folded: `answeredNow` only checks the
// already-resolved ResumedInputs keys, and the freshMatched fold is skipped
// because the status is NodePending, not NodeWaiting. So nothing is
// scheduled and Resume spuriously yields ErrNothingToResume.
//
// EXPECTED: A direct Resume caller supplying a response for an open
// interrupt of a (partial) re-entry node should fold that response into the
// node's ResumedInputs and re-activate the node; it must NOT report
// ErrNothingToResume, and the re-entered node must observe the fresh
// response via ctx.ResumedInput.
//
// LATENT: the shipped runners append the user's response to session history
// BEFORE calling ReconstructRunState, so by rehydration time the response is
// already in `resumed` and the node is classified differently — the
// production path does not hit this. A direct Workflow.Resume caller (the
// public API) does.
//
// HOW THIS TEST DEMONSTRATES IT: It builds the exact RunState that
// inferNodeState's partial-resume branch produces (NodePending, an open
// `later` interrupt, an already-resolved `earlier`) and calls
// Workflow.Resume directly with a fresh `later` response. FAILS on current
// code: Resume returns ErrNothingToResume and the node is never re-run. This
// is a deterministic logic failure (no -race needed).

package workflow

import (
	"errors"
	"iter"
	"sync/atomic"
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// vbugL4ReentryNode is a re-entry-mode node that, when activated, records
// that it ran and echoes the response it observed for the "later" interrupt
// as its output, so the test can confirm the fresh response was folded in.
type vbugL4ReentryNode struct {
	BaseNode
	ran *atomic.Bool
}

func (n *vbugL4ReentryNode) Run(ctx agent.Context, _ any) iter.Seq2[*session.Event, error] {
	later, _ := ctx.ResumedInput("later")
	return func(yield func(*session.Event, error) bool) {
		n.ran.Store(true)
		ev := session.NewEvent(ctx.InvocationID())
		ev.Output = later
		yield(ev, nil)
	}
}

func TestVbugL4_ResumeDoesNotFoldFreshResponseForPendingReentryNode(t *testing.T) {
	tr := true
	node := &vbugL4ReentryNode{
		BaseNode: NewBaseNode("reentry", "", NodeConfig{RerunOnResume: &tr}),
		ran:      &atomic.Bool{},
	}

	w, err := New("wf-l4", []Edge{{From: Start, To: node}})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// This is exactly what inferNodeState's partial-resume branch builds:
	//   len(unresolved) > 0 && reenter && len(resumed) > 0  -> NodePending
	// with ResumedInputs holding the already-resolved earlier interrupt and
	// Interrupts holding the still-open later one.
	state := NewRunState()
	state.Nodes["reentry"] = &NodeState{
		Status:        NodePending,
		Input:         "seed",
		TriggeredBy:   Start.Name(),
		Interrupts:    []string{"later"},
		ResumedInputs: map[string]any{"earlier": "earlier-resp"},
	}

	// A direct Resume caller supplies the response to the still-open
	// interrupt; this response is NOT in session history (the runner-direct
	// path), so it must be folded in by Resume itself.
	responses := map[string]any{"later": "fresh-resp"}

	ctx := newNodeContext(newMockCtx(t), nil)

	var (
		nothingToResume bool
		observed        any
		otherErr        error
	)
	for ev, err := range w.Resume(ctx, state, responses) {
		if err != nil {
			if errors.Is(err, ErrNothingToResume) {
				nothingToResume = true
			} else if otherErr == nil {
				otherErr = err
			}
			continue
		}
		if ev != nil && ev.Output != nil {
			observed = ev.Output
		}
	}
	if otherErr != nil {
		t.Fatalf("unexpected Resume error: %v", otherErr)
	}

	if nothingToResume {
		t.Errorf("L4: Resume of a partial re-entry node (NodePending, open interrupt %q) "+
			"returned ErrNothingToResume; a fresh response not yet in history must be folded "+
			"and the node re-activated", "later")
	}
	if !node.ran.Load() {
		t.Errorf("L4: re-entry node was not re-scheduled; the fresh response to its open " +
			"interrupt was dropped instead of folded into ResumedInputs")
	}
	if observed != "fresh-resp" {
		t.Errorf("L4: re-entered node observed ResumedInput(%q) = %v, want %q "+
			"(fresh response must be folded in)", "later", observed, "fresh-resp")
	}
}
