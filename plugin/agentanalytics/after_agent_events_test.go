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

package agentanalytics

import (
	"fmt"
	"iter"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	baseplugin "google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

// TestGuardlessRetirementKeepsAfterAgentEntry records guardless retirements
// of step under wf (propagatesChildError == true, so no custom ancestor guards
// the retired activation's entry). An unresponded peer that left a credit
// keeps its credit and after-agent action maps across pruning while a peer is
// open and settles only its own credit via consumeAfterAgentCreditLocked; a
// responded activation with after-agent callbacks keeps its entry and maps
// while its callbacks may still run.
func TestGuardlessRetirementKeepsAfterAgentEntry(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	ctx := t.Context()
	wfID := identity{invocationID: "inv-guardless", agentName: "wf"}
	stepID := identity{invocationID: "inv-guardless", agentName: "step"}
	k := llmKey(stepID)

	t.Run("peer retirement with credit stays pinned while peer open", func(t *testing.T) {
		tm := newTraceManager()
		tm.pushAgentSpan(ctx, wfID)
		inv := tm.invocation(ctx, wfID)
		inv.mu.Lock()
		defer inv.mu.Unlock()
		inv.agentStack[1].propagatesChildError = true

		step1 := &spanRecord{spanID: "step-1", kind: spanKindAgent, name: "step", seq: inv.nextSeqLocked(), parent: inv.agentStack[1], parentSpanID: inv.agentStack[1].spanID}
		step2 := &spanRecord{spanID: "step-2", kind: spanKindAgent, name: "step", seq: inv.nextSeqLocked(), parent: inv.agentStack[1], parentSpanID: inv.agentStack[1].spanID, completed: true, hasAfterAgentCallbacks: true}
		inv.agentStack = append(inv.agentStack, step1)

		actions := &session.EventActions{StateDelta: map[string]any{}, ArtifactDelta: map[string]int64{}}
		step2.popRetirement = &popRetirement{
			peerRetirement:       true,
			afterAgentCredit:     true,
			afterAgentActions:    eventActionMaps(&session.Event{Actions: *actions}),
			afterAgentActionsRef: actions,
		}
		inv.unrespondedRetired = map[string]int{k: 1}
		inv.retiredPeerCount = map[string]int{k: 1}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5), step2)

		idx := inv.findRetiredActivationIndexLocked(step2.spanID)
		if idx < 0 {
			t.Fatal("step-2 retired without an entry")
		}
		if ra := inv.retiredActivations[idx]; len(ra.guardSpanIDs) != 0 || !ra.afterAgentCredit || !ra.peerRetirement || !ra.afterAgentActions.shares(eventActionMaps(&session.Event{Actions: *actions})) {
			t.Fatalf("step-2's entry: guards=%v credit=%v peer=%v maps=%+v; want empty guards, true, true and callback maps",
				ra.guardSpanIDs, ra.afterAgentCredit, ra.peerRetirement, ra.afterAgentActions)
		}
		inv.consumeAfterAgentCreditLocked(idx)
		if inv.unrespondedRetired[k] != 0 || inv.retiredActivations[idx].afterAgentCredit {
			t.Fatalf("unrespondedRetired=%d afterAgentCredit=%v after consume; want 0 and false",
				inv.unrespondedRetired[k], inv.retiredActivations[idx].afterAgentCredit)
		}
	})

	t.Run("responded activation keeps its maps while its callbacks may run", func(t *testing.T) {
		tm := newTraceManager()
		tm.pushAgentSpan(ctx, wfID)
		inv := tm.invocation(ctx, wfID)
		inv.mu.Lock()
		defer inv.mu.Unlock()
		inv.agentStack[1].propagatesChildError = true

		actions := &session.EventActions{StateDelta: map[string]any{}}
		step := &spanRecord{
			spanID:                 "step-solo",
			kind:                   spanKindAgent,
			name:                   "step",
			seq:                    inv.nextSeqLocked(),
			parent:                 inv.agentStack[1],
			parentSpanID:           inv.agentStack[1].spanID,
			completed:              true,
			hasResponse:            true,
			ownResponse:            true,
			hasAfterAgentCallbacks: true,
			popRetirement: &popRetirement{
				afterAgentActions:    eventActionMaps(&session.Event{Actions: *actions}),
				afterAgentActionsRef: actions,
			},
		}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5), step)
		inv.pruneSettledRetiredActivationsLocked("step", "")
		idx := inv.findRetiredActivationIndexLocked(step.spanID)
		if idx < 0 {
			t.Fatal("step retired without an entry")
		}
		if ra := inv.retiredActivations[idx]; !ra.afterAgentActions.shares(eventActionMaps(&session.Event{Actions: *actions})) || ra.afterAgentActionsRef == nil {
			t.Fatalf("step's entry keeps maps %+v (ref %v), want the callback's", ra.afterAgentActions, ra.afterAgentActionsRef != nil)
		}
	})
}

// TestSettleRetiredResponseCredit settles the credits that retired
// activations of worker left, for events that a retired activation created in
// its run: such an event settles its own activation's credit, and none once
// that credit was settled, while an event that more than one activation, or
// none, can have created settles one credit of the agent.
func TestSettleRetiredResponseCredit(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	k := llmKey(identity{agentName: "worker"})
	setup := func() *invocationTrace {
		inv := &invocationTrace{unrespondedRetired: map[string]int{k: 2}}
		inv.retiredActivations = []retiredActivation{
			{swallower: retiredSwallower{spanID: "w1", name: "worker"}, afterAgentCredit: true, joinedAtRetire: true, startPlatformTime: at(0), retiredAtPlatformTime: at(2)},
			{swallower: retiredSwallower{spanID: "w2", name: "worker"}, afterAgentCredit: true, joinedAtRetire: true, startPlatformTime: at(3), retiredAtPlatformTime: at(5)},
		}
		return inv
	}
	t.Run("own credit, then none", func(t *testing.T) {
		inv := setup()
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(1), true, true) {
			t.Fatal("w1's run event settled no credit")
		}
		if ra := inv.retiredActivations[0]; ra.afterAgentCredit || !ra.creditSettled || inv.unrespondedRetired[k] != 1 || !inv.retiredActivations[1].afterAgentCredit {
			t.Fatalf("after w1's run event: w1 credit %v settled %v, w2 credit %v, unrespondedRetired %d; want false, true, true, 1", ra.afterAgentCredit, ra.creditSettled, inv.retiredActivations[1].afterAgentCredit, inv.unrespondedRetired[k])
		}
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(2), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("w1's second event settled a credit: unrespondedRetired %d, want 1", inv.unrespondedRetired[k])
		}
		// w1's after-agent event finds its credit settled too.
		inv.consumeAfterAgentCreditLocked(0)
		if inv.unrespondedRetired[k] != 1 {
			t.Fatalf("w1's after-agent event settled w2's credit: unrespondedRetired %d, want 1", inv.unrespondedRetired[k])
		}
	})
	t.Run("custom UUID provider settles by non-empty ID", func(t *testing.T) {
		inv := setup()
		inv.retiredActivations[0].hasCustomUUIDProvider = true
		inv.retiredActivations[1].hasCustomUUIDProvider = true
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(1), false, true) {
			t.Fatal("custom-UUID w1 run event settled no credit")
		}
		if ra := inv.retiredActivations[0]; ra.afterAgentCredit || !ra.creditSettled || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("after custom-UUID w1 event: ra=%+v unrespondedRetired=%d", ra, inv.unrespondedRetired[k])
		}
	})
	t.Run("created while both could have", func(t *testing.T) {
		inv := setup()
		inv.retiredActivations[1].startPlatformTime = at(1)
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(4), false, false) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("unrespondedRetired = %d, want 1", inv.unrespondedRetired[k])
		}
		if !inv.retiredActivations[0].afterAgentCredit || !inv.retiredActivations[1].afterAgentCredit {
			t.Fatal("an event that is no one activation's settled an activation's own credit")
		}
	})
	t.Run("created before both started", func(t *testing.T) {
		// When all outstanding credits are accounted for by pinned entries and none could have created the event,
		// do not steal any credit.
		inv := setup()
		inv.retiredActivations[1].startPlatformTime = at(1)
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", t0.Add(-time.Millisecond), true, false) || inv.unrespondedRetired[k] != 2 {
			t.Fatalf("stole tracked credit: unrespondedRetired = %d, want 2", inv.unrespondedRetired[k])
		}
		if !inv.retiredActivations[0].afterAgentCredit || !inv.retiredActivations[1].afterAgentCredit {
			t.Fatal("an event created before both started cleared an activation's own credit")
		}
		// When an unaccounted earlier credit exists (unrespondedRetired > trackedCredits), settle one credit without touching w1/w2.
		inv.unrespondedRetired[k] = 3
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", t0.Add(-time.Millisecond), true, false) || inv.unrespondedRetired[k] != 2 {
			t.Fatalf("untracked credit unrespondedRetired = %d, want 2", inv.unrespondedRetired[k])
		}
		if !inv.retiredActivations[0].afterAgentCredit || !inv.retiredActivations[1].afterAgentCredit {
			t.Fatal("untracked earlier event cleared w1 or w2 credit")
		}
	})
	t.Run("no credit outstanding", func(t *testing.T) {
		inv := setup()
		inv.unrespondedRetired[k] = 0
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(1), true, true) || inv.unrespondedRetired[k] != 0 || !inv.retiredActivations[0].afterAgentCredit {
			t.Fatal("an event settled a credit while none was outstanding")
		}
	})
	t.Run("createdInRun", func(t *testing.T) {
		ra := retiredActivation{completedOrTask: true, joinedAtRetire: true, startPlatformTime: at(1), retiredAtPlatformTime: at(3)}
		for _, c := range []struct {
			evAt       time.Time
			hasEventID bool
			want       bool
		}{
			{at(0), true, false},
			{at(1), true, true},
			{at(3), true, true},
			{at(4), true, false},
			{at(4), false, true},
			{time.Time{}, true, true},
		} {
			if got := ra.createdInRun(c.evAt, c.hasEventID); got != c.want {
				t.Errorf("createdInRun(%v, %v) = %v, want %v", c.evAt.Sub(t0), c.hasEventID, got, c.want)
			}
		}
		ra.joinedAtRetire = false
		if !ra.createdInRun(at(4), true) {
			t.Error("an unjoined finish_task retirement's run may outlast it, but createdInRun(after retirement) = false")
		}
	})
	t.Run("compacted hull invalidates unique authorship", func(t *testing.T) {
		kP0 := llmKey(identity{agentName: "worker", path: "p0"})
		// Case 1: hull + w with afterAgentCredit=true must not consume w's own credit.
		inv1 := &invocationTrace{
			unrespondedRetired: map[string]int{kP0: 2},
			retiredActivations: []retiredActivation{
				{swallower: retiredSwallower{spanID: "h1", name: "worker", path: ""}, compactedHull: true, joinedAtRetire: true, startPlatformTime: at(0), retiredAtPlatformTime: at(10)},
				{swallower: retiredSwallower{spanID: "w3", name: "worker", path: "p0"}, afterAgentCredit: true, joinedAtRetire: true, startPlatformTime: at(2), retiredAtPlatformTime: at(8)},
			},
		}
		if !inv1.settleRetiredResponseCreditLocked(kP0, "worker", "", "p0", at(4), true, true) {
			t.Fatal("expected ambiguous event matching hull and w3 to settle one credit")
		}
		if inv1.unrespondedRetired[kP0] != 1 || !inv1.retiredActivations[1].afterAgentCredit || inv1.retiredActivations[1].creditSettled {
			t.Fatalf("hull failed to invalidate w3 unique authorship: unresponded=%d w3=%+v", inv1.unrespondedRetired[kP0], inv1.retiredActivations[1])
		}
		// Case 2: hull + w with creditSettled=true must not refuse settlement.
		inv2 := &invocationTrace{
			unrespondedRetired: map[string]int{kP0: 1},
			retiredActivations: []retiredActivation{
				{swallower: retiredSwallower{spanID: "h1", name: "worker", path: ""}, compactedHull: true, joinedAtRetire: true, startPlatformTime: at(0), retiredAtPlatformTime: at(10)},
				{swallower: retiredSwallower{spanID: "w3", name: "worker", path: "p0"}, creditSettled: true, joinedAtRetire: true, startPlatformTime: at(2), retiredAtPlatformTime: at(8)},
			},
		}
		if !inv2.settleRetiredResponseCreditLocked(kP0, "worker", "", "p0", at(4), true, true) || inv2.unrespondedRetired[kP0] != 0 {
			t.Fatalf("hull failed to invalidate creditSettled unique authorship: unresponded=%d", inv2.unrespondedRetired[kP0])
		}
	})
	t.Run("creditSettled stays pinned across pruning and pending folding while credit outstanding", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{k: 1},
		}
		// Populate enough entries to trigger both foldPendingAfterAgentLocked and pruneSettledRetiredActivationsLocked.
		for n := range maxPendingAfterAgent + 4 {
			m := map[string]any{}
			ra := retiredActivation{
				swallower:              retiredSwallower{spanID: fmt.Sprintf("w-%d", n), seq: uint64(1 + n), name: "worker"},
				parentSpanID:           root.spanID,
				guardSpanIDs:           []string{root.spanID},
				retiredAtSeq:           uint64(10 + n),
				completedOrTask:        true,
				joinedAtRetire:         true,
				hasAfterAgentCallbacks: true,
				afterAgentActions:      actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef:   m,
				startPlatformTime:      at(10 * n),
				retiredAtPlatformTime:  at(10*n + 5),
			}
			if n == 0 {
				ra.creditSettled = true
				ra.afterAgentDone = true
				ra.afterAgentDoneAt = at(10*n + 6)
			}
			inv.retiredActivations = append(inv.retiredActivations, ra)
		}
		// Peer holding the outstanding credit at a later non-overlapping time.
		inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
			swallower:             retiredSwallower{spanID: "w-peer", seq: 99, name: "worker"},
			parentSpanID:          root.spanID,
			guardSpanIDs:          []string{root.spanID},
			retiredAtSeq:          999,
			completedOrTask:       true,
			joinedAtRetire:        true,
			afterAgentCredit:      true,
			startPlatformTime:     at(500),
			retiredAtPlatformTime: at(505),
		})
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		idx := inv.findRetiredActivationIndexLocked("w-0")
		if idx < 0 || !inv.retiredActivations[idx].creditSettled {
			t.Fatalf("w-0 creditSettled lost during compaction while peer credit outstanding: idx=%d", idx)
		}
		// A second buffered event from w-0 must NOT steal w-peer's credit.
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(2), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("second event from w-0 stole peer credit: unrespondedRetired=%d, want 1", inv.unrespondedRetired[k])
		}
		// Also test guardless pruning when no open guard exists: both w-0 and w-peer stay pinned.
		inv.agentStack = inv.agentStack[:1]
		inv.retiredActivations[idx].guardSpanIDs = nil
		inv.retiredActivations[idx].dropAfterAgentEvent()
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		if idx2 := inv.findRetiredActivationIndexLocked("w-0"); idx2 < 0 || !inv.retiredActivations[idx2].creditSettled {
			t.Fatalf("w-0 deleted during guardless pruning while peer credit outstanding: idx=%d", idx2)
		}
		if peerIdx := inv.findRetiredActivationIndexLocked("w-peer"); peerIdx < 0 || !inv.retiredActivations[peerIdx].afterAgentCredit {
			t.Fatalf("w-peer deleted during guardless pruning while peer credit outstanding: idx=%d", peerIdx)
		}
		// Once the outstanding credit settles to 0, both w-0 and w-peer can be pruned normally.
		inv.unrespondedRetired[k] = 0
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		if idx3 := inv.findRetiredActivationIndexLocked("w-0"); idx3 >= 0 {
			t.Fatalf("w-0 still pinned after all compatible credits settled: idx=%d", idx3)
		}
		if peerIdx2 := inv.findRetiredActivationIndexLocked("w-peer"); peerIdx2 >= 0 {
			t.Fatalf("w-peer still pinned after all compatible credits settled: idx=%d", peerIdx2)
		}
	})
	t.Run("guardless pruning preserves overlapping model/tool peer so credit is not stranded", func(t *testing.T) {
		for _, peerHasAfterAgentCredit := range []bool{false, true} {
			t.Run(fmt.Sprintf("afterAgentCredit=%v", peerHasAfterAgentCredit), func(t *testing.T) {
				modelCredits := 0
				if !peerHasAfterAgentCredit {
					modelCredits = 1
				}
				inv := &invocationTrace{
					agentStack:         []*spanRecord{{kind: spanKindInvocation}},
					unrespondedRetired: map[string]int{k: 1},
					retiredActivations: []retiredActivation{
						{swallower: retiredSwallower{spanID: "w1", seq: 1, name: "worker"}, retiredAtSeq: 2, completedOrTask: true, joinedAtRetire: true, creditSettled: true, startPlatformTime: at(0), retiredAtPlatformTime: at(10)},
						{swallower: retiredSwallower{spanID: "w2", seq: 3, name: "worker"}, retiredAtSeq: 4, completedOrTask: true, joinedAtRetire: true, afterAgentCredit: peerHasAfterAgentCredit, modelCredits: modelCredits, startPlatformTime: at(2), retiredAtPlatformTime: at(12)},
					},
				}
				inv.pruneSettledRetiredActivationsLocked("worker", "")
				if idx := inv.findRetiredActivationIndexLocked("w2"); idx < 0 {
					t.Fatalf("w2 deleted during guardless pruning while unrespondedRetired=%d", inv.unrespondedRetired[k])
				}
				if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(5), true, true) || inv.unrespondedRetired[k] != 0 {
					t.Fatalf("w2 buffered event failed to settle credit after guardless pruning: unrespondedRetired=%d", inv.unrespondedRetired[k])
				}
				// Once unrespondedRetired reaches 0, guardless pruning drops both entries.
				inv.pruneSettledRetiredActivationsLocked("worker", "")
				if len(inv.retiredActivations) != 0 {
					t.Fatalf("guardless entries retained after credit reached 0: %+v", inv.retiredActivations)
				}
			})
		}
	})
	t.Run("guardless pruning preserves disjoint credit-bearing peers so credit is not stolen", func(t *testing.T) {
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}},
			unrespondedRetired: map[string]int{k: 2},
			retiredActivations: []retiredActivation{
				{swallower: retiredSwallower{spanID: "w1", seq: 1, name: "worker"}, retiredAtSeq: 2, completedOrTask: true, joinedAtRetire: true, afterAgentCredit: true, startPlatformTime: at(0), retiredAtPlatformTime: at(5)},
				{swallower: retiredSwallower{spanID: "w2", seq: 3, name: "worker"}, retiredAtSeq: 4, completedOrTask: true, joinedAtRetire: true, afterAgentCredit: true, startPlatformTime: at(10), retiredAtPlatformTime: at(15)},
			},
		}
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		if inv.findRetiredActivationIndexLocked("w1") < 0 || inv.findRetiredActivationIndexLocked("w2") < 0 {
			t.Fatalf("disjoint credit-bearing entries deleted during guardless pruning: %+v", inv.retiredActivations)
		}
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(2), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("first w1 event did not settle w1 credit: unrespondedRetired=%d", inv.unrespondedRetired[k])
		}
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(3), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("second w1 event stole w2 credit: unrespondedRetired=%d, want 1", inv.unrespondedRetired[k])
		}
		idx2 := inv.findRetiredActivationIndexLocked("w2")
		if idx2 < 0 || !inv.retiredActivations[idx2].afterAgentCredit {
			t.Fatalf("w2 lost afterAgentCredit: %+v", inv.retiredActivations)
		}
	})
	t.Run("ambiguous drain to 0 prevents cross-wave theft and bounds retention", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{k: 2},
			retiredActivations: []retiredActivation{
				{swallower: retiredSwallower{spanID: "w1", seq: 1, name: "worker"}, parentSpanID: root.spanID, guardSpanIDs: []string{root.spanID}, retiredAtSeq: 2, completedOrTask: true, joinedAtRetire: true, afterAgentCredit: true, afterAgentDone: true, startPlatformTime: at(0), retiredAtPlatformTime: at(10)},
				{swallower: retiredSwallower{spanID: "w2", seq: 3, name: "worker"}, parentSpanID: root.spanID, guardSpanIDs: []string{root.spanID}, retiredAtSeq: 4, completedOrTask: true, joinedAtRetire: true, afterAgentCredit: true, afterAgentDone: true, startPlatformTime: at(2), retiredAtPlatformTime: at(12)},
			},
		}
		// Two ambiguous events drain unrespondedRetired[k] from 2 -> 1 -> 0.
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(5), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("first ambiguous event: unrespondedRetired=%d, want 1", inv.unrespondedRetired[k])
		}
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(6), true, true) || inv.unrespondedRetired[k] != 0 {
			t.Fatalf("second ambiguous event: unrespondedRetired=%d, want 0", inv.unrespondedRetired[k])
		}
		for _, ra := range inv.retiredActivations {
			if ra.afterAgentCredit || !ra.creditSettled {
				t.Fatalf("after ambiguous drain to 0, entry %s has afterAgentCredit=%v creditSettled=%v; want false, true", ra.swallower.spanID, ra.afterAgentCredit, ra.creditSettled)
			}
		}
		// A new wave adds fresh credit for w3; late w1 event must NOT steal it.
		inv.unrespondedRetired[k] = 1
		inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
			swallower:             retiredSwallower{spanID: "w3", seq: 5, name: "worker"},
			parentSpanID:          root.spanID,
			guardSpanIDs:          []string{root.spanID},
			retiredAtSeq:          6,
			completedOrTask:       true,
			joinedAtRetire:        true,
			afterAgentCredit:      true,
			startPlatformTime:     at(20),
			retiredAtPlatformTime: at(30),
		})
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(1), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("late w1 run event stole wave-2 credit: unrespondedRetired=%d, want 1", inv.unrespondedRetired[k])
		}
		inv.consumeAfterAgentCreditLocked(0)
		if inv.unrespondedRetired[k] != 1 || !inv.retiredActivations[2].afterAgentCredit {
			t.Fatalf("late w1 after-agent event stole wave-2 credit: unrespondedRetired=%d w3=%+v", inv.unrespondedRetired[k], inv.retiredActivations[2])
		}

		// Draining via consumeAfterAgentCreditLocked also clears remaining exact-key peers and releases maps.
		m1, m2 := map[string]any{}, map[string]any{}
		invConsume := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{k: 1},
			retiredActivations: []retiredActivation{
				{swallower: retiredSwallower{spanID: "c1", seq: 1, name: "worker"}, parentSpanID: root.spanID, guardSpanIDs: []string{root.spanID}, retiredAtSeq: 2, afterAgentCredit: true, afterAgentDone: true, afterAgentActions: actionMaps{stateDelta: mapPointer(reflect.ValueOf(m1))}, afterAgentActionsRef: m1},
				{swallower: retiredSwallower{spanID: "c2", seq: 3, name: "worker"}, parentSpanID: root.spanID, guardSpanIDs: []string{root.spanID}, retiredAtSeq: 4, afterAgentCredit: true, afterAgentDone: true, afterAgentActions: actionMaps{stateDelta: mapPointer(reflect.ValueOf(m2))}, afterAgentActionsRef: m2},
			},
		}
		invConsume.consumeAfterAgentCreditLocked(0)
		if invConsume.unrespondedRetired[k] != 0 {
			t.Fatalf("consumeAfterAgentCreditLocked left unrespondedRetired=%d, want 0", invConsume.unrespondedRetired[k])
		}
		for _, ra := range invConsume.retiredActivations {
			if ra.afterAgentCredit || !ra.creditSettled || ra.holdsAfterAgentEvent() {
				t.Fatalf("after consume drain to 0, entry %s kept credit/maps: %+v", ra.swallower.spanID, ra)
			}
		}

		// Under an open custom guard, draining to 0 allows maps to drop and entries to compact to <= 3.
		invBounded := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{k: 1},
		}
		for n := range maxPendingAfterAgent + 4 {
			m := map[string]any{}
			invBounded.retiredActivations = append(invBounded.retiredActivations, retiredActivation{
				swallower:             retiredSwallower{spanID: fmt.Sprintf("b-%d", n), seq: uint64(1 + n), name: "worker"},
				parentSpanID:          root.spanID,
				guardSpanIDs:          []string{root.spanID},
				retiredAtSeq:          uint64(10 + n),
				completedOrTask:       true,
				joinedAtRetire:        true,
				afterAgentCredit:      true,
				afterAgentDone:        true,
				afterAgentActions:     actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef:  m,
				startPlatformTime:     at(0),
				retiredAtPlatformTime: at(10),
			})
		}
		if !invBounded.settleRetiredResponseCreditLocked(k, "worker", "", "", at(5), true, true) || invBounded.unrespondedRetired[k] != 0 {
			t.Fatalf("ambiguous drain failed: unrespondedRetired=%d", invBounded.unrespondedRetired[k])
		}
		invBounded.pruneSettledRetiredActivationsLocked("worker", "")
		if len(invBounded.retiredActivations) != 3 {
			t.Fatalf("after ambiguous drain under open guard: len=%d, want 3 compacted entries", len(invBounded.retiredActivations))
		}
		for _, ra := range invBounded.retiredActivations {
			if ra.holdsAfterAgentEvent() {
				t.Fatalf("retained afterAgentActionsRef after drain to 0: %+v", ra)
			}
		}

		// Even if stale afterAgentCredit remains set while unrespondedRetired[k] == 0,
		// needsAfterAgentActionsLocked returns false and foldPendingAfterAgentLocked folds.
		mStale := map[string]any{}
		invStale := &invocationTrace{
			agentStack: []*spanRecord{{kind: spanKindInvocation}, root},
			retiredActivations: []retiredActivation{{
				swallower:            retiredSwallower{spanID: "stale-1", name: "worker"},
				afterAgentCredit:     true,
				afterAgentDone:       true,
				afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(mStale))},
				afterAgentActionsRef: mStale,
			}},
		}
		if invStale.needsAfterAgentActionsLocked(0, invStale.afterAgentStateLocked()) {
			t.Fatal("stale afterAgentCredit with unrespondedRetired=0 kept needsAfterAgentActionsLocked=true")
		}
		invFold := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
		for n := range maxPendingAfterAgent + 2 {
			m := map[string]any{}
			invFold.retiredActivations = append(invFold.retiredActivations, retiredActivation{
				swallower:              retiredSwallower{spanID: fmt.Sprintf("stale-fold-%d", n), seq: uint64(1 + n), name: "worker", path: fmt.Sprintf("p%d", n%2)},
				parentSpanID:           root.spanID,
				guardSpanIDs:           []string{root.spanID},
				retiredAtSeq:           uint64(10 + n),
				hasAfterAgentCallbacks: true,
				afterAgentCredit:       true,
				afterAgentActions:      actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef:   m,
			})
		}
		invFold.foldPendingAfterAgentLocked("worker", "", invFold.afterAgentStateLocked())
		if len(invFold.retiredActivations) != maxPendingAfterAgent {
			t.Fatalf("stale afterAgentCredit with unrespondedRetired=0 blocked folding: len=%d, want %d", len(invFold.retiredActivations), maxPendingAfterAgent)
		}
		hullIdx := slices.IndexFunc(invFold.retiredActivations, func(ra retiredActivation) bool { return ra.compactedHull })
		if hullIdx < 0 {
			t.Fatal("expected a folded compactedHull entry")
		}
		for i, ra := range invFold.retiredActivations {
			if ra.compactedHull && ra.afterAgentCredit {
				t.Fatalf("folded compactedHull at %d retained stale afterAgentCredit=true: %+v", i, ra)
			}
		}

		// A later wave adding an outstanding credit on "p2" must neither be
		// blocked from folding into the widened hull nor pin the hull once
		// its pending after-agent state is dropped.
		kP2 := llmKey(identity{agentName: "worker", path: "p2"})
		invFold.unrespondedRetired = map[string]int{kP2: 1}
		for n := range 3 {
			m := map[string]any{}
			invFold.retiredActivations = append(invFold.retiredActivations, retiredActivation{
				swallower:              retiredSwallower{spanID: fmt.Sprintf("wave2-%d", n), seq: uint64(50 + n), name: "worker", path: "p2"},
				parentSpanID:           root.spanID,
				guardSpanIDs:           []string{root.spanID},
				retiredAtSeq:           uint64(60 + n),
				hasAfterAgentCallbacks: true,
				afterAgentActions:      actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef:   m,
			})
		}
		invFold.foldPendingAfterAgentLocked("worker", "", invFold.afterAgentStateLocked())
		if len(invFold.retiredActivations) != maxPendingAfterAgent {
			t.Fatalf("later-wave folding with widened hull: len=%d, want %d", len(invFold.retiredActivations), maxPendingAfterAgent)
		}
		hulls := 0
		for _, ra := range invFold.retiredActivations {
			if ra.compactedHull {
				hulls++
				if ra.afterAgentCredit {
					t.Fatalf("widened hull kept afterAgentCredit=true after wave-2 fold: %+v", ra)
				}
			}
		}
		if hulls != 1 {
			t.Fatalf("expected 1 merged hull in wave-2 fold, got %d", hulls)
		}
		root.completed = true
		invFold.agentStack = invFold.agentStack[:1]
		invFold.recordRetiredActivationsAtLocked(invFold.openSpanSetLocked(), at(20), root)
		invFold.unrespondedRetired[kP2] = 0
		invFold.pruneSettledRetiredActivationsLocked("worker", "")
		if len(invFold.retiredActivations) != 0 {
			t.Fatalf("widened folded hull stayed pinned after wave-2 credit cleared: %+v", invFold.retiredActivations)
		}

		// A concrete-path ("p0") folded hull also drops cleanly even while "p2" still holds a credit.
		root2 := &spanRecord{spanID: "root-2", kind: spanKindAgent, name: "root", seq: 1}
		invP0 := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root2}}
		for n := range maxPendingAfterAgent + 2 {
			m := map[string]any{}
			invP0.retiredActivations = append(invP0.retiredActivations, retiredActivation{
				swallower:              retiredSwallower{spanID: fmt.Sprintf("p0-fold-%d", n), seq: uint64(1 + n), name: "worker", path: "p0"},
				parentSpanID:           root2.spanID,
				guardSpanIDs:           []string{root2.spanID},
				retiredAtSeq:           uint64(10 + n),
				hasAfterAgentCallbacks: true,
				afterAgentCredit:       true,
				afterAgentActions:      actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef:   m,
			})
		}
		invP0.foldPendingAfterAgentLocked("worker", "", invP0.afterAgentStateLocked())
		invP0.unrespondedRetired = map[string]int{kP2: 1}
		root2.completed = true
		invP0.agentStack = invP0.agentStack[:1]
		invP0.recordRetiredActivationsAtLocked(invP0.openSpanSetLocked(), at(20), root2)
		for _, ra := range invP0.retiredActivations {
			if ra.compactedHull {
				t.Fatalf("p0 folded hull stayed pinned under p2 credit: %+v", ra)
			}
		}
	})
	t.Run("wildcard path does not mutate or block another llmKey", func(t *testing.T) {
		kWild := llmKey(identity{agentName: "worker"})
		kP0 := llmKey(identity{agentName: "worker", path: "p0"})

		// Entry path "" vs call path "p0": afterAgentCredit must not consume kWild.
		inv1 := &invocationTrace{
			unrespondedRetired: map[string]int{kWild: 1, kP0: 1},
			retiredActivations: []retiredActivation{{
				swallower:             retiredSwallower{spanID: "w-wild", name: "worker", path: ""},
				afterAgentCredit:      true,
				joinedAtRetire:        true,
				startPlatformTime:     at(0),
				retiredAtPlatformTime: at(10),
			}},
		}
		if !inv1.settleRetiredResponseCreditLocked(kP0, "worker", "", "p0", at(5), true, true) {
			t.Fatal("expected kP0 credit settlement to succeed")
		}
		if inv1.unrespondedRetired[kP0] != 0 || inv1.unrespondedRetired[kWild] != 1 || !inv1.retiredActivations[0].afterAgentCredit {
			t.Fatalf("wildcard entry mutated kWild: kP0=%d kWild=%d ra=%+v", inv1.unrespondedRetired[kP0], inv1.unrespondedRetired[kWild], inv1.retiredActivations[0])
		}

		// Entry path "" vs call path "p0": creditSettled on kWild must not block kP0.
		inv2 := &invocationTrace{
			unrespondedRetired: map[string]int{kWild: 0, kP0: 1},
			retiredActivations: []retiredActivation{{
				swallower:             retiredSwallower{spanID: "w-wild", name: "worker", path: ""},
				creditSettled:         true,
				joinedAtRetire:        true,
				startPlatformTime:     at(0),
				retiredAtPlatformTime: at(10),
			}},
		}
		if !inv2.settleRetiredResponseCreditLocked(kP0, "worker", "", "p0", at(5), true, true) || inv2.unrespondedRetired[kP0] != 0 {
			t.Fatalf("wildcard creditSettled blocked kP0: kP0=%d", inv2.unrespondedRetired[kP0])
		}

		// Entry path "p0" vs call path "": afterAgentCredit must not consume kP0.
		inv3 := &invocationTrace{
			unrespondedRetired: map[string]int{kWild: 1, kP0: 1},
			retiredActivations: []retiredActivation{{
				swallower:             retiredSwallower{spanID: "w-p0", name: "worker", path: "p0"},
				afterAgentCredit:      true,
				joinedAtRetire:        true,
				startPlatformTime:     at(0),
				retiredAtPlatformTime: at(10),
			}},
		}
		if !inv3.settleRetiredResponseCreditLocked(kWild, "worker", "", "", at(5), true, true) {
			t.Fatal("expected kWild credit settlement to succeed")
		}
		if inv3.unrespondedRetired[kWild] != 0 || inv3.unrespondedRetired[kP0] != 1 || !inv3.retiredActivations[0].afterAgentCredit {
			t.Fatalf("concrete-path entry mutated kP0: kWild=%d kP0=%d ra=%+v", inv3.unrespondedRetired[kWild], inv3.unrespondedRetired[kP0], inv3.retiredActivations[0])
		}

		// Entry path "p0" vs call path "": creditSettled on kP0 must not block kWild.
		inv4 := &invocationTrace{
			unrespondedRetired: map[string]int{kWild: 1, kP0: 0},
			retiredActivations: []retiredActivation{{
				swallower:             retiredSwallower{spanID: "w-p0", name: "worker", path: "p0"},
				creditSettled:         true,
				joinedAtRetire:        true,
				startPlatformTime:     at(0),
				retiredAtPlatformTime: at(10),
			}},
		}
		if !inv4.settleRetiredResponseCreditLocked(kWild, "worker", "", "", at(5), true, true) || inv4.unrespondedRetired[kWild] != 0 {
			t.Fatalf("concrete-path creditSettled blocked kWild: kWild=%d", inv4.unrespondedRetired[kWild])
		}
	})
}

func TestParseLLMKey(t *testing.T) {
	for _, tc := range []struct {
		name       string
		key        string
		wantPath   string
		wantName   string
		wantBranch string
		wantOK     bool
	}{
		{"round-trip standard", llmKey(identity{path: "p0", agentName: "worker", branch: "b0"}), "p0", "worker", "b0", true},
		{"round-trip empty path and branch", llmKey(identity{agentName: "worker"}), "", "worker", "", true},
		{"round-trip all empty", llmKey(identity{}), "", "", "", true},
		{"round-trip NULs and colons", llmKey(identity{path: "p0\x00:sub", agentName: "work:er\x001", branch: "b0\x00:extra"}), "p0\x00:sub", "work:er\x001", "b0\x00:extra", true},
		{"missing first colon", "worker", "", "", "", false},
		{"legacy NUL-delimited rejected", "p0\x00worker\x00b0", "", "", "", false},
		{"non-numeric path length", "x:p06:worker:b0", "", "", "", false},
		{"negative path length", "-1:p06:worker:b0", "", "", "", false},
		{"truncated path", "5:p0", "", "", "", false},
		{"missing name length colon", "2:p0worker", "", "", "", false},
		{"non-numeric name length", "2:p0x:worker:b0", "", "", "", false},
		{"negative name length", "2:p0-1:worker:b0", "", "", "", false},
		{"truncated name", "2:p09:worker", "", "", "", false},
		{"missing branch separator colon", "2:p06:workerXb0", "", "", "", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, n, b, ok := parseLLMKey(tc.key)
			if p != tc.wantPath || n != tc.wantName || b != tc.wantBranch || ok != tc.wantOK {
				t.Fatalf("parseLLMKey(%q) = (%q, %q, %q, %v), want (%q, %q, %q, %v)",
					tc.key, p, n, b, ok, tc.wantPath, tc.wantName, tc.wantBranch, tc.wantOK)
			}
		})
	}
}

// TestPendingAfterAgentHull folds the entries of child under an open custom
// root that keep their after-agent action maps only because their callbacks
// may still run into a hull once there are more than maxPendingAfterAgent of
// them. The hull stands for their after-agent events: an event created after
// they retired may be one, so a later activation's failure is not cleared by
// it, and the open activation that root may hold events for clears none with
// its own events either. Once root's run returned, the hull no longer stands
// for them.
func TestPendingAfterAgentHull(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	setup := func(t *testing.T) (*invocationTrace, *spanRecord) {
		t.Helper()
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
		for n := range maxPendingAfterAgent + 2 {
			m := map[string]any{}
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:              retiredSwallower{spanID: fmt.Sprintf("child-%d", n), seq: uint64(10 + n), name: "child", path: fmt.Sprintf("p%d", n%2)},
				parentSpanID:           root.spanID,
				guardSpanIDs:           []string{root.spanID},
				retiredAtSeq:           uint64(100 + n),
				completedOrTask:        true,
				joinedAtRetire:         true,
				hasAfterAgentCallbacks: true,
				afterAgentActions:      actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef:   m,
				startPlatformTime:      at(n),
				retiredAtPlatformTime:  at(n + 1),
			})
		}
		return inv, root
	}
	inv, root := setup(t)
	inv.pruneSettledRetiredActivationsLocked("child", "")
	var hull *retiredActivation
	for idx := range inv.retiredActivations {
		if ra := &inv.retiredActivations[idx]; ra.afterAgentPending {
			if hull != nil {
				t.Fatal("more than one hull")
			}
			hull = ra
		}
	}
	if len(inv.retiredActivations) != maxPendingAfterAgent || hull == nil {
		t.Fatalf("%d entries, hull %v; want %d with one hull", len(inv.retiredActivations), hull != nil, maxPendingAfterAgent)
	}
	if !hull.compactedHull || !hull.holdsAfterAgentEvent() || hull.afterAgentActionsRef != nil || hull.swallower.path != "" || hull.swallower.spanID != "child-2" || hull.retiredAtSeq != 102 || !hull.startPlatformTime.Equal(at(0)) || !hull.retiredAtPlatformTime.Equal(at(3)) {
		t.Fatalf("hull = %+v, want the three oldest entries on any path, without maps", *hull)
	}
	// Fold one more entry into the existing hull to exercise the hullIdx >= 0 path.
	extraMap := map[string]any{}
	inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
		swallower:              retiredSwallower{spanID: "child-extra", seq: 50, name: "child", path: "p0"},
		parentSpanID:           root.spanID,
		guardSpanIDs:           []string{root.spanID},
		retiredAtSeq:           150,
		completedOrTask:        true,
		joinedAtRetire:         true,
		goroutineID:            99,
		hasAfterAgentCallbacks: true,
		afterAgentActions:      actionMaps{stateDelta: mapPointer(reflect.ValueOf(extraMap))},
		afterAgentActionsRef:   extraMap,
		startPlatformTime:      at(15),
		retiredAtPlatformTime:  at(16),
	})
	inv.pruneSettledRetiredActivationsLocked("child", "")
	if len(inv.retiredActivations) != maxPendingAfterAgent {
		t.Fatalf("after adding one more entry: len=%d, want %d", len(inv.retiredActivations), maxPendingAfterAgent)
	}
	// A later activation of child holds a failure and created an event
	// after the hull's members retired.
	later := &spanRecord{spanID: "child-later", parentSpanID: root.spanID, parent: root, kind: spanKindAgent, name: "child", path: "p0", seq: 200, startPlatformTime: at(20)}
	if hull.excludedByEventTimestamp(later.spanID, later.seq, 0, false, later.startPlatformTime, at(21), true) {
		t.Fatal("a later activation excluded the hull by timestamp")
	}
	inv.agentStack = append(inv.agentStack, later)
	if !inv.hasGuardedRetiredPeerForEventLocked(later, "p0", at(21), true, true) {
		t.Fatal("the open activation's own event clears its failure while the hull may stand for a held after-agent event")
	}
	// Once root's run returned, the hull's members' after-agent events, if
	// any can still be delivered, were created, and nothing else needs it.
	root.completed = true
	inv.agentStack = inv.agentStack[:1]
	inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(30), root)
	for _, ra := range inv.retiredActivations {
		if ra.holdsAfterAgentEvent() {
			t.Fatalf("entry %s still holds an after-agent event once root retired: %+v", ra.swallower.spanID, ra)
		}
	}
}

// TestAfterAgentCallbacksReturnedOn records that the after-agent callbacks of
// the activations that retired on a goroutine returned once an agent starts
// or ends on it, and drops their action maps unless something still needs
// them, such as the activation's outstanding credit.
func TestAfterAgentCallbacksReturnedOn(t *testing.T) {
	m1, m2, m3, m4 := map[string]any{}, map[string]any{}, map[string]any{}, map[string]any{}
	mapsOf := func(m map[string]any) actionMaps { return actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))} }
	root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
	inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}, unrespondedRetired: map[string]int{llmKey(identity{agentName: "worker"}): 1}}
	inv.retiredActivations = []retiredActivation{
		{swallower: retiredSwallower{spanID: "a", name: "child"}, guardSpanIDs: []string{root.spanID}, popGoroutineID: 7, afterAgentActions: mapsOf(m1), afterAgentActionsRef: m1},
		{swallower: retiredSwallower{spanID: "b", name: "worker"}, guardSpanIDs: []string{root.spanID}, popGoroutineID: 7, afterAgentCredit: true, afterAgentActions: mapsOf(m2), afterAgentActionsRef: m2},
		{swallower: retiredSwallower{spanID: "c", name: "other"}, guardSpanIDs: []string{root.spanID}, popGoroutineID: 8, afterAgentActions: mapsOf(m3), afterAgentActionsRef: m3},
		{swallower: retiredSwallower{spanID: "d", name: "worker"}, guardSpanIDs: []string{root.spanID}, popGoroutineID: 7, afterAgentActions: mapsOf(m4), afterAgentActionsRef: m4},
	}
	inv.afterAgentCallbacksReturnedOnLocked(0, time.Time{}, 0)
	for _, ra := range inv.retiredActivations {
		if ra.afterAgentDone {
			t.Fatal("an unknown goroutine showed that callbacks returned")
		}
	}
	inv.afterAgentCallbacksReturnedOnLocked(7, time.Unix(1_700_000_000, 0), 0)
	a, b, c := inv.retiredActivations[0], inv.retiredActivations[1], inv.retiredActivations[2]
	if !a.afterAgentDone || !b.afterAgentDone || c.afterAgentDone {
		t.Fatalf("afterAgentDone = %v, %v, %v; want true, true, false", a.afterAgentDone, b.afterAgentDone, c.afterAgentDone)
	}
	d := inv.retiredActivations[3]
	if a.holdsAfterAgentEvent() || !b.holdsAfterAgentEvent() || !c.holdsAfterAgentEvent() || !d.holdsAfterAgentEvent() {
		t.Fatalf("maps kept = %v, %v, %v, %v; want false (settled), true (own credit outstanding), true (callbacks may run), true (another activation's credit outstanding)", a.holdsAfterAgentEvent(), b.holdsAfterAgentEvent(), c.holdsAfterAgentEvent(), d.holdsAfterAgentEvent())
	}
}

// TestAfterAgentDoneBoundsRetiredEvents pins that a retired activation whose
// after-agent callbacks were seen to have returned at afterAgentDoneAt can
// own no event created after that time (see canOwnEventCreatedAt), so that
// neither hasGuardedRetiredPeerForEventLocked nor soleOpenAuthorLocked takes
// it for a possible author of an open activation's later event, whether its
// after-agent event was delivered unrecognized or never came, while an event
// that carries its after-agent action maps stays its own whenever it was
// created, as when the callbacks ran an agent themselves. It also pins how
// that time is recorded and merged.
func TestAfterAgentDoneBoundsRetiredEvents(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	retired := func(spanID string) retiredActivation {
		return retiredActivation{
			swallower:              retiredSwallower{spanID: spanID, seq: 2, name: "worker"},
			parentSpanID:           "root-1",
			guardSpanIDs:           []string{"root-1"},
			retiredAtSeq:           3,
			completedOrTask:        true,
			joinedAtRetire:         true,
			unrespondedAtPop:       true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(0),
			retiredAtPlatformTime:  at(10),
		}
	}
	t.Run("canOwnEventCreatedAt", func(t *testing.T) {
		ra := retired("w-1")
		if !ra.canOwnEventCreatedAt(at(30), true) {
			t.Fatal("an activation whose after-agent event may still come cannot own a later event")
		}
		ra.markAfterAgentDone(at(20))
		for _, c := range []struct {
			ev   time.Time
			want bool
		}{{at(15), true}, {at(20), true}, {at(30), false}} {
			if got := ra.canOwnEventCreatedAt(c.ev, true); got != c.want {
				t.Errorf("canOwnEventCreatedAt(%v) with callbacks seen to return at 20ms = %v, want %v", c.ev.Sub(t0), got, c.want)
			}
		}
		if !ra.canOwnEventAt(at(30), true, true) {
			t.Error("an event that carries the activation's after-agent maps is not its own once created after the callbacks were seen to return")
		}
		ra.markAfterAgentDone(at(25))
		if !ra.afterAgentDoneAt.Equal(at(20)) {
			t.Errorf("afterAgentDoneAt = %v after a later sighting, want the earliest, 20ms", ra.afterAgentDoneAt.Sub(t0))
		}
		ra.markAfterAgentDone(at(15))
		ra.markAfterAgentDone(time.Time{})
		if !ra.afterAgentDone || !ra.afterAgentDoneAt.Equal(at(15)) {
			t.Errorf("afterAgentDone %v at %v, want true at the earliest known time, 15ms", ra.afterAgentDone, ra.afterAgentDoneAt.Sub(t0))
		}
	})
	t.Run("recorded where the callbacks are seen to return", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top"}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top}}
		on7, on8 := retired("w-1"), retired("w-2")
		on7.popGoroutineID, on8.popGoroutineID = 7, 8
		// root-1 runs both under the open custom agent top.
		on7.guardSpanIDs, on8.guardSpanIDs = []string{"root-1", top.spanID}, []string{"root-1", top.spanID}
		inv.retiredActivations = []retiredActivation{on7, on8}
		inv.afterAgentCallbacksReturnedOnLocked(7, at(20), 0)
		if got := inv.retiredActivations[0].afterAgentDoneAt; !got.Equal(at(20)) {
			t.Errorf("afterAgentDoneAt of the activation retired on the goroutine = %v, want 20ms", got.Sub(t0))
		}
		if got := inv.retiredActivations[1]; got.afterAgentDone || !got.afterAgentDoneAt.IsZero() {
			t.Errorf("activation retired on another goroutine: done %v at %v, want neither", got.afterAgentDone, got.afterAgentDoneAt)
		}
		// The parent's run returned once it retired.
		parent := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", completed: true}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(40), parent)
		if idx := inv.findRetiredActivationIndexLocked("w-2"); idx < 0 || !inv.retiredActivations[idx].afterAgentDoneAt.Equal(at(40)) {
			t.Fatalf("afterAgentDoneAt once the parent's run returned at 40ms: entry %d %+v", idx, inv.retiredActivations)
		}
	})
	t.Run("hulls keep the latest time of their members, and none when one has none", func(t *testing.T) {
		for _, oneUnknown := range []bool{false, true} {
			inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, {spanID: "root-1", kind: spanKindAgent, name: "root"}}}
			for n := range maxPendingAfterAgent + 2 {
				m := map[string]any{}
				ra := retired(fmt.Sprintf("w-%d", n))
				ra.retiredAtSeq = uint64(10 + n)
				ra.afterAgentActions, ra.afterAgentActionsRef = actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))}, m
				ra.markAfterAgentDone(at(20 + n))
				if oneUnknown && n == 1 {
					ra.afterAgentDoneAt = time.Time{}
				}
				inv.retiredActivations = append(inv.retiredActivations, ra)
			}
			inv.foldPendingAfterAgentLocked("worker", "", afterAgentState{swallowers: map[string]bool{}, underOpen: map[string]bool{}})
			hull := inv.retiredActivations[slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.afterAgentPending })]
			want := at(22)
			if oneUnknown {
				want = time.Time{}
			}
			if !hull.afterAgentDone || !hull.afterAgentDoneAt.Equal(want) {
				t.Errorf("pending hull of the three oldest (one without a time: %v): done %v at %v, want true at %v", oneUnknown, hull.afterAgentDone, hull.afterAgentDoneAt, want)
			}
		}
		for _, oneUnknown := range []bool{false, true} {
			inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, {spanID: "root-1", kind: spanKindAgent, name: "root"}}}
			for n := range 5 {
				ra := retired(fmt.Sprintf("w-%d", n))
				ra.hasAfterAgentCallbacks, ra.unrespondedAtPop = false, false
				ra.retiredAtSeq = uint64(10 + n)
				ra.markAfterAgentDone(at(20 + n))
				if oneUnknown && n == 2 {
					ra.afterAgentDoneAt = time.Time{}
				}
				inv.retiredActivations = append(inv.retiredActivations, ra)
			}
			inv.pruneSettledRetiredActivationsLocked("worker", "")
			idx := slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.compactedHull })
			if len(inv.retiredActivations) != 3 || idx < 0 {
				t.Fatalf("%d entries after compaction, hull at %d; want 3 with one hull", len(inv.retiredActivations), idx)
			}
			want := at(23)
			if oneUnknown {
				want = time.Time{}
			}
			if hull := inv.retiredActivations[idx]; !hull.afterAgentDoneAt.Equal(want) {
				t.Errorf("compacted hull of the three middle entries (one without a time: %v): done at %v, want %v", oneUnknown, hull.afterAgentDoneAt, want)
			}
		}
	})
	t.Run("an open activation's later event", func(t *testing.T) {
		for _, tc := range []struct {
			name            string
			unresponded     bool
			settledByParent bool
		}{
			{"unresponded", true, false},
			{"responded", false, false},
			{"settledByParent", false, true},
		} {
			t.Run(tc.name, func(t *testing.T) {
				for _, done := range []bool{false, true} {
					root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root"}
					peer := &spanRecord{spanID: "w-2", kind: spanKindAgent, name: "worker", parent: root, parentSpanID: root.spanID, seq: 4, startPlatformTime: at(25)}
					inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root, peer}}
					ra := retired("w-1")
					ra.unrespondedAtPop = tc.unresponded
					ra.settledByParent = tc.settledByParent
					if done {
						ra.markAfterAgentDone(at(20))
					}
					inv.retiredActivations = []retiredActivation{ra}
					if got := inv.hasGuardedRetiredPeerForEventLocked(peer, "", at(30), true, true); got == done {
						t.Errorf("callbacks seen to return before the event %v: hasGuardedRetiredPeerForEventLocked = %v, want %v", done, got, !done)
					}
					want := -1
					if done {
						want = 2
					}
					if got := inv.soleOpenAuthorLocked([]int{2}, "", at(30), func(*spanRecord) bool { return true }, func(retiredActivation) bool { return true }); got != want {
						t.Errorf("callbacks seen to return before the event %v: soleOpenAuthorLocked = %d, want %d", done, got, want)
					}
					wantProven := done && (!tc.unresponded || tc.settledByParent)
					if got := inv.provenAfterAllRetiredLocked(peer, "", at(30), true, true); got != wantProven {
						t.Errorf("callbacks seen to return before the event %v: provenAfterAllRetiredLocked = %v, want %v", done, got, wantProven)
					}
				}
			})
		}
	})

	t.Run("mapless callback-bearing hull with unknown afterAgentDoneAt stays guarded", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
		for n := range 5 {
			ra := retired(fmt.Sprintf("w-%d", n))
			ra.swallower.seq = uint64(2 + n)
			ra.retiredAtSeq = uint64(10 + n)
			ra.unrespondedAtPop = false
			if n != 2 {
				ra.markAfterAgentDone(at(20 + n))
			}
			inv.retiredActivations = append(inv.retiredActivations, ra)
		}
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		idx := slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.compactedHull })
		if len(inv.retiredActivations) != 3 || idx < 0 {
			t.Fatalf("expected 3 entries with hull, got len=%d idx=%d", len(inv.retiredActivations), idx)
		}
		hull := inv.retiredActivations[idx]
		if hull.afterAgentPending || hull.afterAgentActions != (actionMaps{}) || !hull.afterAgentDoneAt.IsZero() {
			t.Fatalf("unexpected mapless hull state: %+v", hull)
		}
		peer := &spanRecord{spanID: "w-later", kind: spanKindAgent, name: "worker", parent: root, parentSpanID: root.spanID, seq: 30, startPlatformTime: at(25)}
		inv.agentStack = append(inv.agentStack, peer)
		if hull.excludedByEventTimestamp(peer.spanID, peer.seq, 0, false, peer.startPlatformTime, at(30), true) {
			t.Fatal("mapless callback-bearing hull falsely excluded at at(30) with unknown afterAgentDoneAt")
		}
		if !inv.hasGuardedRetiredPeerForEventLocked(peer, "", at(30), true, true) {
			t.Fatal("hasGuardedRetiredPeerForEventLocked = false for mapless callback-bearing hull")
		}
		if inv.provenAfterAllRetiredLocked(peer, "", at(30), true, true) {
			t.Fatal("provenAfterAllRetiredLocked = true despite mapless callback-bearing hull with unknown afterAgentDoneAt")
		}
	})
}

// TestPeerRetirementsStayUncompactedWhilePeerOpen retires concurrent peers
// of an open activation of worker under an open custom root. Their entries
// are neither deleted nor compacted while the peer is open, so that
// soleOpenAuthorLocked still finds an entry for every retired peer that
// retiredPeerCount counts and gives the open activation its event created
// after they all retired. Once no peer is open, they compact as usual.
//
// That holds when more than maxPendingAfterAgent of them keep the maps of
// their after-agent callbacks, whether those callbacks may still run or were
// seen to return: foldPendingAfterAgentLocked folds none of them into a hull
// while the peer is open. Nor does it fold, while the peer is open, entries
// that retired before it started and keep their maps because their callbacks
// returned and the open peer could be taken for the author of their
// after-agent events. Once no peer is open, the entries whose callbacks may
// still run fold as usual, and the others lose their maps and compact.
func TestPeerRetirementsStayUncompactedWhilePeerOpen(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	k := llmKey(identity{agentName: "worker"})
	type kind int
	const (
		plain kind = iota
		pendingMaps
		returnedMaps
		returnedBeforePeer
	)
	setup := func(n int, kd kind) *invocationTrace {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root"}
		peer := &spanRecord{spanID: "w-open", kind: spanKindAgent, name: "worker", parent: root, parentSpanID: root.spanID, concurrentPeer: true, seq: 1, startPlatformTime: at(0)}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root, peer}, retiredPeerCount: map[string]int{}}
		for i := range n {
			ra := retiredActivation{
				swallower:             retiredSwallower{spanID: fmt.Sprintf("w-%d", i), seq: uint64(2 + i), name: "worker"},
				parentSpanID:          root.spanID,
				guardSpanIDs:          []string{root.spanID},
				retiredAtSeq:          uint64(100 + i),
				completedOrTask:       true,
				joinedAtRetire:        true,
				concurrentPeer:        kd != returnedBeforePeer,
				peerRetirement:        kd != returnedBeforePeer,
				startPlatformTime:     at(1 + i),
				retiredAtPlatformTime: at(2 + i),
			}
			if kd != plain {
				m := map[string]any{}
				ra.hasAfterAgentCallbacks = true
				ra.afterAgentActions, ra.afterAgentActionsRef = actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))}, m
			}
			if kd == returnedMaps || kd == returnedBeforePeer {
				ra.markAfterAgentDone(at(2 + i))
			}
			if ra.peerRetirement {
				inv.retiredPeerCount[k]++
			}
			inv.retiredActivations = append(inv.retiredActivations, ra)
		}
		return inv
	}
	uncompacted := func(t *testing.T, inv *invocationTrace, n int, wantPeers bool) {
		t.Helper()
		if len(inv.retiredActivations) != n || slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool {
			return ra.compactedHull || ra.afterAgentPending || ra.peerRetirement != wantPeers
		}) {
			t.Fatalf("entries while the peer is open: %+v, want the %d entries as they were", inv.retiredActivations, n)
		}
		if !inv.hasOpenPeerLocked(inv.retiredActivations[0]) {
			t.Fatal("hasOpenPeerLocked = false while the peer is open")
		}
	}
	t.Run("without after-agent maps", func(t *testing.T) {
		inv := setup(4, plain)
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		uncompacted(t, inv, 4, true)
		if got := inv.soleOpenAuthorLocked([]int{2}, "", at(20), func(*spanRecord) bool { return true }, func(retiredActivation) bool { return true }); got != 2 {
			t.Fatalf("soleOpenAuthorLocked = %d, want the open peer, 2", got)
		}
		inv.agentStack = inv.agentStack[:2]
		if inv.hasOpenPeerLocked(inv.retiredActivations[0]) {
			t.Fatal("hasOpenPeerLocked = true once no peer is open")
		}
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		if len(inv.retiredActivations) != 3 {
			t.Fatalf("%d entries once no peer is open, want 3 after compaction", len(inv.retiredActivations))
		}
	})
	for _, n := range []int{maxPendingAfterAgent + 1, 15} {
		t.Run(fmt.Sprintf("%d peer retirements whose callbacks may still run", n), func(t *testing.T) {
			inv := setup(n, pendingMaps)
			inv.pruneSettledRetiredActivationsLocked("worker", "")
			uncompacted(t, inv, n, true)
			inv.agentStack = inv.agentStack[:2]
			inv.pruneSettledRetiredActivationsLocked("worker", "")
			if len(inv.retiredActivations) != maxPendingAfterAgent || !slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.afterAgentPending }) {
				t.Fatalf("%d entries once no peer is open, want %d with one hull standing for the oldest", len(inv.retiredActivations), maxPendingAfterAgent)
			}
		})
		t.Run(fmt.Sprintf("%d peer retirements whose callbacks returned", n), func(t *testing.T) {
			inv := setup(n, returnedMaps)
			inv.pruneSettledRetiredActivationsLocked("worker", "")
			uncompacted(t, inv, n, true)
			if got := inv.soleOpenAuthorLocked([]int{2}, "", at(40), func(*spanRecord) bool { return true }, func(retiredActivation) bool { return true }); got != 2 {
				t.Fatalf("soleOpenAuthorLocked = %d, want the open peer, 2", got)
			}
			inv.agentStack = inv.agentStack[:2]
			inv.pruneSettledRetiredActivationsLocked("worker", "")
			if len(inv.retiredActivations) != 3 || slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.holdsAfterAgentEvent() }) {
				t.Fatalf("entries once no peer is open: %+v, want 3 after compaction, none with maps", inv.retiredActivations)
			}
		})
	}
	t.Run("entries that retired before the peer started and whose callbacks returned", func(t *testing.T) {
		n := maxPendingAfterAgent + 1
		inv := setup(n, returnedBeforePeer)
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		uncompacted(t, inv, n, false)
		inv.agentStack = inv.agentStack[:2]
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		if len(inv.retiredActivations) != 3 || slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.holdsAfterAgentEvent() }) {
			t.Fatalf("entries once no peer is open: %+v, want 3 after compaction, none with maps", inv.retiredActivations)
		}
	})
}

// TestAfterAgentCallbacksStillRunning pins how the runs of after-agent
// callbacks that enclose a start or end on a goroutine are counted in its
// stack trace (see afterAgentCallbackRunsOfTrace), that incomplete, elided, or
// malformed traces fail closed with ok == false, and that the activations
// retired on that goroutine whose callbacks still run, the ones that retired
// first among those not recorded yet, are not recorded as returned (see
// markAfterAgentCallbacksReturnedOnLocked).
func TestAfterAgentCallbacksStillRunning(t *testing.T) {
	frame := func(fn string) string {
		return fn + "(0x1, 0x2)\n\t/src/file.go:10 +0x20\n"
	}
	run := agentPkgPath + ".runAfterAgentCallbacks"
	for _, tc := range []struct {
		name     string
		trace    string
		complete bool
		want     int
		wantOK   bool
	}{
		{"none", "goroutine 7 [running]:\n" + frame(agentPkgPath+".runBeforeAgentCallbacks") + frame(agentPkgPath+".(*agent).Run.func1"), true, 0, true},
		{"one", "goroutine 7 [running]:\n" + frame(agentPkgPath+".runBeforeAgentCallbacks") + frame(run) + frame(agentPkgPath+".(*agent).Run.func1"), true, 1, true},
		{"one with created by", "goroutine 7 [running]:\n" + frame(run) + "created by main.f in goroutine 1\n\t/src/file.go:3 +0x1\n", true, 1, true},
		{"nested", "goroutine 7 [running]:\n" + frame(run) + frame("main.callback") + frame(run), true, 2, true},
		{"another package's function of that name", "goroutine 7 [running]:\n" + frame("example.com/other.runAfterAgentCallbacks"), true, 0, true},
		{"a function inside it", "goroutine 7 [running]:\n" + frame(run+".func1"), true, 0, true},
		{"after the created by line", "goroutine 7 [running]:\n" + frame("main.f") + "created by " + run + " in goroutine 1\n\t/src/file.go:3 +0x1\n", true, 0, true},
		{"tab-indented file line", "goroutine 7 [running]:\n" + frame("main.f") + "\t/dir/" + run + "(0x1)/file.go:10 +0x20\n", true, 0, true},
		{"no frames", "goroutine 7 [running]:\n", true, 0, true},
		{"header without newline", "goroutine 7 [running]:", true, 0, false},
		{"empty", "", true, 0, false},
		{"frame without arguments", "goroutine 7 [running]:\n" + run + "\n\t/src/file.go:10 +0x20\n", true, 0, false},
		{"frame starting with paren", "goroutine 7 [running]:\n(0x1)\n\t/src/file.go:10 +0x20\n", true, 0, false},
		{"elided between two run frames", "goroutine 7 [running]:\n" + frame(run) + "...42 frames elided...\n" + frame(run), true, 0, false},
		{"additional frames elided between two run frames", "goroutine 7 [running]:\n" + frame(run) + "...additional frames elided...\n" + frame(run), true, 0, false},
		{"elided with zero visible run frames", "goroutine 7 [running]:\n" + frame(agentPkgPath+".runBeforeAgentCallbacks") + "...42 frames elided...\n" + frame("main.outer"), true, 0, false},
		{"elided after the created by line", "goroutine 7 [running]:\n" + frame(run) + "created by main.f in goroutine 1\n\t/src/file.go:3 +0x1\n...5 frames elided...\n", true, 1, true},
		{"truncated with visible run frame", "goroutine 7 [running]:\n" + frame(run), false, 0, false},
		{"truncated with zero visible run frames", "goroutine 7 [running]:\n" + frame(agentPkgPath+".runBeforeAgentCallbacks"), false, 0, false},
	} {
		got, ok := afterAgentCallbackRunsOfTrace([]byte(tc.trace), tc.complete)
		if got != tc.want || ok != tc.wantOK {
			t.Errorf("afterAgentCallbackRunsOfTrace(%s) = (%d, %v), want (%d, %v)", tc.name, got, ok, tc.want, tc.wantOK)
		}
	}
	if trace, complete := currentGoroutineTrace(); !complete {
		t.Fatalf("currentGoroutineTrace() complete = false on test goroutine")
	} else if got, ok := afterAgentCallbackRunsOfTrace(trace, complete); got != 0 || !ok {
		t.Errorf("afterAgentCallbackRunsOfTrace(this goroutine) = (%d, %v), want (0, true)", got, ok)
	}
	var callDepth func(int, func()) int
	callDepth = func(depth int, fn func()) int {
		if depth <= 0 {
			fn()
			return 0
		}
		return callDepth(depth-1, fn) + 1
	}
	callDepth(110, func() {
		trace, complete := currentGoroutineTrace()
		if !complete || !strings.Contains(string(trace), "frames elided") {
			t.Fatalf("deep stack trace complete = %v, elided = %v; trace:\n%s", complete, strings.Contains(string(trace), "frames elided"), trace)
		}
		if got, ok := afterAgentCallbackRunsOfTrace(trace, complete); got != 0 || ok {
			t.Errorf("afterAgentCallbackRunsOfTrace(elided goroutine) = (%d, %v), want (0, false)", got, ok)
		}
	})
	setup := func() *invocationTrace {
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, {spanID: "root-1", kind: spanKindAgent, name: "root"}}}
		for _, seq := range []uint64{30, 10, 20} {
			m := map[string]any{}
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:            retiredSwallower{spanID: fmt.Sprintf("r%d", seq), name: "inner"},
				guardSpanIDs:         []string{"root-1"},
				retiredAtSeq:         seq,
				popGoroutineID:       7,
				afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef: m,
			})
		}
		inv.retiredActivations = append(inv.retiredActivations, retiredActivation{swallower: retiredSwallower{spanID: "other", name: "inner"}, retiredAtSeq: 5, popGoroutineID: 8})
		return inv
	}
	done := func(inv *invocationTrace) map[string]bool {
		got := map[string]bool{}
		for _, ra := range inv.retiredActivations {
			got[ra.swallower.spanID] = ra.afterAgentDone
		}
		return got
	}
	at := time.Unix(1_700_000_000, 0)
	for _, tc := range []struct {
		running    int
		want       map[string]bool
		wantMarked bool
	}{
		{0, map[string]bool{"r10": true, "r20": true, "r30": true, "other": false}, true},
		{1, map[string]bool{"r10": false, "r20": true, "r30": true, "other": false}, true},
		{2, map[string]bool{"r10": false, "r20": false, "r30": true, "other": false}, true},
		{3, map[string]bool{"r10": false, "r20": false, "r30": false, "other": false}, false},
	} {
		inv := setup()
		if marked := inv.markAfterAgentCallbacksReturnedOnLocked(7, at, tc.running); marked != tc.wantMarked {
			t.Errorf("running %d: marked = %v, want %v", tc.running, marked, tc.wantMarked)
		}
		if got := done(inv); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("running %d: afterAgentDone = %v, want %v", tc.running, got, tc.want)
		}
	}
	// An elided or truncated trace leaves the count unknown, so pushAgentSpan
	// skips markAfterAgentCallbacksReturnedOnLocked and its unconditional
	// settleAfterAgentActionsOnLocked keeps the action maps of still-pending
	// activations intact.
	for _, tc := range []struct {
		name     string
		trace    string
		complete bool
	}{
		{"elided between two run frames", "goroutine 7 [running]:\n" + frame(run) + "...42 frames elided...\n" + frame(run), true},
		{"elided with zero visible run frames", "goroutine 7 [running]:\n" + frame(agentPkgPath+".runBeforeAgentCallbacks") + "...42 frames elided...\n" + frame("main.outer"), true},
		{"truncated", "goroutine 7 [running]:\n" + frame(run), false},
	} {
		inv := setup()
		goid, _ := parseGoroutineTrace([]byte(tc.trace), tc.complete)
		if goid != 7 {
			t.Fatalf("%s: parseGoroutineTrace() goid = %d, want 7", tc.name, goid)
		}
		if runs, ok := afterAgentCallbackRunsOfTrace([]byte(tc.trace), tc.complete); ok {
			inv.markAfterAgentCallbacksReturnedOnLocked(goid, at, runs)
			inv.afterAgentCallbacksReturnedOnLocked(goid, at, max(runs-1, 0))
		}
		inv.settleAfterAgentActionsOnLocked(goid)
		wantUnmarked := map[string]bool{"r10": false, "r20": false, "r30": false, "other": false}
		if got := done(inv); !reflect.DeepEqual(got, wantUnmarked) {
			t.Errorf("%s: afterAgentDone = %v, want %v", tc.name, got, wantUnmarked)
		}
		for _, ra := range inv.retiredActivations[:3] {
			if !ra.holdsAfterAgentEvent() {
				t.Errorf("%s: %s dropped its after-agent action maps", tc.name, ra.swallower.spanID)
			}
		}
	}
	// Recorded ones count no more: a later start with one run of callbacks
	// enclosing it keeps the earliest of the rest.
	inv := setup()
	inv.retiredActivations[1].markAfterAgentDone(at)
	inv.markAfterAgentCallbacksReturnedOnLocked(7, at, 1)
	if got, want := done(inv), (map[string]bool{"r10": true, "r20": false, "r30": true, "other": false}); !reflect.DeepEqual(got, want) {
		t.Errorf("with r10 recorded earlier: afterAgentDone = %v, want %v", got, want)
	}
}

func TestMultiLevelGuardRetirementSettlesAndPrunes(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }

	t.Run("unjoined mid retires under open root, then root completes", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		mid := &spanRecord{spanID: "mid-1", kind: spanKindAgent, name: "mid", seq: 2, parent: root, parentSpanID: root.spanID}
		m := map[string]any{}
		inner := &spanRecord{
			spanID:                 "inner-1",
			kind:                   spanKindAgent,
			name:                   "inner",
			seq:                    3,
			parent:                 mid,
			parentSpanID:           mid.spanID,
			hasAfterAgentCallbacks: true,
			hasResponse:            true,
			ownResponse:            true,
			startPlatformTime:      at(1),
			popRetirement: &popRetirement{
				afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef: m,
			},
		}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root, mid}}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5), inner)

		// mid retires unjoined while root remains open.
		inv.agentStack = inv.agentStack[:2]
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10), mid)
		idx := inv.findRetiredActivationIndexLocked("inner-1")
		if idx < 0 {
			t.Fatal("inner-1 was prematurely pruned while guard root-1 was still open")
		}
		if ra := inv.retiredActivations[idx]; ra.afterAgentDone || !ra.holdsAfterAgentEvent() {
			t.Fatalf("while root-1 open after unjoined mid-1: afterAgentDone=%v holds=%v, want false, true", ra.afterAgentDone, ra.holdsAfterAgentEvent())
		}

		// Now root completes; inner-1 must be marked afterAgentDone, drop its maps, and prune.
		root.completed = true
		inv.agentStack = inv.agentStack[:1]
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(20), root)
		if len(inv.retiredActivations) != 0 {
			t.Fatalf("after root-1 retired: %d entries remain (%+v), want 0", len(inv.retiredActivations), inv.retiredActivations)
		}
	})

	t.Run("top -> root -> mid -> inner clears pending hull and does not block subsequent inner", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top", seq: 1, propagatesChildError: true}
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 2, parent: top, parentSpanID: top.spanID}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top, root}}

		for i := range maxPendingAfterAgent + 2 {
			mid := &spanRecord{spanID: fmt.Sprintf("mid-%d", i), kind: spanKindAgent, name: "mid", seq: uint64(10 + 2*i), parent: root, parentSpanID: root.spanID}
			m := map[string]any{}
			inner := &spanRecord{
				spanID:                 fmt.Sprintf("inner-%d", i),
				kind:                   spanKindAgent,
				name:                   "inner",
				seq:                    uint64(11 + 2*i),
				parent:                 mid,
				parentSpanID:           mid.spanID,
				completed:              true,
				hasResponse:            true,
				ownResponse:            true,
				hasAfterAgentCallbacks: true,
				startPlatformTime:      at(1 + i),
				popRetirement: &popRetirement{
					afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
					afterAgentActionsRef: m,
				},
			}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(2+i), inner)
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(3+i), mid)
		}
		if !slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.afterAgentPending }) {
			t.Fatal("expected pending hull while root-1 was open")
		}

		// root finishes while top remains open.
		root.completed = true
		inv.agentStack = inv.agentStack[:2]
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(50), root)
		for _, ra := range inv.retiredActivations {
			if ra.swallower.name == "inner" && (ra.holdsAfterAgentEvent() || ra.afterAgentPending) {
				t.Fatalf("inner entry stranded after root-1 retired: %+v", ra)
			}
		}
		if len(inv.retiredActivations) != 0 {
			t.Fatalf("after root-1 retired under top: %d entries remain (%+v), want 0", len(inv.retiredActivations), inv.retiredActivations)
		}

		// A later inner under top has no stranded hull blocking provenAfterAllRetiredLocked.
		laterInner := &spanRecord{spanID: "inner-later", kind: spanKindAgent, name: "inner", seq: 200, parent: top, parentSpanID: top.spanID, startPlatformTime: at(60)}
		inv.agentStack = append(inv.agentStack, laterInner)
		if slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool {
			return !ra.excludedByEventTimestamp(laterInner.spanID, laterInner.seq, 0, false, laterInner.startPlatformTime, at(61), true)
		}) {
			t.Fatal("stranded entry failed excludedByEventTimestamp for later inner under top")
		}
	})

	t.Run("open custom top keeps settled non-pending entries and non-vacuously excludes later inner", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top", seq: 1}
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 2, parent: top, parentSpanID: top.spanID}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top, root}}

		for i := range maxPendingAfterAgent + 2 {
			mid := &spanRecord{spanID: fmt.Sprintf("mid-%d", i), kind: spanKindAgent, name: "mid", seq: uint64(10 + 2*i), parent: root, parentSpanID: root.spanID}
			m := map[string]any{}
			inner := &spanRecord{
				spanID:                 fmt.Sprintf("inner-%d", i),
				kind:                   spanKindAgent,
				name:                   "inner",
				seq:                    uint64(11 + 2*i),
				parent:                 mid,
				parentSpanID:           mid.spanID,
				completed:              true,
				hasResponse:            true,
				ownResponse:            true,
				hasAfterAgentCallbacks: true,
				startPlatformTime:      at(1 + i),
				popRetirement: &popRetirement{
					afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
					afterAgentActionsRef: m,
				},
			}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(2+i), inner)
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(3+i), mid)
		}
		if !slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.afterAgentPending }) {
			t.Fatal("expected pending hull while root-1 was open")
		}

		// root finishes while custom top (propagatesChildError=false) remains open.
		root.completed = true
		inv.agentStack = inv.agentStack[:2]
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(50), root)
		innerCount := 0
		hasInnerHull := false
		for _, ra := range inv.retiredActivations {
			if ra.swallower.name != "inner" {
				continue
			}
			innerCount++
			if ra.compactedHull {
				hasInnerHull = true
			}
			if ra.holdsAfterAgentEvent() || ra.afterAgentPending || !ra.afterAgentDone {
				t.Fatalf("settled entry under custom top kept pending/maps or missed afterAgentDone: %+v", ra)
			}
			if !slices.Equal(ra.guardSpanIDs, []string{top.spanID}) {
				t.Fatalf("retained entry guardSpanIDs = %v, want [%s]", ra.guardSpanIDs, top.spanID)
			}
		}
		if innerCount == 0 || innerCount > 3 || !hasInnerHull || len(inv.retiredActivations) > 7 {
			t.Fatalf("after root-1 retired under open custom top: innerCount=%d hasInnerHull=%v totalLen=%d, want 1..3 inner with hull and total<=7", innerCount, hasInnerHull, len(inv.retiredActivations))
		}

		// Non-vacuously verify excludedByEventTimestamp and provenAfterAllRetiredLocked against retained entries.
		laterInner := &spanRecord{spanID: "inner-later", kind: spanKindAgent, name: "inner", seq: 200, parent: top, parentSpanID: top.spanID, startPlatformTime: at(60)}
		inv.agentStack = append(inv.agentStack, laterInner)
		if slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool {
			return ra.swallower.name == "inner" && !ra.excludedByEventTimestamp(laterInner.spanID, laterInner.seq, 0, false, laterInner.startPlatformTime, at(61), true)
		}) {
			t.Fatal("retained settled entry/hull failed excludedByEventTimestamp for later inner under custom top")
		}
		if !inv.provenAfterAllRetiredLocked(laterInner, "", at(61), true, true) {
			t.Fatal("provenAfterAllRetiredLocked = false despite retained settled entries/hull ending at at(50) before later inner at(60)")
		}
	})

	t.Run("mid completes while root open, peer clears while root open, then root retires", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		mid := &spanRecord{spanID: "mid-1", kind: spanKindAgent, name: "mid", seq: 2, parent: root, parentSpanID: root.spanID, completed: true}
		innerPeer := &spanRecord{spanID: "inner-peer", kind: spanKindAgent, name: "inner", seq: 4, parent: root, parentSpanID: root.spanID}
		inner1 := &spanRecord{
			spanID:            "inner-1",
			kind:              spanKindAgent,
			name:              "inner",
			seq:               3,
			parent:            mid,
			parentSpanID:      mid.spanID,
			completed:         true,
			hasResponse:       true,
			ownResponse:       true,
			startPlatformTime: at(1),
		}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root, mid, innerPeer}}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5), inner1)
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, root, innerPeer}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10), mid)

		// Peer closes while root stays open; inner-1 stays guarded by root-1.
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, root}
		innerPeer.completed = true
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(15), innerPeer)
		if inv.findRetiredActivationIndexLocked("inner-1") < 0 {
			t.Fatal("inner-1 disappeared while root-1 was still open")
		}

		// Now root retires; inner-1 must be pruned automatically.
		root.completed = true
		inv.agentStack = inv.agentStack[:1]
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(25), root)
		if len(inv.retiredActivations) != 0 {
			t.Fatalf("after root-1 retired: %d entries remain (%+v), want 0", len(inv.retiredActivations), inv.retiredActivations)
		}
	})

	t.Run("unjoined root retires while swallower survives, then top recovers", func(t *testing.T) {
		for _, joinedInner := range []bool{true, false} {
			top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top", seq: 1, propagatesChildError: true}
			root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 2, parent: top, parentSpanID: top.spanID}
			mid := &spanRecord{spanID: "mid-1", kind: spanKindAgent, name: "mid", seq: 3, parent: root, parentSpanID: root.spanID}
			m := map[string]any{}
			inner := &spanRecord{
				spanID:            "inner-1",
				kind:              spanKindAgent,
				name:              "inner",
				seq:               4,
				parent:            mid,
				parentSpanID:      mid.spanID,
				completed:         joinedInner,
				startPlatformTime: at(1),
				popRetirement: &popRetirement{
					afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
					afterAgentActionsRef: m,
				},
			}
			inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top, root, mid}}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5), inner)
			top.retiredCauses = []retiredCause{{err: fmt.Errorf("child err"), name: "inner", swallowers: []retiredSwallower{{spanID: "inner-1", seq: 4, name: "inner"}}}}

			// Both mid and root retire unjoined while inner-1 is still an active swallower on top.
			inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, root}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10), mid)
			inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(15), root)
			if idx := inv.findRetiredActivationIndexLocked("inner-1"); idx < 0 || !inv.retiredActivations[idx].holdsAfterAgentEvent() {
				t.Fatalf("joinedInner=%v: active swallower lost its maps before top recovered: idx=%d", joinedInner, idx)
			}

			// top recovers and settles/prunes inner-1.
			top.retiredCauses = nil
			if joinedInner {
				inv.markSettledRetiredActivationsLocked(top.spanID, true)
			} else {
				inv.pruneSettledRetiredActivationsLocked("inner", "")
			}
			if len(inv.retiredActivations) != 0 {
				t.Fatalf("joinedInner=%v: after top recovered: %d entries remain (%+v), want 0", joinedInner, len(inv.retiredActivations), inv.retiredActivations)
			}
		}
	})

	t.Run("mid retires unjoined, then root completes while outer guard stays open", func(t *testing.T) {
		outerGuard := &spanRecord{spanID: "outer-1", kind: spanKindAgent, name: "outer", seq: 1}
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 2, parent: outerGuard, parentSpanID: outerGuard.spanID}
		mid := &spanRecord{spanID: "mid-1", kind: spanKindAgent, name: "mid", seq: 3, parent: root, parentSpanID: root.spanID}
		m := map[string]any{}
		inner := &spanRecord{
			spanID:                 "inner-1",
			kind:                   spanKindAgent,
			name:                   "inner",
			seq:                    4,
			parent:                 mid,
			parentSpanID:           mid.spanID,
			hasAfterAgentCallbacks: true,
			hasResponse:            true,
			ownResponse:            true,
			startPlatformTime:      at(1),
			popRetirement: &popRetirement{
				afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef: m,
			},
		}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, outerGuard, root, mid}}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5), inner)

		// mid retires unjoined while root and outerGuard stay open.
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, outerGuard, root}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10), mid)
		if idx := inv.findRetiredActivationIndexLocked("inner-1"); idx < 0 || !inv.retiredActivations[idx].holdsAfterAgentEvent() {
			t.Fatalf("expected inner-1 to keep maps while root and outerGuard open: idx=%d", idx)
		}

		// root completes while outerGuard remains open: inner-1 stays guarded by outerGuard,
		// so only settleAfterAgentActionsLocked(affectedByRetired) drops its callback maps.
		root.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, outerGuard}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(20), root)
		idx := inv.findRetiredActivationIndexLocked("inner-1")
		if idx < 0 {
			t.Fatal("inner-1 was prematurely pruned while outerGuard was still open")
		}
		if ra := inv.retiredActivations[idx]; !ra.afterAgentDone || ra.holdsAfterAgentEvent() || ra.afterAgentActionsRef != nil {
			t.Fatalf("after root-1 completed under open outerGuard: afterAgentDone=%v holds=%v ref=%v; want true, false, nil",
				ra.afterAgentDone, ra.holdsAfterAgentEvent(), ra.afterAgentActionsRef != nil)
		}
	})
}

func TestDeadParentSpanIDLoopBoundedUnderOpenGuard(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }

	countRefs := func(inv *invocationTrace) int {
		refs := 0
		for _, ra := range inv.retiredActivations {
			if ra.afterAgentActionsRef != nil {
				refs++
			}
		}
		return refs
	}

	t.Run("active swallower across distinct retired seq-m parents stays bounded", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
		root.retiredCauses = []retiredCause{{
			err:        fmt.Errorf("swallowed"),
			name:       "worker",
			swallowers: []retiredSwallower{{spanID: "worker-0", seq: 11, name: "worker"}},
		}}

		for m := range 50 {
			seq := &spanRecord{
				spanID:       fmt.Sprintf("seq-%d", m),
				kind:         spanKindAgent,
				name:         "seq",
				seq:          uint64(10 + 2*m),
				parent:       root,
				parentSpanID: root.spanID,
				completed:    true,
			}
			delta := map[string]any{}
			worker := &spanRecord{
				spanID:                 fmt.Sprintf("worker-%d", m),
				kind:                   spanKindAgent,
				name:                   "worker",
				seq:                    uint64(11 + 2*m),
				parent:                 seq,
				parentSpanID:           seq.spanID,
				completed:              true,
				hasResponse:            true,
				ownResponse:            true,
				hasAfterAgentCallbacks: true,
				startPlatformTime:      at(1 + m),
				popRetirement: &popRetirement{
					afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(delta))},
					afterAgentActionsRef: delta,
				},
			}
			inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, root, seq}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(2+m), worker)
			inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, root}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(3+m), seq)
		}

		// 1 active swallower (worker-0) + at most maxPendingAfterAgent folded entries for worker + bounded seq entries.
		workerEntries := 0
		for _, ra := range inv.retiredActivations {
			if ra.swallower.name == "worker" {
				workerEntries++
			}
		}
		if workerEntries > maxPendingAfterAgent+1 {
			t.Fatalf("worker retiredActivations leaked across dead seq-m parents: got %d, want <= %d", workerEntries, maxPendingAfterAgent+1)
		}
		if refs := countRefs(inv); refs > maxPendingAfterAgent {
			t.Fatalf("afterAgentActionsRef leaked across dead seq-m parents: got %d, want <= %d", refs, maxPendingAfterAgent)
		}
		for _, ra := range inv.retiredActivations {
			if len(ra.guardSpanIDs) > 1 || (len(ra.guardSpanIDs) == 1 && ra.guardSpanIDs[0] != root.spanID) {
				t.Fatalf("entry %s retained closed guardSpanIDs: %v, want only [%s]", ra.swallower.spanID, ra.guardSpanIDs, root.spanID)
			}
			if len(ra.ancestorSpanIDs) > 2 {
				t.Fatalf("entry %s retained unbounded ancestorSpanIDs: %v", ra.swallower.spanID, ra.ancestorSpanIDs)
			}
		}
	})

	t.Run("unjoined seq-m retiring with runOver false stays bounded", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}

		for m := range 50 {
			seq := &spanRecord{
				spanID:       fmt.Sprintf("seq-%d", m),
				kind:         spanKindAgent,
				name:         "seq",
				seq:          uint64(10 + 2*m),
				parent:       root,
				parentSpanID: root.spanID,
				completed:    false,
			}
			delta := map[string]any{}
			worker := &spanRecord{
				spanID:                 fmt.Sprintf("worker-%d", m),
				kind:                   spanKindAgent,
				name:                   "worker",
				seq:                    uint64(11 + 2*m),
				parent:                 seq,
				parentSpanID:           seq.spanID,
				completed:              true,
				hasResponse:            true,
				ownResponse:            true,
				hasAfterAgentCallbacks: true,
				startPlatformTime:      at(1 + m),
				popRetirement: &popRetirement{
					afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(delta))},
					afterAgentActionsRef: delta,
				},
			}
			inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, root, seq}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(2+m), worker)
			inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, root}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(3+m), seq)
		}

		workerEntries := 0
		for _, ra := range inv.retiredActivations {
			if ra.swallower.name == "worker" {
				workerEntries++
			}
		}
		if workerEntries > maxPendingAfterAgent {
			t.Fatalf("worker entries leaked under unjoined seq-m loop: got %d, want <= %d", workerEntries, maxPendingAfterAgent)
		}
		if refs := countRefs(inv); refs > maxPendingAfterAgent-1 {
			t.Fatalf("afterAgentActionsRef leaked under unjoined seq-m loop: got %d, want <= %d", refs, maxPendingAfterAgent-1)
		}
		for _, ra := range inv.retiredActivations {
			if len(ra.guardSpanIDs) > 1 || (len(ra.guardSpanIDs) == 1 && ra.guardSpanIDs[0] != root.spanID) {
				t.Fatalf("entry %s retained closed guardSpanIDs: %v, want only [%s]", ra.swallower.spanID, ra.guardSpanIDs, root.spanID)
			}
			if len(ra.ancestorSpanIDs) > 2 {
				t.Fatalf("entry %s retained unbounded ancestorSpanIDs: %v", ra.swallower.spanID, ra.ancestorSpanIDs)
			}
		}
	})

	t.Run("direct foldPendingAfterAgentLocked strips closed guardSpanIDs", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
		for m := range maxPendingAfterAgent + 4 {
			delta := map[string]any{"m": m}
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:              retiredSwallower{spanID: fmt.Sprintf("w-%d", m), seq: uint64(10 + m), name: "worker"},
				parentSpanID:           root.spanID,
				guardSpanIDs:           []string{fmt.Sprintf("closed-seq-%d", m), root.spanID, fmt.Sprintf("closed-seq-%d", m)},
				ancestorSpanIDs:        []string{fmt.Sprintf("closed-seq-%d", m), root.spanID},
				retiredAtSeq:           uint64(20 + m),
				completedOrTask:        true,
				joinedAtRetire:         true,
				hasAfterAgentCallbacks: true,
				afterAgentActions:      actionMaps{stateDelta: mapPointer(reflect.ValueOf(delta))},
				afterAgentActionsRef:   delta,
				startPlatformTime:      at(m),
			})
		}
		inv.foldPendingAfterAgentLocked("worker", "", inv.afterAgentStateLocked())
		hullIdx := slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.afterAgentPending })
		if hullIdx < 0 {
			t.Fatal("expected folded afterAgentPending hull")
		}
		if got := inv.retiredActivations[hullIdx].guardSpanIDs; !slices.Equal(got, []string{root.spanID}) {
			t.Fatalf("folded hull guardSpanIDs = %v, want [%s]", got, root.spanID)
		}
	})

	t.Run("multiple open-parent hulls merge cleanly once their parents retire", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		seqA := &spanRecord{spanID: "seq-a", kind: spanKindAgent, name: "seq", seq: 2, parent: root, parentSpanID: root.spanID}
		seqB := &spanRecord{spanID: "seq-b", kind: spanKindAgent, name: "seq", seq: 3, parent: root, parentSpanID: root.spanID}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root, seqA, seqB}}

		addWorkers := func(parent *spanRecord, count int, seqBase uint64) {
			for i := range count {
				delta := map[string]any{}
				w := &spanRecord{
					spanID:                 fmt.Sprintf("%s-w-%d", parent.spanID, i),
					kind:                   spanKindAgent,
					name:                   "worker",
					seq:                    seqBase + uint64(i),
					parent:                 parent,
					parentSpanID:           parent.spanID,
					hasAfterAgentCallbacks: true,
					startPlatformTime:      at(1 + i),
					popRetirement: &popRetirement{
						afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(delta))},
						afterAgentActionsRef: delta,
					},
				}
				inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5+i), w)
			}
		}
		addWorkers(seqA, maxPendingAfterAgent+2, 10)
		addWorkers(seqB, maxPendingAfterAgent+2, 100)
		// Deliver after-agent events for two newer seqB workers so hullB has fewer than maxPendingAfterAgent-1 newer peers.
		for _, id := range []string{"seq-b-w-8", "seq-b-w-9"} {
			idx := inv.findRetiredActivationIndexLocked(id)
			inv.retiredActivations[idx].afterAgentEventSeen = true
			inv.retiredActivations[idx].dropAfterAgentEvent()
		}

		// Retire both seqA and seqB unjoined so both hulls merge into the retired-parent group.
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, root}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(50), seqA, seqB)

		workerEntries, hulls := 0, 0
		for _, ra := range inv.retiredActivations {
			if ra.swallower.name == "worker" && ra.holdsAfterAgentEvent() {
				workerEntries++
				if ra.compactedHull {
					hulls++
					if !slices.Equal(ra.ancestorSpanIDs, []string{root.spanID}) {
						t.Fatalf("merged hull ancestorSpanIDs = %v, want [%s]", ra.ancestorSpanIDs, root.spanID)
					}
					if !slices.Contains(ra.guardSpanIDs, root.spanID) {
						t.Fatalf("merged hull guardSpanIDs = %v, missing %s", ra.guardSpanIDs, root.spanID)
					}
				}
			}
		}
		if workerEntries > maxPendingAfterAgent || hulls != 1 {
			t.Fatalf("after merging two parent hulls: workerEntries=%d (want <= %d), hulls=%d (want 1)", workerEntries, maxPendingAfterAgent, hulls)
		}
	})

	t.Run("out-of-order retirement with zero parentRetiredAtPlatformTime compacts to 3", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}

		for m := range 50 {
			seqID := fmt.Sprintf("seq-%d", m)
			// seq-m already retired before worker-m is recorded, so parentRetiredAtPlatformTime is zero while !openSet[seq-m].
			inv.recordRetiredActivationEntryLocked(retiredActivation{
				swallower:              retiredSwallower{spanID: fmt.Sprintf("worker-%d", m), seq: uint64(10 + 2*m), name: "worker"},
				parentSpanID:           seqID,
				guardSpanIDs:           []string{root.spanID},
				ancestorSpanIDs:        []string{root.spanID, seqID},
				retiredAtSeq:           uint64(11 + 2*m),
				completedOrTask:        true,
				joinedAtRetire:         true,
				hasAfterAgentCallbacks: true,
				afterAgentDone:         true,
				startPlatformTime:      at(1 + m),
				retiredAtPlatformTime:  at(2 + m),
			})
			inv.pruneSettledRetiredActivationsLocked("worker", "")
		}
		if len(inv.retiredActivations) != 3 {
			t.Fatalf("out-of-order dead parentSpanID leaked entries: len=%d, want 3", len(inv.retiredActivations))
		}
	})
}

func TestUnjoinedParentDoesNotStampParentRetiredTime(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }

	t.Run("unjoined mid leaves parentRetiredAtPlatformTime zero until joined ancestor retires", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top", seq: 1}
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 2, parent: top, parentSpanID: top.spanID}
		mid := &spanRecord{spanID: "mid-1", kind: spanKindAgent, name: "mid", seq: 3, parent: root, parentSpanID: root.spanID}
		m := map[string]any{}
		inner1 := &spanRecord{
			spanID:                 "inner-1",
			kind:                   spanKindAgent,
			name:                   "inner",
			seq:                    4,
			parent:                 mid,
			parentSpanID:           mid.spanID,
			completed:              true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(1),
			popRetirement: &popRetirement{
				afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef: m,
			},
		}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top, root, mid}}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5), inner1)

		// mid retires unjoined (completed: false, runDone: false) at at(10) while root-1 remains open.
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, root}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10), mid)
		idx := inv.findRetiredActivationIndexLocked("inner-1")
		if idx < 0 {
			t.Fatal("inner-1 missing after unjoined mid retirement")
		}
		ra := inv.retiredActivations[idx]
		if !ra.parentRetiredAtPlatformTime.IsZero() {
			t.Fatalf("unjoined mid-1 stamped parentRetiredAtPlatformTime = %v, want zero", ra.parentRetiredAtPlatformTime.Sub(t0))
		}
		if ra.afterAgentDone {
			t.Fatal("unjoined mid-1 marked inner-1 afterAgentDone=true")
		}

		// Peer inner-2 starts under root-1 at at(12); at at(15) inner-1's still-running
		// AfterAgentCallback creates an event with evCreatedAt = at(15), hasEventID = true.
		inner2 := &spanRecord{
			spanID:            "inner-2",
			kind:              spanKindAgent,
			name:              "inner",
			seq:               10,
			parent:            root,
			parentSpanID:      root.spanID,
			startPlatformTime: at(12),
		}
		inv.agentStack = append(inv.agentStack, inner2)
		if !ra.canOwnEventCreatedAt(at(15), true) {
			t.Fatal("canOwnEventCreatedAt(at(15), true) = false after unjoined parent retirement, want true")
		}
		if !ra.canOwnEventAt(at(15), true, true) {
			t.Fatal("canOwnEventAt(at(15), true, true) = false after unjoined parent retirement, want true")
		}
		if !inv.hasGuardedRetiredPeerForEventLocked(inner2, "", at(15), true, true) {
			t.Fatal("hasGuardedRetiredPeerForEventLocked = false, want true while inner-1's parent retired unjoined")
		}
		if got := inv.soleOpenAuthorLocked([]int{len(inv.agentStack) - 1}, "", at(15), func(*spanRecord) bool { return true }, func(retiredActivation) bool { return true }); got != -1 {
			t.Fatalf("soleOpenAuthorLocked = %d, want -1 (must not attribute inner-1's event to inner-2)", got)
		}

		// When root-1 later retires joined (runOver: true) at at(20), parentRetiredAtPlatformTime
		// must be stamped at at(20) (not frozen at at(10)), while afterAgentEvent=true still owns at(25).
		root.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(20), root)
		idx = inv.findRetiredActivationIndexLocked("inner-1")
		if idx < 0 {
			t.Fatal("inner-1 missing while outer guard top-1 remains open")
		}
		ra = inv.retiredActivations[idx]
		if !ra.parentRetiredAtPlatformTime.Equal(at(20)) {
			t.Fatalf("parentRetiredAtPlatformTime = %v, want %v (at(20))", ra.parentRetiredAtPlatformTime.Sub(t0), at(20).Sub(t0))
		}
		if !ra.afterAgentDone {
			t.Fatal("joined root-1 failed to mark afterAgentDone=true")
		}
		if !ra.canOwnEventAt(at(25), true, true) {
			t.Fatal("canOwnEventAt(at(25), true, true) = false with afterAgentEvent=true, want true")
		}
		if ra.canOwnEventAt(at(25), true, false) {
			t.Fatal("canOwnEventAt(at(25), true, false) = true after parentRetiredAtPlatformTime at(20), want false")
		}
	})

	t.Run("unjoined task child under unjoined mid and runDone outer ancestor", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top", seq: 1}
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 2, parent: top, parentSpanID: top.spanID}
		mid := &spanRecord{spanID: "mid-1", kind: spanKindAgent, name: "mid", seq: 3, parent: root, parentSpanID: root.spanID}
		innerTask := &spanRecord{
			spanID:                 "inner-task",
			kind:                   spanKindAgent,
			name:                   "inner",
			seq:                    4,
			goroutineID:            11,
			parent:                 mid,
			parentSpanID:           mid.spanID,
			taskMode:               true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(1),
		}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top, root, mid}}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5), innerTask)

		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, root}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10), mid)
		idx := inv.findRetiredActivationIndexLocked("inner-task")
		if idx < 0 || !inv.retiredActivations[idx].parentRetiredAtPlatformTime.IsZero() {
			t.Fatalf("unjoined task got parentRetiredAtPlatformTime on unjoined mid: idx=%d ra=%+v", idx, inv.retiredActivations)
		}

		inner2 := &spanRecord{
			spanID:            "inner-2",
			kind:              spanKindAgent,
			name:              "inner",
			seq:               10,
			goroutineID:       22,
			parent:            root,
			parentSpanID:      root.spanID,
			startPlatformTime: at(12),
		}
		inv.agentStack = append(inv.agentStack, inner2)
		ra := inv.retiredActivations[idx]
		if !ra.canOwnEventCreatedAt(at(15), true) || !ra.canOwnEventAt(at(15), true, true) {
			t.Fatal("unjoined task could not own event at at(15) under unjoined parent")
		}
		if !inv.hasGuardedRetiredPeerForEventLocked(inner2, "", at(15), true, true) {
			t.Fatal("hasGuardedRetiredPeerForEventLocked = false for unjoined task under unjoined parent")
		}
		if got := inv.soleOpenAuthorLocked([]int{len(inv.agentStack) - 1}, "", at(15), func(*spanRecord) bool { return true }, func(retiredActivation) bool { return true }); got != -1 {
			t.Fatalf("soleOpenAuthorLocked = %d, want -1 for unjoined task peer", got)
		}

		// root retires via runDone=true (even with completed=false).
		root.runDone = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(20), root)
		idx = inv.findRetiredActivationIndexLocked("inner-task")
		if idx < 0 || !inv.retiredActivations[idx].parentRetiredAtPlatformTime.Equal(at(20)) {
			t.Fatalf("runDone ancestor did not stamp parentRetiredAtPlatformTime at at(20): idx=%d", idx)
		}
	})

	t.Run("outer ancestor completing stops at still-open intermediate ancestor", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top", seq: 1}
		a := &spanRecord{spanID: "a-1", kind: spanKindAgent, name: "a", seq: 2, parent: top, parentSpanID: top.spanID}
		b := &spanRecord{spanID: "b-1", kind: spanKindAgent, name: "b", seq: 3, parent: a, parentSpanID: a.spanID}
		c := &spanRecord{spanID: "c-1", kind: spanKindAgent, name: "c", seq: 4, parent: b, parentSpanID: b.spanID}
		m1 := map[string]any{"k": 1}
		m2 := map[string]any{"k": 2}
		inner1 := &spanRecord{
			spanID:                 "inner-1",
			kind:                   spanKindAgent,
			name:                   "inner",
			seq:                    5,
			parent:                 b,
			parentSpanID:           b.spanID,
			completed:              true,
			ownResponse:            true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(1),
			popRetirement: &popRetirement{
				afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m1))},
				afterAgentActionsRef: m1,
			},
		}
		inner2 := &spanRecord{
			spanID:                 "inner-2",
			kind:                   spanKindAgent,
			name:                   "inner",
			seq:                    6,
			parent:                 c,
			parentSpanID:           c.spanID,
			completed:              true,
			ownResponse:            true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(2),
			popRetirement: &popRetirement{
				afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m2))},
				afterAgentActionsRef: m2,
			},
		}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top, a, b, c}}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5), inner1, inner2)

		// Unjoined c retires while b remains open on agentStack.
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, a, b}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(7), c)

		// Now outer ancestor a completes while intermediate ancestor b remains open on agentStack.
		a.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, b}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10), a)

		for _, id := range []string{"inner-1", "inner-2"} {
			idx := inv.findRetiredActivationIndexLocked(id)
			if idx < 0 {
				t.Fatalf("%s missing while intermediate ancestor b-1 remains open", id)
			}
			ra := inv.retiredActivations[idx]
			if ra.afterAgentDone {
				t.Fatalf("%s marked afterAgentDone=true while intermediate ancestor b-1 is still open", id)
			}
			if !ra.parentRetiredAtPlatformTime.IsZero() {
				t.Fatalf("%s stamped parentRetiredAtPlatformTime=%v while intermediate ancestor b-1 is still open", id, ra.parentRetiredAtPlatformTime.Sub(t0))
			}
			if !ra.holdsAfterAgentEvent() || ra.afterAgentActionsRef == nil {
				t.Fatalf("%s dropped afterAgentActions while intermediate ancestor b-1 is still open: %+v", id, ra)
			}
		}

		// Once b itself completes, both inner-1 and inner-2 become afterAgentDone and drop maps.
		b.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(20), b)
		for _, id := range []string{"inner-1", "inner-2"} {
			idx := inv.findRetiredActivationIndexLocked(id)
			if idx < 0 {
				t.Fatalf("%s missing while top-1 remains open", id)
			}
			ra := inv.retiredActivations[idx]
			if !ra.afterAgentDone || !ra.afterAgentDoneAt.Equal(at(20)) || !ra.parentRetiredAtPlatformTime.Equal(at(20)) || ra.holdsAfterAgentEvent() {
				t.Fatalf("after b-1 completed, %s state unexpected: %+v", id, ra)
			}
		}
	})

	t.Run("multi-parent hull respects still-open intermediate ancestor on one branch", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top", seq: 1}
		a := &spanRecord{spanID: "a-1", kind: spanKindAgent, name: "a", seq: 2, parent: top, parentSpanID: top.spanID}
		b := &spanRecord{spanID: "b-1", kind: spanKindAgent, name: "b", seq: 3, parent: a, parentSpanID: a.spanID}
		c := &spanRecord{spanID: "c-1", kind: spanKindAgent, name: "c", seq: 4, parent: b, parentSpanID: b.spanID}
		d := &spanRecord{spanID: "d-1", kind: spanKindAgent, name: "d", seq: 5, parent: a, parentSpanID: a.spanID}

		m0 := map[string]any{"k": 0}
		w0 := &spanRecord{
			spanID:                 "worker-0",
			kind:                   spanKindAgent,
			name:                   "worker",
			seq:                    6,
			parent:                 c,
			parentSpanID:           c.spanID,
			completed:              true,
			ownResponse:            true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(1),
			popRetirement: &popRetirement{
				afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m0))},
				afterAgentActionsRef: m0,
			},
		}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top, a, b, c}}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(2), w0)
		// c retires unjoined while b, a, top stay open.
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, a, b}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(3), c)

		// Under d (which retires unjoined under a), retire maxPendingAfterAgent workers so worker-0 and worker-1.. fold into a multi-parent hull.
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, a, b, d}
		for i := 1; i <= maxPendingAfterAgent; i++ {
			mi := map[string]any{"k": i}
			wi := &spanRecord{
				spanID:                 fmt.Sprintf("worker-%d", i),
				kind:                   spanKindAgent,
				name:                   "worker",
				seq:                    uint64(10 + i),
				parent:                 d,
				parentSpanID:           d.spanID,
				completed:              true,
				ownResponse:            true,
				hasAfterAgentCallbacks: true,
				startPlatformTime:      at(4 + i),
				popRetirement: &popRetirement{
					afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(mi))},
					afterAgentActionsRef: mi,
				},
			}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5+i), wi)
		}
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, a, b}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(15), d)
		inv.foldPendingAfterAgentLocked("worker", "", inv.afterAgentStateLocked())

		hullIdx := slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool {
			return ra.swallower.name == "worker" && ra.compactedHull
		})
		if hullIdx < 0 {
			t.Fatalf("expected multi-parent pending hull for worker: %+v", inv.retiredActivations)
		}

		// a completes while b remains open under top: the hull MUST NOT be marked afterAgentDone or drop pending!
		a.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, b}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(20), a)

		hullIdx = slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool {
			return ra.swallower.name == "worker" && ra.compactedHull
		})
		if hullIdx < 0 {
			t.Fatal("multi-parent hull missing while b-1 remains open")
		}
		if h := inv.retiredActivations[hullIdx]; h.afterAgentDone || !h.parentRetiredAtPlatformTime.IsZero() || !h.holdsAfterAgentEvent() {
			t.Fatalf("completing a-1 while b-1 remained open prematurely settled multi-parent hull: %+v", h)
		}

		// Once b itself completes while top stays open, the hull settles cleanly.
		b.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(30), b)
		hullIdx = slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool {
			return ra.swallower.name == "worker" && ra.compactedHull
		})
		if hullIdx < 0 {
			t.Fatal("multi-parent hull missing under open top-1 after b-1 completed")
		}
		if h := inv.retiredActivations[hullIdx]; !h.afterAgentDone || !h.afterAgentDoneAt.Equal(at(30)) || !h.parentRetiredAtPlatformTime.Equal(at(30)) || h.holdsAfterAgentEvent() {
			t.Fatalf("multi-parent hull failed to settle once b-1 completed: %+v", h)
		}
	})

	t.Run("multi-parent hull across sibling subtrees settles when both branch ancestors complete under open top", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top", seq: 1}
		a1 := &spanRecord{spanID: "a-1", kind: spanKindAgent, name: "a", seq: 2, parent: top, parentSpanID: top.spanID}
		c1 := &spanRecord{spanID: "c-1", kind: spanKindAgent, name: "c", seq: 3, parent: a1, parentSpanID: a1.spanID}
		a2 := &spanRecord{spanID: "a-2", kind: spanKindAgent, name: "a", seq: 4, parent: top, parentSpanID: top.spanID}
		c2 := &spanRecord{spanID: "c-2", kind: spanKindAgent, name: "c", seq: 5, parent: a2, parentSpanID: a2.spanID}

		m0 := map[string]any{"k": 0}
		w0 := &spanRecord{
			spanID:                 "worker-0",
			kind:                   spanKindAgent,
			name:                   "worker",
			seq:                    6,
			parent:                 c1,
			parentSpanID:           c1.spanID,
			completed:              true,
			ownResponse:            true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(1),
			popRetirement: &popRetirement{
				afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m0))},
				afterAgentActionsRef: m0,
			},
		}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top, a1, c1}}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(2), w0)
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, a1}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(3), c1)

		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, a1, a2, c2}
		for i := 1; i <= maxPendingAfterAgent; i++ {
			mi := map[string]any{"k": i}
			wi := &spanRecord{
				spanID:                 fmt.Sprintf("worker-%d", i),
				kind:                   spanKindAgent,
				name:                   "worker",
				seq:                    uint64(10 + i),
				parent:                 c2,
				parentSpanID:           c2.spanID,
				completed:              true,
				ownResponse:            true,
				hasAfterAgentCallbacks: true,
				startPlatformTime:      at(4 + i),
				popRetirement: &popRetirement{
					afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(mi))},
					afterAgentActionsRef: mi,
				},
			}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5+i), wi)
		}
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, a1, a2}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(15), c2)
		inv.foldPendingAfterAgentLocked("worker", "", inv.afterAgentStateLocked())

		// a1 completes while a2 stays open under top: hull stays pending.
		a1.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, a2}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(20), a1)
		hullIdx := slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool {
			return ra.swallower.name == "worker" && ra.compactedHull
		})
		if hullIdx < 0 {
			t.Fatal("sibling-subtree hull disappeared while a2 remained open")
		}
		if h := inv.retiredActivations[hullIdx]; h.afterAgentDone || !h.holdsAfterAgentEvent() {
			t.Fatalf("completing only a1 settled sibling-subtree hull while a2 stayed open: %+v", h)
		}

		// Now a2 completes while top STILL remains open: hull MUST settle immediately!
		a2.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(30), a2)
		hullIdx = slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool {
			return ra.swallower.name == "worker" && ra.compactedHull
		})
		if hullIdx < 0 {
			t.Fatal("sibling-subtree hull disappeared while top remained open")
		}
		if h := inv.retiredActivations[hullIdx]; !h.afterAgentDone || !h.afterAgentDoneAt.Equal(at(30)) || !h.parentRetiredAtPlatformTime.Equal(at(30)) || h.holdsAfterAgentEvent() || h.afterAgentPending {
			t.Fatalf("sibling-subtree hull wedged after both a1 and a2 completed under open top: %+v", h)
		}
	})

	t.Run("earlier completed ancestor does not settle descendant retiring under open outer guard", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top", seq: 1}
		a := &spanRecord{spanID: "a-1", kind: spanKindAgent, name: "a", seq: 2, parent: top, parentSpanID: top.spanID}
		m := map[string]any{"k": 1}
		b := &spanRecord{
			spanID:                 "b-1",
			kind:                   spanKindAgent,
			name:                   "b",
			seq:                    3,
			parent:                 a,
			parentSpanID:           a.spanID,
			ownResponse:            true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(2),
			popRetirement: &popRetirement{
				afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef: m,
			},
		}
		bTask := &spanRecord{
			spanID:            "b-task",
			kind:              spanKindAgent,
			name:              "b",
			seq:               4,
			parent:            a,
			parentSpanID:      a.spanID,
			taskMode:          true,
			startPlatformTime: at(3),
		}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top, a, b, bTask}}

		// a completes in an earlier retirement at at(10) while b and bTask remain open under top.
		a.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, b, bTask}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10), a)

		// Now b finishes Run and bTask retires unjoined at at(20) while top remains open.
		b.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(20), b, bTask)

		for _, id := range []string{"b-1", "b-task"} {
			idx := inv.findRetiredActivationIndexLocked(id)
			if idx < 0 {
				t.Fatalf("%s missing while outer guard top-1 remains open", id)
			}
			ra := inv.retiredActivations[idx]
			if !slices.Equal(ra.openEnclosingSpanIDs, []string{top.spanID}) {
				t.Fatalf("%s openEnclosingSpanIDs = %v, want [%s]", id, ra.openEnclosingSpanIDs, top.spanID)
			}
			if ra.afterAgentDone || !ra.parentRetiredAtPlatformTime.IsZero() {
				t.Fatalf("%s prematurely settled against earlier completed a-1 under open top-1: %+v", id, ra)
			}
		}
		if idxB := inv.findRetiredActivationIndexLocked("b-1"); !inv.retiredActivations[idxB].holdsAfterAgentEvent() {
			t.Fatalf("b-1 dropped afterAgentActions while top-1 remained open: %+v", inv.retiredActivations[idxB])
		}

		// Once top completes at at(30), b-1 settles cleanly.
		top.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(30), top)
		if len(inv.retiredActivations) != 0 {
			t.Fatalf("after top-1 completed, expected empty ledger, got %d entries", len(inv.retiredActivations))
		}
	})

	t.Run("descendant under still-open b advances past earlier completed a to open top when b retires unjoined", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top", seq: 1}
		a := &spanRecord{spanID: "a-1", kind: spanKindAgent, name: "a", seq: 2, parent: top, parentSpanID: top.spanID}
		b := &spanRecord{spanID: "b-1", kind: spanKindAgent, name: "b", seq: 3, parent: a, parentSpanID: a.spanID}
		m := map[string]any{"k": 1}
		inner1 := &spanRecord{
			spanID:                 "inner-1",
			kind:                   spanKindAgent,
			name:                   "inner",
			seq:                    4,
			parent:                 b,
			parentSpanID:           b.spanID,
			completed:              true,
			ownResponse:            true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(1),
			popRetirement: &popRetirement{
				afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef: m,
			},
		}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top, a, b}}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5), inner1)

		// a completes at at(10) while b remains open.
		a.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, b}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10), a)

		// b retires unjoined at at(20) while top remains open.
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(20), b)

		idx := inv.findRetiredActivationIndexLocked("inner-1")
		if idx < 0 {
			t.Fatal("inner-1 missing while top-1 remains open")
		}
		ra := inv.retiredActivations[idx]
		if !slices.Equal(ra.openEnclosingSpanIDs, []string{top.spanID}) {
			t.Fatalf("inner-1 openEnclosingSpanIDs = %v, want [%s]", ra.openEnclosingSpanIDs, top.spanID)
		}
		if ra.afterAgentDone || !ra.parentRetiredAtPlatformTime.IsZero() || !ra.holdsAfterAgentEvent() {
			t.Fatalf("inner-1 prematurely settled past earlier completed a-1 under open top-1: %+v", ra)
		}
	})

	t.Run("multi-parent hull fails closed when unshared openEnclosingSpanID leaves stack outside recordRetiredActivationsAtLocked", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top", seq: 1}
		a := &spanRecord{spanID: "a-1", kind: spanKindAgent, name: "a", seq: 2, parent: top, parentSpanID: top.spanID}
		b := &spanRecord{spanID: "b-1", kind: spanKindAgent, name: "b", seq: 3, parent: a, parentSpanID: a.spanID}
		m := &spanRecord{spanID: "m-1", kind: spanKindAgent, name: "m", seq: 4, parent: b, parentSpanID: b.spanID}
		c := &spanRecord{spanID: "c-1", kind: spanKindAgent, name: "c", seq: 5, parent: m, parentSpanID: m.spanID}
		d := &spanRecord{spanID: "d-1", kind: spanKindAgent, name: "d", seq: 6, parent: a, parentSpanID: a.spanID}

		// Branch 1: top -> a -> b -> m -> c -> w0 (c retires unjoined, leaving w0.openEnclosingSpanIDs = ["m-1"]).
		m0 := map[string]any{"k": 0}
		w0 := &spanRecord{
			spanID:                 "worker-0",
			kind:                   spanKindAgent,
			name:                   "worker",
			seq:                    7,
			parent:                 c,
			parentSpanID:           c.spanID,
			completed:              true,
			ownResponse:            true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(1),
			popRetirement: &popRetirement{
				afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(m0))},
				afterAgentActionsRef: m0,
			},
		}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top, a, b, m, c}}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(2), w0)
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, a, b, m}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(3), c)

		// Branch 2: top -> a -> d -> w1..wMax (d retires unjoined, forming a multi-parent hull with ancestorSpanIDs = ["a-1", "top-1"] and openEnclosingSpanIDs = ["a-1", "m-1"]).
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, a, b, m, d}
		for i := 1; i <= maxPendingAfterAgent; i++ {
			mi := map[string]any{"k": i}
			wi := &spanRecord{
				spanID:                 fmt.Sprintf("worker-%d", i),
				kind:                   spanKindAgent,
				name:                   "worker",
				seq:                    uint64(10 + i),
				parent:                 d,
				parentSpanID:           d.spanID,
				completed:              true,
				ownResponse:            true,
				hasAfterAgentCallbacks: true,
				startPlatformTime:      at(4 + i),
				popRetirement: &popRetirement{
					afterAgentActions:    actionMaps{stateDelta: mapPointer(reflect.ValueOf(mi))},
					afterAgentActionsRef: mi,
				},
			}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5+i), wi)
		}
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, a, b, m}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(15), d)
		inv.foldPendingAfterAgentLocked("worker", "", inv.afterAgentStateLocked())

		// m-1 leaves agentStack outside recordRetiredActivationsAtLocked (e.g. purgeActivationLocked),
		// while b-1 and top-1 remain open; then a-1 completes.
		a.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top, b}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(20), a)

		hullIdx := slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool {
			return ra.swallower.name == "worker" && ra.compactedHull
		})
		if hullIdx < 0 {
			t.Fatal("multi-parent hull missing while b-1 and top-1 remain open")
		}
		h := inv.retiredActivations[hullIdx]
		if !h.enclosingRetiredUnjoined || h.afterAgentDone || !h.parentRetiredAtPlatformTime.IsZero() || !h.holdsAfterAgentEvent() {
			t.Fatalf("completing a-1 after unshared m-1 left stack outside recordRetiredActivationsAtLocked failed open: %+v", h)
		}

		// Once outer guard top-1 retires, the hull still drops pending state and prunes cleanly.
		b.completed, top.completed = true, true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(30), b, top)
		if idx := slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool {
			return ra.swallower.name == "worker"
		}); idx >= 0 {
			t.Fatalf("hull leaked after outer guard top-1 retired: %+v", inv.retiredActivations[idx])
		}
	})
}

func TestCustomUUIDProvenancePreservedInHulls(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	k := llmKey(identity{agentName: "worker"})

	t.Run("compacted hull preserves custom UUID provenance and prevents cross-peer credit theft", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{},
		}
		// Compact w0..w4 (at(0)..at(10)) into w0, hull, w4.
		for i := range 5 {
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:             retiredSwallower{spanID: fmt.Sprintf("w-%d", i), seq: uint64(2 + i), name: "worker"},
				parentSpanID:          root.spanID,
				guardSpanIDs:          []string{root.spanID},
				ancestorSpanIDs:       []string{root.spanID},
				retiredAtSeq:          uint64(10 + i),
				completedOrTask:       true,
				joinedAtRetire:        true,
				hasCustomUUIDProvider: true,
				startPlatformTime:     at(2 * i),
				retiredAtPlatformTime: at(2*i + 1),
			})
		}
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		if len(inv.retiredActivations) != 3 {
			t.Fatalf("after compacting w0..w4: len=%d, want 3", len(inv.retiredActivations))
		}
		hull := inv.retiredActivations[1]
		if !hull.compactedHull || !hull.hasCustomUUIDProvider {
			t.Fatalf("compacted hull wiped custom UUID provenance: compactedHull=%v hasCustomUUIDProvider=%v", hull.compactedHull, hull.hasCustomUUIDProvider)
		}

		// Later disjoint w5 (at(20)..at(25)) and w6 (at(30)..at(35)) retire with credits.
		inv.unrespondedRetired[k] = 2
		inv.retiredActivations = append(inv.retiredActivations,
			retiredActivation{
				swallower:             retiredSwallower{spanID: "w-5", seq: 20, name: "worker"},
				parentSpanID:          root.spanID,
				guardSpanIDs:          []string{root.spanID},
				ancestorSpanIDs:       []string{root.spanID},
				retiredAtSeq:          21,
				completedOrTask:       true,
				joinedAtRetire:        true,
				hasCustomUUIDProvider: true,
				afterAgentCredit:      true,
				startPlatformTime:     at(20),
				retiredAtPlatformTime: at(25),
			},
			retiredActivation{
				swallower:             retiredSwallower{spanID: "w-6", seq: 30, name: "worker"},
				parentSpanID:          root.spanID,
				guardSpanIDs:          []string{root.spanID},
				ancestorSpanIDs:       []string{root.spanID},
				retiredAtSeq:          31,
				completedOrTask:       true,
				joinedAtRetire:        true,
				hasCustomUUIDProvider: true,
				afterAgentCredit:      true,
				startPlatformTime:     at(30),
				retiredAtPlatformTime: at(35),
			},
		)

		// w5's run event at at(22) with non-UUID non-empty ID (hasEventID=false, evNonEmptyID=true)
		// must attribute directly to w5 without matching hull as an ambiguous author.
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(22), false, true) {
			t.Fatal("w5 custom-UUID run event settled no credit")
		}
		idx5 := inv.findRetiredActivationIndexLocked("w-5")
		idx6 := inv.findRetiredActivationIndexLocked("w-6")
		if idx5 < 0 || idx6 < 0 {
			t.Fatalf("missing w5/w6: idx5=%d idx6=%d", idx5, idx6)
		}
		if w5 := inv.retiredActivations[idx5]; !w5.creditSettled || w5.afterAgentCredit || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("w5 event did not settle w5 directly: w5=%+v unresponded=%d", w5, inv.unrespondedRetired[k])
		}
		// Second w5 run event and consumeAfterAgentCreditLocked must not steal w6's credit.
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(24), false, true) {
			t.Fatal("second w5 run event stole w6 credit")
		}
		inv.consumeAfterAgentCreditLocked(idx5)
		if inv.unrespondedRetired[k] != 1 || !inv.retiredActivations[idx6].afterAgentCredit || inv.retiredActivations[idx6].creditSettled {
			t.Fatalf("w6 credit was stolen: unresponded=%d w6=%+v", inv.unrespondedRetired[k], inv.retiredActivations[idx6])
		}
	})

	t.Run("compacted hull preserves afterAgentEventSeen for canOwnEventCreatedAt and soleOpenAuthor", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
		for i := range 5 {
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:              retiredSwallower{spanID: fmt.Sprintf("w-%d", i), seq: uint64(2 + i), name: "worker"},
				parentSpanID:           root.spanID,
				guardSpanIDs:           []string{root.spanID},
				ancestorSpanIDs:        []string{root.spanID},
				retiredAtSeq:           uint64(10 + i),
				completedOrTask:        true,
				joinedAtRetire:         true,
				settledByParent:        true,
				hasAfterAgentCallbacks: true,
				afterAgentEventSeen:    true,
				afterAgentDone:         false,
				startPlatformTime:      at(2 * i),
				retiredAtPlatformTime:  at(2*i + 1),
			})
		}
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		if len(inv.retiredActivations) != 3 {
			t.Fatalf("after compacting w0..w4: len=%d, want 3", len(inv.retiredActivations))
		}
		hull := inv.retiredActivations[1]
		if !hull.compactedHull || !hull.afterAgentEventSeen {
			t.Fatalf("compacted hull cleared afterAgentEventSeen: compactedHull=%v afterAgentEventSeen=%v", hull.compactedHull, hull.afterAgentEventSeen)
		}
		if hull.canOwnEventCreatedAt(at(25), true) {
			t.Fatalf("compacted hull with afterAgentEventSeen=true falsely claimed event at at(25) after retiredAtPlatformTime=%v", hull.retiredAtPlatformTime.Sub(t0))
		}
		w5 := &spanRecord{
			spanID:            "w-5",
			parentSpanID:      root.spanID,
			parent:            root,
			kind:              spanKindAgent,
			name:              "worker",
			seq:               30,
			startPlatformTime: at(20),
		}
		inv.agentStack = append(inv.agentStack, w5)
		if got := inv.soleOpenAuthorLocked([]int{len(inv.agentStack) - 1}, "", at(25), func(*spanRecord) bool { return true }, func(retiredActivation) bool { return true }); got != len(inv.agentStack)-1 {
			t.Fatalf("soleOpenAuthorLocked = %d, want %d when hull has afterAgentEventSeen=true", got, len(inv.agentStack)-1)
		}
	})

	t.Run("custom-UUID hull retains timestamp provenance in guard, soleOpenAuthor, and provenAfterAllRetired", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
		for i := range 5 {
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:             retiredSwallower{spanID: fmt.Sprintf("w-%d", i), seq: uint64(2 + i), name: "worker"},
				parentSpanID:          root.spanID,
				guardSpanIDs:          []string{root.spanID},
				ancestorSpanIDs:       []string{root.spanID},
				retiredAtSeq:          uint64(10 + i),
				completedOrTask:       true,
				joinedAtRetire:        true,
				hasCustomUUIDProvider: true,
				startPlatformTime:     at(2 * i),
				retiredAtPlatformTime: at(2*i + 1),
			})
		}
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		if len(inv.retiredActivations) != 3 || !inv.retiredActivations[1].hasCustomUUIDProvider {
			t.Fatalf("expected 3 entries with hull hasCustomUUIDProvider=true, got %+v", inv.retiredActivations)
		}

		later := &spanRecord{
			spanID:                "w-later",
			kind:                  spanKindAgent,
			name:                  "worker",
			seq:                   50,
			parent:                root,
			parentSpanID:          root.spanID,
			hasCustomUUIDProvider: true,
			startPlatformTime:     at(20),
		}
		inv.agentStack = append(inv.agentStack, later)

		if inv.hasGuardedRetiredPeerForEventLocked(later, "", at(25), false, true) {
			t.Fatal("hasGuardedRetiredPeerForEventLocked = true, want false when compacted hull has custom UUID provenance")
		}
		hasOpen := func(r *spanRecord) bool { return r.hasCustomUUIDProvider }
		hasRetired := func(ra retiredActivation) bool { return ra.hasCustomUUIDProvider }
		if got := inv.soleOpenAuthorLocked([]int{2}, "", at(25), hasOpen, hasRetired); got != 2 {
			t.Fatalf("soleOpenAuthorLocked = %d, want 2 (later open activation)", got)
		}
		if !inv.provenAfterAllRetiredLocked(later, "", at(25), false, true) {
			t.Fatal("provenAfterAllRetiredLocked = false, want true when compacted hull has custom UUID provenance")
		}
	})

	t.Run("mixed custom-UUID entries stay in separate compactKey buckets", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
		for i := range 10 {
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:             retiredSwallower{spanID: fmt.Sprintf("w-%d", i), seq: uint64(2 + i), name: "worker"},
				parentSpanID:          root.spanID,
				guardSpanIDs:          []string{root.spanID},
				ancestorSpanIDs:       []string{root.spanID},
				retiredAtSeq:          uint64(20 + i),
				completedOrTask:       true,
				joinedAtRetire:        true,
				hasCustomUUIDProvider: i%2 == 0,
				startPlatformTime:     at(2 * i),
				retiredAtPlatformTime: at(2*i + 1),
			})
		}
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		var customHulls, nonCustomHulls int
		for _, ra := range inv.retiredActivations {
			if ra.compactedHull {
				if ra.hasCustomUUIDProvider {
					customHulls++
				} else {
					nonCustomHulls++
				}
			}
		}
		if len(inv.retiredActivations) != 6 || customHulls != 1 || nonCustomHulls != 1 {
			t.Fatalf("mixed compactKey buckets: len=%d customHulls=%d nonCustomHulls=%d, want 6, 1, 1", len(inv.retiredActivations), customHulls, nonCustomHulls)
		}
	})

	t.Run("foldPendingAfterAgentLocked preserves custom UUID iff all folded members have it", func(t *testing.T) {
		for _, oneFalse := range []bool{false, true} {
			t.Run(fmt.Sprintf("oneFalse=%v", oneFalse), func(t *testing.T) {
				root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
				inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
				for n := range maxPendingAfterAgent + 2 {
					m := map[string]any{}
					inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
						swallower:              retiredSwallower{spanID: fmt.Sprintf("p-%d", n), seq: uint64(10 + n), name: "worker"},
						parentSpanID:           root.spanID,
						guardSpanIDs:           []string{root.spanID},
						ancestorSpanIDs:        []string{root.spanID},
						retiredAtSeq:           uint64(100 + n),
						completedOrTask:        true,
						joinedAtRetire:         true,
						hasAfterAgentCallbacks: true,
						hasCustomUUIDProvider:  !(oneFalse && n == 1),
						afterAgentActions:      actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
						afterAgentActionsRef:   m,
						startPlatformTime:      at(2 * n),
						retiredAtPlatformTime:  at(2*n + 1),
					})
				}
				inv.foldPendingAfterAgentLocked("worker", "", inv.afterAgentStateLocked())
				idx := slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.afterAgentPending })
				if idx < 0 {
					t.Fatal("expected a pending hull")
				}
				got := inv.retiredActivations[idx].hasCustomUUIDProvider
				want := !oneFalse
				if got != want {
					t.Fatalf("folded hull hasCustomUUIDProvider = %v, want %v", got, want)
				}
			})
		}
	})
}

func TestCompactedHullClearsUnsharedParentSpanID(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }

	t.Run("distinct retired parents clear parentSpanID and reject out-of-order runOver match", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
		for m := range 5 {
			seqID := fmt.Sprintf("seq-%d", m)
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:              retiredSwallower{spanID: fmt.Sprintf("worker-%d", m), seq: uint64(10 + 2*m), name: "worker"},
				parentSpanID:           seqID,
				guardSpanIDs:           []string{root.spanID},
				ancestorSpanIDs:        []string{seqID, root.spanID},
				retiredAtSeq:           uint64(11 + 2*m),
				completedOrTask:        true,
				joinedAtRetire:         true,
				hasAfterAgentCallbacks: true,
				afterAgentDone:         false,
				startPlatformTime:      at(2 * m),
				retiredAtPlatformTime:  at(2*m + 1),
			})
		}
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		if len(inv.retiredActivations) != 3 {
			t.Fatalf("after compaction: len=%d, want 3", len(inv.retiredActivations))
		}
		hull := &inv.retiredActivations[1]
		if !hull.compactedHull {
			t.Fatal("expected middle entry to be compactedHull")
		}
		if hull.parentSpanID != "" {
			t.Fatalf("multi-parent hull kept parentSpanID = %q, want empty", hull.parentSpanID)
		}
		if !slices.Equal(hull.ancestorSpanIDs, []string{root.spanID}) {
			t.Fatalf("multi-parent hull ancestorSpanIDs = %v, want [%s]", hull.ancestorSpanIDs, root.spanID)
		}

		// An out-of-order runOver retirement for seq-3 must NOT match hull.parentSpanID.
		seq3 := &spanRecord{spanID: "seq-3", kind: spanKindAgent, name: "seq", seq: 16, parent: root, parentSpanID: root.spanID, completed: true}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(50), seq3)
		hullIdx := slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.compactedHull })
		if hullIdx < 0 {
			t.Fatal("hull disappeared after out-of-order seq-3 retirement")
		}
		if h := inv.retiredActivations[hullIdx]; h.afterAgentDone || !h.parentRetiredAtPlatformTime.IsZero() {
			t.Fatalf("out-of-order seq-3 falsely marked multi-parent hull done: afterAgentDone=%v parentRetiredAt=%v", h.afterAgentDone, h.parentRetiredAtPlatformTime)
		}
	})

	t.Run("shared retired parent keeps parentSpanID", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
		for m := range 5 {
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:             retiredSwallower{spanID: fmt.Sprintf("worker-%d", m), seq: uint64(10 + 2*m), name: "worker"},
				parentSpanID:          "seq-shared",
				guardSpanIDs:          []string{root.spanID},
				ancestorSpanIDs:       []string{"seq-shared", root.spanID},
				retiredAtSeq:          uint64(11 + 2*m),
				completedOrTask:       true,
				joinedAtRetire:        true,
				afterAgentDone:        true,
				startPlatformTime:     at(2 * m),
				retiredAtPlatformTime: at(2*m + 1),
			})
		}
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		if len(inv.retiredActivations) != 3 {
			t.Fatalf("after compaction: len=%d, want 3", len(inv.retiredActivations))
		}
		if hull := inv.retiredActivations[1]; !hull.compactedHull || hull.parentSpanID != "seq-shared" {
			t.Fatalf("shared-parent hull parentSpanID = %q, want %q", hull.parentSpanID, "seq-shared")
		}
	})
}

func TestUnderOpenRetiredAncestorSettlesAndPrunesWhenDescendantRetires(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }

	t.Run("guardless workflow loop prunes settled retired ancestors once open descendant retires", func(t *testing.T) {
		loop := &spanRecord{spanID: "loop-1", kind: spanKindAgent, name: "loop", seq: 1, propagatesChildError: true, isWorkflow: true}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, loop}}
		const steps = 12
		for m := range steps {
			wID := fmt.Sprintf("worker-%d", m)
			actions := &session.EventActions{StateDelta: map[string]any{"m": m}}
			worker := &spanRecord{
				spanID:                 wID,
				kind:                   spanKindAgent,
				name:                   "worker",
				path:                   "loop/worker",
				seq:                    uint64(2 + 2*m),
				parent:                 loop,
				parentSpanID:           loop.spanID,
				completed:              true,
				ownResponse:            true,
				hasAfterAgentCallbacks: true,
				startPlatformTime:      at(2 * m),
				popRetirement: &popRetirement{
					afterAgentActions:    eventActionMaps(&session.Event{Actions: *actions}),
					afterAgentActionsRef: actions,
				},
			}
			helper := &spanRecord{
				spanID:            fmt.Sprintf("helper-%d", m),
				kind:              spanKindAgent,
				name:              "helper",
				path:              "loop/worker/helper",
				seq:               uint64(3 + 2*m),
				parent:            worker,
				parentSpanID:      worker.spanID,
				completed:         true,
				ownResponse:       true,
				startPlatformTime: at(2*m + 1),
			}
			inv.agentStack = append(inv.agentStack, helper)
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(2*m+1), worker)
			if idx := inv.findRetiredActivationIndexLocked(wID); idx >= 0 {
				inv.retiredActivations[idx].markAfterAgentDone(at(2*m + 1))
			}
			inv.agentStack = inv.agentStack[:2]
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(2*m+2), helper)
		}
		if len(inv.retiredActivations) != 0 {
			t.Fatalf("guardless loop leaked %d retired entries after descendant retirement: %+v", len(inv.retiredActivations), inv.retiredActivations)
		}
	})

	t.Run("stale afterAgentPending hull settles once retired ancestor loses last open descendant", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		p := &spanRecord{
			spanID:                 "worker-parent",
			kind:                   spanKindAgent,
			name:                   "worker",
			path:                   "p0",
			seq:                    2,
			parent:                 root,
			parentSpanID:           root.spanID,
			completed:              true,
			ownResponse:            true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(1),
		}
		helper := &spanRecord{
			spanID:            "helper-open",
			kind:              spanKindAgent,
			name:              "helper",
			path:              "p0/helper",
			seq:               3,
			parent:            p,
			parentSpanID:      p.spanID,
			completed:         true,
			ownResponse:       true,
			startPlatformTime: at(2),
		}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root, helper}}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(3), p)
		if idx := inv.findRetiredActivationIndexLocked(p.spanID); idx >= 0 {
			inv.retiredActivations[idx].markAfterAgentDone(at(3))
		}
		for m := range maxPendingAfterAgent + 1 {
			actions := &session.EventActions{StateDelta: map[string]any{"m": m}}
			w := &spanRecord{
				spanID:                 fmt.Sprintf("worker-%d", m),
				kind:                   spanKindAgent,
				name:                   "worker",
				path:                   "p0",
				seq:                    uint64(10 + m),
				parent:                 root,
				parentSpanID:           root.spanID,
				completed:              true,
				ownResponse:            true,
				hasAfterAgentCallbacks: true,
				startPlatformTime:      at(10 + m),
				popRetirement: &popRetirement{
					afterAgentActions:    eventActionMaps(&session.Event{Actions: *actions}),
					afterAgentActionsRef: actions,
				},
			}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10+m), w)
			if idx := inv.findRetiredActivationIndexLocked(w.spanID); idx >= 0 {
				inv.retiredActivations[idx].markAfterAgentDone(at(10 + m))
			}
			inv.pruneSettledRetiredActivationsLocked("worker", "")
		}
		// Helper still open keeps p in st.underOpen, so folding must have marked the hull afterAgentPending.
		if !slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool {
			return ra.swallower.name == "worker" && ra.afterAgentPending
		}) {
			t.Fatal("expected worker hull to stay afterAgentPending while helper remained open under retired p")
		}

		// Retiring helper removes p from st.underOpen and must settle the hull's afterAgentPending.
		inv.agentStack = inv.agentStack[:2]
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(50), helper)
		for _, ra := range inv.retiredActivations {
			if ra.swallower.name == "worker" && (ra.afterAgentPending || ra.holdsAfterAgentEvent()) {
				t.Fatalf("worker entry remained pending after helper retired: %+v", ra)
			}
		}
		wLater := &spanRecord{
			spanID:            "worker-later",
			kind:              spanKindAgent,
			name:              "worker",
			path:              "p0",
			seq:               100,
			parent:            root,
			parentSpanID:      root.spanID,
			startPlatformTime: at(60),
		}
		if !inv.provenAfterAllRetiredLocked(wLater, "p0", at(61), true, true) {
			t.Fatal("provenAfterAllRetiredLocked = false after last open descendant retired, want true")
		}
	})
}

func TestSoleOpenAuthor_WildcardAndSiblingPathRetiredPeers(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	allowAll := func(*spanRecord) bool { return true }
	allowAllRetired := func(retiredActivation) bool { return true }

	t.Run("guardless peers on empty path stay pinned while path p0 peer open and fail closed when absent", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1, propagatesChildError: true, isWorkflow: true}
		rec := &spanRecord{
			spanID:            "w-open",
			kind:              spanKindAgent,
			name:              "worker",
			path:              "p0",
			seq:               2,
			parent:            root,
			parentSpanID:      root.spanID,
			startPlatformTime: at(0),
		}
		w1 := &spanRecord{
			spanID:            "w1",
			kind:              spanKindAgent,
			name:              "worker",
			path:              "",
			seq:               3,
			parent:            root,
			parentSpanID:      root.spanID,
			completed:         true,
			ownResponse:       true,
			concurrentPeer:    true,
			startPlatformTime: at(1),
			popRetirement:     &popRetirement{peerRetirement: true},
		}
		w2 := &spanRecord{
			spanID:            "w2",
			kind:              spanKindAgent,
			name:              "worker",
			path:              "",
			seq:               4,
			parent:            root,
			parentSpanID:      root.spanID,
			completed:         true,
			ownResponse:       true,
			concurrentPeer:    true,
			startPlatformTime: at(2),
			popRetirement:     &popRetirement{peerRetirement: true},
		}
		kWild := llmKey(identity{agentName: "worker", path: ""})
		inv := &invocationTrace{
			agentStack:       []*spanRecord{{kind: spanKindInvocation}, root, rec},
			retiredPeerCount: map[string]int{kWild: 1},
		}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(5), w1)
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(6), w2)

		// While rec (path p0) is open, wildcard peers w1/w2 stay pinned so w1's trailer (at(4)) fails closed
		// and rec's post-retirement event (at(7)) still proves authorship.
		if got := inv.soleOpenAuthorLocked([]int{2}, "", at(4), allowAll, allowAllRetired); got != -1 {
			t.Fatalf("soleOpenAuthorLocked(w1 trailer at(4), path=\"\") = %d, want -1", got)
		}
		if got := inv.soleOpenAuthorLocked([]int{2}, "p0", at(7), allowAll, allowAllRetired); got != 2 {
			t.Fatalf("soleOpenAuthorLocked(rec post-retirement at(7), path=\"p0\") = %d, want 2", got)
		}

		// If retired entries on empty path were already pruned while retiredPeerCount[kWild] > 0,
		// soleOpenAuthorLocked must fail closed for both "" and "p0".
		inv.retiredActivations = nil
		if got := inv.soleOpenAuthorLocked([]int{2}, "", at(4), allowAll, allowAllRetired); got != -1 {
			t.Fatalf("soleOpenAuthorLocked(pruned empty-path peer, path=\"\") = %d, want -1", got)
		}
		if got := inv.soleOpenAuthorLocked([]int{2}, "p0", at(4), allowAll, allowAllRetired); got != -1 {
			t.Fatalf("soleOpenAuthorLocked(pruned empty-path peer, path=\"p0\") = %d, want -1", got)
		}
	})

	t.Run("compacted peers on empty path fail closed against open p0 peer and pruned p1 fails closed for wildcard path", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		rec := &spanRecord{
			spanID:            "w-open",
			kind:              spanKindAgent,
			name:              "worker",
			path:              "p0",
			seq:               10,
			parent:            root,
			parentSpanID:      root.spanID,
			startPlatformTime: at(0),
		}
		kWild := llmKey(identity{agentName: "worker", path: ""})
		inv := &invocationTrace{
			agentStack:       []*spanRecord{{kind: spanKindInvocation}, root},
			retiredPeerCount: map[string]int{kWild: 4},
		}
		for m := range 4 {
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:             retiredSwallower{spanID: fmt.Sprintf("w-%d", m), seq: uint64(2 + m), name: "worker"},
				parentSpanID:          root.spanID,
				guardSpanIDs:          []string{root.spanID},
				retiredAtSeq:          uint64(20 + m),
				completedOrTask:       true,
				joinedAtRetire:        true,
				concurrentPeer:        true,
				peerRetirement:        true,
				startPlatformTime:     at(1 + m),
				retiredAtPlatformTime: at(3 + m),
			})
		}
		// Compact 4 guarded peers to 3 entries before rec starts (middle hull clears peerRetirement).
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		if len(inv.retiredActivations) != 3 {
			t.Fatalf("compacted len = %d, want 3", len(inv.retiredActivations))
		}
		inv.agentStack = append(inv.agentStack, rec)
		if got := inv.soleOpenAuthorLocked([]int{2}, "", at(10), allowAll, allowAllRetired); got != -1 {
			t.Fatalf("soleOpenAuthorLocked(compacted empty-path peers, path=\"\") = %d, want -1", got)
		}

		// Sibling path p1 pruned peer: fails closed for wildcard event path "", does not block exact path "p0".
		kP1 := llmKey(identity{agentName: "worker", path: "p1"})
		inv2 := &invocationTrace{
			agentStack:       []*spanRecord{{kind: spanKindInvocation}, root, rec},
			retiredPeerCount: map[string]int{kP1: 1},
		}
		if got := inv2.soleOpenAuthorLocked([]int{2}, "", at(4), allowAll, allowAllRetired); got != -1 {
			t.Fatalf("soleOpenAuthorLocked(pruned p1 peer, path=\"\") = %d, want -1", got)
		}
		if got := inv2.soleOpenAuthorLocked([]int{2}, "p0", at(4), allowAll, allowAllRetired); got != 2 {
			t.Fatalf("soleOpenAuthorLocked(pruned p1 peer, path=\"p0\") = %d, want 2", got)
		}
	})
}

func TestPruneSettledRetiredActivations_SettlesSiblingPathsBeforeFold(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }

	root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
	inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
	for m := range maxPendingAfterAgent + 1 {
		actions := &session.EventActions{StateDelta: map[string]any{"p1": m}}
		inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
			swallower:              retiredSwallower{spanID: fmt.Sprintf("p1-%d", m), seq: uint64(2 + m), name: "worker", path: "p1"},
			parentSpanID:           root.spanID,
			guardSpanIDs:           []string{root.spanID},
			retiredAtSeq:           uint64(10 + m),
			completedOrTask:        true,
			joinedAtRetire:         true,
			hasAfterAgentCallbacks: true,
			afterAgentDone:         true,
			afterAgentDoneAt:       at(20 + m),
			afterAgentActions:      eventActionMaps(&session.Event{Actions: *actions}),
			afterAgentActionsRef:   actions,
			startPlatformTime:      at(1 + m),
			retiredAtPlatformTime:  at(2 + m),
		})
	}
	actionsP0 := &session.EventActions{StateDelta: map[string]any{"p0": 0}}
	inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
		swallower:              retiredSwallower{spanID: "p0-0", seq: 50, name: "worker", path: "p0"},
		parentSpanID:           root.spanID,
		guardSpanIDs:           []string{root.spanID},
		retiredAtSeq:           51,
		completedOrTask:        true,
		joinedAtRetire:         true,
		hasAfterAgentCallbacks: true,
		afterAgentDone:         false,
		afterAgentActions:      eventActionMaps(&session.Event{Actions: *actionsP0}),
		afterAgentActionsRef:   actionsP0,
		startPlatformTime:      at(50),
		retiredAtPlatformTime:  at(51),
	})

	inv.pruneSettledRetiredActivationsLocked("worker", "")
	for _, ra := range inv.retiredActivations {
		if ra.afterAgentPending {
			t.Fatalf("sibling path p1 entries folded into afterAgentPending hull during p0 prune: %+v", ra)
		}
		if ra.swallower.path == "p1" && ra.holdsAfterAgentEvent() {
			t.Fatalf("settled p1 entry %s kept afterAgentActions across p0 prune", ra.swallower.spanID)
		}
	}
	if idx := inv.findRetiredActivationIndexLocked("p0-0"); idx < 0 || !inv.retiredActivations[idx].holdsAfterAgentEvent() {
		t.Fatalf("pending p0-0 entry should retain its after-agent action maps: idx=%d", idx)
	}
}

func TestSettleRetiredResponseCredit_ZeroAndModelCredits(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	k := llmKey(identity{agentName: "worker"})

	t.Run("zero-credit predecessor does not steal disjoint peer credit", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{k: 1},
		}
		w1 := &spanRecord{
			spanID:            "w1",
			kind:              spanKindAgent,
			name:              "worker",
			seq:               2,
			parent:            root,
			parentSpanID:      root.spanID,
			completed:         true,
			ownResponse:       true,
			startPlatformTime: at(0),
		}
		w2 := &spanRecord{
			spanID:            "w2",
			kind:              spanKindAgent,
			name:              "worker",
			seq:               3,
			parent:            root,
			parentSpanID:      root.spanID,
			completed:         true,
			startPlatformTime: at(10),
			popRetirement:     &popRetirement{afterAgentCredit: true, peerRetirement: true},
		}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(4), w1)
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(12), w2)

		// Extra buffered event from w1 (at(2)) must NOT decrement w2's credit.
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(2), true, true) {
			t.Fatal("zero-credit w1 run event settled a credit")
		}
		idx2 := inv.findRetiredActivationIndexLocked("w2")
		if idx2 < 0 || inv.unrespondedRetired[k] != 1 || !inv.retiredActivations[idx2].afterAgentCredit {
			t.Fatalf("zero-credit w1 stole w2 credit: unrespondedRetired=%d w2=%+v", inv.unrespondedRetired[k], inv.retiredActivations[idx2])
		}
	})

	t.Run("modelCredits activation settles only its own credits and preserves pinned status", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{k: 2},
		}
		w1 := &spanRecord{
			spanID:            "w1",
			kind:              spanKindAgent,
			name:              "worker",
			seq:               2,
			parent:            root,
			parentSpanID:      root.spanID,
			completed:         true,
			startPlatformTime: at(0),
			popRetirement:     &popRetirement{modelCredits: 1, peerRetirement: true},
		}
		w2 := &spanRecord{
			spanID:            "w2",
			kind:              spanKindAgent,
			name:              "worker",
			seq:               3,
			parent:            root,
			parentSpanID:      root.spanID,
			completed:         true,
			startPlatformTime: at(10),
			popRetirement:     &popRetirement{afterAgentCredit: true, peerRetirement: true},
		}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(4), w1)
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(12), w2)

		// While w1 holds modelCredits=1, compaction/folding must keep w1 uncompacted and pinned
		// even when 4+ peers share its compactKey (before and after w1 in retiredAtSeq).
		for m, rseq := range []uint64{1, 20, 21, 22} {
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:             retiredSwallower{spanID: fmt.Sprintf("same-ck-%d", m), seq: rseq, name: "worker"},
				parentSpanID:          root.spanID,
				guardSpanIDs:          []string{root.spanID},
				retiredAtSeq:          rseq,
				completedOrTask:       true,
				joinedAtRetire:        true,
				unrespondedAtPop:      true,
				startPlatformTime:     at(40 + 2*m),
				retiredAtPlatformTime: at(41 + 2*m),
			})
		}
		for m := range maxPendingAfterAgent + 2 {
			actions := &session.EventActions{StateDelta: map[string]any{"pad": m}}
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:              retiredSwallower{spanID: fmt.Sprintf("pad-%d", m), seq: uint64(30 + m), name: "worker"},
				parentSpanID:           root.spanID,
				guardSpanIDs:           []string{root.spanID},
				retiredAtSeq:           uint64(30 + m),
				completedOrTask:        true,
				joinedAtRetire:         true,
				hasAfterAgentCallbacks: true,
				afterAgentActions:      eventActionMaps(&session.Event{Actions: *actions}),
				afterAgentActionsRef:   actions,
				startPlatformTime:      at(30 + m),
				retiredAtPlatformTime:  at(31 + m),
			})
		}
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		idx1 := inv.findRetiredActivationIndexLocked("w1")
		if idx1 < 0 || inv.retiredActivations[idx1].modelCredits != 1 {
			t.Fatalf("modelCredits peer w1 lost during prune/fold: idx1=%d", idx1)
		}

		// First w1 event consumes w1's 1 model credit; second w1 event must NOT steal w2's credit.
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(1), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("first w1 model-credit event failed: unrespondedRetired=%d", inv.unrespondedRetired[k])
		}
		if inv.retiredActivations[idx1].modelCredits != 0 || !inv.retiredActivations[idx1].creditSettled {
			t.Fatalf("after consuming w1 model credit: w1=%+v", inv.retiredActivations[idx1])
		}
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(2), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("second w1 event stole w2 credit: unrespondedRetired=%d, want 1", inv.unrespondedRetired[k])
		}
		idx2 := inv.findRetiredActivationIndexLocked("w2")
		if idx2 < 0 || !inv.retiredActivations[idx2].afterAgentCredit {
			t.Fatalf("w2 lost afterAgentCredit: %+v", inv.retiredActivations)
		}
	})

	t.Run("cross-wave ambiguous drain clears modelCredits so wave 1 cannot steal wave 2", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{k: 2},
			retiredActivations: []retiredActivation{
				{swallower: retiredSwallower{spanID: "w1", seq: 1, name: "worker"}, parentSpanID: root.spanID, guardSpanIDs: []string{root.spanID}, retiredAtSeq: 2, completedOrTask: true, joinedAtRetire: true, modelCredits: 1, afterAgentDone: true, startPlatformTime: at(0), retiredAtPlatformTime: at(10)},
				{swallower: retiredSwallower{spanID: "w2", seq: 3, name: "worker"}, parentSpanID: root.spanID, guardSpanIDs: []string{root.spanID}, retiredAtSeq: 4, completedOrTask: true, joinedAtRetire: true, afterAgentCredit: true, afterAgentDone: true, startPlatformTime: at(2), retiredAtPlatformTime: at(12)},
			},
		}
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(5), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("first ambiguous event: unrespondedRetired=%d, want 1", inv.unrespondedRetired[k])
		}
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(6), true, true) || inv.unrespondedRetired[k] != 0 {
			t.Fatalf("second ambiguous event: unrespondedRetired=%d, want 0", inv.unrespondedRetired[k])
		}
		if w1 := inv.retiredActivations[0]; w1.modelCredits != 0 || !w1.creditSettled {
			t.Fatalf("after ambiguous drain to 0, w1 retained modelCredits=%d creditSettled=%v", w1.modelCredits, w1.creditSettled)
		}

		// Wave 2 adds w3 credit; late w1 event at at(1) must NOT steal it.
		inv.unrespondedRetired[k] = 1
		inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
			swallower:             retiredSwallower{spanID: "w3", seq: 5, name: "worker"},
			parentSpanID:          root.spanID,
			guardSpanIDs:          []string{root.spanID},
			retiredAtSeq:          6,
			completedOrTask:       true,
			joinedAtRetire:        true,
			afterAgentCredit:      true,
			startPlatformTime:     at(20),
			retiredAtPlatformTime: at(30),
		})
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(1), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("late wave-1 model-credit w1 event stole wave-2 w3 credit: unrespondedRetired=%d", inv.unrespondedRetired[k])
		}
	})

	t.Run("zero-credit or settled compactedHull does not steal later peer credit", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{},
		}
		// Compact 5 zero-credit wave-1 activations covering [2, 5] into a compactedHull.
		for i := range 5 {
			inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
				swallower:             retiredSwallower{spanID: fmt.Sprintf("z-%d", i), seq: uint64(2 + i), name: "worker"},
				parentSpanID:          root.spanID,
				guardSpanIDs:          []string{root.spanID},
				ancestorSpanIDs:       []string{root.spanID},
				retiredAtSeq:          uint64(10 + i),
				completedOrTask:       true,
				joinedAtRetire:        true,
				startPlatformTime:     at(2),
				retiredAtPlatformTime: at(5),
			})
		}
		inv.pruneSettledRetiredActivationsLocked("worker", "")
		if !slices.ContainsFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.compactedHull }) {
			t.Fatalf("expected compactedHull from wave-1 zero-credit activations: %+v", inv.retiredActivations)
		}

		// Wave-2 w4 retires with afterAgentCredit=true.
		inv.unrespondedRetired[k] = 1
		inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
			swallower:             retiredSwallower{spanID: "w4", seq: 20, name: "worker"},
			parentSpanID:          root.spanID,
			guardSpanIDs:          []string{root.spanID},
			ancestorSpanIDs:       []string{root.spanID},
			retiredAtSeq:          21,
			completedOrTask:       true,
			joinedAtRetire:        true,
			afterAgentCredit:      true,
			startPlatformTime:     at(20),
			retiredAtPlatformTime: at(25),
		})

		// Late buffered wave-1 event at at(3) matches the compactedHull, not w4; must NOT steal w4's credit.
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(3), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("zero-credit compactedHull allowed event to steal w4 credit: unrespondedRetired=%d", inv.unrespondedRetired[k])
		}
		idx4 := inv.findRetiredActivationIndexLocked("w4")
		if idx4 < 0 || !inv.retiredActivations[idx4].afterAgentCredit || inv.retiredActivations[idx4].creditSettled {
			t.Fatalf("w4 credit mutated by wave-1 hull event: %+v", inv.retiredActivations[idx4])
		}
	})

	t.Run("overlapping zero-credit or settled predecessors do not steal later peer credit, while overlapping credit-holders do", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{k: 1},
			retiredActivations: []retiredActivation{
				{swallower: retiredSwallower{spanID: "w1", seq: 2, name: "worker"}, parentSpanID: root.spanID, guardSpanIDs: []string{root.spanID}, retiredAtSeq: 3, completedOrTask: true, joinedAtRetire: true, creditSettled: true, startPlatformTime: at(0), retiredAtPlatformTime: at(10)},
				{swallower: retiredSwallower{spanID: "w2", seq: 4, name: "worker"}, parentSpanID: root.spanID, guardSpanIDs: []string{root.spanID}, retiredAtSeq: 5, completedOrTask: true, joinedAtRetire: true, startPlatformTime: at(2), retiredAtPlatformTime: at(12)},
				{swallower: retiredSwallower{spanID: "w3", seq: 6, name: "worker"}, parentSpanID: root.spanID, guardSpanIDs: []string{root.spanID}, retiredAtSeq: 7, completedOrTask: true, joinedAtRetire: true, afterAgentCredit: true, startPlatformTime: at(20), retiredAtPlatformTime: at(30)},
			},
		}
		// Event at at(5) falls in overlap of settled w1 and zero-credit w2, disjoint from w3.
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(5), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("overlapping wave-1 predecessors stole wave-2 w3 credit: unrespondedRetired=%d", inv.unrespondedRetired[k])
		}
		if !inv.retiredActivations[2].afterAgentCredit || inv.retiredActivations[2].creditSettled {
			t.Fatalf("w3 lost credit to overlapping wave-1 event: %+v", inv.retiredActivations[2])
		}

		// Two overlapping credit-holders (c1, c2) MUST settle one credit on ambiguous overlap event.
		invOverlap := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{k: 2},
			retiredActivations: []retiredActivation{
				{swallower: retiredSwallower{spanID: "c1", seq: 2, name: "worker"}, parentSpanID: root.spanID, guardSpanIDs: []string{root.spanID}, retiredAtSeq: 3, completedOrTask: true, joinedAtRetire: true, afterAgentCredit: true, startPlatformTime: at(20), retiredAtPlatformTime: at(26)},
				{swallower: retiredSwallower{spanID: "c2", seq: 4, name: "worker"}, parentSpanID: root.spanID, guardSpanIDs: []string{root.spanID}, retiredAtSeq: 5, completedOrTask: true, joinedAtRetire: true, afterAgentCredit: true, startPlatformTime: at(24), retiredAtPlatformTime: at(30)},
			},
		}
		if !invOverlap.settleRetiredResponseCreditLocked(k, "worker", "", "", at(25), true, true) || invOverlap.unrespondedRetired[k] != 1 {
			t.Fatalf("overlapping credit-holders failed to settle ambiguous event: unrespondedRetired=%d, want 1", invOverlap.unrespondedRetired[k])
		}
	})

	t.Run("wave-1 zero-credit plus credit-holder overlap settles wave-1 credit without stealing wave-2", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{k: 2},
		}
		z := &spanRecord{
			spanID:            "z",
			kind:              spanKindAgent,
			name:              "worker",
			seq:               2,
			parent:            root,
			parentSpanID:      root.spanID,
			completed:         true,
			ownResponse:       true,
			startPlatformTime: at(0),
		}
		w1 := &spanRecord{
			spanID:            "w1",
			kind:              spanKindAgent,
			name:              "worker",
			seq:               3,
			parent:            root,
			parentSpanID:      root.spanID,
			completed:         true,
			startPlatformTime: at(2),
			popRetirement:     &popRetirement{afterAgentCredit: true, peerRetirement: true},
		}
		w2 := &spanRecord{
			spanID:            "w2",
			kind:              spanKindAgent,
			name:              "worker",
			seq:               4,
			parent:            root,
			parentSpanID:      root.spanID,
			completed:         true,
			startPlatformTime: at(20),
			popRetirement:     &popRetirement{afterAgentCredit: true, peerRetirement: true},
		}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10), z, w1)
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(30), w2)

		// Event at at(5) overlaps zero-credit z and credit-holder w1: must consume w1's credit cleanly.
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(5), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("at(5) failed to settle wave-1 credit: unrespondedRetired=%d, want 1", inv.unrespondedRetired[k])
		}
		idx1 := inv.findRetiredActivationIndexLocked("w1")
		if idx1 < 0 || inv.retiredActivations[idx1].afterAgentCredit || !inv.retiredActivations[idx1].creditSettled {
			t.Fatalf("w1 credit should be settled after at(5): %+v", inv.retiredActivations[idx1])
		}

		// Subsequent w1 after-agent event and another wave-1 event at at(6) MUST preserve wave-2 w2's credit!
		inv.consumeAfterAgentCreditLocked(idx1)
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(6), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("wave-1 stole wave-2 w2 credit: unrespondedRetired=%d, want 1", inv.unrespondedRetired[k])
		}
		idx2 := inv.findRetiredActivationIndexLocked("w2")
		if idx2 < 0 || !inv.retiredActivations[idx2].afterAgentCredit {
			t.Fatalf("w2 lost afterAgentCredit: %+v", inv.retiredActivations[idx2])
		}
	})

	t.Run("wave-1 overlapping credit-holders merge into bounded credit group so wave-1 cannot drain wave-2", func(t *testing.T) {
		for _, viaConsume := range []bool{false, true} {
			t.Run(fmt.Sprintf("viaConsume=%v", viaConsume), func(t *testing.T) {
				root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
				inv := &invocationTrace{
					agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
					unrespondedRetired: map[string]int{k: 3},
				}
				w1 := &spanRecord{
					spanID:            "w1",
					kind:              spanKindAgent,
					name:              "worker",
					seq:               2,
					parent:            root,
					parentSpanID:      root.spanID,
					completed:         true,
					startPlatformTime: at(0),
					popRetirement:     &popRetirement{afterAgentCredit: true, peerRetirement: true},
				}
				w2 := &spanRecord{
					spanID:            "w2",
					kind:              spanKindAgent,
					name:              "worker",
					seq:               3,
					parent:            root,
					parentSpanID:      root.spanID,
					completed:         true,
					startPlatformTime: at(2),
					popRetirement:     &popRetirement{afterAgentCredit: true, peerRetirement: true},
				}
				w3 := &spanRecord{
					spanID:            "w3",
					kind:              spanKindAgent,
					name:              "worker",
					seq:               4,
					parent:            root,
					parentSpanID:      root.spanID,
					completed:         true,
					startPlatformTime: at(20),
					popRetirement:     &popRetirement{afterAgentCredit: true, peerRetirement: true},
				}
				inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10), w1)
				inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(12), w2)
				inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(30), w3)

				// First wave-1 overlap event at at(4): settles 1 of the 2 wave-1 group credits (3 -> 2).
				if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(4), true, true) || inv.unrespondedRetired[k] != 2 {
					t.Fatalf("first overlap event: unrespondedRetired=%d, want 2", inv.unrespondedRetired[k])
				}

				idx1 := inv.findRetiredActivationIndexLocked("w1")
				idx2 := inv.findRetiredActivationIndexLocked("w2")
				if viaConsume {
					inv.consumeAfterAgentCreditLocked(idx1)
				} else {
					if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(5), true, true) {
						t.Fatal("second overlap event failed to settle wave-1 credit")
					}
				}
				if inv.unrespondedRetired[k] != 1 {
					t.Fatalf("after consuming wave-1's 2 credits: unrespondedRetired=%d, want 1", inv.unrespondedRetired[k])
				}

				// Wave-1's 2 credits are now exhausted; neither w1, w2, nor extra wave-1 event at(6) may touch w3!
				inv.consumeAfterAgentCreditLocked(idx1)
				inv.consumeAfterAgentCreditLocked(idx2)
				if inv.settleRetiredResponseCreditLocked(k, "worker", "", "", at(6), true, true) || inv.unrespondedRetired[k] != 1 {
					t.Fatalf("exhausted wave-1 group stole wave-2 w3 credit: unrespondedRetired=%d, want 1", inv.unrespondedRetired[k])
				}
				idx3 := inv.findRetiredActivationIndexLocked("w3")
				if idx3 < 0 || !inv.retiredActivations[idx3].afterAgentCredit {
					t.Fatalf("w3 lost afterAgentCredit: %+v", inv.retiredActivations[idx3])
				}
			})
		}
	})

	t.Run("stale credit-bearing entries via recordRetiredActivationEntryLocked prune creditGroupRem on delete, compact, and fold", func(t *testing.T) {
		// 1. Deletion on guardless prune with unrespondedRetired == 0.
		invDel := &invocationTrace{}
		for i := range 3 {
			invDel.recordRetiredActivationEntryLocked(retiredActivation{
				swallower:        retiredSwallower{spanID: fmt.Sprintf("del-%d", i), seq: uint64(1 + i), name: "worker"},
				retiredAtSeq:     uint64(10 + i),
				afterAgentCredit: true,
			})
		}
		if len(invDel.creditGroupRem) != 3 {
			t.Fatalf("expected 3 allocated credit groups, got %d", len(invDel.creditGroupRem))
		}
		invDel.pruneSettledRetiredActivationsLocked("worker", "")
		if len(invDel.retiredActivations) != 0 || len(invDel.creditGroupRem) != 0 {
			t.Fatalf("after guardless prune: retired=%d creditGroupRem=%v, want 0, empty", len(invDel.retiredActivations), invDel.creditGroupRem)
		}

		// 2. Bucket compaction under open custom guard with unrespondedRetired == 0.
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		invCompact := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
		for i := range 5 {
			invCompact.recordRetiredActivationEntryLocked(retiredActivation{
				swallower:        retiredSwallower{spanID: fmt.Sprintf("cmp-%d", i), seq: uint64(1 + i), name: "worker"},
				parentSpanID:     root.spanID,
				guardSpanIDs:     []string{root.spanID},
				retiredAtSeq:     uint64(10 + i),
				afterAgentCredit: true,
			})
		}
		invCompact.pruneSettledRetiredActivationsLocked("worker", "")
		if len(invCompact.retiredActivations) != 3 || len(invCompact.creditGroupRem) != 2 {
			t.Fatalf("after 5->3 compaction: retired=%d creditGroupRem=%d (want 3, 2 surviving endpoints)", len(invCompact.retiredActivations), len(invCompact.creditGroupRem))
		}
		for gid := range invCompact.creditGroupRem {
			found := slices.ContainsFunc(invCompact.retiredActivations, func(ra retiredActivation) bool {
				return !ra.compactedHull && ra.creditGroupID == gid
			})
			if !found {
				t.Fatalf("orphaned creditGroupRem gid=%d after compaction: %+v", gid, invCompact.retiredActivations)
			}
		}

		// 3. Pending hull folding under open custom guard with unrespondedRetired == 0.
		invFold := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
		for i := range maxPendingAfterAgent + 2 {
			m := map[string]any{}
			invFold.recordRetiredActivationEntryLocked(retiredActivation{
				swallower:              retiredSwallower{spanID: fmt.Sprintf("fld-%d", i), seq: uint64(1 + i), name: "worker"},
				parentSpanID:           root.spanID,
				guardSpanIDs:           []string{root.spanID},
				retiredAtSeq:           uint64(10 + i),
				hasAfterAgentCallbacks: true,
				afterAgentCredit:       true,
				afterAgentActions:      actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef:   m,
			})
		}
		invFold.foldPendingAfterAgentLocked("worker", "", invFold.afterAgentStateLocked())
		if len(invFold.retiredActivations) != maxPendingAfterAgent {
			t.Fatalf("after pending fold: len=%d, want %d", len(invFold.retiredActivations), maxPendingAfterAgent)
		}
		if len(invFold.creditGroupRem) != maxPendingAfterAgent-1 {
			t.Fatalf("after pending fold: creditGroupRem=%d, want %d (orphaned folded members cleaned)", len(invFold.creditGroupRem), maxPendingAfterAgent-1)
		}
		for gid := range invFold.creditGroupRem {
			found := slices.ContainsFunc(invFold.retiredActivations, func(ra retiredActivation) bool {
				return !ra.compactedHull && ra.creditGroupID == gid
			})
			if !found {
				t.Fatalf("orphaned creditGroupRem gid=%d after pending fold", gid)
			}
		}
	})
}

func TestExcludedByEventTimestamp_BoundedPendingHull(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}

	// Build a pending hull whose members all have afterAgentDoneAt bounded by at(18)
	// (kept pending only because a peer credit on another path or open peer kept holdsAfterAgentEvent).
	inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
	for n := range maxPendingAfterAgent + 1 {
		m := map[string]any{}
		inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
			swallower:              retiredSwallower{spanID: fmt.Sprintf("w-%d", n), seq: uint64(2 + n), name: "worker"},
			parentSpanID:           root.spanID,
			guardSpanIDs:           []string{root.spanID},
			retiredAtSeq:           uint64(10 + n),
			completedOrTask:        true,
			joinedAtRetire:         true,
			hasAfterAgentCallbacks: true,
			afterAgentDone:         true,
			afterAgentDoneAt:       at(16 + n%3),
			afterAgentActions:      actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
			afterAgentActionsRef:   m,
			startPlatformTime:      at(1 + n),
			retiredAtPlatformTime:  at(2 + n),
		})
	}
	inv.foldPendingAfterAgentLocked("worker", "", inv.afterAgentStateLocked())
	hullIdx := slices.IndexFunc(inv.retiredActivations, func(ra retiredActivation) bool { return ra.afterAgentPending })
	if hullIdx < 0 {
		t.Fatal("expected folded afterAgentPending hull")
	}
	hull := inv.retiredActivations[hullIdx]
	if hull.afterAgentDoneAt.IsZero() {
		t.Fatalf("expected bounded hull.afterAgentDoneAt, got zero: %+v", hull)
	}

	later := &spanRecord{
		spanID:            "worker-later",
		parentSpanID:      root.spanID,
		parent:            root,
		kind:              spanKindAgent,
		name:              "worker",
		seq:               100,
		startPlatformTime: at(20),
	}
	// Pre-bound event created before hull.afterAgentDoneAt (at(17) <= at(18)) must NOT be excluded.
	if hull.excludedByEventTimestamp(later.spanID, later.seq, 0, false, later.startPlatformTime, at(17), true) {
		t.Fatal("bounded afterAgentPending hull falsely excluded pre-afterAgentDoneAt event")
	}
	// Post-bound event created after hull.afterAgentDoneAt and later.startPlatformTime (at(21)) MUST be excluded.
	if !hull.excludedByEventTimestamp(later.spanID, later.seq, 0, false, later.startPlatformTime, at(21), true) {
		t.Fatal("bounded afterAgentPending hull failed to exclude post-afterAgentDoneAt event")
	}
	// Post-parentRetiredAtPlatformTime bounded hull also excludes later events.
	parentBoundedHull := hull
	parentBoundedHull.afterAgentDoneAt = time.Time{}
	parentBoundedHull.parentRetiredAtPlatformTime = at(19)
	if !parentBoundedHull.excludedByEventTimestamp(later.spanID, later.seq, 0, false, later.startPlatformTime, at(21), true) {
		t.Fatal("parentRetiredAtPlatformTime-bounded afterAgentPending hull failed to exclude later event")
	}
	// Unbounded hull (both afterAgentDoneAt and parentRetiredAtPlatformTime zero) must NOT exclude later events.
	unboundedHull := hull
	unboundedHull.afterAgentDone = false
	unboundedHull.afterAgentDoneAt = time.Time{}
	unboundedHull.parentRetiredAtPlatformTime = time.Time{}
	if unboundedHull.excludedByEventTimestamp(later.spanID, later.seq, 0, false, later.startPlatformTime, at(21), true) {
		t.Fatal("unbounded afterAgentPending hull falsely excluded later event")
	}
}

func TestRetiredActivation_ActionMapPinAcrossLiveReplacement(t *testing.T) {
	var (
		capturedMaps actionMaps
		capturedKeep any
		laterMaps    actionMaps
		emitted      []*session.Event
	)
	root, err := agent.New(agent.Config{
		Name: "worker",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	firstPlugin, err := baseplugin.New(baseplugin.Config{
		Name: "capture-then-replace",
		AfterAgentCallback: func(ctx agent.Context) (*genai.Content, error) {
			capturedMaps, capturedKeep = callbackAfterAgentActions(ctx)
			live := ctx.WithDelta(nil).Actions()
			live.StateDelta = map[string]any{"replaced": true}
			live.ArtifactDelta = map[string]int64{"replaced": 1}
			laterMaps = callbackAfterAgentActionMaps(ctx)
			return textContent("model", "replaced"), nil
		},
	})
	if err != nil {
		t.Fatalf("baseplugin.New: %v", err)
	}
	for ev, runErr := range newTestRunner(t, root, firstPlugin).Run(t.Context(), "u1", "s-pin-replace", textContent("user", "go"), agent.RunConfig{}) {
		if runErr != nil {
			t.Fatalf("runner.Run error = %v", runErr)
		}
		if ev != nil {
			emitted = append(emitted, ev)
		}
	}
	if capturedMaps == (actionMaps{}) || capturedKeep == nil {
		t.Fatalf("callbackAfterAgentActions returned zero: maps=%+v keep=%v", capturedMaps, capturedKeep)
	}
	if len(emitted) != 1 {
		t.Fatalf("got %d events, want 1", len(emitted))
	}
	if laterMaps == (actionMaps{}) || laterMaps.shares(capturedMaps) {
		t.Fatalf("expected replaced maps distinct from captured: captured=%+v later=%+v", capturedMaps, laterMaps)
	}
	if evMaps := eventActionMaps(emitted[0]); evMaps != laterMaps || evMaps.shares(capturedMaps) {
		t.Fatalf("emitted event maps %+v did not track replacement %+v cleanly", evMaps, laterMaps)
	}

	tm := newTraceManager()
	inv := tm.invocation(t.Context(), identity{invocationID: "inv-pin-replace", agentName: "root"})
	inv.mu.Lock()
	defer inv.mu.Unlock()
	inv.recordRetiredActivationEntryLocked(retiredActivation{
		swallower:              retiredSwallower{spanID: "w1", seq: 1, name: "worker"},
		retiredAtSeq:           2,
		completedOrTask:        true,
		joinedAtRetire:         true,
		hasAfterAgentCallbacks: true,
		afterAgentActions:      capturedMaps,
		afterAgentActionsRef:   capturedKeep,
	})
	capturedKeep = nil

	for i := range 2000 {
		if i%200 == 0 {
			runtime.GC()
		}
		sd := map[string]any{"i": i}
		ad := map[string]int64{"i": int64(i)}
		fresh := actionMaps{
			stateDelta:    mapPointer(reflect.ValueOf(sd)),
			artifactDelta: mapPointer(reflect.ValueOf(ad)),
		}
		if fresh.shares(inv.retiredActivations[0].afterAgentActions) {
			t.Fatalf("fresh maps %+v collided with pinned retiredActivation.afterAgentActions %+v at iter %d", fresh, inv.retiredActivations[0].afterAgentActions, i)
		}
	}

	inv.retiredActivations[0].dropAfterAgentEvent()
	if inv.retiredActivations[0].afterAgentActions != (actionMaps{}) || inv.retiredActivations[0].afterAgentActionsRef != nil {
		t.Fatalf("dropAfterAgentEvent did not clear pinned maps/ref: %+v", inv.retiredActivations[0])
	}
}

func TestRetiredActivationLedger_PerInvocationCap4000Retirements(t *testing.T) {
	t0 := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }

	for _, tc := range []struct {
		name string
		mkID func(i int) (agentName, branch, path string)
	}{
		{
			name: "4000 distinct node paths",
			mkID: func(i int) (string, string, string) {
				return "worker", "", fmt.Sprintf("root@1/step@%d", i)
			},
		},
		{
			name: "4000 distinct branches",
			mkID: func(i int) (string, string, string) {
				return "worker", fmt.Sprintf("par.item_%d", i), "root@1/par@1"
			},
		},
		{
			name: "4000 distinct agent names",
			mkID: func(i int) (string, string, string) {
				return fmt.Sprintf("worker_%d", i), "", fmt.Sprintf("root@1/w@%d", i)
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1, nonLLM: true}
			inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, root}}
			openSet := inv.openSpanSetLocked()

			for i := range 4000 {
				aName, br, pth := tc.mkID(i)
				child := &spanRecord{
					spanID:                 fmt.Sprintf("w-%d", i),
					kind:                   spanKindAgent,
					name:                   aName,
					branch:                 br,
					path:                   pth,
					seq:                    uint64(2 + 2*i),
					parent:                 root,
					parentSpanID:           root.spanID,
					completed:              true,
					hasAfterAgentCallbacks: i == 0,
					startPlatformTime:      at(2 * i),
				}
				inv.recordRetiredActivationsAtLocked(openSet, at(2*i+1), child)
				if got := len(inv.retiredActivations); got > maxRetiredActivationsPerInvocation {
					t.Fatalf("after retirement %d: len(retiredActivations) = %d, exceeds cap %d", i, got, maxRetiredActivationsPerInvocation)
				}
			}

			if got := len(inv.retiredActivations); got > maxRetiredActivationsPerInvocation {
				t.Fatalf("after 4000 retirements: len(retiredActivations) = %d, want <= %d", got, maxRetiredActivationsPerInvocation)
			}
			hull := inv.retiredActivations[0]
			if !hull.compactedHull {
				t.Fatalf("retiredActivations[0] = %+v, want compactedHull=true", hull)
			}
			firstName, firstBranch, firstPath := tc.mkID(0)
			if !hull.swallower.matches(firstName, firstBranch, firstPath) {
				t.Fatalf("cap hull %+v failed to match folded first retirement (%q, %q, %q)", hull.swallower, firstName, firstBranch, firstPath)
			}
			if !hull.startPlatformTime.Equal(at(0)) || !hull.hasAfterAgentCallbacks {
				t.Fatalf("cap hull lost earliest start time or hasAfterAgentCallbacks: %+v", hull)
			}
			if !hull.canOwnEventAt(at(2), true, false) {
				t.Fatal("cap hull falsely rejected post-retirement event when folded member had AfterAgentCallbacks")
			}
			last := inv.retiredActivations[len(inv.retiredActivations)-1]
			if last.compactedHull || last.swallower.spanID != "w-3999" {
				t.Fatalf("newest retirement was not preserved intact: %+v", last)
			}

			// Completing the enclosing custom root settles and drops all 128 entries.
			root.completed = true
			inv.agentStack = []*spanRecord{{kind: spanKindInvocation}}
			inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(9000), root)
			if len(inv.retiredActivations) != 0 {
				t.Fatalf("after custom root completed: len(retiredActivations) = %d, want 0", len(inv.retiredActivations))
			}
		})
	}
}

func TestRetiredActivationLedger_ReplayAfterPartialCreditConsumptionDoesNotRestoreCredit(t *testing.T) {
	t0 := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }

	t.Run("modelCredits partial consumption", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		k := llmKey(identity{agentName: "worker", path: "p1"})
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{k: 2},
		}
		entry2 := retiredActivation{
			swallower:             retiredSwallower{spanID: "w1", seq: 2, name: "worker", path: "p1"},
			parentSpanID:          root.spanID,
			guardSpanIDs:          []string{root.spanID},
			ancestorSpanIDs:       []string{root.spanID},
			retiredAtSeq:          3,
			completedOrTask:       true,
			joinedAtRetire:        true,
			modelCredits:          2,
			startPlatformTime:     at(0),
			retiredAtPlatformTime: at(10),
		}
		inv.recordRetiredActivationEntryLocked(entry2)

		idx := inv.findRetiredActivationIndexLocked("w1")
		if idx < 0 {
			t.Fatal("w1 missing after initial recordRetiredActivationEntryLocked")
		}
		gid := inv.retiredActivations[idx].creditGroupID
		if gotRem, gotGroup := inv.retiredActivations[idx].remainingCredits(), inv.creditGroupRem[gid]; gotRem != 2 || gotGroup != 2 {
			t.Fatalf("initial remaining=%d groupRem=%d, want 2 and 2", gotRem, gotGroup)
		}

		// Consume 1 of the 2 granted model credits.
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(5), true, true) {
			t.Fatal("settleRetiredResponseCreditLocked = false, want true")
		}
		if gotRem, gotGroup := inv.retiredActivations[idx].remainingCredits(), inv.creditGroupRem[gid]; gotRem != 1 || gotGroup != 1 {
			t.Fatalf("after consuming 1 credit: remaining=%d groupRem=%d, want 1 and 1", gotRem, gotGroup)
		}

		// Replaying the same 2-credit retirement must NOT restore the consumed credit.
		inv.recordRetiredActivationEntryLocked(entry2)
		if gotRem, gotGroup := inv.retiredActivations[idx].remainingCredits(), inv.creditGroupRem[gid]; gotRem != 1 || gotGroup != 1 {
			t.Fatalf("after replaying 2-credit retirement: remaining=%d groupRem=%d, want 1 and 1", gotRem, gotGroup)
		}

		// A higher-water grant (3 total) adds only the +1 newly granted delta.
		inv.unrespondedRetired[k]++
		entry3 := entry2
		entry3.modelCredits = 3
		inv.recordRetiredActivationEntryLocked(entry3)
		if gotRem, gotGroup := inv.retiredActivations[idx].remainingCredits(), inv.creditGroupRem[gid]; gotRem != 2 || gotGroup != 2 {
			t.Fatalf("after replaying 3-credit high-water retirement: remaining=%d groupRem=%d, want 2 and 2", gotRem, gotGroup)
		}
	})

	t.Run("afterAgentCredit plus modelCredits partial consumption", func(t *testing.T) {
		root := &spanRecord{spanID: "root-1", kind: spanKindAgent, name: "root", seq: 1}
		k := llmKey(identity{agentName: "worker", path: "p1"})
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
			unrespondedRetired: map[string]int{k: 2},
		}
		entry := retiredActivation{
			swallower:             retiredSwallower{spanID: "w1", seq: 2, name: "worker", path: "p1"},
			parentSpanID:          root.spanID,
			guardSpanIDs:          []string{root.spanID},
			ancestorSpanIDs:       []string{root.spanID},
			retiredAtSeq:          3,
			completedOrTask:       true,
			joinedAtRetire:        true,
			afterAgentCredit:      true,
			modelCredits:          1,
			startPlatformTime:     at(0),
			retiredAtPlatformTime: at(10),
		}
		inv.recordRetiredActivationEntryLocked(entry)

		idx := inv.findRetiredActivationIndexLocked("w1")
		gid := inv.retiredActivations[idx].creditGroupID
		inv.consumeAfterAgentCreditLocked(idx)
		if inv.retiredActivations[idx].afterAgentCredit || inv.retiredActivations[idx].remainingCredits() != 1 || inv.creditGroupRem[gid] != 1 {
			t.Fatalf("after consuming afterAgentCredit: entry=%+v groupRem=%d, want 1 remaining", inv.retiredActivations[idx], inv.creditGroupRem[gid])
		}

		// Replaying entry must not restore the spent afterAgentCredit.
		inv.recordRetiredActivationEntryLocked(entry)
		if inv.retiredActivations[idx].afterAgentCredit || inv.retiredActivations[idx].remainingCredits() != 1 || inv.creditGroupRem[gid] != 1 {
			t.Fatalf("replay restored spent afterAgentCredit: entry=%+v groupRem=%d, want 1 remaining", inv.retiredActivations[idx], inv.creditGroupRem[gid])
		}
	})
}

func TestRetiredActivationLedger_DetachedChildGoroutineNotStampedByCustomParentReturn(t *testing.T) {
	t0 := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }

	t.Run("unjoined detached child goroutine is not stamped when custom parent completes", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top", seq: 1, goroutineID: 10, nonLLM: true}
		mid := &spanRecord{spanID: "mid-1", kind: spanKindAgent, name: "mid", seq: 2, goroutineID: 10, parent: top, parentSpanID: top.spanID, nonLLM: true}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top, mid}}

		// detachedEarly retires at at(10) while still running on goroutine 99.
		detachedEarly := &spanRecord{
			spanID:                 "det-early",
			kind:                   spanKindAgent,
			name:                   "worker",
			seq:                    3,
			goroutineID:            99,
			parent:                 mid,
			parentSpanID:           mid.spanID,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(1),
		}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10), detachedEarly)

		// detachedPopped popped its own AfterAgentCallback on goroutine 97
		// (setting both completed via markSettledSubtreeLocked and runDone
		// via sampleSettledProbesLocked) while a later plugin's
		// AfterAgentCallback is still running on goroutine 97.
		detachedPopped := &spanRecord{
			spanID:                 "det-popped",
			kind:                   spanKindAgent,
			name:                   "worker",
			seq:                    4,
			goroutineID:            97,
			parent:                 mid,
			parentSpanID:           mid.spanID,
			completed:              true,
			runDone:                true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(2),
		}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(12), detachedPopped)

		// detachedBatch retires in the same batch as mid completing at at(20),
		// while still running unjoined on goroutine 98 under open top.
		detachedBatch := &spanRecord{
			spanID:                 "det-batch",
			kind:                   spanKindAgent,
			name:                   "worker",
			seq:                    5,
			goroutineID:            98,
			parent:                 mid,
			parentSpanID:           mid.spanID,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(3),
		}
		mid.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(20), detachedBatch, mid)

		for _, spanID := range []string{"det-early", "det-popped", "det-batch"} {
			idx := inv.findRetiredActivationIndexLocked(spanID)
			if idx < 0 {
				t.Fatalf("%s missing under open top", spanID)
			}
			ra := inv.retiredActivations[idx]
			if !ra.parentRetiredAtPlatformTime.IsZero() || ra.afterAgentDone || !ra.afterAgentDoneAt.IsZero() {
				t.Fatalf("%s prematurely stamped on custom parent return: parentRetiredAt=%v afterAgentDone=%v afterAgentDoneAt=%v",
					spanID, ra.parentRetiredAtPlatformTime, ra.afterAgentDone, ra.afterAgentDoneAt)
			}
			if !ra.canOwnEventAt(at(50), true, false) {
				t.Fatalf("%s falsely rejected late event at at(50) while detached goroutine could still emit", spanID)
			}
		}
	})

	t.Run("same-goroutine child, runDone child, and ParallelAgent child stamp parent return bound", func(t *testing.T) {
		top := &spanRecord{spanID: "top-1", kind: spanKindAgent, name: "top", seq: 1, goroutineID: 10, nonLLM: true}
		customMid := &spanRecord{spanID: "mid-1", kind: spanKindAgent, name: "mid", seq: 2, goroutineID: 10, parent: top, parentSpanID: top.spanID, nonLLM: true}
		parMid := &spanRecord{
			spanID:               "par-1",
			kind:                 spanKindAgent,
			name:                 "par",
			seq:                  3,
			goroutineID:          10,
			parent:               top,
			parentSpanID:         top.spanID,
			nonLLM:               true,
			isParallel:           true,
			propagatesChildError: true,
			children:             []string{"worker"},
		}
		inv := &invocationTrace{agentStack: []*spanRecord{{kind: spanKindInvocation}, top, customMid, parMid}}

		sameG := &spanRecord{
			spanID:                 "same-g",
			kind:                   spanKindAgent,
			name:                   "worker",
			seq:                    4,
			goroutineID:            10,
			parent:                 customMid,
			parentSpanID:           customMid.spanID,
			completed:              true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(1),
		}
		runDoneChild := &spanRecord{
			spanID:                 "rundone-g",
			kind:                   spanKindAgent,
			name:                   "worker",
			seq:                    5,
			goroutineID:            99,
			parent:                 customMid,
			parentSpanID:           customMid.spanID,
			runDone:                true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(2),
		}
		parChild := &spanRecord{
			spanID:                 "par-child",
			kind:                   spanKindAgent,
			name:                   "worker",
			branch:                 parallelBranch(parMid, "worker"),
			seq:                    6,
			goroutineID:            88,
			origin:                 goroutineOrigin{creator: errgroupCreator, parent: 10},
			parent:                 parMid,
			parentSpanID:           parMid.spanID,
			completed:              true,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(3),
		}
		completedNoCB := &spanRecord{
			spanID:            "completed-nocb-g",
			kind:              spanKindAgent,
			name:              "worker",
			seq:               7,
			goroutineID:       96,
			parent:            customMid,
			parentSpanID:      customMid.spanID,
			completed:         true,
			startPlatformTime: at(4),
		}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(10), sameG, runDoneChild, parChild, completedNoCB)

		customMid.completed = true
		parMid.completed = true
		inv.agentStack = []*spanRecord{{kind: spanKindInvocation}, top}
		inv.recordRetiredActivationsAtLocked(inv.openSpanSetLocked(), at(20), customMid, parMid)

		for _, spanID := range []string{"same-g", "rundone-g", "par-child", "completed-nocb-g"} {
			idx := inv.findRetiredActivationIndexLocked(spanID)
			if idx < 0 {
				t.Fatalf("%s missing under open top", spanID)
			}
			ra := inv.retiredActivations[idx]
			if !ra.parentRetiredAtPlatformTime.Equal(at(20)) || !ra.afterAgentDone || !ra.afterAgentDoneAt.Equal(at(20)) {
				t.Fatalf("%s failed to stamp parent return bound at at(20): parentRetiredAt=%v afterAgentDone=%v afterAgentDoneAt=%v",
					spanID, ra.parentRetiredAtPlatformTime, ra.afterAgentDone, ra.afterAgentDoneAt)
			}
		}
	})
}

func TestRetiredActivation_PreservesHasUnjoinedTaskAcrossCompaction(t *testing.T) {
	t0 := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }

	t.Run("foldPendingAfterAgentLocked preserves unjoined task when folded with returned non-task", func(t *testing.T) {
		openParent := &spanRecord{spanID: "parent-1", kind: spanKindAgent, name: "parent", seq: 1}
		inv := &invocationTrace{
			agentStack: []*spanRecord{
				{kind: spanKindInvocation},
				openParent,
				{spanID: "open-later", kind: spanKindAgent, name: "worker", seq: 100, startPlatformTime: at(50)},
			},
		}
		for i := range maxPendingAfterAgent + 2 {
			isUnjoinedTask := i == 0
			m := map[string]any{"k": i}
			inv.recordRetiredActivationEntryLocked(retiredActivation{
				swallower:              retiredSwallower{spanID: fmt.Sprintf("r-%d", i), seq: uint64(2 + i), name: "worker"},
				parentSpanID:           openParent.spanID,
				guardSpanIDs:           []string{openParent.spanID},
				ancestorSpanIDs:        []string{openParent.spanID},
				retiredAtSeq:           uint64(10 + i),
				completedOrTask:        isUnjoinedTask,
				joinedAtRetire:         !isUnjoinedTask,
				hasAfterAgentCallbacks: true,
				afterAgentActions:      actionMaps{stateDelta: mapPointer(reflect.ValueOf(m))},
				afterAgentActionsRef:   m,
				startPlatformTime:      at(1 + i),
				retiredAtPlatformTime:  at(20 + i),
			})
		}

		inv.foldPendingAfterAgentLocked("worker", "", inv.afterAgentStateLocked())

		var hull *retiredActivation
		for i := range inv.retiredActivations {
			if inv.retiredActivations[i].compactedHull {
				hull = &inv.retiredActivations[i]
				break
			}
		}
		if hull == nil {
			t.Fatal("expected foldPendingAfterAgentLocked to produce a compacted hull")
		}
		if !hull.hasUnjoinedTask || !hull.unjoinedTaskRetirement() {
			t.Fatalf("hull lost hasUnjoinedTask / unjoinedTaskRetirement: %+v", *hull)
		}
		if !hull.canOwnEventAt(at(100), true, false) {
			t.Fatal("hull.canOwnEventAt(after retirement) = false, want true because an unjoined task was folded into the hull")
		}
	})

	t.Run("foldRetiredCapHull and enforceRetiredActivationCapLocked preserve unjoined task across mixed entries", func(t *testing.T) {
		openParent := &spanRecord{spanID: "parent-cap", kind: spanKindAgent, name: "parent", seq: 1}
		inv := &invocationTrace{
			agentStack: []*spanRecord{{kind: spanKindInvocation}, openParent},
		}
		openSet := inv.openSpanSetLocked()
		members := []retiredActivation{
			{
				swallower:             retiredSwallower{spanID: "task-unjoined", seq: 2, name: "task-agent"},
				parentSpanID:          openParent.spanID,
				guardSpanIDs:          []string{openParent.spanID},
				ancestorSpanIDs:       []string{openParent.spanID},
				retiredAtSeq:          3,
				completedOrTask:       true,
				joinedAtRetire:        false,
				startPlatformTime:     at(1),
				retiredAtPlatformTime: at(10),
			},
			{
				swallower:             retiredSwallower{spanID: "non-task-joined", seq: 4, name: "other-agent"},
				parentSpanID:          openParent.spanID,
				guardSpanIDs:          []string{openParent.spanID},
				ancestorSpanIDs:       []string{openParent.spanID},
				retiredAtSeq:          5,
				completedOrTask:       false,
				joinedAtRetire:        true,
				startPlatformTime:     at(2),
				retiredAtPlatformTime: at(20),
			},
		}
		hull := foldRetiredCapHull(members, openSet)
		if !hull.hasUnjoinedTask || !hull.unjoinedTaskRetirement() {
			t.Fatalf("foldRetiredCapHull lost unjoined task: %+v", hull)
		}
		if !hull.canOwnEventAt(at(100), true, false) {
			t.Fatal("foldRetiredCapHull hull rejected post-retirement event from folded unjoined task")
		}

		// Also verify enforceRetiredActivationCapLocked with 128+ entries and soleOpenAuthorLocked guard.
		inv.recordRetiredActivationEntryLocked(members[0])
		for i := range maxRetiredActivationsPerInvocation + 10 {
			inv.recordRetiredActivationEntryLocked(retiredActivation{
				swallower:             retiredSwallower{spanID: fmt.Sprintf("other-%d", i), seq: uint64(10 + i), name: fmt.Sprintf("other-%d", i)},
				parentSpanID:          openParent.spanID,
				guardSpanIDs:          []string{openParent.spanID},
				ancestorSpanIDs:       []string{openParent.spanID},
				retiredAtSeq:          uint64(200 + i),
				completedOrTask:       false,
				joinedAtRetire:        true,
				startPlatformTime:     at(5 + i),
				retiredAtPlatformTime: at(30 + i),
			})
		}
		inv.enforceRetiredActivationCapLocked()
		var foundUnjoined bool
		for _, ra := range inv.retiredActivations {
			if ra.unjoinedTaskRetirement() && ra.swallower.matches("task-agent", "", "") {
				foundUnjoined = true
				if !ra.canOwnEventAt(at(500), true, false) {
					t.Fatalf("cap hull rejected post-retirement event of folded unjoined task: %+v", ra)
				}
			}
		}
		if !foundUnjoined {
			t.Fatal("enforceRetiredActivationCapLocked dropped unjoined task coverage")
		}
	})
}

func TestRetiredActivationLedger_CompactedReplayDoesNotStealLaterWorkerCredit(t *testing.T) {
	t0 := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }

	root := &spanRecord{spanID: "root-custom", kind: spanKindAgent, name: "root", seq: 1}
	inv := &invocationTrace{
		agentStack: []*spanRecord{{kind: spanKindInvocation}, root},
	}
	w1 := &spanRecord{
		spanID:            "w1",
		kind:              spanKindAgent,
		name:              "worker",
		path:              "p1",
		seq:               2,
		parent:            root,
		parentSpanID:      root.spanID,
		completed:         true,
		startPlatformTime: at(0),
		popRetirement: &popRetirement{
			afterAgentCredit: true,
			modelCredits:     1,
		},
	}
	k := llmKey(identity{agentName: "worker", path: "p1"})
	inv.unrespondedRetired = map[string]int{k: 2}
	openSet := inv.openSpanSetLocked()
	inv.recordRetiredActivationsAtLocked(openSet, at(10), w1)

	// Consume w1's credits cleanly.
	idxW1 := inv.findRetiredActivationIndexLocked("w1")
	if idxW1 < 0 {
		t.Fatal("missing w1 entry")
	}
	inv.consumeAfterAgentCreditLocked(idxW1)
	if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(5), true, true) {
		t.Fatal("failed to settle w1 model credit")
	}
	if inv.unrespondedRetired[k] != 0 || len(inv.creditGroupRem) != 0 {
		t.Fatalf("expected 0 outstanding after consuming w1: unresp=%v groups=%v", inv.unrespondedRetired, inv.creditGroupRem)
	}

	// Force w1 into a compacted hull (> maxRetiredActivationsPerInvocation other entries).
	for i := range maxRetiredActivationsPerInvocation + 10 {
		inv.recordRetiredActivationEntryLocked(retiredActivation{
			swallower:             retiredSwallower{spanID: fmt.Sprintf("fill-%d", i), seq: uint64(10 + i), name: fmt.Sprintf("fill-%d", i)},
			parentSpanID:          root.spanID,
			guardSpanIDs:          []string{root.spanID},
			ancestorSpanIDs:       []string{root.spanID},
			retiredAtSeq:          uint64(200 + i),
			completedOrTask:       true,
			joinedAtRetire:        true,
			startPlatformTime:     at(1 + i),
			retiredAtPlatformTime: at(20 + i),
		})
	}
	inv.enforceRetiredActivationCapLocked()

	// Later worker w2 retires on the same identity with 1 unresponded model credit.
	inv.unrespondedRetired[k] = 1
	w2 := &spanRecord{
		spanID:            "w2",
		kind:              spanKindAgent,
		name:              "worker",
		path:              "p1",
		seq:               1000,
		parent:            root,
		parentSpanID:      root.spanID,
		completed:         true,
		startPlatformTime: at(500),
		popRetirement: &popRetirement{
			modelCredits: 1,
		},
	}
	inv.recordRetiredActivationsAtLocked(openSet, at(510), w2)
	groupsBeforeReplay := len(inv.creditGroupRem)
	if groupsBeforeReplay != 1 {
		t.Fatalf("len(creditGroupRem) = %d, want 1 for w2", groupsBeforeReplay)
	}

	// Replay w1's retirement with its original grant after compaction.
	w1.popRetirement = &popRetirement{
		afterAgentCredit: true,
		modelCredits:     1,
	}
	inv.recordRetiredActivationsAtLocked(openSet, at(10), w1)

	if len(inv.creditGroupRem) != 1 || inv.unrespondedRetired[k] != 1 {
		t.Fatalf("w1 replay created extra credit group or altered w2 count: groups=%v unresp=%v", inv.creditGroupRem, inv.unrespondedRetired)
	}
	idxW2 := inv.findRetiredActivationIndexLocked("w2")
	if idxW2 < 0 || inv.retiredActivations[idxW2].remainingCredits() != 1 || inv.creditGroupRem[inv.retiredActivations[idxW2].creditGroupID] != 1 {
		t.Fatalf("w1 replay corrupted w2 credit: entry=%+v groups=%v", inv.retiredActivations[idxW2], inv.creditGroupRem)
	}
	// Attempting to settle an old event timed before w2 cannot steal w2's credit
	// because w1 granted 0 new credits on replay.
	if inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(5), true, false) {
		t.Fatal("stale w1-era event settled a credit against w2 after compacted replay")
	}
	if inv.unrespondedRetired[k] != 1 {
		t.Fatalf("w2 unrespondedRetired = %d, want 1", inv.unrespondedRetired[k])
	}

	// Replaying w1 with a strictly higher grant (modelCredits=2, +1 new above watermark 1) adds only +1.
	inv.unrespondedRetired[k]++
	w1.popRetirement = &popRetirement{
		afterAgentCredit: true,
		modelCredits:     2,
	}
	inv.recordRetiredActivationsAtLocked(openSet, at(10), w1)
	idxW1Replay := inv.findRetiredActivationIndexLocked("w1")
	if idxW1Replay < 0 || inv.retiredActivations[idxW1Replay].remainingCredits() != 1 || w1.recordedModelCredits != 2 {
		t.Fatalf("increased w1 grant did not add exact +1 delta: idx=%d entry=%+v watermark=%d",
			idxW1Replay, inv.retiredActivations[idxW1Replay], w1.recordedModelCredits)
	}
}

func TestRetiredActivationLedger_IndexAndShrinkCapacity(t *testing.T) {
	t0 := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }

	root := &spanRecord{spanID: "root-shrink", kind: spanKindAgent, name: "root", seq: 1}
	inv := &invocationTrace{
		agentStack:         []*spanRecord{{kind: spanKindInvocation}, root},
		unrespondedRetired: make(map[string]int),
	}
	// Populate > 500 unrespondedRetired entries plus exact and wildcard hulls.
	for i := range 600 {
		inv.unrespondedRetired[llmKey(identity{agentName: fmt.Sprintf("a-%d", i), branch: fmt.Sprintf("b-%d", i), path: fmt.Sprintf("p-%d", i)})] = 1
	}
	inv.unrespondedRetired[llmKey(identity{agentName: "wild-name", branch: "b-0", path: ""})] = 1

	idx := inv.newUnrespondedRetiredIndexLocked()
	probes := []retiredSwallower{
		{name: "a-0", branch: "b-0", path: "p-0"},
		{name: "a-0", branch: "b-0", path: "wrong"},
		{name: "a-0", branch: "b-0", path: ""},
		{name: "wild-name", branch: "b-0", path: "any-path"},
		{name: "a-12", anyBranch: true, path: "p-12"},
		{name: "a-12", anyBranch: true, path: "other"},
		{anyName: true, branch: "b-99", path: "p-99"},
		{anyName: true, branch: "b-99", path: "other"},
		{anyName: true, anyBranch: true, path: "p-555"},
		{anyName: true, anyBranch: true, path: "none"},
	}
	for _, s := range probes {
		if got, want := idx.matches(s), inv.hasUnrespondedRetiredSwallowerLocked(s, false); got != want {
			t.Errorf("unrespondedRetiredIndex.matches(%+v) = %v, want %v", s, got, want)
		}
	}

	// Oversize retiredActivations capacity > 1024, then prune down and verify shrinkRetiredActivationsLocked.
	inv.retiredActivations = make([]retiredActivation, 0, 2048)
	for i := range 1000 {
		inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
			swallower:             retiredSwallower{spanID: fmt.Sprintf("s-%d", i), seq: uint64(i + 2), name: "gone"},
			parentSpanID:          "closed-parent",
			startPlatformTime:     at(i),
			retiredAtPlatformTime: at(i + 1),
		})
	}
	for range 10 {
		inv.retiredActivations = append(inv.retiredActivations, retiredActivation{
			swallower:    retiredSwallower{spanID: "keep-x", seq: 2000, name: "keep"},
			guardSpanIDs: []string{root.spanID},
		})
	}
	if cap(inv.retiredActivations) < 1024 {
		t.Fatalf("setup cap = %d, want >= 1024", cap(inv.retiredActivations))
	}
	inv.pruneSettledRetiredActivationsLocked("gone", "")
	if len(inv.retiredActivations) != 10 {
		t.Fatalf("len(retiredActivations) = %d, want 10", len(inv.retiredActivations))
	}
	if gotCap := cap(inv.retiredActivations); gotCap > maxRetiredActivationsPerInvocation {
		t.Fatalf("cap(retiredActivations) after prune = %d, want <= %d", gotCap, maxRetiredActivationsPerInvocation)
	}

	inv.retiredActivations = inv.retiredActivations[:0]
	inv.shrinkRetiredActivationsLocked()
	if inv.retiredActivations != nil {
		t.Fatalf("expected nil slice after emptying and shrinking, got cap=%d", cap(inv.retiredActivations))
	}
}

func TestRetiredActivationLedger_PartialCapFoldSharedGroupAfterAgent(t *testing.T) {
	t0 := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	k := llmKey(identity{agentName: "worker", path: "p1"})

	buildPartialFoldedAfterAgent := func(t *testing.T) (*invocationTrace, int, int) {
		t.Helper()
		guard := &spanRecord{spanID: "guard-open", kind: spanKindAgent, name: "custom", seq: 1}
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, guard},
			unrespondedRetired: make(map[string]int),
		}
		openSet := map[string]bool{guard.spanID: true}
		// Index 0: one pinned settled filler (tier1).
		inv.recordRetiredActivationsAtLocked(openSet, at(0), &spanRecord{
			spanID:       "filler-pinned",
			kind:         spanKindAgent,
			name:         "other",
			path:         "p-other",
			seq:          2,
			parentSpanID: guard.spanID,
			parent:       guard,
		})
		// Indices 1, 2: overlapping A0 and A1 on k (tier3), merged into one shared credit group (rem=1).
		a0 := &spanRecord{
			spanID:            "a0",
			kind:              spanKindAgent,
			name:              "worker",
			path:              "p1",
			seq:               3,
			parentSpanID:      guard.spanID,
			parent:            guard,
			startPlatformTime: at(10),
			popRetirement: &popRetirement{
				afterAgentCredit: true,
			},
		}
		a1 := &spanRecord{
			spanID:            "a1",
			kind:              spanKindAgent,
			name:              "worker",
			path:              "p1",
			seq:               4,
			parentSpanID:      guard.spanID,
			parent:            guard,
			startPlatformTime: at(20),
			popRetirement: &popRetirement{
				afterAgentCredit: true,
			},
		}
		inv.unrespondedRetired[k] = 2
		inv.recordRetiredActivationsAtLocked(openSet, at(30), a0)
		inv.recordRetiredActivationsAtLocked(openSet, at(40), a1)
		// Overlapping ambiguous run event merges A0 and A1 into one shared credit group (rem=1).
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(25), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("ambiguous wave settle failed: unresponded=%d, want 1", inv.unrespondedRetired[k])
		}
		idxA0 := inv.findRetiredActivationIndexLocked("a0")
		idxA1 := inv.findRetiredActivationIndexLocked("a1")
		if idxA0 < 0 || idxA1 < 0 {
			t.Fatalf("missing a0/a1 before cap fold: idxA0=%d idxA1=%d", idxA0, idxA1)
		}
		if gid0, gid1 := inv.retiredActivations[idxA0].creditGroupID, inv.retiredActivations[idxA1].creditGroupID; gid0 == 0 || gid0 != gid1 {
			t.Fatalf("expected shared non-zero group for a0/a1, got %d vs %d", gid0, gid1)
		}
		// Fill ledger exactly to maxRetiredActivationsPerInvocation (128) with pinned tier3 pads.
		for i := range maxRetiredActivationsPerInvocation - 3 {
			inv.unrespondedRetired[llmKey(identity{
				agentName: fmt.Sprintf("pad-agent-%d", i),
				path:      fmt.Sprintf("p-pad-%d", i),
			})] = 1
			inv.recordRetiredActivationsAtLocked(openSet, at(41), &spanRecord{
				spanID:       fmt.Sprintf("pad-%d", i),
				kind:         spanKindAgent,
				name:         fmt.Sprintf("pad-agent-%d", i),
				path:         fmt.Sprintf("p-pad-%d", i),
				seq:          uint64(10 + i),
				parentSpanID: guard.spanID,
				parent:       guard,
				popRetirement: &popRetirement{
					afterAgentCredit: true,
				},
			})
		}
		// One extra retired entry hits len=129 (need=2): folds [filler-pinned, a0] into hull, keeping a1 uncompacted.
		inv.unrespondedRetired[llmKey(identity{agentName: "pad-trigger", path: "p-trigger"})] = 1
		inv.recordRetiredActivationsAtLocked(openSet, at(45), &spanRecord{
			spanID:       "trigger-cap",
			kind:         spanKindAgent,
			name:         "pad-trigger",
			path:         "p-trigger",
			seq:          1002,
			parentSpanID: guard.spanID,
			parent:       guard,
			popRetirement: &popRetirement{
				afterAgentCredit: true,
			},
		})
		if inv.findRetiredActivationIndexLocked("a0") != -1 {
			t.Fatalf("expected a0 to be folded into cap hull")
		}
		idxA1 = inv.findRetiredActivationIndexLocked("a1")
		if idxA1 < 0 {
			t.Fatalf("expected a1 to survive cap fold uncompacted")
		}
		hullIdx := -1
		for idx, ra := range inv.retiredActivations {
			if ra.compactedHull && len(ra.foldedCredits) > 0 {
				hullIdx = idx
				break
			}
		}
		if hullIdx < 0 {
			t.Fatalf("expected compacted hull to retain foldedCredits")
		}
		return inv, hullIdx, idxA1
	}

	t.Run("surviving_peer_consumes_after_agent_credit", func(t *testing.T) {
		inv, _, idxA1 := buildPartialFoldedAfterAgent(t)
		if !inv.consumeAfterAgentCreditLocked(idxA1) {
			t.Fatal("consumeAfterAgentCreditLocked(a1) = false, want true")
		}
		if inv.unrespondedRetired[k] != 0 {
			t.Fatalf("unrespondedRetired[k] = %d, want 0 (leaked after partial fold)", inv.unrespondedRetired[k])
		}
	})

	t.Run("surviving_peer_uniquely_matched_after_hull_retired", func(t *testing.T) {
		inv, _, idxA1 := buildPartialFoldedAfterAgent(t)
		// Created after A0 retired (at(30)) and within A1's window [at(20), at(40)].
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(35), true, true) {
			t.Fatal("settleRetiredResponseCreditLocked(a1) = false, want true")
		}
		if inv.unrespondedRetired[k] != 0 {
			t.Fatalf("unrespondedRetired[k] = %d, want 0", inv.unrespondedRetired[k])
		}
		if !inv.retiredActivations[idxA1].creditSettled {
			t.Fatal("expected surviving a1 to be marked creditSettled")
		}
	})

	t.Run("hull_buffered_event_settles_without_stranding_or_double_spending", func(t *testing.T) {
		inv, _, idxA1 := buildPartialFoldedAfterAgent(t)
		// Event at at(15) can only belong to folded A0, not A1 (started at(20)).
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(15), true, true) {
			t.Fatal("settleRetiredResponseCreditLocked(hull a0) = false, want true")
		}
		if inv.unrespondedRetired[k] != 0 {
			t.Fatalf("unrespondedRetired[k] = %d, want 0", inv.unrespondedRetired[k])
		}
		if !inv.retiredActivations[idxA1].creditSettled {
			t.Fatal("expected surviving a1 to be settled when shared hull group exhausted")
		}
		if inv.consumeAfterAgentCreditLocked(idxA1) {
			t.Fatal("consumeAfterAgentCreditLocked(a1) = true after group exhaustion, want false")
		}
	})
}

func TestRetiredActivationLedger_PartialCapFoldSharedGroupModelCredits(t *testing.T) {
	t0 := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	k := llmKey(identity{agentName: "worker", path: "p1"})

	guard := &spanRecord{spanID: "guard-open", kind: spanKindAgent, name: "custom", seq: 1}
	inv := &invocationTrace{
		agentStack:         []*spanRecord{{kind: spanKindInvocation}, guard},
		unrespondedRetired: make(map[string]int),
	}
	openSet := map[string]bool{guard.spanID: true}
	inv.recordRetiredActivationsAtLocked(openSet, at(0), &spanRecord{
		spanID:       "filler-pinned",
		kind:         spanKindAgent,
		name:         "other",
		path:         "p-other",
		seq:          2,
		parentSpanID: guard.spanID,
		parent:       guard,
	})
	a0 := &spanRecord{
		spanID:            "a0",
		kind:              spanKindAgent,
		name:              "worker",
		path:              "p1",
		seq:               3,
		parentSpanID:      guard.spanID,
		parent:            guard,
		startPlatformTime: at(10),
		popRetirement:     &popRetirement{modelCredits: 1},
	}
	a1 := &spanRecord{
		spanID:            "a1",
		kind:              spanKindAgent,
		name:              "worker",
		path:              "p1",
		seq:               4,
		parentSpanID:      guard.spanID,
		parent:            guard,
		startPlatformTime: at(20),
		popRetirement:     &popRetirement{modelCredits: 1},
	}
	inv.unrespondedRetired[k] = 2
	inv.recordRetiredActivationsAtLocked(openSet, at(30), a0)
	inv.recordRetiredActivationsAtLocked(openSet, at(40), a1)
	// Ambiguous wave merges {A0, A1} into gid (creditGroupRem[gid] = 1, unrespondedRetired[k] = 1).
	if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(25), true, true) || inv.unrespondedRetired[k] != 1 {
		t.Fatalf("ambiguous wave failed: unresponded=%d, want 1", inv.unrespondedRetired[k])
	}
	// Later peer A2 retires on k after hull window with its own model credit (unrespondedRetired[k] = 2).
	a2 := &spanRecord{
		spanID:            "a2",
		kind:              spanKindAgent,
		name:              "worker",
		path:              "p1",
		seq:               5,
		parentSpanID:      guard.spanID,
		parent:            guard,
		startPlatformTime: at(50),
		popRetirement:     &popRetirement{modelCredits: 1},
	}
	inv.unrespondedRetired[k] = 2
	inv.recordRetiredActivationsAtLocked(openSet, at(60), a2)
	for i := range maxRetiredActivationsPerInvocation - 4 {
		inv.unrespondedRetired[llmKey(identity{
			agentName: fmt.Sprintf("pad-agent-%d", i),
			path:      fmt.Sprintf("p-pad-%d", i),
		})] = 1
		inv.recordRetiredActivationsAtLocked(openSet, at(61), &spanRecord{
			spanID:        fmt.Sprintf("pad-%d", i),
			kind:          spanKindAgent,
			name:          fmt.Sprintf("pad-agent-%d", i),
			path:          fmt.Sprintf("p-pad-%d", i),
			seq:           uint64(10 + i),
			parentSpanID:  guard.spanID,
			parent:        guard,
			popRetirement: &popRetirement{modelCredits: 1},
		})
	}
	// Trigger cap-128 folding: folds [filler-pinned, a0] while a1 and a2 survive uncompacted.
	inv.unrespondedRetired[llmKey(identity{agentName: "pad-trigger", path: "p-trigger"})] = 1
	inv.recordRetiredActivationsAtLocked(openSet, at(65), &spanRecord{
		spanID:        "trigger-cap",
		kind:          spanKindAgent,
		name:          "pad-trigger",
		path:          "p-trigger",
		seq:           1003,
		parentSpanID:  guard.spanID,
		parent:        guard,
		popRetirement: &popRetirement{modelCredits: 1},
	})
	if inv.findRetiredActivationIndexLocked("a0") != -1 {
		t.Fatal("expected a0 to be folded into cap hull")
	}
	idxA1 := inv.findRetiredActivationIndexLocked("a1")
	idxA2 := inv.findRetiredActivationIndexLocked("a2")
	if idxA1 < 0 || idxA2 < 0 {
		t.Fatalf("expected both a1 and a2 to survive cap fold: idxA1=%d idxA2=%d", idxA1, idxA2)
	}

	// Buffered event from A0 (at(15) matches only hull) settles {A0, A1}'s sole remaining credit.
	if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(15), true, true) {
		t.Fatal("settleRetiredResponseCreditLocked(hull a0) = false, want true")
	}
	if inv.unrespondedRetired[k] != 1 {
		t.Fatalf("after A0 event unrespondedRetired[k] = %d, want 1", inv.unrespondedRetired[k])
	}
	// A secondary event from A1 (at(35) uniquely matches A1) must NOT steal A2's credit.
	if inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(35), true, true) {
		t.Fatal("secondary event from A1 stole peer A2's credit")
	}
	if inv.unrespondedRetired[k] != 1 {
		t.Fatalf("after A1 secondary event unrespondedRetired[k] = %d, want 1", inv.unrespondedRetired[k])
	}
	// A2's own response still settles cleanly.
	if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(55), true, true) || inv.unrespondedRetired[k] != 0 {
		t.Fatalf("A2 response failed to settle remaining credit: unresponded=%d", inv.unrespondedRetired[k])
	}
}

func TestRetiredActivationLedger_IncrementalCreditGrantAfterSettled(t *testing.T) {
	t0 := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	k := llmKey(identity{agentName: "worker", path: "p1"})

	guard := &spanRecord{spanID: "guard-open", kind: spanKindAgent, name: "custom", seq: 1}
	inv := &invocationTrace{
		agentStack:         []*spanRecord{{kind: spanKindInvocation}, guard},
		unrespondedRetired: make(map[string]int),
	}
	openSet := map[string]bool{guard.spanID: true}
	w := &spanRecord{
		spanID:            "w-inc",
		kind:              spanKindAgent,
		name:              "worker",
		path:              "p1",
		seq:               2,
		parentSpanID:      guard.spanID,
		parent:            guard,
		startPlatformTime: at(1),
		popRetirement: &popRetirement{
			modelCredits: 1,
		},
	}
	inv.unrespondedRetired[k] = 1
	inv.recordRetiredActivationsAtLocked(openSet, at(2), w)
	// Settle initial modelCredits = 1 while entry remains uncompacted (guarded by open parent).
	if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(1), true, true) || inv.unrespondedRetired[k] != 0 {
		t.Fatalf("initial settle failed: unresponded=%d", inv.unrespondedRetired[k])
	}
	idx := inv.findRetiredActivationIndexLocked("w-inc")
	if idx < 0 || !inv.retiredActivations[idx].creditSettled {
		t.Fatalf("expected uncompacted settled entry, got idx=%d entry=%+v", idx, inv.retiredActivations)
	}

	// Re-record same uncompacted span with increased modelCredits = 2 (+1 incremental delta).
	inv.unrespondedRetired[k] = 1
	w.popRetirement = &popRetirement{modelCredits: 2}
	inv.recordRetiredActivationsAtLocked(openSet, at(3), w)
	idx = inv.findRetiredActivationIndexLocked("w-inc")
	if idx < 0 {
		t.Fatal("missing uncompacted w-inc after incremental model grant")
	}
	ra := inv.retiredActivations[idx]
	if ra.creditSettled || ra.modelCredits != 1 || ra.creditGroupID == 0 || inv.creditGroupRem[ra.creditGroupID] != 1 {
		t.Fatalf("incremental model grant failed on settled entry: ra=%+v rem=%d", ra, inv.creditGroupRem[ra.creditGroupID])
	}
	if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(2), true, true) || inv.unrespondedRetired[k] != 0 {
		t.Fatalf("incremental model credit failed to settle: unresponded=%d", inv.unrespondedRetired[k])
	}

	// Re-record again once settled with newly added afterAgentCredit = true (+1 incremental delta).
	inv.unrespondedRetired[k] = 1
	w.popRetirement = &popRetirement{afterAgentCredit: true, modelCredits: 2}
	inv.recordRetiredActivationsAtLocked(openSet, at(4), w)
	idx = inv.findRetiredActivationIndexLocked("w-inc")
	ra = inv.retiredActivations[idx]
	if ra.creditSettled || !ra.afterAgentCredit || ra.modelCredits != 0 || ra.creditGroupID == 0 || inv.creditGroupRem[ra.creditGroupID] != 1 {
		t.Fatalf("incremental afterAgentCredit grant failed on settled entry: ra=%+v rem=%d", ra, inv.creditGroupRem[ra.creditGroupID])
	}
	if !inv.consumeAfterAgentCreditLocked(idx) || inv.unrespondedRetired[k] != 0 {
		t.Fatalf("incremental afterAgentCredit failed to settle: unresponded=%d", inv.unrespondedRetired[k])
	}
}

func TestRetiredActivationLedger_CapFoldPostRetirementAfterAgentCredit(t *testing.T) {
	t0 := time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC)
	at := func(ms int) time.Time { return t0.Add(time.Duration(ms) * time.Millisecond) }
	k := llmKey(identity{agentName: "worker", path: "p1"})

	buildCapFolded := func(t *testing.T, withLaterPeer bool) *invocationTrace {
		t.Helper()
		guard := &spanRecord{spanID: "guard-open", kind: spanKindAgent, name: "custom", seq: 1}
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, guard},
			unrespondedRetired: make(map[string]int),
		}
		openSet := map[string]bool{guard.spanID: true}
		// Index 0: pinned settled filler (tier1) at at(0).
		inv.recordRetiredActivationsAtLocked(openSet, at(0), &spanRecord{
			spanID:       "filler-pinned",
			kind:         spanKindAgent,
			name:         "other",
			path:         "p-other",
			seq:          2,
			parentSpanID: guard.spanID,
			parent:       guard,
		})
		// Index 1: A0 with afterAgentCallbacks and afterAgentCredit retiring at at(30).
		inv.unrespondedRetired[k] = 1
		inv.recordRetiredActivationsAtLocked(openSet, at(30), &spanRecord{
			spanID:                 "a0",
			kind:                   spanKindAgent,
			name:                   "worker",
			path:                   "p1",
			seq:                    3,
			parentSpanID:           guard.spanID,
			parent:                 guard,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(10),
			popRetirement: &popRetirement{
				afterAgentCredit: true,
			},
		})
		numPads := maxRetiredActivationsPerInvocation - 2
		if withLaterPeer {
			// Index 2: later peer A2 starting at at(50), well after A0 retired at at(30).
			inv.unrespondedRetired[k] = 2
			inv.recordRetiredActivationsAtLocked(openSet, at(60), &spanRecord{
				spanID:            "a2",
				kind:              spanKindAgent,
				name:              "worker",
				path:              "p1",
				seq:               4,
				parentSpanID:      guard.spanID,
				parent:            guard,
				startPlatformTime: at(50),
				popRetirement:     &popRetirement{modelCredits: 1},
			})
			numPads = maxRetiredActivationsPerInvocation - 3
		}
		for i := range numPads {
			inv.unrespondedRetired[llmKey(identity{
				agentName: fmt.Sprintf("pad-agent-%d", i),
				path:      fmt.Sprintf("p-pad-%d", i),
			})] = 1
			inv.recordRetiredActivationsAtLocked(openSet, at(41), &spanRecord{
				spanID:       fmt.Sprintf("pad-%d", i),
				kind:         spanKindAgent,
				name:         fmt.Sprintf("pad-agent-%d", i),
				path:         fmt.Sprintf("p-pad-%d", i),
				seq:          uint64(10 + i),
				parentSpanID: guard.spanID,
				parent:       guard,
				popRetirement: &popRetirement{
					afterAgentCredit: true,
				},
			})
		}
		// Trigger cap-128 folding: folds [filler-pinned, a0] into hull while keeping remaining entries uncompacted.
		inv.unrespondedRetired[llmKey(identity{agentName: "pad-trigger", path: "p-trigger"})] = 1
		inv.recordRetiredActivationsAtLocked(openSet, at(65), &spanRecord{
			spanID:       "trigger-cap",
			kind:         spanKindAgent,
			name:         "pad-trigger",
			path:         "p-trigger",
			seq:          1002,
			parentSpanID: guard.spanID,
			parent:       guard,
			popRetirement: &popRetirement{
				afterAgentCredit: true,
			},
		})
		if inv.findRetiredActivationIndexLocked("a0") != -1 {
			t.Fatal("expected a0 to be folded into cap hull")
		}
		return inv
	}

	t.Run("alone_settles_post_retirement_after_agent_event", func(t *testing.T) {
		inv := buildCapFolded(t, false)
		// Event created at at(35) > hull.retiredAtPlatformTime (at(30)).
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(35), true, true) {
			t.Fatal("settleRetiredResponseCreditLocked(post-retirement hull afterAgentCredit) = false, want true")
		}
		if inv.unrespondedRetired[k] != 0 {
			t.Fatalf("unrespondedRetired[k] = %d, want 0", inv.unrespondedRetired[k])
		}
		for _, ra := range inv.retiredActivations {
			if ra.compactedHull && len(ra.foldedCredits) > 0 {
				t.Fatalf("expected hull foldedCredits cleared after settlement, got %+v", ra.foldedCredits)
			}
		}
	})

	t.Run("post_retirement_after_agent_settlement_does_not_steal_later_peer_credit", func(t *testing.T) {
		inv := buildCapFolded(t, true)
		// Event created at at(35) > hull.retiredAtPlatformTime (at(30)) and before A2 started (at(50)).
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(35), true, true) {
			t.Fatal("settleRetiredResponseCreditLocked(post-retirement hull afterAgentCredit) = false, want true")
		}
		if inv.unrespondedRetired[k] != 1 {
			t.Fatalf("unrespondedRetired[k] = %d, want 1 (a2 credit preserved)", inv.unrespondedRetired[k])
		}
		idxA2 := inv.findRetiredActivationIndexLocked("a2")
		if idxA2 < 0 || inv.retiredActivations[idxA2].modelCredits != 1 {
			t.Fatalf("expected a2 to keep modelCredits=1, got idx=%d entry=%+v", idxA2, inv.retiredActivations)
		}
		// Duplicate late event matching settled hull must NOT steal A2's credit.
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(36), true, true) {
			t.Fatal("duplicate late hull event stole peer A2's credit")
		}
		if inv.unrespondedRetired[k] != 1 {
			t.Fatalf("unrespondedRetired[k] = %d, want 1 after duplicate late hull event", inv.unrespondedRetired[k])
		}
		// A2's own response within [at(50), at(60)] consumes its credit cleanly.
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(55), true, true) || inv.unrespondedRetired[k] != 0 {
			t.Fatalf("A2 response failed to settle remaining credit: unresponded=%d", inv.unrespondedRetired[k])
		}
	})

	t.Run("uncompacted_mapless_entry_settles_post_retirement_after_agent_event", func(t *testing.T) {
		guard := &spanRecord{spanID: "guard-open", kind: spanKindAgent, name: "custom", seq: 1}
		inv := &invocationTrace{
			agentStack:         []*spanRecord{{kind: spanKindInvocation}, guard},
			unrespondedRetired: map[string]int{k: 1},
		}
		openSet := map[string]bool{guard.spanID: true}
		inv.recordRetiredActivationsAtLocked(openSet, at(30), &spanRecord{
			spanID:                 "a0-mapless",
			kind:                   spanKindAgent,
			name:                   "worker",
			path:                   "p1",
			seq:                    2,
			parentSpanID:           guard.spanID,
			parent:                 guard,
			hasAfterAgentCallbacks: true,
			startPlatformTime:      at(10),
			popRetirement: &popRetirement{
				afterAgentCredit: true,
			},
		})
		if !inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(35), true, true) || inv.unrespondedRetired[k] != 0 {
			t.Fatalf("mapless uncompacted post-retirement settlement failed: unresponded=%d", inv.unrespondedRetired[k])
		}
		idx := inv.findRetiredActivationIndexLocked("a0-mapless")
		if idx < 0 || !inv.retiredActivations[idx].creditSettled {
			t.Fatalf("expected a0-mapless creditSettled=true, got idx=%d entry=%+v", idx, inv.retiredActivations)
		}
		// Adding later peer A2 must remain isolated from duplicate late events of settled a0-mapless.
		inv.unrespondedRetired[k] = 1
		inv.recordRetiredActivationsAtLocked(openSet, at(60), &spanRecord{
			spanID:            "a2",
			kind:              spanKindAgent,
			name:              "worker",
			path:              "p1",
			seq:               3,
			parentSpanID:      guard.spanID,
			parent:            guard,
			startPlatformTime: at(50),
			popRetirement:     &popRetirement{modelCredits: 1},
		})
		if inv.settleRetiredResponseCreditLocked(k, "worker", "", "p1", at(36), true, true) || inv.unrespondedRetired[k] != 1 {
			t.Fatalf("late event after mapless settlement stole peer credit: unresponded=%d", inv.unrespondedRetired[k])
		}
	})
}
