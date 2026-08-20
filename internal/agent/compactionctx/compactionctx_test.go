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

package compactionctx

import (
	"context"
	"sync"
	"testing"

	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

func TestFromContextWithoutRuntime(t *testing.T) {
	t.Parallel()

	rt := FromContext(context.Background())
	if rt != nil {
		t.Errorf("FromContext() = %v on a bare context, want nil", rt)
	}
	// The nil receiver must answer, not panic: every caller reaches these
	// through a context that may not carry a runtime.
	if rt.Configured() || rt.Enabled() || rt.AlreadyCompacted() {
		t.Error("a nil runtime reported itself as usable")
	}
	rt.MarkCompacted() // must not panic
}

func TestRoundTrip(t *testing.T) {
	t.Parallel()

	want := New(&compaction.Config{CompactionInterval: 2}, session.InMemoryService())
	got := FromContext(ToContext(context.Background(), want))
	if got != want {
		t.Fatalf("FromContext() returned %v, want the runtime that was stored", got)
	}
	if !got.Configured() {
		t.Error("Configured() = false for a runtime with a config")
	}
}

// TestMarkCompactedIsSafeUnderConcurrency covers the reason this is an atomic
// rather than a plain bool: sub-agents in a parallel workflow share one runtime.
func TestMarkCompactedIsSafeUnderConcurrency(t *testing.T) {
	t.Parallel()

	rt := New(&compaction.Config{CompactionInterval: 1}, nil)
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rt.MarkCompacted()
			_ = rt.AlreadyCompacted()
		}()
	}
	wg.Wait()
	if !rt.AlreadyCompacted() {
		t.Error("AlreadyCompacted() = false after MarkCompacted()")
	}
}

// TestProgressGateReArmsOnceThePromptRecovers pins that one compaction does not
// disarm compaction for the rest of a long turn.
//
// The gate used to compare prompt sizes, refusing anything not smaller than the
// size the last compaction ran at. A turn that kept growing therefore never
// compacted again, however large it got, which is the opposite of what the gate
// is for: a tool loop ran to 45,056 tokens against a 2,000 threshold after two
// compactions had visibly worked.
func TestProgressGateReArmsOnceThePromptRecovers(t *testing.T) {
	t.Parallel()

	rt := New(&compaction.Config{TokenThreshold: 1000, EventRetentionSize: 2}, nil)

	if !rt.GateFor("a", "", "").AllowAt(2000) {
		t.Fatal("AllowAt() = false before any compaction, want true")
	}
	rt.GateFor("a", "", "").RecordAt(2000)

	// Still above the threshold, so the compaction has not been shown to work
	// and another one would summarize a little more to no effect.
	if rt.GateFor("a", "", "").AllowAt(2500) {
		t.Error("AllowAt() = true straight after a compaction, want false")
	}

	// The prompt came back under the threshold, so the compaction did work.
	rt.GateFor("a", "", "").Recovered()
	if !rt.GateFor("a", "", "").AllowAt(2500) {
		t.Error("AllowAt() = false after the prompt recovered, want true: a turn that grows again must be able to compact again")
	}
}

// TestProgressGateStaysClosedWhileCompactionCannotHelp pins the case the gate
// exists for: a retained tail that already exceeds the threshold, where the
// prompt never recovers and every further compaction is a wasted model call.
func TestProgressGateStaysClosedWhileCompactionCannotHelp(t *testing.T) {
	t.Parallel()

	rt := New(&compaction.Config{TokenThreshold: 1000, EventRetentionSize: 2}, nil)
	rt.GateFor("a", "", "").RecordAt(5000)

	for _, tokens := range []int{4900, 5200, 12000} {
		if rt.GateFor("a", "", "").AllowAt(tokens) {
			t.Errorf("AllowAt(%d) = true, want false while the prompt has never come back under the threshold", tokens)
		}
	}
}

// TestProgressGateRecordAtKeepsAZeroCountDistinct pins that a compaction
// recorded at a zero prompt size still closes the gate, rather than reading as
// "nothing recorded yet".
func TestProgressGateRecordAtKeepsAZeroCountDistinct(t *testing.T) {
	t.Parallel()

	rt := New(&compaction.Config{TokenThreshold: 1000, EventRetentionSize: 2}, nil)
	rt.GateFor("a", "", "").RecordAt(0)
	if rt.GateFor("a", "", "").AllowAt(1) {
		t.Error("AllowAt() = true after RecordAt(0), want false")
	}
}

// TestGateIsScopedPerAgent pins that one agent's compaction does not suppress
// another's inside the same invocation.
//
// One invocation can run several agents and they do not share a prompt: a loop
// agent alternating a large-context worker with a small-context critic is two
// prompt sizes, and parallel sub-agents are as many as there are children. A
// single marker for all of them let one agent's compaction close the gate for
// every sibling, and let every parallel sibling read it as open before any of
// them closed it.
func TestGateIsScopedPerAgent(t *testing.T) {
	t.Parallel()

	rt := New(&compaction.Config{TokenThreshold: 1000, EventRetentionSize: 2}, nil)
	worker, critic := rt.GateFor("worker", "root", ""), rt.GateFor("critic", "root", "")

	if !worker.AllowAt(2000) || !critic.AllowAt(2000) {
		t.Fatal("both agents should start allowed")
	}
	worker.RecordAt(2000)
	if worker.AllowAt(2500) {
		t.Error("the worker compacted, so it should be gated")
	}
	if !critic.AllowAt(2500) {
		t.Error("the critic did not compact, so the worker's compaction must not gate it")
	}

	// A failure gates that agent alone, and Recovered does not undo it: the
	// prompt never dropped, so nothing reports recovery.
	critic.Failed()
	if critic.AllowAt(2500) {
		t.Error("the critic's attempt failed, so it should not try again this invocation")
	}
	critic.Recovered()
	if critic.AllowAt(2500) {
		t.Error("Recovered reopened a gate closed by a failure, which the prompt size cannot justify")
	}
}
