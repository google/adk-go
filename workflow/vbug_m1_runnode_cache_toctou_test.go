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

// FINDING M1 — RunNode idempotency cache is check-then-act (TOCTOU).
//
// BUG: dynamicSubScheduler.runNode (dynamic_scheduler.go) reads the
// idempotency cache with lookupCachedOutput and writes it with
// storeCachedOutput, each under its own short-lived lock, but the child's
// Run executes UNLOCKED in between. Two concurrent RunNode calls that share
// the same explicit WithRunID both miss the cache (nothing stored yet) and
// both execute the child, defeating the idempotency the cache promises.
//
// EXPECTED: A given (childPath = parent/<name>@<runID>) executes its child
// at most once per activation; a second RunNode with the same WithRunID must
// observe the cached output without re-running the child. (The sequential
// case is asserted by TestRunNode_WithRunID_IdempotentReplay; this is the
// concurrent case.)
//
// HOW THIS TEST DEMONSTRATES IT: The orchestrator launches two goroutines
// that both call RunNode with WithRunID("dup") on the same child. The child
// increments an atomic counter and then briefly waits (bounded, with a
// timeout so correct single-execution never deadlocks) so both callers are
// guaranteed to be inside child.Run — i.e. both passed lookupCachedOutput
// before either reached storeCachedOutput. With the bug the child runs
// twice; the test asserts exactly one execution, so it FAILS (counter == 2).
// This is a deterministic logic failure independent of -race; run with
// -count to confirm stability.

package workflow

import (
	"iter"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// vbugM1CountingNode counts each Run invocation and emits no events. The
// bounded wait widens (and makes deterministic) the check-then-act window:
// when the bug lets both callers in, both reach runs>=2 immediately; when
// the cache works correctly only one caller runs and it falls through after
// the timeout rather than deadlocking.
type vbugM1CountingNode struct {
	BaseNode
	runs *atomic.Int64
}

func (n *vbugM1CountingNode) Run(_ agent.Context, _ any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		n.runs.Add(1)
		deadline := time.Now().Add(500 * time.Millisecond)
		for n.runs.Load() < 2 && time.Now().Before(deadline) {
			time.Sleep(time.Millisecond)
		}
		// Emit nothing: keeps the parent yield single-use so this test
		// isolates the cache TOCTOU from the concurrent-yield issue (H5).
	}
}

func TestVbugM1_ConcurrentRunNodeSameRunIDExecutesChildTwice(t *testing.T) {
	var runs atomic.Int64
	child := &vbugM1CountingNode{
		BaseNode: NewBaseNode("dupchild", "", NodeConfig{}),
		runs:     &runs,
	}

	orchestrator := NewDynamicNode[string, string]("orch",
		func(ctx NodeContext, _ string, _ func(*session.Event) error) (string, error) {
			start := make(chan struct{})
			var wg sync.WaitGroup
			wg.Add(2)
			for i := 0; i < 2; i++ {
				go func() {
					defer wg.Done()
					<-start
					// Same explicit run id => same childPath => same
					// cache key for both callers.
					_, _ = RunNode[string](ctx, child, "x", WithRunID("dup"))
				}()
			}
			close(start)
			wg.Wait()
			return "", nil
		},
		NodeConfig{},
	)

	if _, err := drainDynamicWithErr(t, orchestrator, ""); err != nil {
		t.Fatalf("unexpected Run error: %v", err)
	}

	if got := runs.Load(); got != 1 {
		t.Errorf("M1: child with WithRunID(%q) ran %d times; idempotency cache "+
			"must execute it exactly once. The check-then-act gap between "+
			"lookupCachedOutput and storeCachedOutput (child.Run runs unlocked "+
			"in between) lets concurrent callers double-execute.", "dup", got)
	}
}
