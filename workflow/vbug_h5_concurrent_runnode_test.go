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

// FINDING H5 — concurrent RunNode children share one unsynchronized
// parent iterator yield.
//
// BUG: A DynamicNode forwards every child event up to its parent via the
// `emit` closure built by makeEmit (dynamic_node.go), which is also the
// `emitUp` used by the sub-scheduler (dynamic_scheduler.go runNode). That
// closure wraps the SINGLE parent range-over-func `yield` with no
// synchronization. WithUseSubBranch's own doc invites running "multiple
// concurrent or independent children". When the orchestrator body launches
// several goroutines that each call RunNode and the children emit events,
// those goroutines invoke the SAME `yield` concurrently. Go's
// range-over-func contract requires `yield` to be called only from the
// goroutine running the loop, never concurrently; violating it races on the
// compiler-generated loop state (and on anything the loop body touches, e.g.
// the consumer's event slice) and can panic.
//
// EXPECTED: Either the API must serialize child events before calling the
// parent yield (so concurrent children are safe as documented), or
// concurrent RunNode must be rejected. A program that runs the documented
// "multiple concurrent children" pattern must not data-race or panic.
//
// HOW THIS TEST DEMONSTRATES IT: The orchestrator launches several
// goroutines that each RunNode a distinct child (one event each) with
// WithUseSubBranch. Their events funnel through the one parent yield
// concurrently. FAILS under `go test -race` (DATA RACE on the
// range-over-func loop state / consumer event slice). It may also surface a
// recovered "range function continued iteration after loop body returned"
// panic, which is recorded and reported via t.Errorf even without -race;
// because the panic is timing-sensitive, -race is the reliable signal.

package workflow

import (
	"fmt"
	"sync"
	"testing"

	"google.golang.org/adk/session"
)

func TestVbugH5_ConcurrentRunNodeRacesOnParentYield(t *testing.T) {
	const workers = 8

	var (
		mu     sync.Mutex
		panics []any
	)

	orchestrator := NewDynamicNode[string, string]("orch",
		func(ctx NodeContext, _ string, _ func(*session.Event) error) (string, error) {
			start := make(chan struct{})
			var wg sync.WaitGroup
			wg.Add(workers)
			for i := 0; i < workers; i++ {
				// Distinct child per goroutine so the only shared
				// mutable state is the parent yield path, keeping the
				// race attributable to the workflow's unsynchronized
				// emitUp rather than to a shared test fixture.
				child := newStubNode(fmt.Sprintf("child-%d", i), fmt.Sprintf("out-%d", i))
				go func() {
					defer wg.Done()
					defer func() {
						if r := recover(); r != nil {
							mu.Lock()
							panics = append(panics, r)
							mu.Unlock()
						}
					}()
					<-start // release all goroutines together
					_, _ = RunNode[string](ctx, child, "x", WithUseSubBranch())
				}()
			}
			close(start)
			wg.Wait()
			return "", nil
		},
		NodeConfig{},
	)

	// Drives the dynamic node and ranges over its events; the range body
	// (and the compiler-generated loop state) is what the concurrent child
	// yields collide on.
	_, _ = drainDynamicWithErr(t, orchestrator, "")

	mu.Lock()
	defer mu.Unlock()
	if len(panics) > 0 {
		t.Errorf("H5: %d recovered panic(s) from concurrent RunNode children "+
			"sharing one unsynchronized parent yield; first: %v", len(panics), panics[0])
	}
}
