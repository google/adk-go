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

package llminternal_test

import (
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
)

// mixedTurnStepWait bounds each wait for a sibling call's step. Two such
// waits stay under psoGuard, so an order in which call 1 waits for call 0
// still ends the run and records its order.
const mixedTurnStepWait = 2 * time.Second

// waitStep waits for ch to close, or moves on after mixedTurnStepWait.
func waitStep(ch <-chan struct{}) {
	select {
	case <-ch:
	case <-time.After(mixedTurnStepWait):
	}
}

// TestParallelCalls_SequentialCallDoesNotHoldBackLaterPlainCall pins the
// Known gap FC-COD-GAP-2: in a turn whose call 0 is sequential and call 1 is
// plain, call 1 runs while call 0 runs, so their steps interleave as
// [async_C sync_A async_D sync_B]. adk-python's
// test_sync_function_blocks_async_functions
// (tests/unittests/flows/llm_flows/tools/test_functions_simple.py:905-946)
// asserts ['sync_A', 'sync_B', 'async_C', 'async_D'] for the same turn,
// because its sync tool blocks the event loop until it returns. Each step
// here waits for the sibling call's step before it, and a wait that times
// out moves on, so the test fails with Python's order if call 1 waits for
// call 0 to return.
func TestParallelCalls_SequentialCallDoesNotHoldBackLaterPlainCall(t *testing.T) {
	var mu sync.Mutex
	var order []string
	record := func(step string) {
		mu.Lock()
		defer mu.Unlock()
		order = append(order, step)
	}
	asyncC, syncA, asyncD := make(chan struct{}), make(chan struct{}), make(chan struct{})
	syncTool := psoSeqTool(t, "blocking_sync_function", func(agent.Context, struct{}) (map[string]any, error) {
		waitStep(asyncC)
		record("sync_A")
		close(syncA)
		waitStep(asyncD)
		record("sync_B")
		return map[string]any{"result": "sync_done"}, nil
	})
	asyncTool := psoTool(t, "yielding_async_function", func(agent.Context, struct{}) (map[string]any, error) {
		record("async_C")
		close(asyncC)
		waitStep(syncA)
		record("async_D")
		close(asyncD)
		return map[string]any{"result": "async_done"}, nil
	})

	psoRun(t, []tool.Tool{syncTool, asyncTool}, nil,
		psoCall("c0", "blocking_sync_function", nil),
		psoCall("c1", "yielding_async_function", nil))

	mu.Lock()
	defer mu.Unlock()
	if diff := cmp.Diff([]string{"async_C", "sync_A", "async_D", "sync_B"}, order); diff != "" {
		t.Errorf("execution order mismatch (-want +got):\n%s", diff)
	}
}
