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
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
)

type bumpArgs struct{ DelayMS int }

// lockedBump returns a tool body that sleeps DelayMS, then increments the
// state key "count" under mu, which it holds across State().Get and
// State().Set.
func lockedBump(mu *sync.Mutex) func(agent.Context, bumpArgs) (map[string]any, error) {
	return func(ctx agent.Context, a bumpArgs) (map[string]any, error) {
		time.Sleep(time.Duration(a.DelayMS) * time.Millisecond)
		mu.Lock()
		defer mu.Unlock()
		n, err := psoGet(ctx, "count", 0)
		if err != nil {
			return nil, err
		}
		if err := ctx.State().Set("count", n+1); err != nil {
			return nil, err
		}
		return map[string]any{"count": n + 1}, nil
	}
}

// TestParallelCalls_LockAroundStateDoesNotHang pins that a tool that holds a
// mutex across State().Get and Set, called twice in one turn with either call
// delayed, answers both calls. It fails when a call's State use waits for the
// earlier calls to return: the later call then holds the mutex while it waits
// for an earlier call that waits for the mutex.
func TestParallelCalls_LockAroundStateDoesNotHang(t *testing.T) {
	for _, tc := range []struct {
		name           string
		delay0, delay1 int
	}{
		{"first call delayed", 100, 0},
		{"second call delayed", 0, 100},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			bump := psoTool(t, "bump", lockedBump(&mu))

			turn := psoRun(t, []tool.Tool{bump}, nil,
				psoCall("c0", "bump", map[string]any{"DelayMS": tc.delay0}),
				psoCall("c1", "bump", map[string]any{"DelayMS": tc.delay1}))

			var counts []float64
			for _, id := range []string{"c0", "c1"} {
				r, ok := turn.responses[id]
				if !ok {
					t.Fatalf("call %s got no response", id)
				}
				if e, isErr := r["error"]; isErr {
					t.Fatalf("call %s answered with an error: %v", id, e)
				}
				// A function tool's result reaches the response through JSON.
				n, _ := r["count"].(float64)
				counts = append(counts, n)
			}
			slices.Sort(counts)
			if diff := cmp.Diff([]float64{1, 2}, counts); diff != "" {
				t.Errorf("the two calls' counts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestParallelCalls_LockAroundStateAllDoesNotHang pins that State().All, used
// under a mutex by two calls of one turn with call 0 delayed, does not wait
// for the earlier call: both calls answer and see the two initial keys.
func TestParallelCalls_LockAroundStateAllDoesNotHang(t *testing.T) {
	var mu sync.Mutex
	list := psoTool(t, "list", func(ctx agent.Context, a bumpArgs) (map[string]any, error) {
		time.Sleep(time.Duration(a.DelayMS) * time.Millisecond)
		mu.Lock()
		defer mu.Unlock()
		n := 0
		for range ctx.State().All() {
			n++
		}
		return map[string]any{"seen": n}, nil
	})

	turn := psoRun(t, []tool.Tool{list}, map[string]any{"a": 1, "b": 2},
		psoCall("c0", "list", map[string]any{"DelayMS": 100}),
		psoCall("c1", "list", map[string]any{"DelayMS": 0}))

	for _, id := range []string{"c0", "c1"} {
		if diff := cmp.Diff(map[string]any{"seen": float64(2)}, turn.responses[id]); diff != "" {
			t.Errorf("%s response mismatch (-want +got):\n%s", id, diff)
		}
	}
}

// TestParallelCalls_SequentialToolWithLockRunsInCallOrder pins that the tool
// of TestParallelCalls_LockAroundStateDoesNotHang, declared sequential, runs
// its calls one after the other in call order even when call 0 is delayed:
// the counts are 1 then 2 and the stored count is 2. It fails if a sequential
// call starts before the earlier calls have returned (counts 2 and 1) or
// waits inside the tool's critical section (a hang).
func TestParallelCalls_SequentialToolWithLockRunsInCallOrder(t *testing.T) {
	var mu sync.Mutex
	bump := psoSeqTool(t, "bump", lockedBump(&mu))

	turn := psoRun(t, []tool.Tool{bump}, nil,
		psoCall("c0", "bump", map[string]any{"DelayMS": 100}),
		psoCall("c1", "bump", map[string]any{"DelayMS": 0}))

	type counts struct{ C0, C1, Stored any }
	got := counts{turn.responses["c0"]["count"], turn.responses["c1"]["count"], psoStored(t, turn.state, "count")}
	want := counts{C0: float64(1), C1: float64(2), Stored: 2}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("counts mismatch (-want +got):\n%s", diff)
	}
}
