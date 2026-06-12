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
	"fmt"
	"iter"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// retryLoopTestNode simulates a node that fails twice in each execution
// and succeeds on the third attempt.
type retryLoopTestNode struct {
	BaseNode
	calls atomic.Int32
}

func (n *retryLoopTestNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		calls := n.calls.Add(1)

		// Make 2 failing calls and one successful call in each execution of the node.
		if calls%3 != 0 {
			yield(nil, fmt.Errorf("fail attempt %d", calls))
			return
		}

		ev := session.NewEvent(ctx.InvocationID())
		// Finish after 2 loop executions.
		if calls >= 6 {
			ev.Routes = []string{"finish"}
		} else {
			ev.Routes = []string{"loop"}
		}
		ev.Output = fmt.Sprintf("%v:%s", input, n.Name())
		yield(ev, nil)
	}
}

func TestScheduler_RetryLoop(t *testing.T) {
	mockCtx := newSeededMockCtx(t)
	retryConfig := DefaultRetryConfig()
	retryConfig.MaxAttempts = 3
	retryConfig.InitialDelay = 0
	retryConfig.Jitter = 0.0

	cfg := NodeConfig{
		RetryConfig: retryConfig,
	}

	nodeA := &retryLoopTestNode{
		BaseNode: NewBaseNode("A", "", cfg),
	}

	nodeFinish := newRecordingNode("Finish")
	nodeFinish.release()

	edges := []Edge{
		{From: Start, To: nodeA},
		{From: nodeA, To: nodeA, Route: StringRoute("loop")},
		{From: nodeA, To: nodeFinish, Route: StringRoute("finish")},
	}

	w := mustNew(t, edges)

	// Verify that ns.Attempt is reset on success, allowing full retries
	// in subsequent executions in a loop.

	var events []*session.Event
	for ev, err := range w.Run(mockCtx) {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
	}

	wantOutputs := []string{"seed:A", "seed:A:A", "seed:A:A:Finish"}
	gotOutputs := outputsOf(events)

	if diff := cmp.Diff(wantOutputs, gotOutputs); diff != "" {
		t.Errorf("outputs mismatch (-want +got):\n%s", diff)
	}

	if got := nodeA.calls.Load(); got != 6 {
		t.Errorf("node A calls = %d, want 6", got)
	}
}
