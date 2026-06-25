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
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/agent"
)

// bumpPeak increments inFlight, updates peak if the new value is
// higher, and returns a closure that decrements inFlight when
// called. Used by tests that observe peak concurrency.
func bumpPeak(inFlight, peak *atomic.Int32) func() {
	now := inFlight.Add(1)
	for {
		p := peak.Load()
		if now <= p || peak.CompareAndSwap(p, now) {
			break
		}
	}
	return func() { inFlight.Add(-1) }
}

// fanOutToJoin builds a START -> nodes -> join topology: every node
// fans out from START and feeds a single JoinNode. Returns the edges
// and the join.
func fanOutToJoin(nodes []Node) ([]Edge, *JoinNode) {
	join := NewJoinNode("join")
	edges := make([]Edge, 0, len(nodes)+1)
	for _, n := range nodes {
		edges = append(edges, Edge{From: Start, To: n})
		edges = append(edges, Edge{From: n, To: join})
	}
	return edges, join
}

// TestMaxConcurrency_Caps verifies that a fan-out wider than the
// configured cap never exceeds the cap in the number of nodes
// running simultaneously.
func TestMaxConcurrency_Caps(t *testing.T) {
	const fanOut = 6
	const cap = 2

	mockCtx := newSeededMockCtx(t)
	var inFlight, peak atomic.Int32

	releases := make([]chan struct{}, fanOut)
	nodes := make([]Node, fanOut)
	for i := range fanOut {
		i := i
		releases[i] = make(chan struct{})
		nodes[i] = newBlockingNode(fmt.Sprintf("N%d", i), func() {
			defer bumpPeak(&inFlight, &peak)()
			<-releases[i]
		})
	}

	edges, _ := fanOutToJoin(nodes)
	w, err := New("", edges, WithMaxConcurrency(cap))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		drain(t, w.Run(t.Context(), mockCtx))
	}()

	// Wait until cap nodes are admitted, then release them
	// one by one so the scheduler dispatches queued nodes
	// into their slots without ever exceeding the cap.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && inFlight.Load() != cap {
		time.Sleep(time.Millisecond)
	}
	if got := inFlight.Load(); got != cap {
		t.Fatalf("after warmup: inFlight = %d, want %d (cap not enforced)", got, cap)
	}
	for i := range fanOut {
		close(releases[i])
	}
	<-done

	if got := peak.Load(); got > cap {
		t.Errorf("peak in-flight = %d, want <= %d (cap violated)", got, cap)
	}
}

// TestMaxConcurrency_PendingDispatchedFIFO verifies that queued
// activations are dispatched in arrival order: a fan-out with
// cap=1 must run in strict sequence.
func TestMaxConcurrency_PendingDispatchedFIFO(t *testing.T) {
	const fanOut = 4
	mockCtx := newSeededMockCtx(t)

	var order atomic.Int32
	starts := make([]int32, fanOut)
	nodes := make([]Node, fanOut)
	for i := range fanOut {
		i := i
		nodes[i] = NewFunctionNode(
			fmt.Sprintf("N%d", i),
			func(_ context.Context, _ agent.Context, input any) (string, error) {
				starts[i] = order.Add(1)
				return "ok", nil
			},
			defaultNodeConfig,
		)
	}
	edges, _ := fanOutToJoin(nodes)
	w, err := New("", edges, WithMaxConcurrency(1))
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	drain(t, w.Run(t.Context(), mockCtx))

	for i := range fanOut {
		if want := int32(i + 1); starts[i] != want {
			t.Errorf("N%d start order = %d, want %d (FIFO violated)", i, starts[i], want)
		}
	}
}

// TestMaxConcurrency_RetryRespectsLimit verifies that a node
// scheduled via the retry path also goes through the
// concurrency-cap gate. Without the gate, a retry could squeeze
// in while a sibling is still in-flight under cap=1.
func TestMaxConcurrency_RetryRespectsLimit(t *testing.T) {
	mockCtx := newSeededMockCtx(t)
	var inFlight, peak atomic.Int32

	var flakyCalls atomic.Int32
	flaky := NewFunctionNode("flaky",
		func(_ context.Context, _ agent.Context, input any) (string, error) {
			defer bumpPeak(&inFlight, &peak)()
			// Hold long enough that, if the cap were broken, a
			// sibling could sneak in.
			time.Sleep(20 * time.Millisecond)
			if flakyCalls.Add(1) == 1 {
				return "", errors.New("retryable")
			}
			return "ok", nil
		},
		NodeConfig{
			RetryConfig: &RetryConfig{
				MaxAttempts:   3,
				InitialDelay:  10 * time.Millisecond,
				MaxDelay:      10 * time.Millisecond,
				BackoffFactor: 1.0,
				ShouldRetry:   func(error) bool { return true },
			},
		},
	)
	stable := NewFunctionNode("stable",
		func(_ context.Context, _ agent.Context, input any) (string, error) {
			defer bumpPeak(&inFlight, &peak)()
			time.Sleep(20 * time.Millisecond)
			return "ok", nil
		},
		defaultNodeConfig,
	)

	edges, _ := fanOutToJoin([]Node{flaky, stable})
	w, err := New("", edges, WithMaxConcurrency(1))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	drain(t, w.Run(t.Context(), mockCtx))

	if got := peak.Load(); got > 1 {
		t.Errorf("peak in-flight = %d, want <= 1 (retry bypassed cap)", got)
	}
	if got := flakyCalls.Load(); got < 2 {
		t.Errorf("flaky calls = %d, want >= 2 (retry never happened)", got)
	}
}
