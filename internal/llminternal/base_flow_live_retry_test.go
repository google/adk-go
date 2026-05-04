// Copyright 2025 Google LLC
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

package llminternal

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// TestRunLive_RetryBudgetExhaustion verifies that when every connect attempt
// fails, RunLive yields a "failed to connect live after N retries" error and
// stops after exactly maxReconnectRetries+1 attempts (one initial attempt
// plus maxReconnectRetries retries).
func TestRunLive_RetryBudgetExhaustion(t *testing.T) {
	withFastReconnectVars(t, 2, 10*time.Millisecond, 10*time.Millisecond)

	var calls atomic.Int32
	connectFn := func(_ string) (model.LiveConnection, error) {
		calls.Add(1)
		return nil, errors.New("synthetic connect failure")
	}

	ctx := newTestInvocationContext(t, context.Background())
	queue := agent.NewLiveRequestQueue(1)
	flow := &LiveFlow{}

	var lastErr error
	for ev, err := range flow.RunLive(ctx, connectFn, queue) {
		_ = ev
		if err != nil {
			lastErr = err
		}
	}

	if lastErr == nil || !strings.Contains(lastErr.Error(), "failed to connect live after") {
		t.Fatalf("expected retry-exhaustion error, got %v", lastErr)
	}
	if got, want := calls.Load(), int32(maxReconnectRetries+1); got != want {
		t.Errorf("connectFn called %d times, want %d", got, want)
	}
}

// TestRunLive_ContextCancelDuringReconnectBackoff verifies that cancelling
// the parent context during retry backoff propagates promptly and RunLive
// returns a context.Canceled chain — not the retry-exhaustion message —
// well before the next backoff timer would have fired.
//
// We use a 1s base backoff and a 50ms cancel deadline so a non-interruptible
// sleep would yield wall time ≥ 1s, while a select-on-Done backoff yields
// well under 100ms. The 200ms threshold sits comfortably between the two
// modes, so a regression to bare time.Sleep will trip this assertion.
func TestRunLive_ContextCancelDuringReconnectBackoff(t *testing.T) {
	withFastReconnectVars(t, 5, time.Second, 10*time.Millisecond)

	parent, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	var calls atomic.Int32
	connectFn := func(_ string) (model.LiveConnection, error) {
		n := calls.Add(1)
		if n == 1 {
			// Schedule a cancel well before the 1s backoff timer fires.
			time.AfterFunc(50*time.Millisecond, cancel)
		}
		return nil, errors.New("synthetic")
	}

	ctx := newTestInvocationContext(t, parent)
	queue := agent.NewLiveRequestQueue(1)
	flow := &LiveFlow{}
	start := time.Now()

	var lastErr error
	for ev, err := range flow.RunLive(ctx, connectFn, queue) {
		_ = ev
		if err != nil {
			lastErr = err
		}
	}
	elapsed := time.Since(start)

	if !errors.Is(lastErr, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", lastErr)
	}
	if elapsed > 200*time.Millisecond {
		t.Errorf("expected fast cancel, elapsed = %s (likely waited out a non-interruptible sleep)", elapsed)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("expected exactly 1 connect attempt before cancel, got %d", got)
	}
}

// TestRunLive_ContextCancelDuringPostGoAwaySleep verifies that cancelling
// the parent context during the post-GoAway pause (between the broken
// session ending and the reconnect attempt) yields context.Canceled to the
// iterator — not a silent close. A regression to a bare `return` would
// drop the cancel error so consumers couldn't distinguish a clean exit
// from a cancellation.
//
// Setup: stretch reconnectGoAwaySleep to 1s (well above the cancel window)
// and schedule the cancel for 100ms after the first connect; with the fix,
// wall time stays well under 300ms and the iterator yields ctx.Err().
func TestRunLive_ContextCancelDuringPostGoAwaySleep(t *testing.T) {
	withFastReconnectVars(t, 5, 10*time.Millisecond, time.Second)

	parent, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	var connects atomic.Int32
	connectFn := func(_ string) (model.LiveConnection, error) {
		n := connects.Add(1)
		if n == 1 {
			// Schedule cancel to fire well into the 1s post-GoAway sleep,
			// but well before it would naturally elapse.
			time.AfterFunc(100*time.Millisecond, cancel)
		}
		return &goAwayThenBlockConn{}, nil
	}

	ctx := newTestInvocationContext(t, parent)
	queue := agent.NewLiveRequestQueue(1)
	flow := &LiveFlow{}
	start := time.Now()

	var lastErr error
	for ev, err := range flow.RunLive(ctx, connectFn, queue) {
		_ = ev
		if err != nil {
			lastErr = err
		}
	}
	elapsed := time.Since(start)

	if !errors.Is(lastErr, context.Canceled) {
		t.Fatalf("expected context.Canceled to be yielded after post-GoAway cancel, got %v", lastErr)
	}
	if elapsed > 300*time.Millisecond {
		t.Errorf("expected fast cancel, elapsed = %s (likely waited out the post-GoAway sleep)", elapsed)
	}
	if got := connects.Load(); got != 1 {
		t.Errorf("expected exactly 1 connect attempt, got %d", got)
	}
}

// goAwayThenBlockConn is a minimal model.LiveConnection that yields one
// GoAway response on the first Receive and then blocks on context
// cancellation, simulating a server that signalled GoAway and stopped
// sending. It records nothing else — sufficient for tests that exercise
// the post-GoAway code path.
type goAwayThenBlockConn struct {
	receives atomic.Int32
}

func (c *goAwayThenBlockConn) Send(_ context.Context, _ *model.LiveRequest) error {
	return nil
}

func (c *goAwayThenBlockConn) Receive(ctx context.Context) (*model.LLMResponse, error) {
	if c.receives.Add(1) == 1 {
		return &model.LLMResponse{GoAway: &genai.LiveServerGoAway{}}, nil
	}
	<-ctx.Done()
	return nil, ctx.Err()
}

func (c *goAwayThenBlockConn) Close() error { return nil }

// withFastReconnectVars overrides the package-level reconnect knobs for the
// duration of a test and restores them via t.Cleanup. Tests use small values
// so they run in tens of milliseconds rather than seconds.
func withFastReconnectVars(t *testing.T, retries int, baseBackoff, goAwaySleep time.Duration) {
	t.Helper()
	prevR, prevB, prevG := maxReconnectRetries, reconnectBaseBackoff, reconnectGoAwaySleep
	maxReconnectRetries = retries
	reconnectBaseBackoff = baseBackoff
	reconnectGoAwaySleep = goAwaySleep
	t.Cleanup(func() {
		maxReconnectRetries = prevR
		reconnectBaseBackoff = prevB
		reconnectGoAwaySleep = prevG
	})
}

// newTestInvocationContext builds a minimal agent.InvocationContext suitable
// for driving LiveFlow.RunLive in unit tests. The Agent is set so receiver-
// side processing (which reads invCtx.Agent().Name() to label events) does
// not nil-panic when a session actually starts.
func newTestInvocationContext(t *testing.T, parent context.Context) agent.InvocationContext {
	t.Helper()
	resp, err := session.InMemoryService().Create(parent, &session.CreateRequest{
		AppName:   "test",
		UserID:    "u",
		SessionID: "s",
	})
	if err != nil {
		t.Fatalf("session create: %v", err)
	}
	a, err := agent.New(agent.Config{Name: "test_agent"})
	if err != nil {
		t.Fatalf("agent create: %v", err)
	}
	return icontext.NewInvocationContext(parent, icontext.InvocationContextParams{
		Session: resp.Session,
		Agent:   a,
	})
}
