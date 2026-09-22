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

package llminternal

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"math"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// drainTimeout bounds how long a test waits for RunLive's iterator to finish.
// Every reconnect test drives a bounded policy, so an iterator that has not
// drained by now is not terminating at all.
const drainTimeout = 30 * time.Second

// testLiveReconnectPolicy is the default policy with the delays shrunk, so a
// test that exercises the bounds does not also sleep through the production
// backoff.
func testLiveReconnectPolicy(maxAttempts, maxTotal int) *liveReconnectPolicy {
	return &liveReconnectPolicy{
		initialBackoff: time.Millisecond,
		maxBackoff:     10 * time.Millisecond,
		jitter:         0,
		maxAttempts:    maxAttempts,
		maxTotal:       maxTotal,
		// Every connection a fake server makes dies in milliseconds, so this
		// only has to exceed that for them all to count as short-lived.
		healthyUptime: time.Second,
	}
}

func newReconnectFlow(client *genai.Client, policy *liveReconnectPolicy) *Flow {
	return &Flow{
		Model:             &fakeLiveModel{client: client},
		RequestProcessors: []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{liveConfigProcessor},
		reconnect:         policy,
	}
}

type drainResult struct {
	err    error
	events int
}

// startDrain consumes seq to exhaustion on its own goroutine. Draining has to
// begin before the flow is expected to make progress: outputCh is unbuffered,
// so an undrained iterator parks the flow in pushEvent on the first event
// rather than wherever the test means to observe it.
func startDrain(seq iter.Seq2[*session.Event, error]) <-chan drainResult {
	done := make(chan drainResult, 1)
	go func() {
		var r drainResult
		for ev, err := range seq {
			if err != nil {
				r.err = err
			}
			if ev != nil {
				r.events++
			}
		}
		done <- r
	}()
	return done
}

func awaitDrain(t *testing.T, done <-chan drainResult) (events int, lastErr error) {
	t.Helper()
	select {
	case r := <-done:
		return r.events, r.err
	case <-time.After(drainTimeout):
		t.Fatal("iterator never drained: the flow is not terminating")
		return 0, nil
	}
}

// drainLive drains seq and waits for it to finish.
func drainLive(t *testing.T, seq iter.Seq2[*session.Event, error]) (events int, lastErr error) {
	t.Helper()
	return awaitDrain(t, startDrain(seq))
}

// waitForReconnectBackoff blocks until a RunLive goroutine is parked in the
// reconnect wait, so a teardown test is observing the wait and not some earlier
// point in the loop.
func waitForReconnectBackoff(t *testing.T) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		_, stacks := runLiveStacks()
		if strings.Contains(stacks, "llminternal.waitBeforeReconnect") {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("no RunLive goroutine ever reached the reconnect backoff")
}

// TestRunLiveBoundsADeadEndpoint pins the bug this policy exists for: an
// endpoint that completes the websocket handshake and then hangs up must not
// redial without limit, and the caller must be told the session is over.
func TestRunLiveBoundsADeadEndpoint(t *testing.T) {
	client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		// Hard close straight after setupComplete.
	})
	f := newReconnectFlow(client, testLiveReconnectPolicy(4, 20))
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	defer func() { _ = sess.Close() }()

	_, lastErr := drainLive(t, seq)

	if !errors.Is(lastErr, errLiveReconnectExhausted) {
		t.Errorf("caller got err %v, want one wrapping errLiveReconnectExhausted", lastErr)
	}
	if !strings.Contains(fmt.Sprint(lastErr), "consecutive") {
		t.Errorf("err %q does not name the consecutive-attempt bound", lastErr)
	}
	// One initial dial plus maxAttempts retries.
	if got, want := connCount.Load(), int32(5); got != want {
		t.Errorf("dialled %d times, want %d", got, want)
	}
}

// TestRunLiveBoundsAOneFramePerConnectionBackend covers the hole the
// consecutive-attempt budget alone leaves: a backend that serves one content
// frame and then hangs up resets that budget on every connection, so only the
// invocation-wide ceiling ends the session.
func TestRunLiveBoundsAOneFramePerConnectionBackend(t *testing.T) {
	client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		_ = conn.WriteMessage(websocket.TextMessage, []byte(serverContentPong))
		// Give the flow time to consume the frame and reset the consecutive
		// budget before the connection dies.
		time.Sleep(20 * time.Millisecond)
	})
	f := newReconnectFlow(client, testLiveReconnectPolicy(4, 6))
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	defer func() { _ = sess.Close() }()

	events, lastErr := drainLive(t, seq)

	if !errors.Is(lastErr, errLiveReconnectExhausted) {
		t.Errorf("caller got err %v, want one wrapping errLiveReconnectExhausted", lastErr)
	}
	if !strings.Contains(fmt.Sprint(lastErr), "short-lived connections") {
		t.Errorf("err %q does not name the invocation-wide bound", lastErr)
	}
	// The consecutive budget keeps resetting, so the session must stop on the
	// ceiling: 1 initial dial + maxTotal retries.
	if got, want := connCount.Load(), int32(7); got != want {
		t.Errorf("dialled %d times, want %d", got, want)
	}
	if events == 0 {
		t.Error("caller received no events, so the reset path was never taken")
	}
}

// TestRunLiveReconnectSurvivesATransientDrop guards the other direction: the
// bounds must not kill a session that recovers.
func TestRunLiveReconnectSurvivesATransientDrop(t *testing.T) {
	client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		if connNum == 1 {
			return // one abrupt drop
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(serverContentPong)); err != nil {
			return
		}
		blockUntilClientCloses(conn)
	})
	f := newReconnectFlow(client, testLiveReconnectPolicy(4, 20))
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}

	// Key on content, not on any event: the connection that drops still yields
	// a content-less event for the setup exchange before it dies.
	sawTurn := make(chan struct{})
	var wg sync.WaitGroup
	var streamErr error
	wg.Add(1)
	go func() {
		defer wg.Done()
		once := false
		for ev, err := range seq {
			if err != nil {
				streamErr = err
			}
			if ev != nil && ev.LLMResponse.Content != nil && !once {
				once = true
				close(sawTurn)
			}
		}
	}()
	select {
	case <-sawTurn:
	case <-time.After(drainTimeout):
		t.Fatal("never served a turn after the transient drop")
	}
	_ = sess.Close()
	// Wait for the drain goroutine before asserting, so it cannot log after
	// the test returns and so streamErr is safe to read.
	wg.Wait()

	if streamErr != nil {
		t.Errorf("caller got an error after a recoverable drop: %v", streamErr)
	}
	if got := connCount.Load(); got != 2 {
		t.Errorf("dialled %d times, want 2", got)
	}
}

// TestRunLiveReconnectStopsOnTeardown pins that neither Close nor cancellation
// waits out the backoff, and that neither dials again afterwards.
func TestRunLiveReconnectStopsOnTeardown(t *testing.T) {
	tests := []struct {
		name     string
		teardown func(sess agent.LiveSession, cancel context.CancelFunc)
		wantErr  bool
	}{
		{
			name:     "close",
			teardown: func(sess agent.LiveSession, cancel context.CancelFunc) { _ = sess.Close() },
		},
		{
			name:     "cancel",
			teardown: func(sess agent.LiveSession, cancel context.CancelFunc) { cancel() },
			wantErr:  true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			baseline, _ := runLiveStacks()
			client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {})
			// A backoff far longer than the test, so the flow is certain to be
			// parked in the wait when teardown lands.
			p := testLiveReconnectPolicy(100, 100)
			p.initialBackoff = time.Hour
			p.maxBackoff = time.Hour
			f := newReconnectFlow(client, p)
			ctx, cancel := newLiveInvocationContext(t)
			defer cancel()

			sess, seq, err := f.RunLive(ctx)
			if err != nil {
				t.Fatalf("RunLive failed: %v", err)
			}
			drained := startDrain(seq)
			waitForConns(t, connCount, 1)
			waitForReconnectBackoff(t)
			tc.teardown(sess, cancel)

			_, lastErr := awaitDrain(t, drained)
			if tc.wantErr && !errors.Is(lastErr, context.Canceled) {
				t.Errorf("caller got err %v, want context.Canceled", lastErr)
			}
			if !tc.wantErr && lastErr != nil {
				t.Errorf("Close reported err %v, want none", lastErr)
			}
			if got := connCount.Load(); got != 1 {
				t.Errorf("dialled %d times, want 1: a torn-down session must not redial", got)
			}
			// The iterator ends on Close whether or not the flow goroutine
			// noticed, so the drain above proves nothing on its own. This is
			// what pins the wait as interruptible: an uninterruptible one
			// leaves the goroutine parked for the full hour.
			assertNoRunLiveLeak(t, baseline)
		})
	}
}

// TestRunLiveChargesAFailedRedialToTheBudget covers the path where the endpoint
// stops accepting connections partway through a session. Without this the
// backoff is unreachable in the outage it exists for, because the first failed
// dial would end the session outright.
func TestRunLiveChargesAFailedRedialToTheBudget(t *testing.T) {
	client, connCount, closeServer := startClosableLiveServer(t, func(connNum int, conn *websocket.Conn) {})
	f := newReconnectFlow(client, testLiveReconnectPolicy(3, 20))
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	defer func() { _ = sess.Close() }()

	waitForConns(t, connCount, 1)
	closeServer() // every later dial is refused

	_, lastErr := drainLive(t, seq)
	if !errors.Is(lastErr, errLiveReconnectExhausted) {
		t.Errorf("caller got err %v, want one wrapping errLiveReconnectExhausted: a refused redial must be charged to the budget, not end the session", lastErr)
	}
}

// TestRunLiveFirstConnectFailureIsFatal pins the opposite rule: nothing ever
// connected, so this is bad credentials or an unknown model, and retrying only
// delays a permanent error.
func TestRunLiveFirstConnectFailureIsFatal(t *testing.T) {
	client, connCount, closeServer := startClosableLiveServer(t, func(connNum int, conn *websocket.Conn) {})
	closeServer()

	f := newReconnectFlow(client, testLiveReconnectPolicy(5, 20))
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	defer func() { _ = sess.Close() }()

	_, lastErr := drainLive(t, seq)
	if lastErr == nil {
		t.Fatal("caller got no error for a connection that never succeeded")
	}
	if errors.Is(lastErr, errLiveReconnectExhausted) {
		t.Errorf("err %v means the first dial was retried; it must fail fast", lastErr)
	}
	if got := connCount.Load(); got != 0 {
		t.Errorf("server accepted %d connections, want 0", got)
	}
}

func TestLiveReconnectPolicyNextBackoff(t *testing.T) {
	tests := []struct {
		name string
		p    *liveReconnectPolicy
		in   time.Duration
		want time.Duration
	}{
		{
			name: "doubles",
			p:    &liveReconnectPolicy{maxBackoff: time.Minute},
			in:   250 * time.Millisecond,
			want: 500 * time.Millisecond,
		},
		{
			name: "saturates at the cap",
			p:    &liveReconnectPolicy{maxBackoff: 5 * time.Second},
			in:   4 * time.Second,
			want: 5 * time.Second,
		},
		{
			name: "saturates rather than overshooting the cap",
			p:    &liveReconnectPolicy{maxBackoff: 5 * time.Second},
			in:   3 * time.Second,
			want: 5 * time.Second,
		},
		{
			// Doubling near the top of the range wraps negative, which would
			// fire the next timer at once and restore the hot loop.
			name: "a delay at the top of the range cannot wrap",
			p:    &liveReconnectPolicy{maxBackoff: math.MaxInt64},
			in:   time.Duration(math.MaxInt64 - 1),
			want: time.Duration(math.MaxInt64),
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.p.nextBackoff(tc.in); got != tc.want {
				t.Errorf("nextBackoff(%v) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

func TestLiveReconnectPolicyJittered(t *testing.T) {
	t.Run("stays within the band and varies", func(t *testing.T) {
		p := &liveReconnectPolicy{jitter: 0.2}
		const base = time.Second
		lo, hi := time.Duration(0.8*float64(base)), time.Duration(1.2*float64(base))
		seen := map[time.Duration]bool{}
		for range 200 {
			got := p.jittered(base)
			if got < lo || got > hi {
				t.Fatalf("jittered(%v) = %v, want within [%v, %v]", base, got, lo, hi)
			}
			seen[got] = true
		}
		// Without spread every client that dropped together redials together,
		// which is the behaviour the jitter exists to prevent.
		if len(seen) < 2 {
			t.Error("jittered returned a single value: delays are not being spread")
		}
	})

	degenerate := []struct {
		name string
		p    *liveReconnectPolicy
		in   time.Duration
		want time.Duration
	}{
		{name: "no jitter is exact", p: &liveReconnectPolicy{}, in: time.Second, want: time.Second},
		{name: "zero delay", p: &liveReconnectPolicy{jitter: 0.2}, in: 0, want: 0},
		{name: "negative delay", p: &liveReconnectPolicy{jitter: 0.2}, in: -time.Second, want: 0},
	}
	for _, tc := range degenerate {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.p.jittered(tc.in); got != tc.want {
				t.Errorf("jittered(%v) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

func TestDefaultLiveReconnectPolicy(t *testing.T) {
	p := defaultLiveReconnectPolicy()
	if p.maxAttempts <= 0 || p.maxTotal <= 0 {
		t.Fatalf("policy has no bound: maxAttempts=%d maxTotal=%d", p.maxAttempts, p.maxTotal)
	}
	if p.initialBackoff <= 0 {
		t.Errorf("initialBackoff = %v: a non-positive delay restores the hot loop", p.initialBackoff)
	}
	if p.jitter <= 0 {
		t.Errorf("jitter = %v: without spread every client redials in lockstep", p.jitter)
	}
	// The cap has to be reachable within the consecutive budget, or it is dead
	// configuration that only bites if maxAttempts is later raised.
	d := p.initialBackoff
	for range p.maxAttempts - 1 {
		d = p.nextBackoff(d)
	}
	if d < p.maxBackoff {
		t.Errorf("backoff reaches only %v in %d attempts, so maxBackoff %v is unreachable", d, p.maxAttempts, p.maxBackoff)
	}
	// The worst case a caller waits before learning the session is dead, which
	// is also how long Send can stall, since no writer goroutine exists while
	// the flow is backing off.
	var worst time.Duration
	d = p.initialBackoff
	for range p.maxAttempts {
		worst += d
		d = p.nextBackoff(d)
	}
	if worst > 30*time.Second {
		t.Errorf("worst-case reconnect window is %v; that is how long Send blocks with no way for the caller to bound it", worst)
	}
}

func TestTornDown(t *testing.T) {
	t.Run("open session", func(t *testing.T) {
		if tornDown(t.Context(), newLiveSessionImpl()) {
			t.Error("tornDown = true for a live session and an uncancelled context")
		}
	})
	t.Run("closed session", func(t *testing.T) {
		sess := newLiveSessionImpl()
		_ = sess.Close()
		if !tornDown(t.Context(), sess) {
			t.Error("tornDown = false after Close")
		}
	})
	t.Run("cancelled context", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if !tornDown(ctx, newLiveSessionImpl()) {
			t.Error("tornDown = false after the invocation was cancelled")
		}
	})
}

func TestWaitBeforeReconnect(t *testing.T) {
	t.Run("proceeds when nothing tore down", func(t *testing.T) {
		if !waitBeforeReconnect(t.Context(), newLiveSessionImpl(), time.Millisecond) {
			t.Error("waitBeforeReconnect = false with a live session and an uncancelled context")
		}
	})

	// A teardown that lands as the timer fires leaves the select with two ready
	// cases, and select picks uniformly among them. Without the re-check after
	// the timer this reports "proceed" about half the time and dials a socket
	// nobody reads, so one call proves nothing: repeat until a coin-flip bug
	// cannot survive.
	const flips = 200
	t.Run("refuses a closed session even when the timer is already ready", func(t *testing.T) {
		for i := range flips {
			sess := newLiveSessionImpl()
			_ = sess.Close()
			if waitBeforeReconnect(t.Context(), sess, 0) {
				t.Fatalf("iteration %d: proceeded with a closed session", i)
			}
		}
	})
	t.Run("refuses a cancelled context even when the timer is already ready", func(t *testing.T) {
		for i := range flips {
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			if waitBeforeReconnect(ctx, newLiveSessionImpl(), 0) {
				cancel()
				t.Fatalf("iteration %d: proceeded with a cancelled invocation", i)
			}
			cancel()
		}
	})
}

// TestRunLiveCeilingSparesLongLivedConnections covers the other side of the
// invocation-wide ceiling. The Live API cycles a connection as ordinary
// lifecycle, so counting every reconnect would end a long healthy call once it
// had been cycled maxTotal times.
func TestRunLiveCeilingSparesLongLivedConnections(t *testing.T) {
	const connLifetime = 120 * time.Millisecond
	client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		_ = conn.WriteMessage(websocket.TextMessage, []byte(serverContentPong))
		time.Sleep(connLifetime)
	})
	p := testLiveReconnectPolicy(100, 2)
	p.healthyUptime = connLifetime / 4
	f := newReconnectFlow(client, p)
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	var wg sync.WaitGroup
	var streamErr error
	wg.Add(1)
	go func() {
		defer wg.Done()
		for _, err := range seq {
			if err != nil {
				streamErr = err
			}
		}
	}()

	// Long enough for many more cycles than maxTotal.
	time.Sleep(10 * connLifetime)
	got := connCount.Load()
	_ = sess.Close()
	wg.Wait()

	if streamErr != nil {
		t.Errorf("session ended with %v; connections that served past healthyUptime must not spend the ceiling", streamErr)
	}
	if want := int32(p.maxTotal + 1); got <= want {
		t.Errorf("dialled %d times, want more than %d: the ceiling stopped a healthy session", got, want)
	}
}

// TestRunLiveConsecutiveBudgetSparesASilentSession covers the session the
// consecutive budget must not end: nobody is speaking, so the model sends no
// content, and the Live API cycles the connection as ordinary lifecycle. With
// no content to reset that budget, how long each connection lasts is the only
// thing separating this session from a backend that hangs up on every dial.
func TestRunLiveConsecutiveBudgetSparesASilentSession(t *testing.T) {
	const connLifetime = 120 * time.Millisecond
	client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		// No content on any connection, only the setup exchange and a drop.
		time.Sleep(connLifetime)
	})
	// A ceiling far above the dials this test can reach, so the
	// consecutive-attempt bound is the only one that can fire.
	p := testLiveReconnectPolicy(2, 1000)
	p.healthyUptime = connLifetime / 4
	f := newReconnectFlow(client, p)
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	var wg sync.WaitGroup
	var streamErr error
	wg.Add(1)
	go func() {
		defer wg.Done()
		for _, err := range seq {
			if err != nil {
				streamErr = err
			}
		}
	}()

	// Long enough for many more cycles than maxAttempts.
	time.Sleep(10 * connLifetime)
	got := connCount.Load()
	_ = sess.Close()
	wg.Wait()

	if streamErr != nil {
		t.Errorf("session ended with %v; a connection that served past healthyUptime must not spend the consecutive budget", streamErr)
	}
	if want := int32(p.maxAttempts + 1); got <= want {
		t.Errorf("dialled %d times, want more than %d: the consecutive budget killed a silent session", got, want)
	}
}

// TestRunLiveCeilingIgnoresConsumerLatency pins where a connection's life is
// measured. The consumer loop blocks on the caller between dequeuing the error
// and acting on it, so reading the clock there charges the caller's own
// slowness to the connection: a backend that dies at once then reads as healthy
// and spends none of the ceiling.
//
// The server sends nothing, so each connection produces exactly one event, the
// setup exchange. The reader hands that off and is free again before it meets
// the error, which is what puts the caller's delay after the error is stamped
// rather than before.
func TestRunLiveCeilingIgnoresConsumerLatency(t *testing.T) {
	client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {})
	// maxAttempts well above maxTotal, so the ceiling is what fires.
	p := testLiveReconnectPolicy(100, 3)
	p.healthyUptime = 20 * time.Millisecond
	f := newReconnectFlow(client, p)
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	defer func() { _ = sess.Close() }()

	var wg sync.WaitGroup
	var streamErr error
	wg.Add(1)
	go func() {
		defer wg.Done()
		for _, err := range seq {
			if err != nil {
				streamErr = err
			}
			// Far slower than healthyUptime, which is the whole point.
			time.Sleep(10 * p.healthyUptime)
		}
	}()

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(drainTimeout):
		t.Fatalf("still reconnecting after %v (%d dials): the caller's latency is being credited as connection uptime",
			drainTimeout, connCount.Load())
	}
	// Name the bound, not just the sentinel: with the clock read at the wrong
	// place these connections look healthy, the ceiling never fires, and the
	// session ends far later on the consecutive-attempt bound instead.
	if !strings.Contains(fmt.Sprint(streamErr), "short-lived connections") {
		t.Errorf("session ended with %v, want the short-lived-connection ceiling to fire", streamErr)
	}
}
