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
	"iter"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// drainTimeout bounds every wait for RunLive's iterator to finish. Without a
// deadline a regression that removes the budget hangs these tests until the
// package-wide timeout, which panics across the whole package instead of
// naming the test that broke.
const drainTimeout = 20 * time.Second

// testReconnectPolicy has delays short enough for a test to sit through, with
// jitter off so timing assertions see only scheduling noise.
//
// resetAfter defaults to unreachable: on a loaded machine the delay between
// opening a connection and noticing it died can exceed a small threshold,
// making a dead connection look long-lived. Tests needing the reset set it
// explicitly, with a wide margin.
func testReconnectPolicy(initialDelay time.Duration, maxRetries int) *reconnectPolicy {
	return &reconnectPolicy{
		initialDelay: initialDelay,
		maxDelay:     time.Minute,
		backoff:      2,
		jitter:       0,
		maxRetries:   maxRetries,
		resetAfter:   time.Hour,
	}
}

func newReconnectFlow(client *genai.Client, policy *reconnectPolicy) *Flow {
	return &Flow{
		Model:             &fakeLiveModel{client: client},
		RequestProcessors: []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{liveConfigProcessor},
		reconnect:         policy,
	}
}

// drainAsync consumes seq on its own goroutine, reporting the last error seen.
func drainAsync(seq iter.Seq2[*session.Event, error]) <-chan error {
	drained := make(chan error, 1)
	go func() {
		var last error
		for _, e := range seq {
			if e != nil {
				last = e
			}
		}
		drained <- last
	}()
	return drained
}

// awaitDrain waits for the iterator to finish, failing rather than hanging.
func awaitDrain(t *testing.T, drained <-chan error) error {
	t.Helper()
	select {
	case err := <-drained:
		return err
	case <-time.After(drainTimeout):
		t.Fatal("iterator never drained: the flow is not terminating")
		return nil
	}
}

// waitForConns blocks until the fake server has accepted at least n
// connections, failing the test rather than hanging if it never does.
func waitForConns(t *testing.T, connCount *atomic.Int32, n int32) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for connCount.Load() < n {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %d connection(s); saw %d", n, connCount.Load())
		}
		time.Sleep(time.Millisecond)
	}
}

// startRefusingLiveServer serves a normal Live handshake for the first
// acceptFirst dials and refuses the upgrade after that, so client.Live.Connect
// itself fails — the shape of an outage. The counter counts every dial,
// refused ones included.
func startRefusingLiveServer(t *testing.T, acceptFirst int32, serve func(conn *websocket.Conn)) (*genai.Client, *atomic.Int32) {
	t.Helper()
	var upgrader websocket.Upgrader
	var dials atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if dials.Add(1) > acceptFirst {
			http.Error(w, "backend gone", http.StatusServiceUnavailable)
			return
		}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			t.Errorf("websocket upgrade failed: %v", err)
			return
		}
		defer func() { _ = conn.Close() }()
		if _, _, err := conn.ReadMessage(); err != nil {
			t.Errorf("reading setup frame failed: %v", err)
			return
		}
		if err := conn.WriteMessage(websocket.TextMessage, []byte(`{"setupComplete":{}}`)); err != nil {
			t.Errorf("writing setupComplete failed: %v", err)
			return
		}
		serve(conn)
	}))
	t.Cleanup(ts.Close)

	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		Backend:     genai.BackendGeminiAPI,
		APIKey:      "test-api-key",
		HTTPOptions: genai.HTTPOptions{BaseURL: strings.Replace(ts.URL, "http", "ws", 1)},
	})
	if err != nil {
		t.Fatalf("NewClient failed: %v", err)
	}
	return client, &dials
}

// TestRunLiveReconnectBudget tests that RunLive redials a connection that
// keeps dropping. The test uses an endpoint that completes the handshake then
// hangs up, causing RunLive to redial ~1200 times/sec forever with the caller's
// stream silent. Attempts must now be bounded, paced, and the failure reported.
func TestRunLiveReconnectBudget(t *testing.T) {
	const (
		initialDelay = 40 * time.Millisecond
		maxRetries   = 3
	)
	baseline, _ := runLiveStacks()

	// Hard close right after setupComplete: a resumable 1006, and too
	// short-lived to count as progress.
	client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {})

	f := newReconnectFlow(client, testReconnectPolicy(initialDelay, maxRetries))
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	start := time.Now()
	_, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}

	gotErr := awaitDrain(t, drainAsync(seq))
	elapsed := time.Since(start)

	// Terminating at all is half the fix: before, it never did.
	if gotErr == nil {
		t.Fatal("giving up on a dead endpoint must surface an error; the caller was left with a silent stream")
	}
	if !strings.Contains(gotErr.Error(), "giving up") {
		t.Errorf("surfaced error = %v, want it to report the exhausted reconnect budget", gotErr)
	}
	// 1 initial connection + maxRetries.
	if got, want := connCount.Load(), int32(1+maxRetries); got != want {
		t.Errorf("connection count = %d, want %d", got, want)
	}
	// Delays are 40/80/160ms; finishing sooner would mean a hot loop.
	var wantMin time.Duration
	for i := range maxRetries {
		wantMin += initialDelay << i
	}
	if elapsed < wantMin {
		t.Errorf("run took %v, want at least %v; reconnects are not being paced by the backoff", elapsed, wantMin)
	}
	assertNoRunLiveLeak(t, baseline)
}

// TestRunLiveReconnectDialFailureUsesBudget covers the outage the backoff
// exists for: the redial itself fails because the network is still down. Those
// dials must go on the budget — terminating on the first would leave the
// backoff unreachable in the case it was written for.
func TestRunLiveReconnectDialFailureUsesBudget(t *testing.T) {
	const maxRetries = 3
	baseline, _ := runLiveStacks()

	// Dial 1 connects and dies; every later dial is refused outright.
	client, dials := startRefusingLiveServer(t, 1, func(conn *websocket.Conn) {})

	f := newReconnectFlow(client, testReconnectPolicy(time.Millisecond, maxRetries))
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	_, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}

	gotErr := awaitDrain(t, drainAsync(seq))
	if gotErr == nil || !strings.Contains(gotErr.Error(), "giving up") {
		t.Errorf("error = %v, want the exhausted reconnect budget; a failed redial must not end the session outright", gotErr)
	}
	if got, want := dials.Load(), int32(1+maxRetries); got != want {
		t.Errorf("dial count = %d, want %d", got, want)
	}
	assertNoRunLiveLeak(t, baseline)
}

// TestRunLiveFirstDialFailureIsFatal is the other side of that rule: with
// nothing ever connected, a failed dial is bad credentials or an unknown
// model, not a blip. Retrying those only delays a permanent error.
func TestRunLiveFirstDialFailureIsFatal(t *testing.T) {
	baseline, _ := runLiveStacks()
	client, dials := startRefusingLiveServer(t, 0, func(conn *websocket.Conn) {})

	f := newReconnectFlow(client, testReconnectPolicy(time.Millisecond, 5))
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	_, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}

	gotErr := awaitDrain(t, drainAsync(seq))
	if gotErr == nil || !strings.Contains(gotErr.Error(), "failed to connect live session") {
		t.Fatalf("error = %v, want the connect failure reported directly", gotErr)
	}
	if strings.Contains(gotErr.Error(), "giving up") {
		t.Errorf("error = %v, want no retries burned on a connection that never worked", gotErr)
	}
	if got := dials.Load(); got != 1 {
		t.Errorf("dial count = %d, want 1", got)
	}
	assertNoRunLiveLeak(t, baseline)
}

// TestRunLiveReconnectBudgetSurvivesLongLivedSession is the other direction: a
// connection that stayed up must restore the budget.
//
// It asserts the exact total rather than just "kept going", which would also
// hold for the unbounded pre-fix loop.
//
// The idle subtest is the important one — gating the reset on model output as
// well as lifetime would never reset a session where the user is silent, so a
// few ordinary GoAways would kill a healthy call.
func TestRunLiveReconnectBudgetSurvivesLongLivedSession(t *testing.T) {
	const (
		maxRetries = 2
		// 1 long-lived + 1 immediate reconnect once the budget resets + maxRetries.
		wantConns = 2 + maxRetries

		// The exact count holds only if exactly one connection counts as
		// progress. resetAfter sits far above the cost of noticing a hard close
		// (sub-millisecond on loopback) and far below the long connection's
		// lifetime: a loaded machine that took longer than resetAfter to notice
		// a dead socket would score it healthy and re-arm the budget again.
		resetAfter   = 500 * time.Millisecond
		longLifetime = 3 * resetAfter
	)
	tests := []struct {
		name    string
		firstUp func(conn *websocket.Conn)
	}{
		{
			name: "connection delivers output",
			firstUp: func(conn *websocket.Conn) {
				_ = conn.WriteMessage(websocket.TextMessage, []byte(serverContentPong))
				time.Sleep(longLifetime)
			},
		},
		{
			// No model output at all.
			name:    "connection is idle",
			firstUp: func(conn *websocket.Conn) { time.Sleep(longLifetime) },
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			baseline, _ := runLiveStacks()
			// Connection 1 outlives resetAfter; everything after it dies at
			// once, so only the first can restore the budget.
			client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
				if connNum == 1 {
					tc.firstUp(conn)
				}
			})

			policy := testReconnectPolicy(time.Millisecond, maxRetries)
			policy.resetAfter = resetAfter
			f := newReconnectFlow(client, policy)
			ctx, cancel := newLiveInvocationContext(t)
			defer cancel()

			_, seq, err := f.RunLive(ctx)
			if err != nil {
				t.Fatalf("RunLive failed: %v", err)
			}

			gotErr := awaitDrain(t, drainAsync(seq))
			if gotErr == nil || !strings.Contains(gotErr.Error(), "giving up") {
				t.Errorf("error = %v, want the exhausted reconnect budget", gotErr)
			}
			// Fewer means the long-lived connection did not restore the
			// budget; more means it never re-armed and the loop is unbounded.
			if got := connCount.Load(); got != wantConns {
				t.Errorf("connection count = %d, want %d", got, wantConns)
			}
			assertNoRunLiveLeak(t, baseline)
		})
	}
}

// TestRunLiveReconnectBudgetBoundsCrashLoop is why the reset is gated on
// lifetime: a backend that accepts and dies immediately made no progress and
// must not hand the budget back. Without the gate this never terminated.
func TestRunLiveReconnectBudgetBoundsCrashLoop(t *testing.T) {
	const maxRetries = 3

	tests := []struct {
		name  string
		serve func(connNum int, conn *websocket.Conn)
	}{
		{
			name:  "dies right after the handshake",
			serve: func(connNum int, conn *websocket.Conn) {},
		},
		{
			// Real output must still not reset the budget if the connection
			// dies immediately.
			name: "serves one turn then dies",
			serve: func(connNum int, conn *websocket.Conn) {
				_ = conn.WriteMessage(websocket.TextMessage, []byte(serverContentPong))
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			baseline, _ := runLiveStacks()
			client, connCount := startFakeLiveServer(t, tc.serve)

			// The default resetAfter is unreachable, so nothing counts as
			// progress.
			f := newReconnectFlow(client, testReconnectPolicy(time.Millisecond, maxRetries))
			ctx, cancel := newLiveInvocationContext(t)
			defer cancel()

			_, seq, err := f.RunLive(ctx)
			if err != nil {
				t.Fatalf("RunLive failed: %v", err)
			}

			gotErr := awaitDrain(t, drainAsync(seq))
			if gotErr == nil || !strings.Contains(gotErr.Error(), "giving up") {
				t.Errorf("error = %v, want the exhausted reconnect budget", gotErr)
			}
			if got, want := connCount.Load(), int32(1+maxRetries); got != want {
				t.Errorf("connection count = %d, want %d", got, want)
			}
			assertNoRunLiveLeak(t, baseline)
		})
	}
}

// TestRunLiveUsesDefaultPolicyWhenUnset covers the only configuration that
// ships: a Flow built without a reconnect policy. Every other test here injects
// one, so without this the production path never runs through a retry.
func TestRunLiveUsesDefaultPolicyWhenUnset(t *testing.T) {
	baseline, _ := runLiveStacks()
	// Connection 1 dies immediately; connection 2 is held open, so exactly one
	// default-paced backoff separates the two timestamps.
	connAt := make(chan time.Time, 4)
	client, _ := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		select {
		case connAt <- time.Now():
		default:
		}
		if connNum > 1 {
			blockUntilClientCloses(conn)
		}
	})

	f := &Flow{
		Model:             &fakeLiveModel{client: client},
		RequestProcessors: []func(ctx agent.InvocationContext, req *model.LLMRequest, f *Flow) iter.Seq2[*session.Event, error]{liveConfigProcessor},
	}
	if f.reconnect != nil {
		t.Fatal("this test exists to exercise the nil-policy path")
	}
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	_, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	drained := drainAsync(seq)

	// Timed dial to dial: measuring from RunLive would fold the first dial's
	// own latency into the gap, so a slow machine could satisfy the bound with
	// no backoff at all.
	var first, second time.Time
	for _, dst := range []*time.Time{&first, &second} {
		select {
		case *dst = <-connAt:
		case <-time.After(drainTimeout):
			t.Fatal("timed out waiting for the reconnect")
		}
	}

	// Tear down with cancel, not Close: cancellation is what the consumer loop
	// selects on, so it releases a connection that is up and idle.
	cancel()
	_ = awaitDrain(t, drained)

	// delay(1) with the shipped defaults is initialDelay ±jitter, so its floor
	// is initialDelay*(1-jitter). A nil policy that silently paced at zero
	// would reconnect far sooner than this.
	wantMin := time.Duration(float64(defaultReconnectInitialDelay) * (1 - defaultReconnectJitter))
	if gap := second.Sub(first); gap < wantMin {
		t.Errorf("reconnected %v after the first connection, want at least %v; the default policy is not being applied", gap, wantMin)
	}
	assertNoRunLiveLeak(t, baseline)
}

// TestRunLiveNoDialAfterTeardownOnResetPath covers the reconnect that skips the
// backoff: a restored budget means no wait, so waitBeforeReconnect's teardown
// checks are never reached. Without the loop-top check a closed session still
// gets dialled, and genai's dialler ignores the context.
func TestRunLiveNoDialAfterTeardownOnResetPath(t *testing.T) {
	baseline, _ := runLiveStacks()
	// The server holds the connection until the test releases it, so "Close
	// arrives while still connected" is ordered by the handshake rather than by
	// a sleep, which a loaded machine could reorder.
	connUp := make(chan struct{}, 1)
	release := make(chan struct{})
	client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {
		select {
		case connUp <- struct{}{}:
		default:
		}
		<-release
	})

	policy := testReconnectPolicy(time.Hour, 5) // a delay this long would be obvious
	// Positive but unreachably small, so every connection counts as progress
	// and attempt stays 0. Zero would mean "never reset" (see madeProgress).
	policy.resetAfter = time.Nanosecond
	f := newReconnectFlow(client, policy)
	ctx, cancel := newLiveInvocationContext(t)
	defer cancel()

	sess, seq, err := f.RunLive(ctx)
	if err != nil {
		t.Fatalf("RunLive failed: %v", err)
	}
	drained := drainAsync(seq)

	<-connUp         // connection 1 is established
	_ = sess.Close() // close while it is still up
	close(release)   // only now let the server drop it

	_ = awaitDrain(t, drained)
	time.Sleep(300 * time.Millisecond) // give any stray dial time to land
	if got := connCount.Load(); got != 1 {
		t.Errorf("connection count = %d, want 1; a websocket was dialled after the session was closed", got)
	}
	assertNoRunLiveLeak(t, baseline)
}

// TestRunLiveReconnectBackoffIsInterruptible pins teardown latency: a backoff
// must not delay Close or cancellation.
func TestRunLiveReconnectBackoffIsInterruptible(t *testing.T) {
	// Long enough that finishing the wait is distinct from interrupting it.
	const backoff = 30 * time.Second
	// Teardown is a handful of channel operations. Generous for a loaded
	// machine, still far below the backoff, so this measures promptness rather
	// than merely "not 30s".
	const wantWithin = 2 * time.Second

	tests := []struct {
		name string
		stop func(sess agent.LiveSession, cancel func())
		// wantErr is a substring of the error the caller should observe, or
		// empty if the teardown should stay silent.
		wantErr string
	}{
		{
			name:    "Close during backoff",
			stop:    func(sess agent.LiveSession, cancel func()) { _ = sess.Close() },
			wantErr: "", // an explicit Close is a clean teardown, not a failure
		},
		{
			name:    "context cancel during backoff",
			stop:    func(sess agent.LiveSession, cancel func()) { cancel() },
			wantErr: "context canceled",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			baseline, _ := runLiveStacks()
			client, connCount := startFakeLiveServer(t, func(connNum int, conn *websocket.Conn) {})

			// The unreachable default forces the backoff path, not the reset
			// path.
			f := newReconnectFlow(client, testReconnectPolicy(backoff, 5))
			ctx, cancel := newLiveInvocationContext(t)
			defer cancel()

			sess, seq, err := f.RunLive(ctx)
			if err != nil {
				t.Fatalf("RunLive failed: %v", err)
			}
			drained := drainAsync(seq)

			waitForConns(t, connCount, 1)
			time.Sleep(200 * time.Millisecond)

			stopped := time.Now()
			tc.stop(sess, cancel)

			select {
			case gotErr := <-drained:
				if took := time.Since(stopped); took > wantWithin {
					t.Errorf("iterator drained %v after teardown, want within %v; the backoff sleep is not interruptible", took, wantWithin)
				}
				switch {
				case tc.wantErr == "" && gotErr != nil:
					t.Errorf("got error %v, want a silent teardown", gotErr)
				case tc.wantErr != "" && (gotErr == nil || !strings.Contains(gotErr.Error(), tc.wantErr)):
					t.Errorf("got error %v, want one containing %q", gotErr, tc.wantErr)
				}
			case <-time.After(backoff / 2):
				t.Fatal("iterator never drained: teardown is blocked on the backoff sleep")
			}
			if got := connCount.Load(); got != 1 {
				t.Errorf("connection count = %d, want 1 (no reconnect after teardown)", got)
			}
			assertNoRunLiveLeak(t, baseline)
		})
	}
}

// TestWaitBeforeReconnectPrefersTeardown covers what a test can decide: a wait
// entered on an already-dead session must not report "proceed".
//
// The post-timer re-check is deliberately not tested — it needs two select
// cases ready within nanoseconds of each other, which cannot be provoked from
// outside. TestRunLiveNoDialAfterTeardownOnResetPath covers the ordinary path.
func TestWaitBeforeReconnectPrefersTeardown(t *testing.T) {
	t.Run("already torn down", func(t *testing.T) {
		ctx, cancel := newLiveInvocationContext(t)
		defer cancel()
		sess := newLiveSessionImpl()
		_ = sess.Close()

		// A delay this long proves the answer came from the closed session,
		// not from waiting the timer out.
		start := time.Now()
		if waitBeforeReconnect(ctx, sess, time.Hour) {
			t.Fatal("said proceed with the session already closed")
		}
		if took := time.Since(start); took > 5*time.Second {
			t.Errorf("returned after %v; teardown is not short-circuiting the wait", took)
		}
	})
}

func TestReconnectPolicyMadeProgress(t *testing.T) {
	tests := []struct {
		name       string
		resetAfter time.Duration
		lifetime   time.Duration
		want       bool
	}{
		{name: "outlived the threshold", resetAfter: time.Second, lifetime: 2 * time.Second, want: true},
		{name: "exactly the threshold", resetAfter: time.Second, lifetime: time.Second, want: true},
		{name: "died early", resetAfter: time.Second, lifetime: time.Millisecond, want: false},
		{name: "unset resetAfter never resets", resetAfter: 0, lifetime: time.Hour, want: false},
		{name: "negative resetAfter never resets", resetAfter: -time.Second, lifetime: time.Hour, want: false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p := &reconnectPolicy{resetAfter: tc.resetAfter}
			if got := p.madeProgress(tc.lifetime); got != tc.want {
				t.Errorf("madeProgress(%v) with resetAfter %v = %v, want %v", tc.lifetime, tc.resetAfter, got, tc.want)
			}
		})
	}
}

func TestReconnectPolicyDelay(t *testing.T) {
	// Jitter off so the growth and the cap can be asserted exactly.
	p := &reconnectPolicy{
		initialDelay: 100 * time.Millisecond,
		maxDelay:     time.Second,
		backoff:      2,
		jitter:       0,
		maxRetries:   10,
	}
	tests := []struct {
		attempt int
		want    time.Duration
	}{
		{attempt: -1, want: 0},
		{attempt: 0, want: 0},
		{attempt: 1, want: 100 * time.Millisecond},
		{attempt: 2, want: 200 * time.Millisecond},
		{attempt: 3, want: 400 * time.Millisecond},
		{attempt: 4, want: 800 * time.Millisecond},
		{attempt: 5, want: time.Second}, // capped
		{attempt: 6, want: time.Second},
		// A long outage must not overflow the doubling into a negative or
		// absurd duration.
		{attempt: 5000, want: time.Second},
	}
	for _, tc := range tests {
		if got := p.delay(tc.attempt); got != tc.want {
			t.Errorf("delay(%d) = %v, want %v", tc.attempt, got, tc.want)
		}
	}

	// An initialDelay above the cap must still be capped, even though attempt 1
	// skips the doubling loop entirely.
	over := &reconnectPolicy{initialDelay: time.Minute, maxDelay: time.Second, backoff: 2, maxRetries: 3}
	if got := over.delay(1); got != time.Second {
		t.Errorf("delay(1) with initialDelay > maxDelay = %v, want %v", got, time.Second)
	}
}

// TestReconnectPolicyDelayDegenerateFields pins what an unset or nonsensical
// field means. Several used to yield 0 or a negative duration, which fires a
// timer immediately: a policy that looks configured but paces nothing.
//
// The answers match workflow.CalculateDelay, which cannot be called from here
// (workflow imports this package) but must not drift from it.
func TestReconnectPolicyDelayDegenerateFields(t *testing.T) {
	tests := []struct {
		name    string
		p       *reconnectPolicy
		attempt int
		want    time.Duration
	}{
		{
			// Not "cap at zero", which would remove all pacing.
			name:    "maxDelay unset means no cap",
			p:       &reconnectPolicy{initialDelay: 100 * time.Millisecond, backoff: 2},
			attempt: 3,
			want:    400 * time.Millisecond,
		},
		{
			name:    "maxDelay negative means no cap",
			p:       &reconnectPolicy{initialDelay: 100 * time.Millisecond, maxDelay: -time.Second, backoff: 2},
			attempt: 3,
			want:    400 * time.Millisecond,
		},
		{
			// Not zero, which would collapse every retry after the first.
			name:    "backoff unset means constant delay",
			p:       &reconnectPolicy{initialDelay: 100 * time.Millisecond, maxDelay: time.Second},
			attempt: 4,
			want:    100 * time.Millisecond,
		},
		{
			name:    "backoff negative means constant delay",
			p:       &reconnectPolicy{initialDelay: 100 * time.Millisecond, maxDelay: time.Second, backoff: -2},
			attempt: 4,
			want:    100 * time.Millisecond,
		},
		{
			name:    "negative initialDelay clamps to zero",
			p:       &reconnectPolicy{initialDelay: -time.Second, maxDelay: time.Second, backoff: 2},
			attempt: 1,
			want:    0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.p.delay(tc.attempt); got != tc.want {
				t.Errorf("delay(%d) = %v, want %v", tc.attempt, got, tc.want)
			}
		})
	}

	// No field combination may produce a negative wait: time.NewTimer fires a
	// non-positive duration immediately, which is a hot loop wearing a policy.
	for _, p := range []*reconnectPolicy{
		{initialDelay: -time.Hour, maxDelay: -time.Hour, backoff: -2, jitter: -1},
		{initialDelay: -time.Hour, maxDelay: time.Second, backoff: 2, jitter: 0.5},
		{initialDelay: time.Second, maxDelay: -time.Second, backoff: 2, jitter: 0.5},
		{initialDelay: time.Second, maxDelay: time.Second, backoff: math.NaN(), jitter: 0.2},
		{initialDelay: time.Second, maxDelay: time.Second, backoff: math.Inf(1), jitter: 0.2},
		{initialDelay: time.Second, maxDelay: time.Second, backoff: 2, jitter: math.NaN()},
	} {
		for a := 1; a <= 12; a++ {
			if got := p.delay(a); got < 0 {
				t.Errorf("delay(%d) = %v for %+v, want a non-negative duration", a, got, p)
			}
		}
	}
}

func TestReconnectPolicyDelayJitter(t *testing.T) {
	p := &reconnectPolicy{initialDelay: time.Second, maxDelay: time.Minute, backoff: 2, jitter: 0.2, maxRetries: 5}
	low, high := time.Duration(0.8*float64(time.Second)), time.Duration(1.2*float64(time.Second))
	seen := make(map[time.Duration]bool)
	for range 500 {
		d := p.delay(1)
		if d < low || d > high {
			t.Fatalf("delay(1) = %v, want within ±20%% of %v", d, time.Second)
		}
		seen[d] = true
	}
	if len(seen) < 2 {
		t.Error("delay is constant across 500 calls: jitter is not being applied")
	}

	// At the cap the delay must stay spread: clamping jittered values to
	// maxDelay would pile half of them onto it exactly, so half a fleet would
	// retry on the same millisecond.
	capped := &reconnectPolicy{initialDelay: time.Second, maxDelay: time.Second, backoff: 2, jitter: 0.2, maxRetries: 10}
	atCap := 0
	spread := make(map[time.Duration]bool)
	const n = 2000
	for range n {
		d := capped.delay(6) // deep enough that the nominal delay is saturated
		if d > time.Second {
			t.Fatalf("delay = %v, want <= maxDelay %v", d, time.Second)
		}
		if d == time.Second {
			atCap++
		}
		spread[d] = true
	}
	if frac := float64(atCap) / n; frac > 0.05 {
		t.Errorf("%.1f%% of saturated delays landed on exactly maxDelay, want a spread; "+
			"jitter is being clamped into a point mass at the cap", frac*100)
	}
	// Counting only exact-cap hits would also pass for a constant delay just
	// below the cap, so assert the saturated values are genuinely spread.
	if len(spread) < n/10 {
		t.Errorf("saturated delay took only %d distinct values across %d calls, want a spread", len(spread), n)
	}

	// jitter <= 0 means no jitter, matching workflow.CalculateDelay.
	off := &reconnectPolicy{initialDelay: time.Second, maxDelay: time.Minute, backoff: 2, jitter: 0, maxRetries: 5}
	for range 100 {
		if d := off.delay(1); d != time.Second {
			t.Fatalf("delay(1) with jitter 0 = %v, want exactly %v", d, time.Second)
		}
	}
}

func TestDefaultReconnectPolicy(t *testing.T) {
	p := defaultReconnectPolicy()

	// The cap must bind, or it is dead configuration. Compared against
	// uncapped growth so jitter cannot affect the answer.
	uncapped := float64(p.initialDelay)
	for range p.maxRetries - 1 {
		uncapped *= p.backoff
	}
	if time.Duration(uncapped) <= p.maxDelay {
		t.Errorf("uncapped growth reaches only %v against maxDelay %v; the cap can never bind",
			time.Duration(uncapped), p.maxDelay)
	}
	// maxDelay also bounds how long a backoff can stall Send, so it stays in
	// seconds even though a longer one would ride out a longer outage.
	if p.maxDelay > 10*time.Second {
		t.Errorf("maxDelay = %v; a caller's Send can be stalled that long, keep it short", p.maxDelay)
	}
	if p.resetAfter <= 0 {
		t.Error("resetAfter must be positive, or a crash loop restores the budget on every attempt")
	}
	// Never exceed the cap, jitter included.
	for a := 1; a <= p.maxRetries; a++ {
		for range 200 {
			if d := p.delay(a); d > p.maxDelay {
				t.Fatalf("delay(%d) = %v exceeds maxDelay %v", a, d, p.maxDelay)
			}
		}
	}

	// Pin the window from both ends: long enough to ride out a network handover
	// or a backend drain, short enough that a caller learns a dead endpoint is
	// dead. Bounding the hot-loop fix any tighter trades one failure mode for
	// another.
	var window time.Duration
	for a := 1; a <= p.maxRetries; a++ {
		// delay() is jittered, so sample the worst case rather than one draw.
		var worst time.Duration
		for range 200 {
			worst = max(worst, p.delay(a))
		}
		window += worst
	}
	if window < 15*time.Second {
		t.Errorf("total reconnect window is only %v; a transient outage would kill the session", window)
	}
	if window > 60*time.Second {
		t.Errorf("total reconnect window is %v; a caller waits that long to learn the session is dead", window)
	}
}
