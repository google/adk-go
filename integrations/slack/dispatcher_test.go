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

package slack

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/v2/session"

	"google.golang.org/adk/integrations/slack/internal/slackapitest"
)

func TestKeyedLocks(t *testing.T) {
	var k keyedLocks
	k.lock("a")
	acquired := make(chan string, 2)
	go func() { k.lock("a"); acquired <- "a"; k.unlock("a") }()
	go func() { k.lock("b"); acquired <- "b"; k.unlock("b") }()
	select {
	case key := <-acquired:
		if key != "b" {
			t.Fatalf("lock(%q) acquired while held", key)
		}
	case <-time.After(time.Second):
		t.Fatal("lock(\"b\") blocked behind another key")
	}
	select {
	case <-acquired:
		t.Fatal("lock(\"a\") acquired while held")
	case <-time.After(20 * time.Millisecond):
	}
	k.unlock("a")
	select {
	case <-acquired:
	case <-time.After(time.Second):
		t.Fatal("lock(\"a\") not acquired after unlock")
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	if len(k.locks) != 0 {
		t.Errorf("%d locks left after every unlock, want 0", len(k.locks))
	}
}

func TestStripOwnMention(t *testing.T) {
	tests := []struct{ text, bot, want string }{
		{"<@UBOT> hello <@UOTHER>", "UBOT", "hello <@UOTHER>"},
		{"hi <@UBOT>", "UBOT", "hi"},
		{"<@UBOT>", "UBOT", ""},
		{" <@UBOT> hi ", "", "<@UBOT> hi"},
	}
	for _, tt := range tests {
		if got := stripOwnMention(tt.text, tt.bot); got != tt.want {
			t.Errorf("stripOwnMention(%q, %q) = %q, want %q", tt.text, tt.bot, got, tt.want)
		}
	}
}

func TestSameThreadTurnsRunOneAtATime(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.MaxConcurrentRuns = 2 })
	e.backend.block = true
	const root = "1700000000.000100"
	e.post(mention("E1", root, "one"))
	c1 := e.waitCall(t)
	e.post(mention("E2", "1700000000.000200", "two", "thread_ts", root))
	e.noCall(t, 100*time.Millisecond)
	close(c1.release)
	c2 := e.waitCall(t)
	close(c2.release)
	if c2.sessionID != c1.sessionID {
		t.Errorf("second turn ran in session %s, want %s", c2.sessionID, c1.sessionID)
	}
}

func TestDifferentThreadsOverlap(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.MaxConcurrentRuns = 2 })
	e.backend.block = true
	e.post(mention("E1", "1700000000.000100", "one"))
	e.post(mention("E2", "1700000000.000200", "two"))
	c1 := e.waitCall(t)
	c2 := e.waitCall(t)
	close(c1.release)
	close(c2.release)
}

func TestMaxConcurrentRuns(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.MaxConcurrentRuns = 2; c.MaxQueued = 1 })
	e.backend.block = true
	// The first two sends wait for their turns to start, so the queue holds
	// exactly the third.
	e.post(mention("E1", "1700000000.000001", "one"))
	c1 := e.waitCall(t)
	e.post(mention("E2", "1700000000.000002", "two"))
	c2 := e.waitCall(t)
	e.post(mention("E3", "1700000000.000003", "three"))
	e.noCall(t, 100*time.Millisecond)

	e.post(mention("E4", "1700000000.000004", "four"))
	if !strings.Contains(e.log.String(), "reason=queue_full") || !strings.Contains(e.log.String(), "event_id=E4") {
		t.Errorf("log has no queue_full drop of E4:\n%s", e.log.String())
	}

	close(c1.release)
	c3 := e.waitCall(t)
	if c3.text != "three" {
		t.Errorf("third turn is session %s, want the third event", c3.sessionID)
	}
	close(c2.release)
	close(c3.release)
	e.drain(t)
	if n := e.backend.count(); n != 3 {
		t.Errorf("backend called %d times, want 3", n)
	}
}

// The event ID is recorded before the turn, so a retry that arrives while
// the turn is still running is dropped.
func TestDedupDuringTurn(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.MaxConcurrentRuns = 2 })
	e.backend.block = true
	body := mention("E1", "1700000000.000100", "hi")
	e.post(body)
	c := e.waitCall(t)
	e.post(body)
	e.noCall(t, 100*time.Millisecond)
	close(c.release)
	e.drain(t)
	if n := e.backend.count(); n != 1 {
		t.Errorf("backend called %d times, want 1", n)
	}
}

// Slack sends a channel mention as app_mention and as message, with
// different event IDs.
func TestMentionDeliveredTwiceRunsOnce(t *testing.T) {
	e := newTestService(t)
	const ts = "1700000000.000100"
	e.post(callback("E1", slackEvent("app_mention", "C1", ts, "<@UBOT> hi")))
	e.post(callback("E2", slackEvent("message", "C1", ts, "<@UBOT> hi")))
	close(e.waitCall(t).release)
	e.flush(t)
	if n := e.backend.count(); n != 2 {
		t.Errorf("backend called %d times, want 2 (the mention and the sentinel)", n)
	}
}

func TestTurnTimeout(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.TurnTimeout = 50 * time.Millisecond; c.ErrorMessage = "turn failed" })
	e.backend.block = true
	e.post(mention("E1", "1700000000.000100", "hi"))
	c := e.waitCall(t)
	e.waitAPI(t, hasText("turn failed"))
	if c.ctx.Err() == nil {
		t.Error("backend ctx not done after the timeout")
	}
}

func TestTurnPanic(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.ErrorMessage = "turn failed"; c.NoPlaceholder = true })
	e.backend.panicText = "boom"
	e.backend.events = []*session.Event{textEvent("ok")}
	e.post(mention("E1", "1700000000.000100", "boom"))
	e.post(mention("E2", "1700000000.000200", "fine"))
	// The panicked turn's thread is usable again.
	e.post(mention("E3", "1700000000.000300", "again", "thread_ts", "1700000000.000100"))
	for range 3 {
		e.waitCall(t)
	}
	e.drain(t)
	want := []slackapitest.Call{
		{Method: "chat.postMessage", Channel: "C1", ThreadTS: "1700000000.000100", TS: ts(1), Text: "turn failed"},
		{Method: "chat.postMessage", Channel: "C1", ThreadTS: "1700000000.000200", TS: ts(2), Text: "ok"},
		{Method: "chat.postMessage", Channel: "C1", ThreadTS: "1700000000.000100", TS: ts(3), Text: "ok"},
	}
	if got := e.api.Snapshot(); !slices.Equal(got, want) {
		t.Errorf("calls =\n%+v\nwant\n%+v", got, want)
	}
	for _, want := range []string{"level=ERROR", `msg="slack: turn panicked" stage=turn`, "event_id=E1", "panic=string", "stack="} {
		if !strings.Contains(e.log.String(), want) {
			t.Errorf("log has no %q:\n%s", want, e.log.String())
		}
	}
	if strings.Contains(e.log.String(), "backend exploded") {
		t.Error("log contains the panic value, which may carry user text")
	}
}

func TestCloseCancelsRunningTurns(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.ErrorMessage = "turn failed" })
	e.backend.block = true
	e.post(mention("E1", "1700000000.000100", "hi"))
	c := e.waitCall(t)

	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	err := e.svc.Close(ctx)
	if d := time.Since(start); d > time.Second {
		t.Errorf("Close took %s, want under 1s", d)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Close() = %v, want context.DeadlineExceeded", err)
	}
	if c.ctx.Err() == nil {
		t.Error("backend ctx not done after Close")
	}
	if !hasText("turn failed")(e.api.Snapshot()) {
		t.Errorf("error message not delivered: %+v", e.api.Snapshot())
	}
}

// A worker that finishes its turn after Close must not start a queued one,
// nor run the Policy for it.
func TestCloseDropsQueuedTurns(t *testing.T) {
	var policyCalls atomic.Int32
	e := newTestService(t, func(c *Config) {
		c.MaxQueued = 10
		c.Policy = func(context.Context, *Message) (Decision, error) {
			policyCalls.Add(1)
			return Decision{Allow: true}, nil
		}
	})
	e.backend.block = true
	e.post(mention("E0", "1700000000.000100", "running"))
	c := e.waitCall(t)
	for i := range 8 {
		e.post(mention(fmt.Sprintf("E%d", i+1), fmt.Sprintf("1700000000.00020%d", i), "queued"))
	}
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	closed := make(chan error, 1)
	go func() { closed <- e.svc.Close(ctx) }()
	// Close has closed stopCh once Submit refuses.
	deadline := time.Now().Add(time.Second)
	for e.post(dm("Eprobe", "1700000000.000300", "probe")).Code != http.StatusServiceUnavailable {
		if time.Now().After(deadline) {
			t.Fatal("Submit still accepts 1s after Close")
		}
		time.Sleep(time.Millisecond)
	}
	close(c.release)
	if err := <-closed; err != nil {
		t.Fatalf("Close() = %v", err)
	}
	if n := e.backend.count(); n != 1 {
		t.Errorf("backend called %d times, want 1: queued turns ran after Close", n)
	}
	if n := policyCalls.Load(); n != 1 {
		t.Errorf("Policy called %d times, want 1: queued turns were gated after Close", n)
	}
}

// A worker already waiting for the thread's lock when Close is called must
// not start its turn once the lock is free.
func TestCloseDropsTurnWaitingForThreadLock(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.MaxConcurrentRuns = 2 })
	e.backend.block = true
	const root = "1700000000.000100"
	e.post(mention("E1", root, "one"))
	c := e.waitCall(t)
	e.post(mention("E2", "1700000000.000200", "two", "thread_ts", root))
	waitLockWaiters(t, e.svc.disp, c.sessionID, 2)

	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	closed := make(chan error, 1)
	go func() { closed <- e.svc.Close(ctx) }()
	deadline := time.Now().Add(time.Second)
	for e.post(dm("Eprobe", "1700000000.000300", "probe")).Code != http.StatusServiceUnavailable {
		if time.Now().After(deadline) {
			t.Fatal("Submit still accepts 1s after Close")
		}
		time.Sleep(time.Millisecond)
	}
	close(c.release)
	e.noCall(t, 200*time.Millisecond)
	if err := <-closed; err != nil {
		t.Errorf("Close() = %v", err)
	}
	e.waitLog(t, "reason=shutdown")
}

func TestCloseWaitsForRunningTurns(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.ErrorMessage = "turn failed" })
	e.backend.block = true
	e.backend.events = []*session.Event{textEvent("done")}
	e.post(mention("E1", "1700000000.000100", "hi"))
	c := e.waitCall(t)
	go func() {
		time.Sleep(20 * time.Millisecond)
		close(c.release)
	}()
	e.drain(t)
	if !hasText("done")(e.api.Snapshot()) || hasText("turn failed")(e.api.Snapshot()) {
		t.Errorf("calls = %+v, want the reply and no error message", e.api.Snapshot())
	}
	// A second Close returns at once.
	e.drain(t)
}

func TestPolicyPanic(t *testing.T) {
	e := newTestService(t, func(c *Config) {
		c.Policy = func(_ context.Context, m *Message) (Decision, error) {
			if strings.Contains(m.Text, "boom") {
				panic("policy exploded")
			}
			return Decision{Allow: true}, nil
		}
	})
	e.post(mention("E1", "1700000000.000100", "boom"))
	e.post(mention("E2", "1700000000.000200", "fine"))
	c := e.waitCall(t)
	close(c.release)
	if c.text != "fine" {
		t.Errorf("turn ran for session %s, want the second event", c.sessionID)
	}
	e.drain(t)
	for _, call := range e.api.Snapshot() {
		if call.ThreadTS == "1700000000.000100" || call.TS == "1700000000.000100" {
			t.Errorf("refused message got Slack call %+v, want none", call)
		}
	}
	for _, want := range []string{"level=ERROR", `msg="slack: turn panicked" stage=policy`, "event_id=E1", "panic=string", "stack="} {
		if !strings.Contains(e.log.String(), want) {
			t.Errorf("log has no %q:\n%s", want, e.log.String())
		}
	}
	if strings.Contains(e.log.String(), "policy exploded") {
		t.Error("log contains the panic value")
	}
}

func TestPolicyHangEndsAtTurnTimeout(t *testing.T) {
	e := newTestService(t, func(c *Config) {
		c.TurnTimeout = 50 * time.Millisecond
		c.Policy = func(ctx context.Context, m *Message) (Decision, error) {
			if strings.Contains(m.Text, "hang") {
				<-ctx.Done()
				return Decision{}, ctx.Err()
			}
			return Decision{Allow: true}, nil
		}
	})
	e.post(mention("E1", "1700000000.000100", "hang"))
	e.post(mention("E2", "1700000000.000200", "fine"))
	c := e.waitCall(t)
	close(c.release)
	if c.text != "fine" {
		t.Errorf("turn ran for session %s, want the second event", c.sessionID)
	}
	e.waitLog(t, "reason="+ReasonPolicyError)
}

func TestPolicyHangEndsAtClose(t *testing.T) {
	entered := make(chan struct{})
	e := newTestService(t, func(c *Config) {
		c.Policy = func(ctx context.Context, m *Message) (Decision, error) {
			close(entered)
			<-ctx.Done()
			return Decision{}, ctx.Err()
		}
	})
	e.post(mention("E1", "1700000000.000100", "hi"))
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("Policy not called within 1s")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()
	start := time.Now()
	if err := e.svc.Close(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Close() = %v, want context.DeadlineExceeded", err)
	}
	if d := time.Since(start); d > time.Second {
		t.Errorf("Close took %s with a hanging Policy, want under 1s", d)
	}
}

// A Backend does not have to yield ctx.Err() when it stops, and the user
// must still learn that the turn failed.
func TestTurnTimeoutWithQuietBackend(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.TurnTimeout = 50 * time.Millisecond; c.ErrorMessage = "turn failed" })
	e.backend.block = true
	e.backend.quietCancel = true
	e.post(mention("E1", "1700000000.000100", "hi"))
	e.waitCall(t)
	e.waitAPI(t, hasText("turn failed"))
}

// Top-level channel chatter is refused before it can take the slot a
// mention needs.
func TestChannelChatterDoesNotFillQueue(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.MaxQueued = 1 })
	e.backend.block = true
	e.post(mention("E0", "1700000000.000100", "running"))
	c0 := e.waitCall(t)
	for i := range 100 {
		e.post(callback(fmt.Sprintf("Echat%d", i), slackEvent("message", "C1", fmt.Sprintf("1700000001.%06d", i), gateText)))
	}
	e.post(mention("E1", "1700000002.000100", "answer me"))
	close(c0.release)
	c1 := e.waitCall(t)
	close(c1.release)
	if c1.text != "answer me" {
		t.Errorf("second turn is session %s, want the mention", c1.sessionID)
	}
	if strings.Contains(e.log.String(), "reason=queue_full") {
		t.Errorf("log has a queue_full drop:\n%s", e.log.String())
	}
}

// These refusals need no bot ID, so the handler logs them before answering.
func TestRefusedInSubmit(t *testing.T) {
	const ts = "1700000000.000100"
	tests := []struct {
		name   string
		mut    func(*Config)
		body   []byte
		reason string
	}{
		{"kill switch", func(c *Config) { c.Disabled = true }, mention("E1", ts, gateText), ReasonKillSwitch},
		{"channel allowlist", func(c *Config) { c.Channels = []string{"C2"} }, mention("E1", ts, gateText), ReasonChannel},
		{"unmentioned top-level message", func(*Config) {}, callback("E1", slackEvent("message", "C1", ts, gateText)), ReasonNoMention},
		{"mention echo, bot ID known", func(*Config) {}, callback("E1", slackEvent("message", "C1", ts, "<@UBOT> "+gateText)), ReasonMentionEcho},
		{"mention echo, bot ID unknown", func(*Config) {}, eventNoAuth("E1", slackEvent("message", "C1", ts, "<@UBOT> "+gateText)), ReasonNoMention},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			e := newTestService(t, tt.mut)
			e.post(tt.body)
			want := `msg="slack: message not answered" reason=` + tt.reason + " channel=C1 event_id=E1"
			if !strings.Contains(e.log.String(), want) {
				t.Fatalf("log has no %q right after the handler returned:\n%s", want, e.log.String())
			}
			e.noCall(t, 50*time.Millisecond)
			if n := strings.Count(e.log.String(), "slack: message not answered"); n != 1 {
				t.Errorf("%d refusal records, want 1: the event reached the worker", n)
			}
		})
	}
}

func TestKillSwitchSubmitNeverQueues(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.Disabled = true; c.MaxQueued = 1 })
	m := &Message{EventID: "E1", TeamID: "T1", Channel: "C1", ChannelType: "channel", UserID: "U1", Text: gateText, TS: "1700000000.000100", IsMention: true, kind: "app_mention"}
	if err := e.svc.disp.Submit(t.Context(), m); err != nil {
		t.Fatalf("Submit() = %v, want nil", err)
	}
	if n := len(e.svc.disp.queue); n != 0 {
		t.Errorf("queue holds %d turns, want 0", n)
	}
	e.noCall(t, 50*time.Millisecond)
	if n := strings.Count(e.log.String(), "reason="+ReasonKillSwitch); n != 1 {
		t.Errorf("%d kill_switch records, want 1", n)
	}
}

// The gate and the run share one TurnTimeout budget.
func TestTurnTimeoutCoversPolicy(t *testing.T) {
	var policyStart atomic.Int64
	e := newTestService(t, func(c *Config) {
		c.TurnTimeout = 50 * time.Millisecond
		c.ErrorMessage = "turn failed"
		c.Policy = func(context.Context, *Message) (Decision, error) {
			policyStart.Store(time.Now().UnixNano())
			time.Sleep(40 * time.Millisecond)
			return Decision{Allow: true}, nil
		}
	})
	e.backend.block = true
	e.post(mention("E1", "1700000000.000100", "hi"))
	c := e.waitCall(t)
	deadline, ok := c.ctx.Deadline()
	if !ok {
		t.Fatal("backend ctx has no deadline")
	}
	if budget := deadline.Sub(time.Unix(0, policyStart.Load())); budget > 55*time.Millisecond {
		t.Errorf("backend deadline is %s after the Policy started, want at most 50ms plus slack", budget)
	}
	e.waitAPI(t, hasText("turn failed"))
}

// Waiting for the thread's lock does not spend the turn's budget.
func TestTurnTimeoutExcludesThreadLockWait(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.MaxConcurrentRuns = 2; c.TurnTimeout = 300 * time.Millisecond })
	e.backend.block = true
	const root = "1700000000.000100"
	e.post(mention("E1", root, "one"))
	c1 := e.waitCall(t)
	e.post(mention("E2", "1700000000.000200", "two", "thread_ts", root))
	select {
	case <-time.After(200 * time.Millisecond): // E2 waits for the lock meanwhile
	case c := <-e.backend.started:
		t.Fatalf("second turn started early in session %s", c.sessionID)
	}
	close(c1.release)
	c2 := e.waitCall(t)
	started := time.Now()
	close(c2.release)
	deadline, _ := c2.ctx.Deadline()
	if left := deadline.Sub(started); left < 200*time.Millisecond {
		t.Errorf("second turn had %s left when it started, want most of 300ms", left)
	}
}

// waitLockWaiters waits until n workers hold or wait for key's lock.
func waitLockWaiters(t *testing.T, d *dispatcher, key string, n int) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for {
		d.locks.mu.Lock()
		refs := 0
		if l := d.locks.locks[key]; l != nil {
			refs = l.refs
		}
		d.locks.mu.Unlock()
		if refs == n {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("%d workers on the lock of %s after 1s, want %d", refs, key, n)
		}
		time.Sleep(time.Millisecond)
	}
}
