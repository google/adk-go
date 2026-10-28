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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"

	"google.golang.org/adk/integrations/slack/internal/slackapitest"
)

// TestMain looks for these in every record written through the shared test
// logger (newTestService). The deliverer tests log to private buffers.
const (
	testBotToken = "xoxb-test"
	testSecret   = "shh-secret"
	// gateText is the text of the messages the gate tests refuse.
	gateText = "nobody-should-log-this-7f3a"
)

// TestMain fails the run when any service log record carries a token, the
// signing secret or message text.

var testNow = time.Unix(1_800_000_000, 0)

type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// allLogs receives every record written by a newTestService logger.

var allLogs syncBuffer

type testLog struct{ syncBuffer }

func (l *testLog) Write(p []byte) (int, error) {
	_, _ = allLogs.Write(p)
	return l.syncBuffer.Write(p)
}

type backendCall struct {
	ctx                     context.Context
	userID, sessionID, text string
	// release unblocks a call of a blocking backend.
	release chan struct{}
}

// fakeBackend records each Run and yields scripted events.

type fakeBackend struct {
	// block makes each Run wait until its call is released or its ctx is done.
	block bool
	// quietCancel makes a blocked Run stop without an error when its ctx is done.
	quietCancel bool
	events      []*session.Event
	err         error
	// panicText makes Run panic when it receives exactly this text.
	panicText string
	// started receives each call when its Run begins.
	started chan *backendCall

	mu    sync.Mutex
	calls []*backendCall
}

func newFakeBackend() *fakeBackend { return &fakeBackend{started: make(chan *backendCall, 64)} }

func (b *fakeBackend) Run(ctx context.Context, userID, sessionID string, msg *genai.Content) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		c := &backendCall{ctx: ctx, userID: userID, sessionID: sessionID, text: msg.Parts[0].Text, release: make(chan struct{})}
		b.mu.Lock()
		b.calls = append(b.calls, c)
		b.mu.Unlock()
		b.started <- c
		if b.block {
			select {
			case <-c.release:
			case <-ctx.Done():
				if !b.quietCancel {
					yield(nil, ctx.Err())
				}
				return
			}
		}
		if b.panicText != "" && c.text == b.panicText {
			panic("backend exploded")
		}
		for _, ev := range b.events {
			if !yield(ev, nil) {
				return
			}
		}
		if b.err != nil {
			yield(nil, b.err)
		}
	}
}

func (b *fakeBackend) count() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.calls)
}

type testEnv struct {
	svc     *Service
	api     *slackapitest.Fake
	backend *fakeBackend
	log     *testLog
}

// newTestService runs one worker unless a mutator says otherwise, so turns
// run in the order they were submitted.

func newTestService(t *testing.T, mutators ...func(*Config)) *testEnv {
	t.Helper()
	e := &testEnv{api: slackapitest.New(t), backend: newFakeBackend(), log: &testLog{}}
	cfg := Config{
		BotToken:          testBotToken,
		SigningSecret:     testSecret,
		Backend:           e.backend,
		Logger:            slog.New(slog.NewTextHandler(e.log, &slog.HandlerOptions{Level: slog.LevelDebug})),
		MaxConcurrentRuns: 1,
		apiBaseURL:        e.api.URL(),
		now:               func() time.Time { return testNow },
	}
	for _, m := range mutators {
		m(&cfg)
	}
	svc, err := New(t.Context(), cfg)
	if err != nil {
		t.Fatalf("New() = %v", err)
	}
	e.svc = svc
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		_ = svc.Close(ctx)
	})
	return e
}

func signedRequest(ctx context.Context, secret string, body []byte, at time.Time) *http.Request {
	req := httptest.NewRequestWithContext(ctx, http.MethodPost, "/slack/events", bytes.NewReader(body))
	ts := strconv.FormatInt(at.Unix(), 10)
	req.Header.Set("X-Slack-Request-Timestamp", ts)
	req.Header.Set("X-Slack-Signature", sign(secret, ts, body))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func (e *testEnv) serve(req *http.Request) *httptest.ResponseRecorder {
	rec := httptest.NewRecorder()
	e.svc.Handler().ServeHTTP(rec, req)
	return rec
}

// post delivers body as a correctly signed request. It may run outside the
// test goroutine.

func (e *testEnv) post(body []byte) *httptest.ResponseRecorder {
	return e.serve(signedRequest(context.Background(), testSecret, body, testNow))
}

func (e *testEnv) waitCall(t *testing.T) *backendCall {
	t.Helper()
	select {
	case c := <-e.backend.started:
		return c
	case <-time.After(time.Second):
		t.Fatal("backend not called within 1s")
		return nil
	}
}

// noCall asserts that no turn starts within d.

func (e *testEnv) noCall(t *testing.T, d time.Duration) {
	t.Helper()
	select {
	case c := <-e.backend.started:
		t.Fatalf("unexpected backend call for session %s", c.sessionID)
	case <-time.After(d):
	}
}

var sentinels atomic.Int64

// flush runs one DM turn and asserts it is the next backend call. With one
// worker, every turn submitted earlier has been gated or run by then.

func (e *testEnv) flush(t *testing.T) {
	t.Helper()
	n := sentinels.Add(1)
	e.post(dm(fmt.Sprintf("Esentinel%d", n), fmt.Sprintf("1900000000.%06d", n), "sentinel"))
	c := e.waitCall(t)
	close(c.release)
	if c.text != "sentinel" {
		t.Fatalf("backend call for session %s ran before the sentinel", c.sessionID)
	}
}

func (e *testEnv) waitLog(t *testing.T, substr string) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for !strings.Contains(e.log.String(), substr) {
		if time.Now().After(deadline) {
			t.Fatalf("log has no %q within 1s:\n%s", substr, e.log.String())
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// waitAPI waits until the Slack fake has recorded calls satisfying ok.

func (e *testEnv) waitAPI(t *testing.T, ok func([]slackapitest.Call) bool) []slackapitest.Call {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for {
		calls := e.api.Snapshot()
		if ok(calls) {
			return calls
		}
		if time.Now().After(deadline) {
			t.Fatalf("Slack API calls within 1s: %+v", calls)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// drain waits for every turn already taken by a worker, then stops the service.

func (e *testEnv) drain(t *testing.T) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	if err := e.svc.Close(ctx); err != nil {
		t.Fatalf("Close() = %v", err)
	}
}

func hasText(text string) func([]slackapitest.Call) bool {
	return func(calls []slackapitest.Call) bool {
		for _, c := range calls {
			if c.Text == text {
				return true
			}
		}
		return false
	}
}

// slackEvent returns the inner event of an event_callback. kv adds or
// overrides string fields, such as "thread_ts" or "parent_user_id".

func slackEvent(typ, channel, ts, text string, kv ...string) map[string]any {
	ev := map[string]any{"type": typ, "user": "U1", "channel": channel, "channel_type": "channel", "ts": ts, "text": text}
	if strings.HasPrefix(channel, "D") {
		ev["channel_type"] = "im"
	}
	for i := 0; i+1 < len(kv); i += 2 {
		ev[kv[i]] = kv[i+1]
	}
	return ev
}

func callback(eventID string, event map[string]any) []byte {
	b, err := json.Marshal(map[string]any{
		"type":           "event_callback",
		"event_id":       eventID,
		"team_id":        "T1",
		"authorizations": []any{map[string]any{"user_id": "UBOT", "is_bot": true}},
		"event":          event,
	})
	if err != nil {
		panic(err)
	}
	return b
}

// mention is an app_mention of the bot in channel C1.

func mention(eventID, ts, text string, kv ...string) []byte {
	return callback(eventID, slackEvent("app_mention", "C1", ts, "<@UBOT> "+text, kv...))
}

// textEvent is a final model event with one text part per text.

func textEvent(texts ...string) *session.Event {
	ev := &session.Event{}
	ev.Content = &genai.Content{Role: genai.RoleModel}
	for _, s := range texts {
		ev.Content.Parts = append(ev.Content.Parts, genai.NewPartFromText(s))
	}
	return ev
}

func dm(eventID, ts, text string) []byte {
	return callback(eventID, slackEvent("message", "D1", ts, text))
}

func eventNoAuth(eventID string, event map[string]any) []byte {
	b, err := json.Marshal(map[string]any{"type": "event_callback", "event_id": eventID, "team_id": "T1", "event": event})
	if err != nil {
		panic(err)
	}
	return b
}

// A Slack outage must not cost an auth.test call per event.

// A broken Deduper must not silence the bot.

func TestMain(m *testing.M) {
	code := m.Run()
	logs := allLogs.String()
	for _, secret := range []string{"xoxb-", testSecret, gateText, "hello <@UOTHER>", "\x00"} {
		if strings.Contains(logs, secret) {
			fmt.Fprintf(os.Stderr, "FAIL: service logs contain %q\n", secret)
			code = 1
		}
	}
	os.Exit(code)
}
