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
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func urlVerification(challenge string) []byte {
	b, _ := json.Marshal(map[string]string{"type": "url_verification", "challenge": challenge})
	return b
}

func TestHandlerURLVerification(t *testing.T) {
	e := newTestService(t)
	rec := e.post(urlVerification("abc123"))
	if rec.Code != http.StatusOK || rec.Body.String() != "abc123" {
		t.Fatalf("got %d %q, want 200 %q", rec.Code, rec.Body.String(), "abc123")
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Errorf("Content-Type = %q, want text/plain", ct)
	}
}

func TestHandlerRejectsBadSignature(t *testing.T) {
	e := newTestService(t)
	body := mention("E1", "1700000000.000100", "hi")
	rec := e.serve(signedRequest(t.Context(), "wrong-secret", body, testNow))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("code = %d, want 401", rec.Code)
	}
	if !strings.Contains(e.log.String(), "reason=auth_failed") || !strings.Contains(e.log.String(), "level=WARN") {
		t.Errorf("log has no WARN auth_failed record:\n%s", e.log.String())
	}
	if strings.Contains(e.log.String(), "UBOT") {
		t.Error("log contains the request body")
	}
	e.flush(t)
}

func TestHandlerTimestampWindow(t *testing.T) {
	e := newTestService(t)
	for _, tt := range []struct {
		age  time.Duration
		want int
	}{
		{301 * time.Second, http.StatusUnauthorized},
		{299 * time.Second, http.StatusOK},
	} {
		rec := e.serve(signedRequest(t.Context(), testSecret, urlVerification("c"), testNow.Add(-tt.age)))
		if rec.Code != tt.want {
			t.Errorf("request %s old: code = %d, want %d", tt.age, rec.Code, tt.want)
		}
	}
}

func TestHandlerBodyLimit(t *testing.T) {
	e := newTestService(t)
	// The challenge pads the body to exactly the limit.
	atLimit := urlVerification(strings.Repeat("a", 1<<20-len(urlVerification(""))))
	if len(atLimit) != 1<<20 {
		t.Fatalf("fixture is %d bytes, want %d", len(atLimit), 1<<20)
	}
	if rec := e.post(atLimit); rec.Code != http.StatusOK {
		t.Errorf("body of exactly 1 MiB: code = %d, want 200", rec.Code)
	}
	over := urlVerification(strings.Repeat("a", 1<<20-len(urlVerification(""))+1))
	if rec := e.post(over); rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("body over 1 MiB: code = %d, want 413", rec.Code)
	}
	bigEvent := mention("E1", "1700000000.000100", strings.Repeat("a", 1<<20))
	if rec := e.post(bigEvent); rec.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("event_callback over 1 MiB: code = %d, want 413", rec.Code)
	}
	e.flush(t)
}

func TestHandlerBadJSON(t *testing.T) {
	e := newTestService(t)
	if rec := e.post([]byte("{not json")); rec.Code != http.StatusBadRequest {
		t.Errorf("code = %d, want 400", rec.Code)
	}
}

// Slack retries a request it has not seen answered within three seconds, so
// the handler must answer before the turn ends.
func TestHandlerAnswersBeforeTurnEnds(t *testing.T) {
	e := newTestService(t)
	e.backend.block = true
	done := make(chan *httptest.ResponseRecorder, 1)
	go func() { done <- e.post(mention("E1", "1700000000.000100", "hi")) }()
	var rec *httptest.ResponseRecorder
	select {
	case rec = <-done:
	case <-time.After(time.Second):
		t.Fatal("handler did not return while the turn was blocked")
	}
	c := e.waitCall(t)
	if rec.Code != http.StatusOK || rec.Body.Len() != 0 {
		t.Errorf("got %d %q while the turn runs, want 200 and an empty body", rec.Code, rec.Body.String())
	}
	close(c.release)
}

// net/http cancels a request's context when its handler returns, and the
// turn must not stop with it.
func TestHandlerTurnOutlivesRequest(t *testing.T) {
	e := newTestService(t)
	e.backend.block = true
	ctx, cancel := context.WithCancel(t.Context())
	e.serve(signedRequest(ctx, testSecret, mention("E1", "1700000000.000100", "hi"), testNow))
	cancel()
	c := e.waitCall(t)
	select {
	case <-c.ctx.Done():
		t.Fatal("backend ctx is done after the request ended")
	case <-time.After(50 * time.Millisecond):
	}
	close(c.release)
}

func TestHandlerIgnoresOtherEvents(t *testing.T) {
	e := newTestService(t)
	bodies := map[string][]byte{
		"envelope type": []byte(`{"type":"app_rate_limited"}`),
		"event type":    callback("E1", slackEvent("reaction_added", "C1", "1700000000.000100", "")),
		"subtype":       callback("E2", slackEvent("message", "D1", "1700000000.000100", "edited", "subtype", "message_changed")),
	}
	for name, body := range bodies {
		if rec := e.post(body); rec.Code != http.StatusOK {
			t.Errorf("%s: code = %d, want 200", name, rec.Code)
		}
	}
	e.flush(t)
}

func TestHandlerRetryOfSeenEvent(t *testing.T) {
	e := newTestService(t)
	// gateText lets TestMain catch a drop record that logs the text.
	body := mention("E1", "1700000000.000100", gateText)
	e.post(body)
	close(e.waitCall(t).release)

	req := signedRequest(t.Context(), testSecret, body, testNow)
	req.Header.Set("X-Slack-Retry-Num", "1")
	if rec := e.serve(req); rec.Code != http.StatusOK {
		t.Fatalf("code = %d, want 200", rec.Code)
	}
	for _, want := range []string{"level=WARN", `msg="slack: event dropped"`, "reason=duplicate_event", "event_id=E1"} {
		if !strings.Contains(e.log.String(), want) {
			t.Errorf("log has no %q:\n%s", want, e.log.String())
		}
	}
	e.flush(t)
}

func TestHandlerQueueFull(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.MaxQueued = 1 })
	e.backend.block = true
	e.post(mention("E1", "1700000000.000001", "one"))
	c1 := e.waitCall(t)
	e.post(mention("E2", "1700000000.000002", "two")) // waits in the queue
	rec := e.post(mention("E3", "1700000000.000003", "three"))
	if rec.Code != http.StatusOK {
		t.Errorf("code = %d, want 200", rec.Code)
	}
	for _, want := range []string{"level=WARN", "reason=queue_full", "event_id=E3"} {
		if !strings.Contains(e.log.String(), want) {
			t.Errorf("log has no %q:\n%s", want, e.log.String())
		}
	}
	if n := e.backend.count(); n != 1 {
		t.Errorf("backend called %d times while the first turn runs, want 1", n)
	}
	close(c1.release)
	c2 := e.waitCall(t)
	close(c2.release)
	if c2.text != "two" {
		t.Errorf("second turn is session %s, want the queued event", c2.sessionID)
	}
	e.drain(t)
	if n := e.backend.count(); n != 2 {
		t.Errorf("backend called %d times, want 2", n)
	}
}

func TestHandlerWithoutSigningSecret(t *testing.T) {
	e := newTestService(t, func(c *Config) { c.SigningSecret = "" })
	for _, want := range []string{"level=WARN", `msg="slack: signing secret not configured"`} {
		if !strings.Contains(e.log.String(), want) {
			t.Errorf("New logged no %q:\n%s", want, e.log.String())
		}
	}
	for _, secret := range []string{"", testSecret} {
		rec := e.serve(signedRequest(t.Context(), secret, urlVerification("c"), testNow))
		if rec.Code != http.StatusInternalServerError {
			t.Errorf("code = %d, want 500", rec.Code)
		}
	}
}

// After Close the instance answers 503, so Slack retries the event elsewhere.
func TestHandlerAfterClose(t *testing.T) {
	e := newTestService(t)
	e.drain(t)
	if rec := e.post(mention("E1", "1700000000.000100", "hi")); rec.Code != http.StatusServiceUnavailable {
		t.Errorf("code = %d, want 503", rec.Code)
	}
	if n := e.backend.count(); n != 0 {
		t.Errorf("backend called %d times, want 0", n)
	}
}

// Slack retries a request that is not answered within three seconds, so a
// slow auth.test must never hold back the answer.
func TestHandlerDoesNotWaitForAuthTest(t *testing.T) {
	e := newTestService(t)
	e.api.Delay = map[string]time.Duration{"auth.test": 500 * time.Millisecond}
	for i := range 3 {
		body := eventNoAuth(fmt.Sprintf("E%d", i), slackEvent("message", "D1", fmt.Sprintf("1700000000.00010%d", i), "hi"))
		start := time.Now()
		rec := e.post(body)
		// Well under the 500ms auth.test delay, with room for a loaded -race run.
		if d := time.Since(start); rec.Code != http.StatusOK || d > 400*time.Millisecond {
			t.Errorf("event %d: answered %d after %s, want 200 within 400ms", i, rec.Code, d)
		}
	}
	for range 3 {
		close(e.waitCall(t).release)
	}
	e.drain(t)
	if n := countMethod(e.api.Methods(), "auth.test"); n != 1 {
		t.Errorf("auth.test called %d times, want 1", n)
	}
}

func TestHandlerLogsIgnoredEvents(t *testing.T) {
	e := newTestService(t)
	bodies := map[string][]byte{
		"E1": callback("E1", slackEvent("message", "D1", "1700000000.000100", "", "subtype", "file_share")),
		"E2": callback("E2", slackEvent("reaction_added", "C1", "1700000000.000100", "")),
		"E3": []byte(`{"type":"some_future_envelope","event_id":"E3"}`),
	}
	for id, body := range bodies {
		if rec := e.post(body); rec.Code != http.StatusOK {
			t.Errorf("%s: code = %d, want 200", id, rec.Code)
		}
		want := `level=DEBUG msg="slack: event ignored" event_id=` + id + " reason=unsupported_event"
		if !strings.Contains(e.log.String(), want) {
			t.Errorf("log has no %q:\n%s", want, e.log.String())
		}
	}
	e.flush(t)
}
