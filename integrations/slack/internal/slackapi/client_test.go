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

package slackapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/integrations/slack/internal/slackapitest"
)

const testToken = "xoxb-test"

// recorder sits in front of the fake to capture what Call does not carry: the
// raw Authorization header and the exact JSON keys of each request body.
type recorder struct {
	mu      sync.Mutex
	auth    []string
	bodies  []map[string]any
	handler http.Handler
}

func (r *recorder) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	raw, _ := io.ReadAll(req.Body)
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	r.mu.Lock()
	r.auth = append(r.auth, req.Header.Get("Authorization"))
	r.bodies = append(r.bodies, body)
	r.mu.Unlock()
	req.Body = io.NopCloser(bytes.NewReader(raw))
	r.handler.ServeHTTP(w, req)
}

func newRecorded(t *testing.T) (*Client, *slackapitest.Fake, *recorder) {
	t.Helper()
	f := slackapitest.New(t)
	rec := &recorder{handler: f.Server.Config.Handler}
	srv := httptest.NewServer(rec)
	t.Cleanup(srv.Close)
	return New(testToken, WithBaseURL(srv.URL+"/api/")), f, rec
}

func newClient(t *testing.T) (*Client, *slackapitest.Fake) {
	t.Helper()
	f := slackapitest.New(t)
	return New(testToken, WithBaseURL(f.URL())), f
}

// step is one scripted HTTP response, for what the fake's Response cannot
// express: an arbitrary Retry-After value, or a 2xx other than 200.
type step struct {
	status     int
	retryAfter string
	body       string
}

// scripted serves steps in order and counts the requests it received.
func scripted(t *testing.T, steps ...step) (*Client, *int) {
	t.Helper()
	var mu sync.Mutex
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		s := steps[min(hits, len(steps)-1)]
		hits++
		mu.Unlock()
		if s.retryAfter != "" {
			w.Header().Set("Retry-After", s.retryAfter)
		}
		w.WriteHeader(s.status)
		_, _ = io.WriteString(w, s.body)
	}))
	t.Cleanup(srv.Close)
	return New(testToken, WithBaseURL(srv.URL+"/api/")), &hits
}

// noSleep records each requested retry delay instead of waiting.
func noSleep(c *Client) *[]time.Duration {
	var slept []time.Duration
	c.sleep = func(_ context.Context, d time.Duration) error {
		slept = append(slept, d)
		return nil
	}
	return &slept
}

func TestClient(t *testing.T) {
	ctx := t.Context()

	t.Run("PostMessage sends token and thread", func(t *testing.T) {
		c, f, rec := newRecorded(t)
		ts, err := c.PostMessage(ctx, "C1", "111.222", "hello")
		if err != nil {
			t.Fatalf("PostMessage: %v", err)
		}
		if ts != "1000000000.000001" {
			t.Errorf("ts = %q, want 1000000000.000001", ts)
		}
		if got := rec.auth; !slices.Equal(got, []string{"Bearer " + testToken}) {
			t.Errorf("Authorization = %q, want [Bearer %s]", got, testToken)
		}
		body := rec.bodies[0]
		if body["channel"] != "C1" || body["text"] != "hello" || body["thread_ts"] != "111.222" {
			t.Errorf("body = %v, want channel C1, text hello, thread_ts 111.222", body)
		}
		if body["unfurl_links"] != false {
			t.Errorf("body unfurl_links = %v, want false", body["unfurl_links"])
		}
		want := []slackapitest.Call{{Method: "chat.postMessage", Channel: "C1", ThreadTS: "111.222", TS: ts, Text: "hello"}}
		if !slices.Equal(f.Calls, want) {
			t.Errorf("Calls = %+v, want %+v", f.Calls, want)
		}
	})

	t.Run("PostMessage without thread omits thread_ts", func(t *testing.T) {
		c, _, rec := newRecorded(t)
		if _, err := c.PostMessage(ctx, "C1", "", "hello"); err != nil {
			t.Fatalf("PostMessage: %v", err)
		}
		if v, ok := rec.bodies[0]["thread_ts"]; ok {
			t.Errorf("body has thread_ts = %v, want key absent", v)
		}
	})

	t.Run("UpdateMessage and DeleteMessage", func(t *testing.T) {
		c, f := newClient(t)
		if err := c.UpdateMessage(ctx, "C1", "5.6", "edited"); err != nil {
			t.Fatalf("UpdateMessage: %v", err)
		}
		if err := c.DeleteMessage(ctx, "C2", "7.8"); err != nil {
			t.Fatalf("DeleteMessage: %v", err)
		}
		want := []slackapitest.Call{
			{Method: "chat.update", Channel: "C1", TS: "5.6", Text: "edited"},
			{Method: "chat.delete", Channel: "C2", TS: "7.8"},
		}
		if got := f.Snapshot(); !slices.Equal(got, want) {
			t.Errorf("Snapshot = %+v, want %+v", got, want)
		}
	})

	t.Run("AuthTest returns bot user", func(t *testing.T) {
		c, f := newClient(t)
		f.BotUserID = "UOTHER"
		id, err := c.AuthTest(ctx)
		if err != nil {
			t.Fatalf("AuthTest: %v", err)
		}
		if id != "UOTHER" {
			t.Errorf("AuthTest = %q, want UOTHER", id)
		}
	})

	t.Run("ok false is APIError", func(t *testing.T) {
		c, f := newClient(t)
		f.Fail = map[string][]slackapitest.Response{"chat.postMessage": {{APIError: "channel_not_found"}}}
		_, err := c.PostMessage(ctx, "C1", "", "hi")
		var apiErr *APIError
		if !errors.As(err, &apiErr) || apiErr.Code != "channel_not_found" {
			t.Fatalf("err = %v, want *APIError{channel_not_found}", err)
		}
	})

	t.Run("429 once is retried after Retry-After", func(t *testing.T) {
		c, f := newClient(t)
		slept := noSleep(c)
		f.Fail = map[string][]slackapitest.Response{"chat.postMessage": {{Status: http.StatusTooManyRequests, RetryAfter: 1}}}
		ts, err := c.PostMessage(ctx, "C1", "", "hi")
		if err != nil {
			t.Fatalf("PostMessage: %v", err)
		}
		if ts != "1000000000.000001" {
			t.Errorf("ts = %q, want 1000000000.000001", ts)
		}
		if !slices.Equal(*slept, []time.Duration{time.Second}) {
			t.Errorf("slept %v, want [1s]", *slept)
		}
		if got := f.Methods(); !slices.Equal(got, []string{"chat.postMessage", "chat.postMessage"}) {
			t.Errorf("Methods = %v, want two chat.postMessage", got)
		}
	})

	t.Run("Retry-After parsing", func(t *testing.T) {
		tests := []struct {
			header string
			want   time.Duration
		}{
			{"0", 0},
			{"2", 2 * time.Second},
			{"abc", time.Second},
			{"-5", time.Second},
		}
		for _, tt := range tests {
			c, hits := scripted(t,
				step{status: http.StatusTooManyRequests, retryAfter: tt.header},
				step{status: http.StatusOK, body: `{"ok":true,"ts":"1.2"}`},
			)
			slept := noSleep(c)
			ts, err := c.PostMessage(ctx, "C1", "", "hi")
			if err != nil || ts != "1.2" {
				t.Errorf("Retry-After %q: PostMessage = %q, %v, want 1.2 after one retry", tt.header, ts, err)
			}
			if !slices.Equal(*slept, []time.Duration{tt.want}) {
				t.Errorf("Retry-After %q: slept %v, want [%v]", tt.header, *slept, tt.want)
			}
			if *hits != 2 {
				t.Errorf("Retry-After %q: %d requests, want 2", tt.header, *hits)
			}
		}
	})

	t.Run("429 without Retry-After waits 1s", func(t *testing.T) {
		c, f := newClient(t)
		slept := noSleep(c)
		f.Fail = map[string][]slackapitest.Response{"chat.postMessage": {{Status: http.StatusTooManyRequests}}}
		if _, err := c.PostMessage(ctx, "C1", "", "hi"); err != nil {
			t.Fatalf("PostMessage: %v", err)
		}
		if !slices.Equal(*slept, []time.Duration{time.Second}) {
			t.Errorf("slept %v, want [1s]", *slept)
		}
	})

	t.Run("299 is success", func(t *testing.T) {
		c, _ := scripted(t, step{status: 299, body: `{"ok":true,"ts":"1.2"}`})
		ts, err := c.PostMessage(ctx, "C1", "", "hi")
		if err != nil || ts != "1.2" {
			t.Errorf("PostMessage = %q, %v, want 1.2, nil", ts, err)
		}
	})

	t.Run("empty token is refused", func(t *testing.T) {
		f := slackapitest.New(t)
		_, err := New("", WithBaseURL(f.URL())).PostMessage(ctx, "C1", "", "hi")
		if err == nil || !strings.Contains(err.Error(), "401") {
			t.Errorf("err = %v, want error mentioning 401", err)
		}
		if got := f.Snapshot(); len(got) != 0 {
			t.Errorf("Calls = %+v, want none", got)
		}
	})

	t.Run("429 twice is an error", func(t *testing.T) {
		c, f := newClient(t)
		noSleep(c)
		f.Fail = map[string][]slackapitest.Response{"chat.postMessage": {
			{Status: http.StatusTooManyRequests, RetryAfter: 1},
			{Status: http.StatusTooManyRequests, RetryAfter: 1},
		}}
		if _, err := c.PostMessage(ctx, "C1", "", "hi"); err == nil {
			t.Fatal("PostMessage succeeded, want error")
		}
		if n := len(f.Calls); n != 2 {
			t.Errorf("%d calls, want 2", n)
		}
	})

	t.Run("500 is an error with the status", func(t *testing.T) {
		c, f := newClient(t)
		noSleep(c)
		f.Fail = map[string][]slackapitest.Response{"chat.postMessage": {{Status: http.StatusInternalServerError}}}
		_, err := c.PostMessage(ctx, "C1", "", "hi")
		if err == nil || !strings.Contains(err.Error(), "500") {
			t.Fatalf("err = %v, want error mentioning 500", err)
		}
		if n := len(f.Calls); n != 1 {
			t.Errorf("%d calls, want 1", n)
		}
	})

	t.Run("ctx cancelled during retry sleep", func(t *testing.T) {
		c, f := newClient(t)
		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		c.sleep = func(ctx context.Context, _ time.Duration) error {
			cancel()
			return ctx.Err()
		}
		f.Fail = map[string][]slackapitest.Response{"chat.postMessage": {{Status: http.StatusTooManyRequests, RetryAfter: 1}}}
		_, err := c.PostMessage(ctx, "C1", "", "hi")
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
		// A retry on the canceled ctx also yields context.Canceled, wrapped in a
		// *url.Error, so this is what proves the client stopped after the sleep.
		var ue *url.Error
		if errors.As(err, &ue) {
			t.Errorf("err = %v, want the sleep's error, not a transport error", err)
		}
		if n := len(f.Calls); n != 1 {
			t.Errorf("%d calls, want 1", n)
		}
	})

	t.Run("default sleep honors cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(ctx)
		cancel()
		done := make(chan error, 1)
		go func() { done <- New(testToken).sleep(ctx, time.Hour) }()
		select {
		case err := <-done:
			if !errors.Is(err, context.Canceled) {
				t.Errorf("sleep = %v, want context.Canceled", err)
			}
		case <-time.After(time.Second):
			t.Error("sleep still running after 1s on a canceled ctx")
		}
	})

	t.Run("token never in error", func(t *testing.T) {
		failures := map[string][]slackapitest.Response{
			"api error":  {{APIError: "invalid_auth"}},
			"429 twice":  {{Status: http.StatusTooManyRequests}, {Status: http.StatusTooManyRequests}},
			"500":        {{Status: http.StatusInternalServerError}},
			"403 status": {{Status: http.StatusForbidden}},
		}
		for name, fail := range failures {
			c, f := newClient(t)
			noSleep(c)
			f.Fail = map[string][]slackapitest.Response{"chat.postMessage": fail}
			_, err := c.PostMessage(ctx, "C1", "", "hi")
			if err == nil {
				t.Errorf("%s: PostMessage succeeded, want error", name)
				continue
			}
			if strings.Contains(err.Error(), "xoxb-") {
				t.Errorf("%s: error %q contains the token", name, err)
			}
		}
		// The transport error path: nothing listens at this address.
		c := New(testToken, WithBaseURL("http://127.0.0.1:1/api/"))
		if _, err := c.PostMessage(ctx, "C1", "", "hi"); err == nil || strings.Contains(err.Error(), "xoxb-") {
			t.Errorf("transport error = %v, want non-nil error without the token", err)
		}
	})
}
