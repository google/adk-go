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

// Package slackapitest provides an in-memory Slack Web API for tests.
package slackapitest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// Fake is an in-memory Slack Web API for tests. It records, in order, every
// call that carries a bearer token and a body that is empty or decodable
// JSON, and lets a test script failures.
type Fake struct {
	Server *httptest.Server
	mu     sync.Mutex
	// Calls holds the recorded calls in order, scripted failures included.
	// Read it only after all requests have finished, or use Snapshot.
	Calls []Call
	next  int
	// Fail scripts responses per method, consumed in order. A method with no
	// scripted response left succeeds.
	Fail map[string][]Response
	// BotUserID is returned by auth.test.
	BotUserID string
	// Delay holds back the response per method, or until the request is
	// canceled.
	Delay map[string]time.Duration
}

// Call is one recorded Web API call.
type Call struct{ Method, Channel, ThreadTS, TS, Text string }

// Response is a scripted failure. A non-zero Status is sent as that HTTP
// status, with Retry-After when RetryAfter > 0. Otherwise a non-empty
// APIError is sent as {"ok": false, "error": APIError}.
type Response struct {
	Status     int
	RetryAfter int
	APIError   string
}

// New starts a Fake whose server is closed when the test ends.
func New(t testing.TB) *Fake {
	f := &Fake{BotUserID: "UBOT"}
	f.Server = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.Server.Close)
	return f
}

// URL returns the base URL to configure a client with. It ends in "/api/".
func (f *Fake) URL() string { return f.Server.URL + "/api/" }

// Methods returns the method names of the recorded calls in order.
func (f *Fake) Methods() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	methods := make([]string, len(f.Calls))
	for i, c := range f.Calls {
		methods[i] = c.Method
	}
	return methods
}

// Snapshot returns a copy of Calls that is safe to take while requests are
// in flight.
func (f *Fake) Snapshot() []Call {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.Calls)
}

// Reset forgets recorded calls, scripted failures and the ts counter.
func (f *Fake) Reset() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.Calls = nil
	f.Fail = nil
	f.next = 0
}

func (f *Fake) serve(w http.ResponseWriter, r *http.Request) {
	token, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
	if !ok || token == "" {
		http.Error(w, "missing bearer token", http.StatusUnauthorized)
		return
	}
	method := strings.TrimPrefix(r.URL.Path, "/api/")
	var body struct {
		Channel  string `json:"channel"`
		ThreadTS string `json:"thread_ts"`
		TS       string `json:"ts"`
		Text     string `json:"text"`
	}
	if r.ContentLength != 0 {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	}

	f.mu.Lock()
	call := Call{Method: method, Channel: body.Channel, ThreadTS: body.ThreadTS, TS: body.TS, Text: body.Text}
	var fail *Response
	if queue := f.Fail[method]; len(queue) > 0 {
		fail = &queue[0]
		f.Fail[method] = queue[1:]
	}
	resp := map[string]any{"ok": true}
	if fail == nil {
		switch method {
		case "chat.postMessage":
			f.next++
			call.TS = fmt.Sprintf("1000000000.%06d", f.next)
			resp["ts"] = call.TS
		case "chat.update", "chat.delete":
		case "auth.test":
			resp["user_id"] = f.BotUserID
		default:
			resp = map[string]any{"ok": false, "error": "unknown_method"}
		}
	}
	f.Calls = append(f.Calls, call)
	delay := f.Delay[method]
	f.mu.Unlock()

	if delay > 0 {
		select {
		case <-time.After(delay):
		case <-r.Context().Done():
			return
		}
	}

	if fail != nil {
		if fail.Status != 0 {
			if fail.RetryAfter > 0 {
				w.Header().Set("Retry-After", strconv.Itoa(fail.RetryAfter))
			}
			w.WriteHeader(fail.Status)
			return
		}
		resp = map[string]any{"ok": false, "error": fail.APIError}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
