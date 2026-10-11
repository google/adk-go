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

package agentengine_test

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/server/agentengine"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

// TestNewHandlerRejectsUnusableCompaction checks that Agent Engine refuses a
// compaction config it cannot serve, at construction.
//
// The config is validated inside runner.New, and this surface builds a runner
// per request, so without a check here the handler is created, the process
// reports healthy, and every request fails with the same error instead.
//
// This pins that the check runs, not how. NewHandler delegates to
// launcher.Config.Validate rather than reaching for the compaction field, so
// that a check added there later reaches this surface too, but a hand-rolled
// copy of today's check would satisfy this test just as well.
func TestNewHandlerRejectsUnusableCompaction(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		cfg  *compaction.Config
		ok   bool
	}{
		{name: "nil compaction is fine", cfg: nil, ok: true},
		{name: "overlap with no interval", cfg: &compaction.Config{OverlapSize: 2}},
		{name: "no strategy at all", cfg: &compaction.Config{}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg := &launcher.Config{
				SessionService: session.InMemoryService(),
				Compaction:     tc.cfg,
			}
			_, err := agentengine.NewHandler(cfg, time.Second, 1<<20, "engine")
			if tc.ok {
				if err != nil {
					t.Errorf("NewHandler() = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatal("NewHandler() accepted a compaction config it cannot serve")
			}
			if !strings.Contains(err.Error(), "Compaction") {
				t.Errorf("error %q does not name the field an operator has to change", err)
			}
		})
	}
}

// deadlineRecorder records the write deadlines a handler sets and passes them
// on to the underlying connection.
type deadlineRecorder struct {
	http.ResponseWriter
	mu        sync.Mutex
	deadlines []time.Time
}

func (r *deadlineRecorder) SetWriteDeadline(t time.Time) error {
	r.mu.Lock()
	r.deadlines = append(r.deadlines, t)
	r.mu.Unlock()
	return http.NewResponseController(r.ResponseWriter).SetWriteDeadline(t)
}

func (r *deadlineRecorder) recorded() []time.Time {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]time.Time(nil), r.deadlines...)
}

func (r *deadlineRecorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

// TestQuerySSEWriteTimeout checks the write deadline a query sets for each
// sseWriteTimeout. A zero timeout used to set the deadline to the moment the
// request arrived, so every response was lost. A negative timeout clears the
// deadline, including the http.Server's own WriteTimeout.
func TestQuerySSEWriteTimeout(t *testing.T) {
	tests := []struct {
		name         string
		timeout      time.Duration
		wantDeadline time.Duration // 0 means no deadline
	}{
		{name: "zero uses default", timeout: 0, wantDeadline: 120 * time.Second},
		{name: "positive", timeout: 30 * time.Second, wantDeadline: 30 * time.Second},
		{name: "negative clears deadline", timeout: -1, wantDeadline: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h, err := agentengine.NewHandler(&launcher.Config{SessionService: session.InMemoryService()}, tt.timeout, 1<<20, "engine")
			if err != nil {
				t.Fatalf("NewHandler() error = %v", err)
			}
			rec := &deadlineRecorder{}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				rec.ResponseWriter = w
				h.ServeHTTP(rec, r)
			}))
			defer srv.Close()

			start := time.Now()
			resp, err := http.Post(srv.URL+"/reasoning_engine", "application/json",
				strings.NewReader(`{"class_method":"async_create_session","input":{"user_id":"u"}}`))
			if err != nil {
				t.Fatalf("POST /reasoning_engine error = %v", err)
			}
			defer func() { _ = resp.Body.Close() }()
			got, err := io.ReadAll(resp.Body)
			if err != nil {
				t.Fatalf("reading response body: %v", err)
			}

			if resp.StatusCode != http.StatusOK {
				t.Fatalf("POST /reasoning_engine status = %d, want %d; body: %s", resp.StatusCode, http.StatusOK, got)
			}
			if !strings.Contains(string(got), `"user_id":"u"`) {
				t.Errorf("response does not contain the created session:\n%s", got)
			}
			deadlines := rec.recorded()
			if len(deadlines) != 1 {
				t.Fatalf("SetWriteDeadline called %d times, want 1", len(deadlines))
			}
			d := deadlines[0]
			if tt.wantDeadline == 0 {
				if !d.IsZero() {
					t.Errorf("write deadline = %v, want zero (no deadline)", d)
				}
				return
			}
			if lo, hi := start.Add(tt.wantDeadline), time.Now().Add(tt.wantDeadline); d.Before(lo) || d.After(hi) {
				t.Errorf("write deadline = %v, want between %v and %v", d, lo, hi)
			}
		})
	}
}
