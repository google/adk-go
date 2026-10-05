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

package adkrest_test

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/server/adkrest"
	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/session"
)

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

// TestRunSSEWriteTimeout checks the write deadline /run_sse sets for each
// ServerConfig.SSEWriteTimeout. A zero timeout used to set the deadline to the
// moment the request arrived, so every stream failed before its first byte.
// A negative timeout clears the deadline, including the http.Server's own
// WriteTimeout, so a long run can stream to completion.
func TestRunSSEWriteTimeout(t *testing.T) {
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
			rec := &deadlineRecorder{}
			h := newSSETimeoutServer(t, tt.timeout)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/run_sse" {
					rec.ResponseWriter = w
					w = rec
				}
				h.ServeHTTP(w, r)
			}))
			defer srv.Close()
			sid := createCompactionSession(t, srv.URL)

			start := time.Now()
			got := postRunSSE(t, srv.URL, sid)

			if !strings.Contains(got, "answer 1") {
				t.Errorf("POST /run_sse body does not contain the model's answer:\n%s", got)
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

func newSSETimeoutServer(t *testing.T, timeout time.Duration) *adkrest.Server {
	t.Helper()
	root, err := llmagent.New(llmagent.Config{Name: compactionApp, Model: &echoModel{}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	srv, err := adkrest.NewServer(adkrest.ServerConfig{
		SessionService:  session.InMemoryService(),
		AgentLoader:     agent.NewSingleLoader(root),
		SSEWriteTimeout: timeout,
		Authenticator:   authn.NewCustom(func(r *http.Request) (*authn.Caller, error) { return &authn.Caller{UserID: "u"}, nil }),
	})
	if err != nil {
		t.Fatalf("adkrest.NewServer() error = %v", err)
	}
	return srv
}

func postRunSSE(t *testing.T, baseURL, sessionID string) string {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"appName":    compactionApp,
		"userId":     compactionUser,
		"sessionId":  sessionID,
		"newMessage": genai.NewContentFromText("hi", genai.RoleUser),
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(baseURL+"/run_sse", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST /run_sse error = %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("reading /run_sse body: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /run_sse status = %d, want %d; body: %s", resp.StatusCode, http.StatusOK, got)
	}
	return string(got)
}
