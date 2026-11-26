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

package fallbackmodel

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/openai/openai-go/v3/option"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropicmodel"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/model/openaimodel"
)

// The tests in this file put a real provider model in front of a local HTTP
// server that answers with an error status, so that the error the fallback
// model reads is the one that provider's SDK and ADK model really return.

// fakeServer answers every request with status and body, and records the
// path of each request.
type fakeServer struct {
	status int
	body   string

	mu    sync.Mutex
	paths []string
}

func newFakeServer(t *testing.T, status int, body string) (*fakeServer, *httptest.Server) {
	t.Helper()
	s := &fakeServer{status: status, body: body}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		s.mu.Lock()
		s.paths = append(s.paths, r.URL.Path)
		s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(s.status)
		_, _ = io.WriteString(w, s.body)
	}))
	t.Cleanup(srv.Close)
	return s, srv
}

func (s *fakeServer) requests() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.paths...)
}

func newGemini(t *testing.T, name, url string) model.LLM {
	t.Helper()
	m, err := gemini.NewModel(t.Context(), name, &genai.ClientConfig{
		APIKey:      "test-key",
		Backend:     genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{BaseURL: url},
	})
	if err != nil {
		t.Fatalf("gemini.NewModel() error = %v", err)
	}
	return m
}

func newOpenAI(t *testing.T, url string) model.LLM {
	t.Helper()
	m, err := openaimodel.NewModel(t.Context(), "gpt-test", &openaimodel.ClientConfig{
		APIKey:  "test-key",
		BaseURL: url + "/v1",
		Options: []option.RequestOption{option.WithMaxRetries(0)},
	})
	if err != nil {
		t.Fatalf("openaimodel.NewModel() error = %v", err)
	}
	return m
}

func newAnthropic(t *testing.T, url string) model.LLM {
	t.Helper()
	m, err := anthropicmodel.NewModel(t.Context(), "claude-test", anthropicmodel.ClientConfig{
		Backend:    anthropicmodel.BackendAnthropic,
		APIKey:     "test-key",
		BaseURL:    url,
		MaxRetries: new(0),
	})
	if err != nil {
		t.Fatalf("anthropicmodel.NewModel() error = %v", err)
	}
	return m
}

const (
	geminiError    = `{"error":{"code":%d,"message":"test error","status":"TEST"}}`
	openAIError    = `{"error":{"message":"test error","type":"test_error","code":"test"}}`
	anthropicError = `{"type":"error","error":{"type":"%s","message":"test error"}}`
)

func TestProviderErrorsFallBack(t *testing.T) {
	tests := []struct {
		name         string
		status       int
		body         string
		newPrimary   func(t *testing.T, url string) model.LLM
		wantFallBack bool
	}{
		{
			name:         "gemini 503",
			status:       http.StatusServiceUnavailable,
			body:         fmt.Sprintf(geminiError, 503),
			newPrimary:   func(t *testing.T, url string) model.LLM { return newGemini(t, "gemini-test", url) },
			wantFallBack: true,
		},
		{
			name:       "gemini 400",
			status:     http.StatusBadRequest,
			body:       fmt.Sprintf(geminiError, 400),
			newPrimary: func(t *testing.T, url string) model.LLM { return newGemini(t, "gemini-test", url) },
		},
		{
			name:         "openai 429",
			status:       http.StatusTooManyRequests,
			body:         openAIError,
			newPrimary:   newOpenAI,
			wantFallBack: true,
		},
		{
			name:       "openai 401",
			status:     http.StatusUnauthorized,
			body:       openAIError,
			newPrimary: newOpenAI,
		},
		{
			name:         "anthropic 503",
			status:       http.StatusServiceUnavailable,
			body:         fmt.Sprintf(anthropicError, "api_error"),
			newPrimary:   newAnthropic,
			wantFallBack: true,
		},
		{
			// 529 is not a default status; DefaultStatusCodes shows how to
			// add it.
			name:       "anthropic 529",
			status:     anthropicmodel.StatusOverloaded,
			body:       fmt.Sprintf(anthropicError, "overloaded_error"),
			newPrimary: newAnthropic,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, srv := newFakeServer(t, tt.status, tt.body)
			backup := healthy("backup")
			m := newModel(t, Config{Models: []model.LLM{tt.newPrimary(t, srv.URL), backup}})

			got, err := collect(t, m, request(), false)

			if tt.wantFallBack {
				if err != nil {
					t.Fatalf("GenerateContent() error = %v, want the backup's response", err)
				}
				if diff := cmp.Diff([]string{"backup"}, got); diff != "" {
					t.Errorf("response texts mismatch (-want +got):\n%s", diff)
				}
				return
			}
			if status, _ := statusCode(err); status != tt.status {
				t.Errorf("GenerateContent() error = %v, want the provider's status %d error", err, tt.status)
			}
			if n := backup.callCount(); n != 0 {
				t.Errorf("backup called %d times, want 0", n)
			}
		})
	}
}

func TestRunsInAnAgent(t *testing.T) {
	// The flow names the primary on the request. Each Gemini model sends the
	// name it finds there, so the backup is only reached under its own name if
	// the fallback model renames the request.
	primaryAPI, primarySrv := newFakeServer(t, http.StatusTooManyRequests, fmt.Sprintf(geminiError, 429))
	backupAPI, backupSrv := newFakeServer(t, http.StatusOK,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"answer from the backup"}]},"finishReason":"STOP"}]}`)

	m := newModel(t, Config{Models: []model.LLM{
		newGemini(t, "gemini-primary", primarySrv.URL),
		newGemini(t, "gemini-backup", backupSrv.URL),
	}})
	a, err := llmagent.New(llmagent.Config{Name: "reliable_agent", Model: m, Instruction: "Answer briefly."})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r := testutil.NewTestAgentRunner(t, a)

	var got []string
	for ev, err := range r.Run(t, "session", "hello") {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev.Content != nil {
			got = append(got, texts([]*genai.Content{ev.Content})...)
		}
	}

	if diff := cmp.Diff([]string{"answer from the backup"}, got); diff != "" {
		t.Errorf("event texts mismatch (-want +got):\n%s", diff)
	}
	if paths := primaryAPI.requests(); len(paths) != 1 || !strings.Contains(paths[0], "gemini-primary") {
		t.Errorf("primary requests = %q, want one for gemini-primary", paths)
	}
	if paths := backupAPI.requests(); len(paths) != 1 || !strings.Contains(paths[0], "gemini-backup") {
		t.Errorf("backup requests = %q, want one for gemini-backup", paths)
	}
}
