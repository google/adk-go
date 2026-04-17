// Copyright 2025 Google LLC
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

package controllers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/plugin"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/server/adkrest/internal/fakes"
	"google.golang.org/adk/server/adkrest/internal/models"
	"google.golang.org/adk/session"
)

func TestNewRuntimeAPIController_PluginsAssignment(t *testing.T) {
	p1, err := plugin.New(plugin.Config{Name: "plugin1"})
	if err != nil {
		t.Fatalf("plugin.New() failed for plugin1: %v", err)
	}

	p2, err := plugin.New(plugin.Config{Name: "plugin2"})
	if err != nil {
		t.Fatalf("plugin.New() failed for plugin2: %v", err)
	}

	tc := []struct {
		name        string
		plugins     []*plugin.Plugin
		wantPlugins int
	}{
		{
			name:        "with no plugins",
			plugins:     nil,
			wantPlugins: 0,
		},
		{
			name:        "with empty plugin list",
			plugins:     []*plugin.Plugin{},
			wantPlugins: 0,
		},
		{
			name:        "with single plugin",
			plugins:     []*plugin.Plugin{p1},
			wantPlugins: 1,
		},
		{
			name:        "with multiple plugins",
			plugins:     []*plugin.Plugin{p1, p2},
			wantPlugins: 2,
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			controller := NewRuntimeAPIController(nil, nil, nil, nil, 10*time.Second, runner.PluginConfig{
				Plugins: tt.plugins,
			})

			if controller == nil {
				t.Fatal("NewRuntimeAPIController returned nil")
			}

			if got := len(controller.pluginConfig.Plugins); got != tt.wantPlugins {
				t.Errorf("NewRuntimeAPIController() plugins count = %v, want %v", got, tt.wantPlugins)
			}
		})
	}
}

type recorderWithDeadline struct {
	*httptest.ResponseRecorder
}

func (r *recorderWithDeadline) SetWriteDeadline(t time.Time) error {
	return nil
}

type testAgentResult struct {
	event *session.Event
	err   error
}

func testAgent(results []testAgentResult) func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
	return func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(yield func(*session.Event, error) bool) {
			for _, res := range results {
				if !yield(res.event, res.err) {
					return
				}
			}
		}
	}
}

func makeEvent(id, author, text string) *session.Event {
	e := session.NewEvent(id)
	e.Author = author
	e.LLMResponse.Content = &genai.Content{
		Parts: []*genai.Part{{Text: text}},
	}
	return e
}

func TestRunSSEHandler(t *testing.T) {
	tc := []struct {
		name       string
		results    []testAgentResult
		wantStatus int
		wantBody   []string
	}{
		{
			name: "success case",
			results: []testAgentResult{
				{event: makeEvent("invocation-1", "testApp", "Hello from agent"), err: nil},
			},
			wantStatus: http.StatusOK,
			wantBody:   []string{"data: {", "Hello from agent"},
		},
		{
			name: "error case",
			results: []testAgentResult{
				{err: fmt.Errorf("agent failed")},
			},
			wantStatus: http.StatusOK,
			wantBody:   []string{"event: error\ndata: {\"error\":\"agent failed\"}\n\n"},
		},
		{
			name: "interleaved success and error",
			results: []testAgentResult{
				{event: makeEvent("invocation-1", "testApp", "Hello from agent"), err: nil},
				{err: fmt.Errorf("agent failed")},
				{event: makeEvent("invocation-1", "testApp", "More data"), err: nil},
				{err: fmt.Errorf("agent failed again")},
			},
			wantStatus: http.StatusOK,
			wantBody: []string{
				"data: {", "Hello from agent",
				"event: error\ndata: {\"error\":\"agent failed\"}\n\n",
				"data: {", "More data",
				"event: error\ndata: {\"error\":\"agent failed again\"}\n\n",
			},
		},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			// Setup fake agent with testAgent(tt.results)
			fakeAgent, err := agent.New(agent.Config{
				Name: "testApp",
				Run:  testAgent(tt.results),
			})
			if err != nil {
				t.Fatalf("agent.New failed: %v", err)
			}

			// Setup fake session service
			id := fakes.SessionKey{
				AppName:   "testApp",
				UserID:    "testUser",
				SessionID: "testSession",
			}
			sessionService := fakes.FakeSessionService{
				Sessions: map[fakes.SessionKey]fakes.TestSession{
					id: {
						Id:            id,
						SessionState:  fakes.TestState{},
						SessionEvents: fakes.TestEvents{},
						UpdatedAt:     time.Now(),
					},
				},
			}

			// Setup controller
			controller := NewRuntimeAPIController(
				&sessionService,
				nil,
				agent.NewSingleLoader(fakeAgent),
				nil,
				10*time.Second,
				runner.PluginConfig{},
			)

			// Create request
			reqObj := models.RunAgentRequest{
				AppName:   "testApp",
				UserId:    "testUser",
				SessionId: "testSession",
				Streaming: true,
				NewMessage: genai.Content{
					Parts: []*genai.Part{{Text: "Hello"}},
				},
			}
			reqBytes, _ := json.Marshal(reqObj)
			req := httptest.NewRequest(http.MethodPost, "/run-sse", bytes.NewBuffer(reqBytes))

			// Record response
			rr := httptest.NewRecorder()
			w := &recorderWithDeadline{rr}

			// Call handler
			controller.RunSSEHandler(w, req)

			// Verify response
			if rr.Code != tt.wantStatus {
				t.Errorf("expected status %d, got %d", tt.wantStatus, rr.Code)
			}

			body := rr.Body.String()
			for _, s := range tt.wantBody {
				if !strings.Contains(body, s) {
					t.Errorf("expected body to contain %q, got %s", s, body)
				}
			}
		})
	}
}
