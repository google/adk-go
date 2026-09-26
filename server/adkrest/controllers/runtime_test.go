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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/server/adkrest/internal/fakes"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
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
			}, false)

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
	e := session.NewEvent(context.Background(), id)
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
				false,
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

func TestDecodeRequestBody_AcceptsFunctionCallEventID(t *testing.T) {
	body := `{
		"appName": "a",
		"userId": "u",
		"sessionId": "s",
		"newMessage": {"role": "user", "parts": [{"text": "hi"}]},
		"functionCallEventId": "fce-1"
	}`
	req := httptest.NewRequest(http.MethodPost, "/run", bytes.NewBufferString(body))
	rr := httptest.NewRecorder()

	got, err := decodeRequestBody(rr, req)
	if err != nil {
		t.Fatalf("decodeRequestBody: unexpected error: %v", err)
	}
	if got.FunctionCallEventID == nil || *got.FunctionCallEventID != "fce-1" {
		t.Errorf("FunctionCallEventID = %v, want %q", got.FunctionCallEventID, "fce-1")
	}
}

func TestDecodeRequestBody_RejectsUnknownFields(t *testing.T) {
	body := `{
		"appName": "a",
		"userId": "u",
		"sessionId": "s",
		"newMessage": {},
		"totallyMadeUpField": 123
	}`
	req := httptest.NewRequest(http.MethodPost, "/run", bytes.NewBufferString(body))
	rr := httptest.NewRecorder()

	if _, err := decodeRequestBody(rr, req); err == nil {
		t.Errorf("decodeRequestBody: expected error for unknown field, got nil")
	}
}

func TestDecodeRequestBody_EnforcesMaxBytesReader(t *testing.T) {
	// Create a payload > 10MB
	largeBody := bytes.Repeat([]byte("a"), 11*1024*1024)
	req := httptest.NewRequest(http.MethodPost, "/run", bytes.NewBuffer(largeBody))
	rr := httptest.NewRecorder()

	if _, err := decodeRequestBody(rr, req); err == nil {
		t.Errorf("decodeRequestBody: expected error for payload > 10MB, got nil")
	}
}

func TestRunLiveHandler_WebSocketReadLimit(t *testing.T) {
	fakeAgent, err := agent.New(agent.Config{
		Name: "app",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New failed: %v", err)
	}

	id := fakes.SessionKey{
		AppName:   "app",
		UserID:    "user",
		SessionID: "sess",
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

	controller := NewRuntimeAPIController(
		&sessionService,
		nil,
		agent.NewSingleLoader(fakeAgent),
		nil,
		10*time.Second,
		runner.PluginConfig{},
		false,
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = controller.RunLiveHandler(w, r)
	}))
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "?appName=app&userId=user&sessionId=sess"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("dial failed: %v", err)
	}
	defer func() { _ = conn.Close() }()

	// Send message > 10MB to test read limit enforcement
	largeMsg := bytes.Repeat([]byte("x"), 11*1024*1024)
	err = conn.WriteMessage(websocket.BinaryMessage, largeMsg)
	if err != nil {
		// Connection might be closed early when write/read limit triggers
		return
	}

	// Reading response or next message should fail due to connection closure / read limit violation
	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, _, err = conn.ReadMessage()
	if err == nil {
		t.Errorf("expected error when sending message > 10MB, got nil")
	}
}

func TestRunLiveHandler_SanitizedInternalErrorCloseReason(t *testing.T) {
	fakeAgent, err := agent.New(agent.Config{
		Name: "app",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {}
		},
	})
	if err != nil {
		t.Fatalf("agent.New failed: %v", err)
	}

	sessionService := fakes.FakeSessionService{}
	controller := NewRuntimeAPIController(
		&sessionService,
		nil,
		agent.NewSingleLoader(fakeAgent),
		nil,
		10*time.Second,
		runner.PluginConfig{},
		false,
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = controller.RunLiveHandler(w, r)
	}))
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "?appName=unknownApp&userId=user&sessionId=sess"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		t.Fatalf("dial failed: %v", err)
	}
	defer func() { _ = conn.Close() }()

	_ = conn.SetReadDeadline(time.Now().Add(2 * time.Second))
	_, _, err = conn.ReadMessage()
	if err == nil {
		t.Fatalf("expected close error, got nil")
	}

	var closeErr *websocket.CloseError
	if !errors.As(err, &closeErr) {
		t.Fatalf("expected *websocket.CloseError, got %T: %v", err, err)
	}

	if closeErr.Code != websocket.CloseInternalServerErr {
		t.Errorf("expected close code %d, got %d", websocket.CloseInternalServerErr, closeErr.Code)
	}

	if closeErr.Text != "internal server error" {
		t.Errorf("expected sanitized close text %q, got %q", "internal server error", closeErr.Text)
	}
}

func TestRunSSEHandler_SanitizesInternalServerError(t *testing.T) {
	// Setup session service with valid session
	id := fakes.SessionKey{
		AppName:   "nonExistentApp",
		UserID:    "user",
		SessionID: "sess",
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

	// Controller with agent loader that will fail to find nonExistentApp
	fakeAgent, _ := agent.New(agent.Config{Name: "otherApp"})
	controller := NewRuntimeAPIController(
		&sessionService,
		nil,
		agent.NewSingleLoader(fakeAgent),
		nil,
		10*time.Second,
		runner.PluginConfig{},
		false,
	)

	reqObj := models.RunAgentRequest{
		AppName:   "nonExistentApp",
		UserId:    "user",
		SessionId: "sess",
		Streaming: true,
		NewMessage: genai.Content{
			Parts: []*genai.Part{{Text: "Hello"}},
		},
	}
	reqBytes, _ := json.Marshal(reqObj)
	req := httptest.NewRequest(http.MethodPost, "/run-sse", bytes.NewBuffer(reqBytes))

	rr := httptest.NewRecorder()
	w := &recorderWithDeadline{rr}

	controller.RunSSEHandler(w, req)

	if rr.Code != http.StatusInternalServerError {
		t.Errorf("expected status %d, got %d", http.StatusInternalServerError, rr.Code)
	}

	gotBody := rr.Body.String()
	wantBody := "internal server error\n"
	if gotBody != wantBody {
		t.Errorf("expected sanitized body %q, got %q", wantBody, gotBody)
	}
}

func TestRunSSEHandler_SanitizesBadRequestAndNotFoundErrors(t *testing.T) {
	fakeAgent, _ := agent.New(agent.Config{Name: "app"})
	sessionService := fakes.FakeSessionService{}
	controller := NewRuntimeAPIController(
		&sessionService,
		nil,
		agent.NewSingleLoader(fakeAgent),
		nil,
		10*time.Second,
		runner.PluginConfig{},
		false,
	)

	t.Run("bad request", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/run-sse", bytes.NewBufferString("{invalid json}"))
		rr := httptest.NewRecorder()
		w := &recorderWithDeadline{rr}

		controller.RunSSEHandler(w, req)

		if rr.Code != http.StatusBadRequest {
			t.Errorf("expected status %d, got %d", http.StatusBadRequest, rr.Code)
		}
		if got := rr.Body.String(); got != "bad request\n" {
			t.Errorf("expected body %q, got %q", "bad request\n", got)
		}
	})

	t.Run("not found", func(t *testing.T) {
		reqObj := models.RunAgentRequest{
			AppName:   "app",
			UserId:    "user",
			SessionId: "nonExistentSession",
			Streaming: true,
			NewMessage: genai.Content{
				Parts: []*genai.Part{{Text: "Hello"}},
			},
		}
		reqBytes, _ := json.Marshal(reqObj)
		req := httptest.NewRequest(http.MethodPost, "/run-sse", bytes.NewBuffer(reqBytes))
		rr := httptest.NewRecorder()
		w := &recorderWithDeadline{rr}

		controller.RunSSEHandler(w, req)

		if rr.Code != http.StatusNotFound {
			t.Errorf("expected status %d, got %d", http.StatusNotFound, rr.Code)
		}
		if got := rr.Body.String(); got != "not found\n" {
			t.Errorf("expected body %q, got %q", "not found\n", got)
		}
	})
}
