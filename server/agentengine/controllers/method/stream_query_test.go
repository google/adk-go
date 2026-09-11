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

package method

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

type simpleEvent struct {
	Content *genai.Content `json:"content"`
}

// TestSimpleText checks whether a simple message as string gives the same result as genai.Content.
func TestSimpleText(t *testing.T) {
	agentEngineId := 123
	appName := strconv.Itoa(agentEngineId)
	userID := "u"

	// agent invokes BeforeAgent callback which returns the content as provided as an answer
	a, err := llmagent.New(llmagent.Config{
		Name: "Echo",
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{
			func(cc agent.Context) (*genai.Content, error) {
				return cc.UserContent(), nil
			},
		},
	})
	if err != nil {
		t.Fatalf("failed to create agent: %v", err)
	}

	config := &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(a),
		SessionService: session.InMemoryService(),
	}
	h := NewStreamQueryHandler(config, appName, "async_stream_query", "")

	ctx := t.Context()
	sess, err := config.SessionService.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID})
	if err != nil {
		t.Fatalf("failed to create session: %v", err)
	}

	wantContent := genai.NewContentFromText("Say hello", genai.RoleUser)
	wantBytes, err := json.Marshal(wantContent)
	if err != nil {
		t.Fatalf("json.Marshal() failed: %v", err)
	}
	want := string(wantBytes)

	tests := []struct {
		name    string
		payload string
	}{
		{
			name: "full content",
			payload: `{
"class_method":"async_stream_query",
"input":{
   "message":{
     "parts":[
        {"text":"Say hello"}
      ],
      "role":"user"
   },
   "session_id":"` + sess.Session.ID() + `",
   "user_id":"` + userID + `"}}`,
		},
		{
			name: "simplified content",
			payload: `{
"class_method":"async_stream_query",
"input":{
   "message":"Say hello",
   "session_id":"` + sess.Session.ID() + `",
   "user_id":"` + userID + `"}}`,
		},
	}

	for _, tt := range tests {
		w := newStringWriter()
		b := []byte(tt.payload)
		err := h.streamJSONL(t.Context(), w, b)
		if err != nil {
			t.Fatalf("streamJSONL() failed: %v", err)
		}

		var ev simpleEvent
		p := w.sb.String()

		err = json.Unmarshal([]byte(p), &ev)
		if err != nil {
			t.Fatalf("json.Unmarshal() failed: %v", err)
		}
		gotBytes, err := json.Marshal(ev.Content)
		if err != nil {
			t.Fatalf("json.Marshal() failed: %v", err)
		}
		got := string(gotBytes)
		if got != want {
			t.Errorf("streamJSONL() = %v, want %v", got, want)
		}
	}
}

// mock writer for http
type stringWriter struct {
	sb strings.Builder
	h  http.Header
}

// Header implements [http.ResponseWriter].
func (s *stringWriter) Header() http.Header {
	return s.h
}

// WriteHeader implements [http.ResponseWriter].
func (s *stringWriter) WriteHeader(statusCode int) {
	s.h = http.Header{"Status": []string{http.StatusText(statusCode)}}
}

// Write implements [http.ResponseWriter].
func (s *stringWriter) Write(p []byte) (n int, err error) {
	return s.sb.Write(p)
}

// Flush implements [http.Flusher]
func (s *stringWriter) Flush() {
	// do nothing
}

var (
	_ http.ResponseWriter = (*stringWriter)(nil)
	_ http.Flusher        = (*stringWriter)(nil)
)

func newStringWriter() *stringWriter {
	return &stringWriter{
		sb: strings.Builder{},
		h:  http.Header{},
	}
}

// countingSummarizer records that compaction reached the summarizer.
type countingSummarizer struct {
	mu    sync.Mutex
	calls int
}

func (c *countingSummarizer) SummarizeEvents(context.Context, []*session.Event) (compaction.SummarizeResult, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls++
	return compaction.SummarizeResult{Content: genai.NewContentFromText("a summary", genai.RoleModel)}, nil
}

func (c *countingSummarizer) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

// TestAgentEngineActuallyRunsCompaction pins that the compaction config this
// surface accepts reaches the runners it builds.
//
// Both Agent Engine methods build a runner.Config per request, and deleting the
// line that copies Compaction into it left the whole server suite
// green. A test that only checks a compaction failure is tolerated cannot tell
// that apart: a surface that never runs compaction tolerates its failures
// perfectly. Counting summarizer calls is what distinguishes the two.
func TestAgentEngineActuallyRunsCompaction(t *testing.T) {
	for _, tc := range []struct {
		name    string
		method  string
		stream  func(*launcher.Config, string) func(context.Context, http.ResponseWriter, []byte) error
		payload func(sessionID, userID string) []byte
	}{
		{
			name:   "stream_query",
			method: "async_stream_query",
			stream: func(c *launcher.Config, app string) func(context.Context, http.ResponseWriter, []byte) error {
				return NewStreamQueryHandler(c, app, "async_stream_query", "").streamJSONL
			},
			payload: func(sessionID, userID string) []byte {
				return []byte(`{"class_method":"async_stream_query","input":{"message":{"parts":[{"text":"hi"}],"role":"user",` +
					`"role":"user"},"session_id":"` + sessionID + `","user_id":"` + userID + `"}}`)
			},
		},
		{
			name:   "streaming_agent_run_with_events",
			method: "async_stream_agent_run_with_events",
			stream: func(c *launcher.Config, app string) func(context.Context, http.ResponseWriter, []byte) error {
				return NewStreamingAgentRunWithEventsHandler(c, app, "async_stream_agent_run_with_events", "").streamJSONL
			},
			// This method carries its request as a JSON string inside the
			// envelope rather than as an object.
			payload: func(sessionID, userID string) []byte {
				inner, err := json.Marshal(map[string]any{
					"message":    map[string]any{"role": "user", "parts": []any{map[string]any{"text": "hi"}}},
					"session_id": sessionID,
					"user_id":    userID,
				})
				if err != nil {
					panic(err)
				}
				outer, err := json.Marshal(map[string]any{
					"class_method": "async_stream_agent_run_with_events",
					"input":        map[string]any{"request_json": string(inner)},
				})
				if err != nil {
					panic(err)
				}
				return outer
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const appName, userID = "123", "u"
			a, err := llmagent.New(llmagent.Config{
				Name: "Echo",
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{
					func(cc agent.Context) (*genai.Content, error) { return cc.UserContent(), nil },
				},
			})
			if err != nil {
				t.Fatalf("llmagent.New() error = %v", err)
			}
			summarizer := &countingSummarizer{}
			config := &launcher.Config{
				AgentLoader:    agent.NewSingleLoader(a),
				SessionService: session.InMemoryService(),
				Compaction: &compaction.Config{
					CompactionInterval: 1,
					Summarizer:         summarizer,
				},
			}
			stream := tc.stream(config, appName)

			ctx := t.Context()
			sess, err := config.SessionService.Create(ctx, &session.CreateRequest{AppName: appName, UserID: userID})
			if err != nil {
				t.Fatalf("Create() error = %v", err)
			}
			payload := tc.payload(sess.Session.ID(), userID)
			if err := stream(ctx, newStringWriter(), payload); err != nil {
				t.Fatalf("streamJSONL() error = %v", err)
			}
			if got := summarizer.count(); got == 0 {
				t.Error("the summarizer was never called, so this surface accepts a compaction config and does nothing with it")
			}
		})
	}
}
