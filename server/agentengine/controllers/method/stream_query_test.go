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
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/cmd/launcher"
	"google.golang.org/adk/session"
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
			func(cc agent.CallbackContext) (*genai.Content, error) {
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
