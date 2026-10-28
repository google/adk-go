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
	"iter"
	"strings"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// pongModel answers every request with "pong" and records whether the
// runner asked it to stream.
type pongModel struct{ streamed atomic.Bool }

func (*pongModel) Name() string { return "pong" }

func (m *pongModel) GenerateContent(_ context.Context, _ *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	if stream {
		m.streamed.Store(true)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: genai.NewContentFromText("pong", genai.RoleModel)}, nil)
	}
}

func newPongAgent(t *testing.T, m *pongModel) agent.Agent {
	t.Helper()
	a, err := llmagent.New(llmagent.Config{Name: "pong_agent", Model: m})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	return a
}

// runTexts drains one turn and returns the text of every event part.
func runTexts(t *testing.T, b Backend, userID, sessionID, text string) []string {
	t.Helper()
	var texts []string
	for ev, err := range b.Run(t.Context(), userID, sessionID, genai.NewContentFromText(text, genai.RoleUser)) {
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			texts = append(texts, p.Text)
		}
	}
	return texts
}

func TestLocalBackendRunsTurnsInOneSession(t *testing.T) {
	sessions := session.InMemoryService()
	m := &pongModel{}
	b, err := NewLocalBackend(LocalBackendConfig{AppName: "app", Agent: newPongAgent(t, m), SessionService: sessions})
	if err != nil {
		t.Fatalf("NewLocalBackend: %v", err)
	}

	if got := runTexts(t, b, "U1", "s1", "ping"); len(got) != 1 || got[0] != "pong" {
		t.Fatalf("first turn texts = %q, want [pong]", got)
	}
	// The second turn must find the session the first one auto-created.
	if got := runTexts(t, b, "U1", "s1", "ping again"); len(got) != 1 || got[0] != "pong" {
		t.Fatalf("second turn texts = %q, want [pong]", got)
	}
	// Slack posts whole messages, so a streamed turn would only add partial events.
	if m.streamed.Load() {
		t.Error("model was asked to stream, want StreamingModeNone")
	}

	resp, err := sessions.Get(t.Context(), &session.GetRequest{AppName: "app", UserID: "U1", SessionID: "s1"})
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	var authors []string
	for ev := range resp.Session.Events().All() {
		authors = append(authors, ev.Author)
	}
	want := []string{"user", "pong_agent", "user", "pong_agent"}
	if len(authors) != len(want) {
		t.Fatalf("session authors = %q, want %q", authors, want)
	}
	for i := range want {
		if authors[i] != want[i] {
			t.Fatalf("session authors = %q, want %q", authors, want)
		}
	}
}

func TestLocalBackendDefaultsToInMemorySessions(t *testing.T) {
	b, err := NewLocalBackend(LocalBackendConfig{AppName: "app", Agent: newPongAgent(t, &pongModel{})})
	if err != nil {
		t.Fatalf("NewLocalBackend: %v", err)
	}
	if got := runTexts(t, b, "U1", "s1", "ping"); len(got) != 1 || got[0] != "pong" {
		t.Fatalf("texts = %q, want [pong]", got)
	}
}

func TestNewLocalBackendRejectsIncompleteConfig(t *testing.T) {
	a := newPongAgent(t, &pongModel{})
	tests := []struct {
		name  string
		cfg   LocalBackendConfig
		field string
	}{
		{name: "empty AppName", cfg: LocalBackendConfig{Agent: a}, field: "AppName"},
		{name: "nil Agent", cfg: LocalBackendConfig{AppName: "app"}, field: "Agent"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewLocalBackend(tt.cfg)
			// The error names the missing field rather than leaving it to runner.New.
			if err == nil || !strings.Contains(err.Error(), "LocalBackendConfig."+tt.field) {
				t.Fatalf("NewLocalBackend error = %v, want one naming LocalBackendConfig.%s", err, tt.field)
			}
		})
	}
}
