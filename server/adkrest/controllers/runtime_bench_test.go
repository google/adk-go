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

package controllers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/server/authn"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
)

// These measure one /run_sse and one /run turn against a session that already
// holds N events, which is the shape a long conversation reaches. The handler
// loads the session once to check it exists and Runner.Run loads it again, so
// the cost of the second read is what a single-read design removes. The agent
// is a stub that yields one event, so the numbers are framework overhead only.
//
// Run with:
//
//	go test ./server/adkrest/controllers/ -run '^$' -bench 'BenchmarkRun(SSE|)Handler' -benchtime=10x -count=3
func benchSessionService(b *testing.B, backend string) session.Service {
	b.Helper()
	switch backend {
	case "inmemory":
		return session.InMemoryService()
	case "sqlite":
		path := filepath.Join(b.TempDir(), "sessions.db")
		svc, err := database.NewSessionService(sqlite.Open(path))
		if err != nil {
			b.Fatalf("open sqlite: %v", err)
		}
		if err := database.AutoMigrate(svc); err != nil {
			b.Fatalf("migrate: %v", err)
		}
		return svc
	default:
		b.Fatalf("unknown backend %q", backend)
		return nil
	}
}

// seedSession creates a session holding n alternating user/agent text events.
func seedSession(b *testing.B, svc session.Service, sessionID string, n int) {
	b.Helper()
	ctx := context.Background()
	resp, err := svc.Create(ctx, &session.CreateRequest{AppName: "testApp", UserID: "testUser", SessionID: sessionID})
	if err != nil {
		b.Fatalf("create session: %v", err)
	}
	text := strings.Repeat("lorem ipsum dolor sit amet ", 8) // ~216 chars, a short chat turn
	for i := range n {
		ev := session.NewEvent(ctx, fmt.Sprintf("inv-%d", i/2))
		ev.Author = "user"
		if i%2 == 1 {
			ev.Author = "testApp"
		}
		ev.Timestamp = time.Now()
		ev.LLMResponse.Content = &genai.Content{Parts: []*genai.Part{{Text: text}}}
		if err := svc.AppendEvent(ctx, resp.Session, ev); err != nil {
			b.Fatalf("append event %d: %v", i, err)
		}
	}
}

func benchController(b *testing.B, svc session.Service) *RuntimeAPIController {
	b.Helper()
	stub, err := agent.New(agent.Config{
		Name: "testApp",
		Run:  testAgent([]testAgentResult{{event: makeEvent("inv", "testApp", "ok")}}),
	})
	if err != nil {
		b.Fatalf("agent.New: %v", err)
	}
	return NewRuntimeAPIControllerWithConfig(RuntimeAPIControllerConfig{
		SessionService: svc,
		AgentLoader:    agent.NewSingleLoader(stub),
		SSETimeout:     10 * time.Second,
	})
}

func benchRequest(b *testing.B, path, sessionID string, streaming bool) *http.Request {
	b.Helper()
	body, _ := json.Marshal(models.RunAgentRequest{
		AppName:    "testApp",
		UserId:     "testUser",
		SessionId:  sessionID,
		Streaming:  streaming,
		NewMessage: genai.Content{Parts: []*genai.Part{{Text: "Hello"}}},
	})
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	return req.WithContext(authn.WithCaller(context.Background(), &authn.Caller{UserID: "testUser"}))
}

var benchSizes = []int{100, 500, 2000}

func BenchmarkRunSSEHandler(b *testing.B) {
	for _, backend := range []string{"inmemory", "sqlite"} {
		for _, n := range benchSizes {
			b.Run(fmt.Sprintf("%s/events=%d", backend, n), func(b *testing.B) {
				svc := benchSessionService(b, backend)
				c := benchController(b, svc)
				// Each turn appends two events, so a fresh session per
				// iteration keeps N fixed. Seeding is outside the timer.
				for i := 0; b.Loop(); i++ {
					b.StopTimer()
					sid := fmt.Sprintf("s-%d", i)
					seedSession(b, svc, sid, n)
					req := benchRequest(b, "/run_sse", sid, true)
					w := &recorderWithDeadline{httptest.NewRecorder()}
					b.StartTimer()

					c.RunSSEHandler(w, req)

					if w.Code != http.StatusOK {
						b.Fatalf("status %d: %s", w.Code, w.Body.String())
					}
				}
			})
		}
	}
}

func BenchmarkRunHandler(b *testing.B) {
	for _, backend := range []string{"inmemory", "sqlite"} {
		for _, n := range benchSizes {
			b.Run(fmt.Sprintf("%s/events=%d", backend, n), func(b *testing.B) {
				svc := benchSessionService(b, backend)
				c := benchController(b, svc)
				for i := 0; b.Loop(); i++ {
					b.StopTimer()
					sid := fmt.Sprintf("s-%d", i)
					seedSession(b, svc, sid, n)
					req := benchRequest(b, "/run", sid, false)
					w := httptest.NewRecorder()
					b.StartTimer()

					if err := c.RunHandler(w, req); err != nil {
						b.Fatalf("RunHandler: %v", err)
					}
					if w.Code != http.StatusOK {
						b.Fatalf("status %d: %s", w.Code, w.Body.String())
					}
				}
			})
		}
	}
}
