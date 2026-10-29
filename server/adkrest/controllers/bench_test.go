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
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/glebarez/sqlite"
	"github.com/gorilla/mux"
	"google.golang.org/genai"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/benchutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/server/adkrest/internal/models"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/database"
)

type benchEchoModel struct{}

func (benchEchoModel) Name() string { return "echo" }

func (benchEchoModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content:      genai.NewContentFromText(benchutil.BenchText, genai.RoleModel),
			FinishReason: genai.FinishReasonStop,
			TurnComplete: true,
		}, nil)
	}
}

func benchSQLite(b *testing.B) session.Service {
	b.Helper()
	db, err := gorm.Open(sqlite.Open(b.TempDir()+"/bench.db"), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		b.Fatal(err)
	}
	svc, err := database.NewSessionServiceFromDB(db)
	if err != nil {
		b.Fatal(err)
	}
	if err := database.AutoMigrate(svc); err != nil {
		b.Fatal(err)
	}
	return svc
}

type benchServer struct {
	runtime  *RuntimeAPIController
	sessions *SessionsAPIController
	svc      *benchutil.CountingSessionService
	router   *mux.Router
}

func newBenchServer(b *testing.B, backend session.Service, events int) *benchServer {
	b.Helper()
	svc := &benchutil.CountingSessionService{Service: backend}
	if _, err := benchutil.SeedSession(b.Context(), svc, "app", "user", "s", "app", events); err != nil {
		b.Fatal(err)
	}
	a, err := llmagent.New(llmagent.Config{Name: "app", Model: benchEchoModel{}, Instruction: "You are terse."})
	if err != nil {
		b.Fatal(err)
	}
	rt := NewRuntimeAPIControllerWithConfig(RuntimeAPIControllerConfig{
		SessionService: svc,
		AgentLoader:    agent.NewSingleLoader(a),
	})
	ss := NewSessionsAPIController(svc)
	router := mux.NewRouter()
	router.HandleFunc("/run", NewErrorHandler(rt.RunHandler)).Methods(http.MethodPost)
	router.HandleFunc("/run_sse", rt.RunSSEHandler).Methods(http.MethodPost)
	router.HandleFunc("/apps/{app_name}/users/{user_id}/sessions/{session_id}", ss.GetSessionHandler).Methods(http.MethodGet)
	router.HandleFunc("/apps/{app_name}/users/{user_id}/sessions/{session_id}", ss.UpdateSessionHandler).Methods(http.MethodPatch)
	return &benchServer{runtime: rt, sessions: ss, svc: svc, router: router}
}

func runBody(b *testing.B, streaming bool) []byte {
	b.Helper()
	body, err := json.Marshal(models.RunAgentRequest{
		AppName: "app", UserId: "user", SessionId: "s",
		NewMessage: *genai.NewContentFromText("hello", genai.RoleUser),
		Streaming:  streaming,
	})
	if err != nil {
		b.Fatal(err)
	}
	return body
}

// benchHandler serves path once per iteration against a freshly seeded
// session, outside the timer, so the history is exactly n events rather than
// growing with the iteration count.
func benchHandler(b *testing.B, path string, newService func(*testing.B) session.Service, backend string) {
	for _, n := range []int{0, 100, 1000} {
		b.Run(fmt.Sprintf("%s/events=%d", backend, n), func(b *testing.B) {
			var body []byte
			method := http.MethodPost
			switch path {
			case "/run":
				body = runBody(b, false)
			case "/run_sse":
				body = runBody(b, true)
			default:
				method = http.MethodGet
			}
			var gets, appends, loaded int64
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				srv := newBenchServer(b, newService(b), n)
				srv.svc.Reset()
				req := httptest.NewRequest(method, path, bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				rec := &recorderWithDeadline{httptest.NewRecorder()}
				b.StartTimer()
				srv.router.ServeHTTP(rec, req)
				b.StopTimer()
				if rec.Code != http.StatusOK {
					b.Fatalf("%s: status %d: %s", path, rec.Code, rec.Body.String())
				}
				gets += srv.svc.Gets.Load()
				appends += srv.svc.Appends.Load()
				loaded += srv.svc.EventsLoaded.Load()
				b.StartTimer()
			}
			b.ReportMetric(float64(gets)/float64(b.N), "gets/req")
			b.ReportMetric(float64(loaded)/float64(b.N), "events-loaded/req")
			b.ReportMetric(float64(appends)/float64(b.N), "appends/req")
		})
	}
}

// BenchmarkRunHandler is POST /run end to end, with a model that answers
// instantly, so what remains is the server's own per-request work: decode,
// existence check, runner construction, session load, prompt assembly, two
// appends, and encoding every event of the turn.
func BenchmarkRunHandler(b *testing.B) {
	benchHandler(b, "/run", func(*testing.B) session.Service { return session.InMemoryService() }, "inmemory")
	benchHandler(b, "/run", benchSQLite, "sqlite")
}

func BenchmarkRunSSEHandler(b *testing.B) {
	benchHandler(b, "/run_sse", func(*testing.B) session.Service { return session.InMemoryService() }, "inmemory")
	benchHandler(b, "/run_sse", benchSQLite, "sqlite")
}

// BenchmarkRunSSEChunks streams 100 partial model events through /run_sse to
// isolate per-event DTO mapping, JSON marshalling, and SSE frame writes.
func BenchmarkRunSSEChunks(b *testing.B) {
	b.Run("inmemory/chunks=100", func(b *testing.B) {
		const chunks = 100
		events := make([]*session.Event, chunks)
		for i := range events {
			ev := session.NewEvent(b.Context(), fmt.Sprintf("inv-%d", i))
			ev.Author = "app"
			ev.Partial = true
			ev.LLMResponse.Partial = true
			ev.LLMResponse.Content = genai.NewContentFromText(benchutil.BenchText, genai.RoleModel)
			events[i] = ev
		}
		streamAgent, err := agent.New(agent.Config{
			Name: "app",
			Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for _, ev := range events {
						if !yield(ev, nil) {
							return
						}
					}
				}
			},
		})
		if err != nil {
			b.Fatal(err)
		}
		body := runBody(b, true)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rt := NewRuntimeAPIControllerWithConfig(RuntimeAPIControllerConfig{
				SessionService:    session.InMemoryService(),
				AgentLoader:       agent.NewSingleLoader(streamAgent),
				AutoCreateSession: true,
			})
			req := httptest.NewRequest(http.MethodPost, "/run_sse", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			rec := &recorderWithDeadline{httptest.NewRecorder()}
			rt.RunSSEHandler(rec, req)
			if rec.Code != http.StatusOK {
				b.Fatalf("status %d: %s", rec.Code, rec.Body.String())
			}
		}
	})
}

// BenchmarkGetSessionHandler is what the web UI calls when a conversation is
// opened: one Get plus the JSON encoding of every event.
func BenchmarkGetSessionHandler(b *testing.B) {
	const path = "/apps/app/users/user/sessions/s"
	benchHandler(b, path, func(*testing.B) session.Service { return session.InMemoryService() }, "inmemory")
	benchHandler(b, path, benchSQLite, "sqlite")
}

// BenchmarkUpdateSessionHandler is the rename/state-edit PATCH: it reads the
// session twice around one append.
func BenchmarkUpdateSessionHandler(b *testing.B) {
	for _, n := range []int{0, 100, 1000} {
		b.Run(fmt.Sprintf("inmemory/events=%d", n), func(b *testing.B) {
			var gets, appends, loaded int64
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				srv := newBenchServer(b, session.InMemoryService(), n)
				srv.svc.Reset()
				body := `{"stateDelta":{"__session_metadata__":{"displayName":"renamed"}}}`
				req := httptest.NewRequest(http.MethodPatch, "/apps/app/users/user/sessions/s", bytes.NewBufferString(body))
				rec := httptest.NewRecorder()
				b.StartTimer()
				srv.router.ServeHTTP(rec, req)
				b.StopTimer()
				if rec.Code != http.StatusOK {
					b.Fatalf("status %d: %s", rec.Code, rec.Body.String())
				}
				gets += srv.svc.Gets.Load()
				appends += srv.svc.Appends.Load()
				loaded += srv.svc.EventsLoaded.Load()
				b.StartTimer()
			}
			b.ReportMetric(float64(gets)/float64(b.N), "gets/req")
			b.ReportMetric(float64(loaded)/float64(b.N), "events-loaded/req")
			b.ReportMetric(float64(appends)/float64(b.N), "appends/req")
		})
	}
}
