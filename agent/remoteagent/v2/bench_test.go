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

package remoteagent

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

func BenchmarkRemoteAgentInvocation(b *testing.B) {
	for _, cardDelay := range []time.Duration{0, 20 * time.Millisecond} {
		b.Run(fmt.Sprintf("card-delay=%s", cardDelay), func(b *testing.B) {
			var cardFetches atomic.Int64
			executor := &mockA2AExecutor{
				executeFn: func(_ context.Context, execCtx *a2asrv.ExecutorContext) iter.Seq2[a2a.Event, error] {
					return func(yield func(a2a.Event, error) bool) {
						msg := a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("pong"))
						msg.TaskID = execCtx.TaskID
						msg.ContextID = execCtx.ContextID
						yield(msg, nil)
					}
				},
			}
			handler := a2asrv.NewHandler(executor)

			var srv *httptest.Server
			mux := http.NewServeMux()
			mux.Handle("/invoke", a2asrv.NewJSONRPCHandler(handler))
			mux.HandleFunc("/.well-known/agent-card.json", func(w http.ResponseWriter, _ *http.Request) {
				cardFetches.Add(1)
				if cardDelay > 0 {
					time.Sleep(cardDelay)
				}
				card := &a2a.AgentCard{
					Name: "bench-remote",
					SupportedInterfaces: []*a2a.AgentInterface{
						a2a.NewAgentInterface(srv.URL+"/invoke", a2a.TransportProtocolJSONRPC),
					},
					Capabilities: a2a.AgentCapabilities{Streaming: true},
				}
				_ = json.NewEncoder(w).Encode(card)
			})
			srv = httptest.NewServer(mux)
			b.Cleanup(srv.Close)

			remoteAgent, err := NewA2A(A2AConfig{
				Name:              "bench_a2a",
				AgentCardProvider: NewAgentCardProvider(srv.URL),
			})
			if err != nil {
				b.Fatal(err)
			}

			sessSvc := session.InMemoryService()
			created, err := sessSvc.Create(b.Context(), &session.CreateRequest{
				AppName:   "app",
				UserID:    "user",
				SessionID: "s1",
			})
			if err != nil {
				b.Fatal(err)
			}
			userEvent := session.NewEvent(b.Context(), "init")
			userEvent.Author = "user"
			userEvent.Content = genai.NewContentFromText("hello", genai.RoleUser)
			if err := sessSvc.AppendEvent(b.Context(), created.Session, userEvent); err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			invocations := 0
			for b.Loop() {
				invocations++
				ic := icontext.NewInvocationContext(b.Context(), icontext.InvocationContextParams{
					Agent:        remoteAgent,
					Session:      created.Session,
					UserContent:  userEvent.Content,
					InvocationID: fmt.Sprintf("inv-%d", invocations),
					RunConfig:    &agent.RunConfig{StreamingMode: agent.StreamingModeNone},
				})
				for ev, err := range remoteAgent.Run(ic) {
					if err != nil {
						b.Fatal(err)
					}
					if ev != nil && ev.ErrorMessage != "" {
						b.Fatalf("remoteAgent error event: %s", ev.ErrorMessage)
					}
				}
			}
			b.ReportMetric(float64(cardFetches.Load())/float64(invocations), "card-fetches/invocation")
		})
	}
}
