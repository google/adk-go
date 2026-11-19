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

package a2a

import (
	"fmt"
	"iter"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	a2alegacy "github.com/a2aproject/a2a-go/a2a"
	a2alegacyclient "github.com/a2aproject/a2a-go/a2aclient"
	a2alegacyagentcard "github.com/a2aproject/a2a-go/a2aclient/agentcard"
	a2acore "github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2aclient"
	"github.com/a2aproject/a2a-go/v2/a2aclient/agentcard"
	"github.com/gorilla/mux"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/cmd/launcher"
	"google.golang.org/adk/v2/cmd/launcher/web"
	"google.golang.org/adk/v2/cmd/launcher/web/api"
	"google.golang.org/adk/v2/session"
)

func getFreePort(t *testing.T) int {
	t.Helper()
	addr, err := net.ResolveTCPAddr("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("net.ResolveTCPAddr() error = %v", err)
	}
	listener, err := net.ListenTCP("tcp", addr)
	if err != nil {
		t.Fatalf("net.ListenTCP() error = %v", err)
	}
	tcpAddr, ok := listener.Addr().(*net.TCPAddr)
	if !ok {
		t.Fatalf("listener.Addr() = %T, want net.TCPAddr", listener.Addr())
	}
	port := tcpAddr.Port
	if err := listener.Close(); err != nil {
		t.Fatalf("listener.Close() error = %v", err)
	}
	return port
}

func TestWebLauncher_ServesA2A(t *testing.T) {
	ctx := t.Context()

	port := getFreePort(t)

	l := web.NewLauncher(NewLauncher())
	_, err := l.Parse([]string{
		"--port", strconv.Itoa(port),
		"a2a", "--a2a_agent_url", "http://localhost:" + strconv.Itoa(port),
	})
	if err != nil {
		t.Fatalf("web.NewLauncher() error = %v", err)
	}

	wantMessage := "Hello, world!"
	agnt, err := agent.New(agent.Config{
		Name: "HelloWorldAgent",
		Run: func(ic agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := session.NewEvent(ic, ic.InvocationID())
				event.Content = genai.NewContentFromText(wantMessage, genai.RoleModel)
				yield(event, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	config := &launcher.Config{
		AgentLoader:    agent.NewSingleLoader(agnt),
		SessionService: session.InMemoryService(),
	}

	go func() {
		if err := l.Run(t.Context(), config); err != nil {
			t.Errorf("launcher.Run() error = %v", err)
		}
	}()

	t.Run("A2A v2 client", func(t *testing.T) {
		var card *a2acore.AgentCard
		for retry := range 3 {
			time.Sleep(10 * time.Millisecond) // give server time to start
			card, err = agentcard.DefaultResolver.Resolve(ctx, "http://localhost:"+strconv.Itoa(port))
			if err == nil {
				break
			}
			if retry == 2 {
				t.Fatalf("cardResolver.Resolve() error = %v", err)
			}
		}

		client, err := a2aclient.NewFromCard(ctx, card)
		if err != nil {
			t.Fatalf("a2aclient.NewFromCard() error = %v", err)
		}

		got, err := client.SendMessage(ctx, &a2acore.SendMessageRequest{
			Message: a2acore.NewMessage(a2acore.MessageRoleUser, a2acore.NewTextPart("Hi!")),
		})
		if err != nil {
			t.Fatalf("client.SendMessage() error = %v", err)
		}
		task, ok := got.(*a2acore.Task)
		if !ok {
			t.Fatalf("client.SendMessage() result type = %T, want a2a.Task", got)
		}
		if len(task.Artifacts) != 1 {
			t.Fatalf("len(task.Artifacts) = %d, want 1", len(task.Artifacts))
		}
		parts := task.Artifacts[0].Parts
		if len(parts) != 1 {
			t.Fatalf("len(task.Artifacts[0].Parts) = %d, want 1", len(parts))
		}
		if gotPart := parts[0].Text(); gotPart != wantMessage {
			t.Fatalf("task.Artifacts[0].Parts[0] = %v, want %v", parts[0], wantMessage)
		}
	})

	t.Run("A2A v0 client", func(t *testing.T) {
		var card *a2alegacy.AgentCard
		for retry := range 3 {
			time.Sleep(10 * time.Millisecond) // give server time to start
			card, err = a2alegacyagentcard.DefaultResolver.Resolve(ctx, "http://localhost:"+strconv.Itoa(port))
			if err == nil {
				break
			}
			if retry == 2 {
				t.Fatalf("a2alegacyagentcard.DefaultResolver.Resolve() error = %v", err)
			}
		}

		client, err := a2alegacyclient.NewFromCard(ctx, card)
		if err != nil {
			t.Fatalf("a2alegacyclient.NewFromCard() error = %v", err)
		}

		got, err := client.SendMessage(ctx, &a2alegacy.MessageSendParams{
			Message: a2alegacy.NewMessage(a2alegacy.MessageRoleUser, a2alegacy.TextPart{Text: "Hi!"}),
		})
		if err != nil {
			t.Fatalf("client.SendMessage() error = %v", err)
		}
		task, ok := got.(*a2alegacy.Task)
		if !ok {
			t.Fatalf("client.SendMessage() result type = %T, want a2alegacy.Task", got)
		}
		if len(task.Artifacts) != 1 {
			t.Fatalf("len(task.Artifacts) = %d, want 1", len(task.Artifacts))
		}
		parts := task.Artifacts[0].Parts
		if len(parts) != 1 {
			t.Fatalf("len(task.Artifacts[0].Parts) = %d, want 1", len(parts))
		}
		if gotPart := parts[0].(a2alegacy.TextPart); gotPart.Text != wantMessage {
			t.Fatalf("task.Artifacts[0].Parts[0] = %v, want %v", parts[0], wantMessage)
		}
	})
}

// routerCaptureSublauncher records the root router built by web.NewLauncher so
// a test can send requests through the assembled middleware and sublaunchers.
type routerCaptureSublauncher struct {
	router *mux.Router
}

func (s *routerCaptureSublauncher) Keyword() string                       { return "capture" }
func (s *routerCaptureSublauncher) Parse(args []string) ([]string, error) { return args, nil }
func (s *routerCaptureSublauncher) CommandLineSyntax() string             { return "" }
func (s *routerCaptureSublauncher) SimpleDescription() string             { return "" }
func (s *routerCaptureSublauncher) UserMessage(string, func(v ...any))    {}
func (s *routerCaptureSublauncher) SetupSubrouters(r *mux.Router, _ *launcher.Config) error {
	s.router = r
	return nil
}

// TestA2AAgentURLDoesNotWidenAPIOrigins pins that -a2a_agent_url (whether left
// at its http://localhost:8080 default or set explicitly) does not widen which
// origins /api admits, and that what /api admits does not depend on whether
// a2a is set up before api (as in full.NewLauncher) or after api (as in
// prod.NewLauncher).
func TestA2AAgentURLDoesNotWidenAPIOrigins(t *testing.T) {
	agnt, err := agent.New(agent.Config{Name: "HelloWorldAgent"})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	for _, order := range []struct {
		name  string
		build func(a2aSub, apiSub, capSub web.Sublauncher) launcher.SubLauncher
		args  func(a2aArgs []string) []string
	}{
		{
			name: "a2a before api",
			build: func(a2aSub, apiSub, capSub web.Sublauncher) launcher.SubLauncher {
				return web.NewLauncher(a2aSub, apiSub, capSub)
			},
			args: func(a2aArgs []string) []string {
				out := []string{"-allow_origins", "https://allowed.example.com", "a2a"}
				out = append(out, a2aArgs...)
				return append(out, "api", "-webui_address", "localhost:9000", "capture")
			},
		},
		{
			name: "api before a2a",
			build: func(a2aSub, apiSub, capSub web.Sublauncher) launcher.SubLauncher {
				return web.NewLauncher(apiSub, a2aSub, capSub)
			},
			args: func(a2aArgs []string) []string {
				out := []string{"-allow_origins", "https://allowed.example.com", "api", "-webui_address", "localhost:9000", "a2a"}
				out = append(out, a2aArgs...)
				return append(out, "capture")
			},
		},
	} {
		t.Run(order.name, func(t *testing.T) {
			for _, tc := range []struct {
				name       string
				a2aArgs    []string
				origin     string
				wantStatus int
				wantCORS   string
			}{
				{
					name:       "default a2a_agent_url is refused on /api",
					origin:     "http://localhost:8080",
					wantStatus: http.StatusForbidden,
					wantCORS:   "",
				},
				{
					name:       "explicit a2a_agent_url is refused on /api",
					a2aArgs:    []string{"-a2a_agent_url", "https://agent.example.com"},
					origin:     "https://agent.example.com",
					wantStatus: http.StatusForbidden,
					wantCORS:   "",
				},
				{
					name:       "configured webui_address is served on /api",
					origin:     "http://localhost:9000",
					wantStatus: http.StatusOK,
					wantCORS:   "http://localhost:9000",
				},
				{
					name:       "configured allow_origins is served on /api",
					origin:     "https://allowed.example.com",
					wantStatus: http.StatusOK,
					wantCORS:   "https://allowed.example.com",
				},
			} {
				t.Run(tc.name, func(t *testing.T) {
					ln, err := net.Listen("tcp", "127.0.0.1:0")
					if err != nil {
						t.Fatalf("net.Listen() error = %v", err)
					}
					t.Cleanup(func() { _ = ln.Close() })
					port := ln.Addr().(*net.TCPAddr).Port

					capSub := &routerCaptureSublauncher{}
					l := order.build(NewLauncher(), api.NewLauncher(), capSub)
					args := append([]string{"-port", fmt.Sprint(port)}, order.args(tc.a2aArgs)...)
					if _, err := l.Parse(args); err != nil {
						t.Fatalf("Parse(%v) error = %v", args, err)
					}
					cfg := &launcher.Config{
						AgentLoader:    agent.NewSingleLoader(agnt),
						SessionService: session.InMemoryService(),
					}
					if err := l.Run(t.Context(), cfg); err == nil {
						t.Fatalf("Run() succeeded, want server bind failure")
					}
					if capSub.router == nil {
						t.Fatalf("Run() returned before SetupSubrouters captured the router")
					}

					req := httptest.NewRequest(http.MethodGet, "/api/list-apps", nil)
					req.Host = "127.0.0.1:9000"
					req.Header.Set("Origin", tc.origin)
					rec := httptest.NewRecorder()
					capSub.router.ServeHTTP(rec, req)

					if rec.Code != tc.wantStatus {
						t.Errorf("GET /api/list-apps with Origin %q = %d, want %d", tc.origin, rec.Code, tc.wantStatus)
					}
					if got := rec.Header().Get("Access-Control-Allow-Origin"); got != tc.wantCORS {
						t.Errorf("GET /api/list-apps with Origin %q: Access-Control-Allow-Origin = %q, want %q", tc.origin, got, tc.wantCORS)
					}
				})
			}
		})
	}
}
