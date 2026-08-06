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

package agentregistry

import (
	"encoding/json"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	adka2a "google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
)

// TestRemoteAgent_E2E is an end-to-end round trip: a real in-process A2A server
// (adka2a over an offline canned agent) is fronted by a faked registry record,
// and RemoteAgent resolves that record, connects over A2A, and exchanges a
// message. It exercises both agentCard shapes — an embedded A2A_AGENT_CARD and
// discrete protocols — and both transports (JSONRPC, HTTP+JSON), which also
// covers the HTTP_JSON->HTTP+JSON binding mapping and the 0.3.0 compat transport.
func TestRemoteAgent_E2E(t *testing.T) {
	const reply = "hello from the remote agent"

	tests := []struct {
		name    string
		binding a2a.TransportProtocol // selects the A2A server transport
		record  func(serverURL string) Agent
	}{
		{
			name:    "embedded A2A_AGENT_CARD over JSONRPC",
			binding: a2a.TransportProtocolJSONRPC,
			record: func(serverURL string) Agent {
				card, err := json.Marshal(a2a.AgentCard{
					Name:                "Echo",
					Description:         "echoes a canned reply",
					SupportedInterfaces: []*a2a.AgentInterface{a2a.NewAgentInterface(serverURL, a2a.TransportProtocolJSONRPC)},
					Capabilities:        a2a.AgentCapabilities{Streaming: true},
				})
				if err != nil {
					t.Fatalf("marshaling embedded card: %v", err)
				}
				return Agent{
					DisplayName: "Echo",
					Description: "echoes a canned reply",
					Card:        &Card{Type: cardTypeA2AAgentCard, Content: card},
				}
			},
		},
		{
			// Registry wire binding HTTP_JSON maps to a2a HTTP+JSON; the missing
			// protocolVersion falls back to 0.3.0, for which the factory registers
			// a compat transport.
			name:    "discrete protocols over HTTP+JSON",
			binding: a2a.TransportProtocolHTTPJSON,
			record: func(serverURL string) Agent {
				return Agent{
					DisplayName: "Echo",
					Description: "echoes a canned reply",
					Protocols: []Protocol{{
						Type:       protocolTypeA2AAgent,
						Interfaces: []Interface{{URL: serverURL, ProtocolBinding: bindingHTTPJSON}},
					}},
				}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a2aURL := startCannedA2AServer(t, tc.binding, reply)

			regSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewEncoder(w).Encode(tc.record(a2aURL)); err != nil {
					t.Errorf("encoding registry record: %v", err)
				}
			}))
			defer regSrv.Close()

			remote, err := newTestClient(regSrv).RemoteAgent(t.Context(), "projects/p/locations/l/agents/echo")
			if err != nil {
				t.Fatalf("RemoteAgent() error = %v", err)
			}

			if got := runAndCollectText(t, remote, "ping"); !strings.Contains(got, reply) {
				t.Errorf("remote agent response = %q, want it to contain %q", got, reply)
			}
		})
	}
}

// startCannedA2AServer runs an in-process A2A server, backed by an offline agent
// that always replies with reply, over the given transport. It returns the
// server URL and registers cleanup.
func startCannedA2AServer(t *testing.T, binding a2a.TransportProtocol, reply string) string {
	t.Helper()

	canned, err := agent.New(agent.Config{
		Name: "echo_agent",
		Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				yield(&session.Event{LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromText(reply, genai.RoleModel),
				}}, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}

	handler := a2asrv.NewHandler(adka2a.NewExecutor(adka2a.ExecutorConfig{
		RunnerConfig: runner.Config{
			AppName:        canned.Name(),
			Agent:          canned,
			SessionService: session.InMemoryService(),
		},
	}))

	var h http.Handler
	switch binding {
	case a2a.TransportProtocolJSONRPC:
		h = a2asrv.NewJSONRPCHandler(handler)
	case a2a.TransportProtocolHTTPJSON:
		h = a2asrv.NewRESTHandler(handler)
	default:
		t.Fatalf("unsupported binding %q", binding)
	}

	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return srv.URL
}

// runAndCollectText drives ag with a single user message and returns the
// concatenated text across all model responses.
func runAndCollectText(t *testing.T, ag agent.Agent, prompt string) string {
	t.Helper()

	r, err := runner.New(runner.Config{
		AppName:           ag.Name(),
		Agent:             ag,
		SessionService:    session.InMemoryService(),
		AutoCreateSession: true,
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}

	var sb strings.Builder
	msg := genai.NewContentFromText(prompt, genai.RoleUser)
	for ev, err := range r.Run(t.Context(), "user", "session", msg, agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("runner.Run() yielded error = %v", err)
		}
		if ev.LLMResponse.Content == nil {
			continue
		}
		for _, p := range ev.LLMResponse.Content.Parts {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}
