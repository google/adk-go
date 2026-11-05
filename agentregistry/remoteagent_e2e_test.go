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
	"sync"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2acompat/a2av0"
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
// discrete protocols — both bindings (JSONRPC, HTTP+JSON), and both protocol
// versions: a record without protocolVersion defaults to 0.3.0 and must speak
// A2A 0.3 on the wire, while an explicit 1.0 record speaks 1.0.
func TestRemoteAgent_E2E(t *testing.T) {
	const reply = "hello from the remote agent"

	tests := []struct {
		name    string
		binding a2a.TransportProtocol // selects the A2A server transport
		version a2a.ProtocolVersion   // selects the A2A server protocol version
		record  func(serverURL string) Agent
	}{
		{
			name:    "embedded A2A_AGENT_CARD over JSONRPC 1.0",
			binding: a2a.TransportProtocolJSONRPC,
			version: a2a.Version,
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
			name:    "discrete protocols without version over HTTP_JSON speak 0.3",
			binding: a2a.TransportProtocolHTTPJSON,
			version: a2av0.Version,
			record:  func(serverURL string) Agent { return discreteRecord(serverURL, bindingHTTPJSON, "") },
		},
		{
			name:    "discrete protocols without version over JSONRPC speak 0.3",
			binding: a2a.TransportProtocolJSONRPC,
			version: a2av0.Version,
			record:  func(serverURL string) Agent { return discreteRecord(serverURL, bindingJSONRPC, "") },
		},
		{
			name:    "discrete protocols with version 1.0 over HTTP_JSON",
			binding: a2a.TransportProtocolHTTPJSON,
			version: a2a.Version,
			record:  func(serverURL string) Agent { return discreteRecord(serverURL, bindingHTTPJSON, string(a2a.Version)) },
		},
		{
			name:    "discrete protocols with version 1.0 over JSONRPC",
			binding: a2a.TransportProtocolJSONRPC,
			version: a2a.Version,
			record:  func(serverURL string) Agent { return discreteRecord(serverURL, bindingJSONRPC, string(a2a.Version)) },
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			a2aURL := startCannedA2AServer(t, tc.binding, tc.version, reply, nil)

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

// TestRemoteAgent_OnTheWire pins what a registry record puts on the wire. A
// versionless record speaks 0.3: the A2A-Version header is the 0.3 the a2av0
// server serves (a 1.0 server rejects any other value since a2a-go 2.7.0), and
// the HTTP+JSON binding uses the 0.3 "/v1" route prefix. An explicit 1.0 record
// speaks 1.0. On both versions the caller's egress headers reach the server.
func TestRemoteAgent_OnTheWire(t *testing.T) {
	tests := []struct {
		name          string
		binding       a2a.TransportProtocol
		wireBinding   string
		recordVersion string // the record's protocolVersion; empty means unset
		serverVersion a2a.ProtocolVersion
		wantPath      string
	}{
		{name: "HTTP_JSON without version", binding: a2a.TransportProtocolHTTPJSON, wireBinding: bindingHTTPJSON, serverVersion: a2av0.Version, wantPath: a2av0.RESTPathPrefix + "/message:send"},
		{name: "JSONRPC without version", binding: a2a.TransportProtocolJSONRPC, wireBinding: bindingJSONRPC, serverVersion: a2av0.Version, wantPath: "/"},
		{name: "HTTP_JSON 1.0", binding: a2a.TransportProtocolHTTPJSON, wireBinding: bindingHTTPJSON, recordVersion: string(a2a.Version), serverVersion: a2a.Version, wantPath: "/message:send"},
		{name: "JSONRPC 1.0", binding: a2a.TransportProtocolJSONRPC, wireBinding: bindingJSONRPC, recordVersion: string(a2a.Version), serverVersion: a2a.Version, wantPath: "/"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var (
				mu   sync.Mutex
				seen []*http.Request
			)
			observe := func(r *http.Request) {
				mu.Lock()
				defer mu.Unlock()
				seen = append(seen, r.Clone(r.Context()))
			}
			a2aURL := startCannedA2AServer(t, tc.binding, tc.serverVersion, "pong", observe)

			regSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewEncoder(w).Encode(discreteRecord(a2aURL, tc.wireBinding, tc.recordVersion)); err != nil {
					t.Errorf("encoding registry record: %v", err)
				}
			}))
			defer regSrv.Close()

			remote, err := newTestClient(regSrv).RemoteAgent(t.Context(), "projects/p/locations/l/agents/echo",
				WithA2AHeaders(map[string]string{"X-Egress": "registry"}))
			if err != nil {
				t.Fatalf("RemoteAgent() error = %v", err)
			}
			_ = runAndCollectText(t, remote, "ping")

			mu.Lock()
			defer mu.Unlock()
			if len(seen) == 0 {
				t.Fatal("the A2A server received no request")
			}
			for i, r := range seen {
				if r.URL.Path != tc.wantPath {
					t.Errorf("request %d path = %q, want %q", i, r.URL.Path, tc.wantPath)
				}
				if got := r.Header.Get(a2a.SvcParamVersion); got != string(tc.serverVersion) {
					t.Errorf("request %d %s = %q, want %q", i, a2a.SvcParamVersion, got, tc.serverVersion)
				}
				if got := r.Header.Get("X-Egress"); got != "registry" {
					t.Errorf("request %d X-Egress = %q, want the egress header from WithA2AHeaders", i, got)
				}
			}
		})
	}
}

// discreteRecord is a registry record that describes the agent through discrete
// protocol fields rather than an embedded card. version is the record's
// protocolVersion; empty leaves it unset.
func discreteRecord(serverURL, binding, version string) Agent {
	return Agent{
		DisplayName: "Echo",
		Description: "echoes a canned reply",
		Protocols: []Protocol{{
			Type:            protocolTypeA2AAgent,
			ProtocolVersion: version,
			Interfaces:      []Interface{{URL: serverURL, ProtocolBinding: binding}},
		}},
	}
}

// startCannedA2AServer runs an in-process A2A server, backed by an offline agent
// that always replies with reply, over the given binding and protocol version
// (a2a.Version or a2av0.Version). observe, when non-nil, sees every request
// before the A2A handler does. It returns the server URL and registers cleanup.
func startCannedA2AServer(t *testing.T, binding a2a.TransportProtocol, version a2a.ProtocolVersion, reply string, observe func(*http.Request)) string {
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
	switch {
	case binding == a2a.TransportProtocolJSONRPC && version == a2a.Version:
		h = a2asrv.NewJSONRPCHandler(handler)
	case binding == a2a.TransportProtocolHTTPJSON && version == a2a.Version:
		h = a2asrv.NewRESTHandler(handler)
	case binding == a2a.TransportProtocolJSONRPC && version == a2av0.Version:
		h = a2av0.NewJSONRPCHandler(handler)
	case binding == a2a.TransportProtocolHTTPJSON && version == a2av0.Version:
		h = a2av0.NewRESTHandler(handler)
	default:
		t.Fatalf("unsupported binding %q at version %q", binding, version)
	}
	if observe != nil {
		next := h
		h = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			observe(r)
			next.ServeHTTP(w, r)
		})
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
