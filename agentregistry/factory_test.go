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
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
)

func TestIsGoogleAPI(t *testing.T) {
	tests := []struct {
		url  string
		want bool
	}{
		{url: "https://googleapis.com/v1", want: true},
		{url: "https://agentregistry.googleapis.com/v1", want: true},
		{url: "https://agentregistry.mtls.googleapis.com/v1", want: true},
		{url: "https://evil.com/googleapis.com", want: false},
		{url: "https://notgoogleapis.com", want: false},
		{url: "https://example.com", want: false},
		{url: "://bad", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.url, func(t *testing.T) {
			if got := isGoogleAPI(tc.url); got != tc.want {
				t.Errorf("isGoogleAPI(%q) = %t, want %t", tc.url, got, tc.want)
			}
		})
	}
}

func TestEgressClient(t *testing.T) {
	registryClient := &http.Client{}
	override := &http.Client{}
	c := &Client{httpClient: registryClient}

	tests := []struct {
		name string
		url  string
		ec   egressConfig
		want *http.Client
	}{
		{
			name: "explicit override wins",
			url:  "https://x.googleapis.com",
			ec:   egressConfig{httpClient: override},
			want: override,
		},
		{
			name: "google api reuses registry client",
			url:  "https://x.googleapis.com/mcp",
			want: registryClient,
		},
		{
			name: "non-google uses default",
			url:  "https://third-party.example/mcp",
			want: http.DefaultClient,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := c.egressClient(tc.url, tc.ec); got != tc.want {
				t.Errorf("egressClient() = %p, want %p", got, tc.want)
			}
		})
	}
}

func TestEgressClient_HeadersProduceDistinctClient(t *testing.T) {
	registryClient := &http.Client{}
	c := &Client{httpClient: registryClient}
	got := c.egressClient("https://x.googleapis.com", egressConfig{headers: map[string]string{"X": "y"}})
	if got == registryClient {
		t.Error("egressClient() with headers returned the shared registry client; want a distinct clone")
	}
}

func TestClientWithHeaders_NoHeadersReturnsBase(t *testing.T) {
	base := &http.Client{}
	if got := clientWithHeaders(base, nil); got != base {
		t.Errorf("clientWithHeaders(base, nil) = %p, want base %p", got, base)
	}
}

func TestClientWithHeaders_AppliesAndDoesNotMutateBase(t *testing.T) {
	var gotHeader string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Test")
	}))
	defer srv.Close()

	base := &http.Client{}
	wrapped := clientWithHeaders(base, map[string]string{"X-Test": "v"})
	if wrapped == base {
		t.Fatal("clientWithHeaders returned base; want a clone")
	}
	if base.Transport != nil {
		t.Error("base client Transport was mutated; want it left nil")
	}

	resp, err := wrapped.Get(srv.URL)
	if err != nil {
		t.Fatalf("wrapped.Get() error = %v", err)
	}
	_ = resp.Body.Close()
	if gotHeader != "v" {
		t.Errorf("server saw X-Test = %q, want v", gotHeader)
	}
}

func TestRemoteAgentOptions(t *testing.T) {
	client := &http.Client{}
	ec := applyRemoteAgentOptions([]RemoteAgentOption{
		WithA2AHTTPClient(client),
		WithA2AHeaders(map[string]string{"X": "y"}),
		WithA2AHeaders(map[string]string{"Z": "w"}),
	})
	if ec.httpClient != client {
		t.Errorf("httpClient = %p, want %p", ec.httpClient, client)
	}
	// Repeated WithA2AHeaders calls accumulate rather than replacing the set.
	if ec.headers["X"] != "y" || ec.headers["Z"] != "w" {
		t.Errorf("headers = %v, want X=y and Z=w", ec.headers)
	}
}

func TestMCPToolsetOptions(t *testing.T) {
	client := &http.Client{}
	ec := applyMCPToolsetOptions([]MCPToolsetOption{
		WithMCPHTTPClient(client),
		WithMCPHeaders(map[string]string{"X": "y"}),
		WithMCPHeaders(map[string]string{"Z": "w"}),
	})
	if ec.httpClient != client {
		t.Errorf("httpClient = %p, want %p", ec.httpClient, client)
	}
	// Repeated WithMCPHeaders calls accumulate rather than replacing the set.
	if ec.headers["X"] != "y" || ec.headers["Z"] != "w" {
		t.Errorf("headers = %v, want X=y and Z=w", ec.headers)
	}
}

func TestAgentCard_EmbeddedFastPath(t *testing.T) {
	embedded := a2a.AgentCard{
		Name:                "My Agent",
		Description:         "embedded desc",
		Version:             "9",
		SupportedInterfaces: []*a2a.AgentInterface{a2a.NewAgentInterface("https://a.example", a2a.TransportProtocolJSONRPC)},
	}
	content, err := json.Marshal(embedded)
	if err != nil {
		t.Fatalf("marshal embedded card: %v", err)
	}
	info := &Agent{Card: &Card{Type: "A2A_AGENT_CARD", Content: content}}

	card, name, err := agentCard(info, "projects/p/locations/l/agents/a")
	if err != nil {
		t.Fatalf("agentCard() error = %v", err)
	}
	if name != "My_Agent" {
		t.Errorf("name = %q, want My_Agent (cleaned)", name)
	}
	if card.Description != "embedded desc" {
		t.Errorf("description = %q, want embedded desc", card.Description)
	}
	// The embedded card itself is used verbatim (its name is not rewritten).
	if card.Name != "My Agent" || card.Version != "9" {
		t.Errorf("card = %+v, want the embedded card unchanged", card)
	}
}

func TestAgentCard_EmbeddedNoInterfaces(t *testing.T) {
	// An embedded card without supported interfaces must fail fast (with the
	// resource name) rather than only at the first invocation.
	content, err := json.Marshal(a2a.AgentCard{Name: "My Agent"})
	if err != nil {
		t.Fatalf("marshal embedded card: %v", err)
	}
	info := &Agent{Card: &Card{Type: "A2A_AGENT_CARD", Content: content}}

	if _, _, err := agentCard(info, "projects/p/locations/l/agents/a"); err == nil {
		t.Error("agentCard() error = nil, want an error for an embedded card with no supported interfaces")
	}
}

func TestAgentCard_EmbeddedEmptyName(t *testing.T) {
	// An embedded card must carry a usable name; the embedded path does not fall
	// back to the resource name (parity with adk-python), so an empty name fails.
	content, err := json.Marshal(a2a.AgentCard{
		SupportedInterfaces: []*a2a.AgentInterface{a2a.NewAgentInterface("https://a.example", a2a.TransportProtocolJSONRPC)},
	})
	if err != nil {
		t.Fatalf("marshal embedded card: %v", err)
	}
	info := &Agent{Card: &Card{Type: "A2A_AGENT_CARD", Content: content}}

	if _, _, err := agentCard(info, "projects/p/locations/l/agents/a"); err == nil {
		t.Error("agentCard() error = nil, want an error for an embedded card with an empty name")
	}
}

func TestAgentCard_Synthesized(t *testing.T) {
	info := &Agent{
		DisplayName: "Cool Agent",
		Description: "does cool things",
		Version:     "2.0",
		Protocols: []Protocol{{
			Type:            "A2A_AGENT",
			ProtocolVersion: "0.3.0",
			Interfaces: []Interface{
				{URL: "https://a.example/jsonrpc", ProtocolBinding: "JSONRPC"},
			},
		}},
		Skills: []Skill{{ID: "s1", Name: "Skill One", Tags: []string{"t"}}},
	}

	card, name, err := agentCard(info, "projects/p/locations/l/agents/cool")
	if err != nil {
		t.Fatalf("agentCard() error = %v", err)
	}
	if name != "Cool_Agent" {
		t.Errorf("name = %q, want Cool_Agent", name)
	}
	if card.Description != "does cool things" {
		t.Errorf("description = %q, want does cool things", card.Description)
	}
	if card.Name != "Cool_Agent" || card.Version != "2.0" {
		t.Errorf("card name/version = %q/%q, want Cool_Agent/2.0", card.Name, card.Version)
	}
	if len(card.SupportedInterfaces) != 1 {
		t.Fatalf("supported interfaces = %d, want 1", len(card.SupportedInterfaces))
	}
	iface := card.SupportedInterfaces[0]
	if iface.URL != "https://a.example/jsonrpc" {
		t.Errorf("interface URL = %q, want the jsonrpc URL", iface.URL)
	}
	if iface.ProtocolBinding != a2a.TransportProtocolJSONRPC {
		t.Errorf("interface binding = %q, want JSONRPC", iface.ProtocolBinding)
	}
	if iface.ProtocolVersion != a2a.ProtocolVersion("0.3.0") {
		t.Errorf("interface protocol version = %q, want 0.3.0", iface.ProtocolVersion)
	}
	if len(card.Skills) != 1 || card.Skills[0].ID != "s1" || card.Skills[0].Name != "Skill One" {
		t.Errorf("skills = %+v, want one mapped skill s1", card.Skills)
	}
}

func TestAgentCard_SynthesizedDefaultsAndNameFallback(t *testing.T) {
	// No display name -> fall back to the resource name; no binding -> HTTP+JSON;
	// no protocol version -> default.
	info := &Agent{
		Protocols: []Protocol{{
			Type:       "A2A_AGENT",
			Interfaces: []Interface{{URL: "https://a.example", ProtocolBinding: "SOMETHING_ELSE"}},
		}},
	}
	card, name, err := agentCard(info, "projects/p/locations/l/agents/fallback-id")
	if err != nil {
		t.Fatalf("agentCard() error = %v", err)
	}
	if name != "projects_p_locations_l_agents_fallback_id" {
		t.Errorf("name = %q, want cleaned resource name", name)
	}
	iface := card.SupportedInterfaces[0]
	if iface.ProtocolBinding != a2a.TransportProtocolHTTPJSON {
		t.Errorf("binding = %q, want default HTTP+JSON for unknown binding", iface.ProtocolBinding)
	}
	if iface.ProtocolVersion != a2a.ProtocolVersion(defaultA2AProtocolVersion) {
		t.Errorf("protocol version = %q, want default %q", iface.ProtocolVersion, defaultA2AProtocolVersion)
	}
}

func TestAgentCard_NoConnectionURI(t *testing.T) {
	info := &Agent{DisplayName: "no interfaces"}
	if _, _, err := agentCard(info, "projects/p/locations/l/agents/x"); err == nil {
		t.Error("agentCard() error = nil, want an error when no A2A connection URI is present")
	}
}

func TestAgentCard_EmptyNameError(t *testing.T) {
	// No display name and an empty resource name leave nothing to derive a
	// non-empty agent name from, so agentCard must fail rather than build an
	// agent with an empty (invalid) name.
	info := &Agent{
		Protocols: []Protocol{{
			Type:       "A2A_AGENT",
			Interfaces: []Interface{{URL: "https://a.example", ProtocolBinding: "JSONRPC"}},
		}},
	}
	if _, _, err := agentCard(info, ""); err == nil {
		t.Error("agentCard() error = nil, want an error when no non-empty name can be derived")
	}
}

func TestRemoteAgent_Integration(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"displayName": "Doc Summarizer",
			"description": "summarizes documents",
			"protocols": [{"type": "A2A_AGENT", "protocolVersion": "0.3.0",
				"interfaces": [{"url": "https://a.example/jsonrpc", "protocolBinding": "JSONRPC"}]}]
		}`))
	}))
	defer srv.Close()

	got, err := newTestClient(srv).RemoteAgent(t.Context(), "projects/p/locations/l/agents/doc")
	if err != nil {
		t.Fatalf("RemoteAgent() error = %v", err)
	}
	if got == nil {
		t.Fatal("RemoteAgent() = nil, want an agent")
	}
	if got.Name() != "Doc_Summarizer" {
		t.Errorf("agent Name() = %q, want Doc_Summarizer", got.Name())
	}
	if got.Description() != "summarizes documents" {
		t.Errorf("agent Description() = %q, want summarizes documents", got.Description())
	}
}

func TestA2AClientFactory_CompatProtocolVersion(t *testing.T) {
	// Registry cards commonly advertise an older A2A protocol version than the
	// SDK's current one. The factory must register a compatible transport so the
	// client can be created (this failed with "no compatible transports found"
	// before the compat transports were added).
	tests := []struct {
		name    string
		binding a2a.TransportProtocol
		version a2a.ProtocolVersion
	}{
		{name: "http+json old version", binding: a2a.TransportProtocolHTTPJSON, version: "0.3.0"},
		{name: "jsonrpc old version", binding: a2a.TransportProtocolJSONRPC, version: "0.3.0"},
		{name: "http+json current version", binding: a2a.TransportProtocolHTTPJSON, version: a2a.Version},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			card := &a2a.AgentCard{
				Name: "x",
				SupportedInterfaces: []*a2a.AgentInterface{{
					URL:             "https://x.example/a2a",
					ProtocolBinding: tc.binding,
					ProtocolVersion: tc.version,
				}},
			}
			factory := a2aClientFactory(card, http.DefaultClient)
			client, err := factory.CreateFromCard(t.Context(), card)
			if err != nil {
				t.Fatalf("CreateFromCard(%s@%s) error = %v, want a client", tc.binding, tc.version, err)
			}
			if client == nil {
				t.Fatal("CreateFromCard() = nil client")
			}
			_ = client.Destroy()
		})
	}
}

func TestMCPEndpointURI(t *testing.T) {
	tests := []struct {
		name    string
		server  *MCPServer
		wantURI string
		wantOK  bool
	}{
		{
			name: "prefers jsonrpc over http_json",
			server: &MCPServer{Interfaces: []Interface{
				{URL: "https://s/http", ProtocolBinding: "HTTP_JSON"},
				{URL: "https://s/jsonrpc", ProtocolBinding: "JSONRPC"},
			}},
			wantURI: "https://s/jsonrpc",
			wantOK:  true,
		},
		{
			name: "falls back to http_json",
			server: &MCPServer{Interfaces: []Interface{
				{URL: "https://s/http", ProtocolBinding: "HTTP_JSON"},
			}},
			wantURI: "https://s/http",
			wantOK:  true,
		},
		{
			name: "reads from protocols too",
			server: &MCPServer{Protocols: []Protocol{{
				Interfaces: []Interface{{URL: "https://s/jsonrpc", ProtocolBinding: "JSONRPC"}},
			}}},
			wantURI: "https://s/jsonrpc",
			wantOK:  true,
		},
		{
			name:   "no supported binding",
			server: &MCPServer{Interfaces: []Interface{{URL: "https://s/grpc", ProtocolBinding: "GRPC"}}},
			wantOK: false,
		},
		{
			name:   "empty",
			server: &MCPServer{},
			wantOK: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			uri, ok := mcpEndpointURI(tc.server)
			if ok != tc.wantOK || uri != tc.wantURI {
				t.Errorf("mcpEndpointURI() = (%q, %t), want (%q, %t)", uri, ok, tc.wantURI, tc.wantOK)
			}
		})
	}
}

func TestMCPToolset_Integration(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"displayName": "Data MCP",
			"mcpServerId": "data-mcp",
			"interfaces": [{"url": "https://data.example/mcp", "protocolBinding": "JSONRPC"}]
		}`))
	}))
	defer srv.Close()

	ts, err := newTestClient(srv).MCPToolset(t.Context(), "projects/p/locations/l/mcpServers/data")
	if err != nil {
		t.Fatalf("MCPToolset() error = %v", err)
	}
	if ts == nil {
		t.Fatal("MCPToolset() = nil, want a toolset")
	}
}

func TestMCPToolset_NoEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"displayName": "Data MCP"}`)) // no interfaces
	}))
	defer srv.Close()

	if _, err := newTestClient(srv).MCPToolset(t.Context(), "projects/p/locations/l/mcpServers/data"); err == nil {
		t.Error("MCPToolset() error = nil, want an error when no endpoint URI is present")
	}
}
