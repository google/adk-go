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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"regexp"
	"strings"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
)

func TestIsGoogleAPI(t *testing.T) {
	tests := []struct {
		url  string
		want bool
	}{
		{url: "https://agentregistry.googleapis.com/v1", want: true},
		{url: "https://agentregistry.mtls.googleapis.com/v1", want: true},
		{url: "https://us-central1-aiplatform.googleapis.com/mcp", want: true},
		{url: "https://AgentRegistry.GoogleAPIs.com/v1", want: true},
		{url: "HTTPS://agentregistry.googleapis.com/v1", want: true},
		{url: "https://agentregistry.googleapis.com:443/v1", want: true},
		{url: "https://AZaz09.googleapis.com/v1", want: true},
		// No such host exists, but it has the suffix, as in adk-python.
		{url: "https://.googleapis.com/v1", want: true},
		{url: "https://evil.com/googleapis.com", want: false},
		{url: "https://notgoogleapis.com", want: false},
		{url: "https://googleapis.com.evil.com/v1", want: false},
		{url: "https://example.com", want: false},
		{url: "://bad", want: false},
		{url: "", want: false},
		// The apex is not a service endpoint.
		{url: "https://googleapis.com/v1", want: false},
		// Credentials do not go out in clear text.
		{url: "http://agentregistry.googleapis.com/v1", want: false},
		{url: "https://googleapis.com@evil.com/v1", want: false},
		{url: "https://user:pw@agentregistry.googleapis.com/v1", want: false},
		{url: `https://evil.com\.googleapis.com/v1`, want: false},
		// The zone of an IPv6 address can end in the suffix, and the address
		// is anyone's.
		{url: "https://[::1%25.googleapis.com]/mcp", want: false},
		{url: "https://[::1%25.googleapis.com]:8443/mcp", want: false},
		{url: "https://[2001:db8::1%25.googleapis.com]/mcp", want: false},
		// Not the characters of a DNS name.
		{url: "https://evil.com,foo.googleapis.com/v1", want: false},
		{url: "https://my_api.googleapis.com/v1", want: false},
		{url: "https://\u0130.googleapis.com/v1", want: false},
		{url: "https://agentregistry.googleapis.com./v1", want: false},
	}
	for _, tc := range tests {
		t.Run(tc.url, func(t *testing.T) {
			if got := isGoogleAPI(tc.url); got != tc.want {
				t.Errorf("isGoogleAPI(%q) = %t, want %t", tc.url, got, tc.want)
			}
		})
	}
}

var errDialed = errors.New("dialed")

// Unless the caller sets a client, the registry's credentials go to every URL
// that isGoogleAPI accepts, so a request to one has to connect to a DNS name
// under googleapis.com. The URLs are every string of up to three pieces in
// each template.
func TestIsGoogleAPIDestinationIsDialed(t *testing.T) {
	pieces := []string{"a", "A", "1", ".", "-", "_", ":", "%", "%25", "[", "]", "@", "/", "\\", ",", "#", "?", " ", "\u0130", "::1", "1.2.3.4"}
	templates := []string{
		"https://%s.googleapis.com/v1",
		"https://%s.googleapis.com:443/v1",
		"https://%sgoogleapis.com/v1",
		"https://[%s.googleapis.com]/v1",
		"https://[::1%s.googleapis.com]/v1",
		"https://[::1%s.googleapis.com]:443/v1",
		"https://%sx.googleapis.com/v1",
		"https://x.googleapis.com%s",
	}
	dnsName := regexp.MustCompile(`^[a-z0-9.-]*\.googleapis\.com$`)

	var dialed string
	dial := func(_ context.Context, _, addr string) (net.Conn, error) {
		dialed = addr
		return nil, errDialed
	}
	client := &http.Client{Transport: &http.Transport{DialContext: dial, DialTLSContext: dial, DisableKeepAlives: true}}

	// check reports whether isGoogleAPI accepts rawURL, after checking where
	// a request to it would connect.
	var connects int
	check := func(t *testing.T, rawURL string) bool {
		if !isGoogleAPI(rawURL) {
			return false
		}
		req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, rawURL, http.NoBody)
		if err != nil {
			return true // No request can be sent there.
		}
		dialed = ""
		resp, err := client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			t.Fatalf("request to %q was sent", rawURL)
		}
		if !errors.Is(err, errDialed) {
			return true // Refused before any connection.
		}
		connects++
		host, _, err := net.SplitHostPort(dialed)
		if err != nil {
			t.Errorf("isGoogleAPI(%q) = true, and a request connects to %q, which is not host:port", rawURL, dialed)
			return true
		}
		if _, err := netip.ParseAddr(host); err == nil {
			t.Errorf("isGoogleAPI(%q) = true, and a request connects to the IP address %q", rawURL, dialed)
		}
		if !dnsName.MatchString(strings.ToLower(host)) {
			t.Errorf("isGoogleAPI(%q) = true, and a request connects to %q, which is not a DNS name under googleapis.com", rawURL, dialed)
		}
		return true
	}

	for _, mode := range []struct {
		name, godebug string
		// only is a URL that url.Parse takes only with godebug.
		only string
	}{
		{name: "default"},
		// Since Go 1.26 url.Parse refuses an http or https host that holds
		// more than one colon. The setting turns that off.
		{name: "colons in the host", godebug: "urlstrictcolons=0", only: "https://evil.com:443,foo.googleapis.com:443/v1"},
	} {
		t.Run(mode.name, func(t *testing.T) {
			if mode.godebug != "" {
				t.Setenv("GODEBUG", mode.godebug)
				if _, err := url.Parse(mode.only); err != nil {
					t.Skipf("GODEBUG=%s has no effect: %v", mode.godebug, err)
				}
				if check(t, mode.only) {
					t.Errorf("isGoogleAPI(%q) = true", mode.only)
				}
			}
			connects = 0
			var total, accepted int
			var walk func(prefix string, depth int)
			walk = func(prefix string, depth int) {
				for _, tmpl := range templates {
					total++
					if check(t, fmt.Sprintf(tmpl, prefix)) {
						accepted++
					}
				}
				if depth == 0 {
					return
				}
				for _, piece := range pieces {
					walk(prefix+piece, depth-1)
				}
			}
			walk("", 3)
			// Guards against a generator, or a dialer, that reaches nothing.
			if total < 50000 || accepted < 500 || connects < accepted/2 {
				t.Errorf("checked %d URLs, %d accepted, %d of those connecting: too few to mean anything", total, accepted, connects)
			}
			t.Logf("%d URLs, %d accepted, %d of those connecting", total, accepted, connects)
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
		{
			name: "address with a googleapis.com zone uses default",
			url:  "https://[::1%25.googleapis.com]/mcp",
			want: http.DefaultClient,
		},
		{
			name: "clear text uses default",
			url:  "http://x.googleapis.com/mcp",
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
	// before the compat transports were added). A version with no registered
	// transport, such as a future major, is refused rather than mapped onto one.
	tests := []struct {
		name    string
		binding a2a.TransportProtocol
		version a2a.ProtocolVersion
		wantErr bool
	}{
		{name: "http+json old version", binding: a2a.TransportProtocolHTTPJSON, version: "0.3.0"},
		{name: "jsonrpc old version", binding: a2a.TransportProtocolJSONRPC, version: "0.3.0"},
		{name: "http+json current version", binding: a2a.TransportProtocolHTTPJSON, version: a2a.Version},
		{name: "http+json unknown major", binding: a2a.TransportProtocolHTTPJSON, version: "2.0", wantErr: true},
		{name: "jsonrpc unknown major", binding: a2a.TransportProtocolJSONRPC, version: "2.0", wantErr: true},
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
			factory := a2aClientFactory(http.DefaultClient)
			client, err := factory.CreateFromCard(t.Context(), card)
			if tc.wantErr {
				if err == nil {
					_ = client.Destroy()
					t.Fatalf("CreateFromCard(%s@%s) = client, want an error", tc.binding, tc.version)
				}
				return
			}
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
