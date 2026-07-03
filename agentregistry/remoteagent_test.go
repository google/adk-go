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
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
)

func TestAgentCard_EmbeddedFastPath(t *testing.T) {
	embedded := a2a.AgentCard{Name: "My Agent", Description: "embedded desc", Version: "9"}
	content, err := json.Marshal(embedded)
	if err != nil {
		t.Fatalf("marshal embedded card: %v", err)
	}
	info := &Agent{Card: &Card{Type: "A2A_AGENT_CARD", Content: content}}

	card, name, desc, err := agentCard(info, "projects/p/locations/l/agents/a")
	if err != nil {
		t.Fatalf("agentCard() error = %v", err)
	}
	if name != "My_Agent" {
		t.Errorf("name = %q, want My_Agent (cleaned)", name)
	}
	if desc != "embedded desc" {
		t.Errorf("description = %q, want embedded desc", desc)
	}
	// The embedded card itself is used verbatim (its name is not rewritten).
	if card.Name != "My Agent" || card.Version != "9" {
		t.Errorf("card = %+v, want the embedded card unchanged", card)
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

	card, name, desc, err := agentCard(info, "projects/p/locations/l/agents/cool")
	if err != nil {
		t.Fatalf("agentCard() error = %v", err)
	}
	if name != "Cool_Agent" {
		t.Errorf("name = %q, want Cool_Agent", name)
	}
	if desc != "does cool things" {
		t.Errorf("description = %q, want does cool things", desc)
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
	card, name, _, err := agentCard(info, "projects/p/locations/l/agents/fallback-id")
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
	if _, _, _, err := agentCard(info, "projects/p/locations/l/agents/x"); err == nil {
		t.Error("agentCard() error = nil, want an error when no A2A connection URI is present")
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

	got, err := newTestClient(srv).RemoteAgent(context.Background(), "projects/p/locations/l/agents/doc")
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
			client, err := factory.CreateFromCard(context.Background(), card)
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
