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
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestMcpEndpointURI(t *testing.T) {
	tests := []struct {
		name    string
		server  *McpServer
		wantURI string
		wantOK  bool
	}{
		{
			name: "prefers jsonrpc over http_json",
			server: &McpServer{Interfaces: []Interface{
				{URL: "https://s/http", ProtocolBinding: "HTTP_JSON"},
				{URL: "https://s/jsonrpc", ProtocolBinding: "JSONRPC"},
			}},
			wantURI: "https://s/jsonrpc",
			wantOK:  true,
		},
		{
			name: "falls back to http_json",
			server: &McpServer{Interfaces: []Interface{
				{URL: "https://s/http", ProtocolBinding: "HTTP_JSON"},
			}},
			wantURI: "https://s/http",
			wantOK:  true,
		},
		{
			name: "reads from protocols too",
			server: &McpServer{Protocols: []Protocol{{
				Interfaces: []Interface{{URL: "https://s/jsonrpc", ProtocolBinding: "JSONRPC"}},
			}}},
			wantURI: "https://s/jsonrpc",
			wantOK:  true,
		},
		{
			name:   "no supported binding",
			server: &McpServer{Interfaces: []Interface{{URL: "https://s/grpc", ProtocolBinding: "GRPC"}}},
			wantOK: false,
		},
		{
			name:   "empty",
			server: &McpServer{},
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

func TestMcpToolset_Integration(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"displayName": "Data MCP",
			"mcpServerId": "data-mcp",
			"interfaces": [{"url": "https://data.example/mcp", "protocolBinding": "JSONRPC"}]
		}`))
	}))
	defer srv.Close()

	ts, err := newTestClient(srv).McpToolset(context.Background(), "projects/p/locations/l/mcpServers/data")
	if err != nil {
		t.Fatalf("McpToolset() error = %v", err)
	}
	if ts == nil {
		t.Fatal("McpToolset() = nil, want a toolset")
	}
}

func TestMcpToolset_NoEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"displayName": "Data MCP"}`)) // no interfaces
	}))
	defer srv.Close()

	if _, err := newTestClient(srv).McpToolset(context.Background(), "projects/p/locations/l/mcpServers/data"); err == nil {
		t.Error("McpToolset() error = nil, want an error when no endpoint URI is present")
	}
}
