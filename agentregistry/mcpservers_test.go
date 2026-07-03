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

func TestListMcpServers(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_, _ = w.Write([]byte(`{"mcpServers":[{"displayName":"Data","mcpServerId":"data-mcp"}]}`))
	}))
	defer srv.Close()

	page, err := newTestClient(srv).ListMcpServers(context.Background())
	if err != nil {
		t.Fatalf("ListMcpServers() error = %v", err)
	}
	if want := "/projects/p/locations/l/mcpServers"; gotPath != want {
		t.Errorf("request path = %q, want %q", gotPath, want)
	}
	if len(page.McpServers) != 1 || page.McpServers[0].McpServerID != "data-mcp" {
		t.Errorf("page = %+v, want one server with mcpServerId data-mcp", page)
	}
}

func TestGetMcpServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"displayName":"Data","mcpServerId":"data-mcp"}`))
	}))
	defer srv.Close()

	got, err := newTestClient(srv).GetMcpServer(context.Background(), "projects/p/locations/l/mcpServers/data-mcp")
	if err != nil {
		t.Fatalf("GetMcpServer() error = %v", err)
	}
	if got.DisplayName != "Data" || got.McpServerID != "data-mcp" {
		t.Errorf("server = %+v, want Data/data-mcp", got)
	}
}

func TestAllMcpServers(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"mcpServers":[{"displayName":"one"},{"displayName":"two"}]}`))
	}))
	defer srv.Close()

	var got []string
	for server, err := range newTestClient(srv).AllMcpServers(context.Background()) {
		if err != nil {
			t.Fatalf("AllMcpServers() yielded error = %v", err)
		}
		got = append(got, server.DisplayName)
	}
	if len(got) != 2 || got[0] != "one" || got[1] != "two" {
		t.Errorf("collected %v, want [one two]", got)
	}
}
