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

	"github.com/modelcontextprotocol/go-sdk/mcp"

	icontext "google.golang.org/adk/v2/internal/context"
)

// TestMCPToolset_E2E is an end-to-end round trip: a real in-process MCP server
// over streamable HTTP is fronted by a faked registry record, and MCPToolset
// resolves that record, connects over the resolved endpoint, and lists the
// server's tools. It exercises endpoint resolution, the streamable-HTTP
// transport wiring, and egress-client selection.
func TestMCPToolset_E2E(t *testing.T) {
	const toolName = "get_data"

	mcpServer := mcp.NewServer(&mcp.Implementation{Name: "data_mcp", Version: "v1.0.0"}, nil)
	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: "returns data"},
		func(context.Context, *mcp.CallToolRequest, struct{}) (*mcp.CallToolResult, struct{}, error) {
			return &mcp.CallToolResult{}, struct{}{}, nil
		})

	mcpSrv := httptest.NewServer(mcp.NewStreamableHTTPHandler(
		func(*http.Request) *mcp.Server { return mcpServer }, nil))
	defer mcpSrv.Close()
	// The toolset holds a persistent streamable-HTTP connection with no public
	// close; force-close it (runs before Close) so the server can shut down.
	defer mcpSrv.CloseClientConnections()

	regSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := MCPServer{
			DisplayName: "Data MCP",
			MCPServerID: "data-mcp",
			Interfaces:  []Interface{{URL: mcpSrv.URL, ProtocolBinding: bindingJSONRPC}},
		}
		if err := json.NewEncoder(w).Encode(rec); err != nil {
			t.Errorf("encoding registry record: %v", err)
		}
	}))
	defer regSrv.Close()

	ts, err := newTestClient(regSrv).MCPToolset(t.Context(), "projects/p/locations/l/mcpServers/data")
	if err != nil {
		t.Fatalf("MCPToolset() error = %v", err)
	}

	rctx := icontext.NewReadonlyContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}))
	tools, err := ts.Tools(rctx)
	if err != nil {
		t.Fatalf("Tools() error = %v", err)
	}

	var names []string
	for _, tl := range tools {
		names = append(names, tl.Name())
	}
	found := false
	for _, n := range names {
		if n == toolName {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Tools() = %v, want it to contain %q", names, toolName)
	}
}
