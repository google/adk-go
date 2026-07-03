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
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"
)

// McpToolset resolves a registered MCP server into a [tool.Toolset] backed by a
// streamable-HTTP MCP connection. name is the full MCP server resource name.
//
// The endpoint is resolved preferring the JSONRPC binding, then HTTP_JSON. By
// default, requests to *.googleapis.com endpoints are authenticated with the
// registry's Application Default Credentials; use [WithMcpHTTPClient] and/or
// [WithMcpHeaders] to override or augment egress.
func (c *Client) McpToolset(ctx context.Context, name string, opts ...McpToolsetOption) (tool.Toolset, error) {
	server, err := c.GetMcpServer(ctx, name)
	if err != nil {
		return nil, err
	}

	uri, ok := mcpEndpointURI(server)
	if !ok {
		return nil, fmt.Errorf("agentregistry: MCP server endpoint URI not found for %q", name)
	}

	ec := applyMcpToolsetOptions(opts)
	egress := c.egressClient(uri, ec, true)

	return mcptoolset.New(mcptoolset.Config{
		Transport: &mcp.StreamableClientTransport{
			Endpoint:   uri,
			HTTPClient: egress,
		},
	})
}

// mcpEndpointURI returns the MCP server's connection URI, preferring the JSONRPC
// binding and falling back to HTTP_JSON (parity with adk-python).
func mcpEndpointURI(server *McpServer) (string, bool) {
	if uri, _, _, ok := connectionURI(server.Protocols, server.Interfaces, "", bindingJSONRPC); ok {
		return uri, true
	}
	if uri, _, _, ok := connectionURI(server.Protocols, server.Interfaces, "", bindingHTTPJSON); ok {
		return uri, true
	}
	return "", false
}
