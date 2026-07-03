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
	"iter"
)

// ListMcpServers returns one page of registered MCP servers. Use opts to set a
// filter and pagination; for automatic paging use [Client.AllMcpServers].
func (c *Client) ListMcpServers(ctx context.Context, opts ...ListOption) (*McpServersPage, error) {
	var page McpServersPage
	if err := c.get(ctx, "mcpServers", listValues(opts...), &page); err != nil {
		return nil, err
	}
	return &page, nil
}

// GetMcpServer returns the metadata of a single MCP server. name is the full
// resource name (e.g. "projects/<p>/locations/<l>/mcpServers/<id>").
func (c *Client) GetMcpServer(ctx context.Context, name string) (*McpServer, error) {
	var server McpServer
	if err := c.get(ctx, name, nil, &server); err != nil {
		return nil, err
	}
	return &server, nil
}

// AllMcpServers iterates over every MCP server matching opts, fetching pages on
// demand. If a page fetch fails the iterator yields a single (nil, error) and
// stops.
func (c *Client) AllMcpServers(ctx context.Context, opts ...ListOption) iter.Seq2[*McpServer, error] {
	return pages(ctx, opts, func(ctx context.Context, o ...ListOption) ([]McpServer, string, error) {
		page, err := c.ListMcpServers(ctx, o...)
		if err != nil {
			return nil, "", err
		}
		return page.McpServers, page.NextPageToken, nil
	})
}
