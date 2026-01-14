// Copyright 2025 Google LLC
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

package mcptoolset

import (
	"context"
	"fmt"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/internal/version"
)

// MCPClient abstracts MCP session operations for easier connection management.
type MCPClient interface {
	CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error)
	ListTools(context.Context) ([]*mcp.Tool, error)
}

// connectionRefresher wraps an MCP client/transport and handles automatic reconnection.
// It implements MCPClient and transparently retries operations after reconnecting
// when the underlying session fails.
type connectionRefresher struct {
	client    *mcp.Client
	transport mcp.Transport

	mu      sync.Mutex
	session *mcp.ClientSession
}

// newConnectionRefresher creates a new connectionRefresher with the given client and transport.
// If client is nil, a default MCP client will be created.
func newConnectionRefresher(client *mcp.Client, transport mcp.Transport) *connectionRefresher {
	if client == nil {
		client = mcp.NewClient(&mcp.Implementation{Name: "adk-mcp-client", Version: version.Version}, nil)
	}
	return &connectionRefresher{
		client:    client,
		transport: transport,
	}
}

// CallTool calls a tool on the MCP server, automatically reconnecting if needed.
func (c *connectionRefresher) CallTool(ctx context.Context, params *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	result, _, err := withRetry(ctx, c, func(session *mcp.ClientSession) (*mcp.CallToolResult, error) {
		return session.CallTool(ctx, params)
	})
	return result, err
}

// ListTools lists all available tools from the MCP server, handling pagination
// and automatically reconnecting if needed. Per MCP spec, cursors do not persist
// across sessions, so pagination restarts from scratch after reconnection.
func (c *connectionRefresher) ListTools(ctx context.Context) ([]*mcp.Tool, error) {
	var tools []*mcp.Tool
	cursor := ""
	hasReconnected := false

	for {
		resp, reconnected, err := withRetry(ctx, c, func(session *mcp.ClientSession) (*mcp.ListToolsResult, error) {
			return session.ListTools(ctx, &mcp.ListToolsParams{Cursor: cursor})
		})
		if err != nil {
			return nil, fmt.Errorf("failed to list MCP tools: %w", err)
		}
		if reconnected {
			if hasReconnected {
				return nil, fmt.Errorf("failed to list MCP tools: connection lost again after reconnection")
			}
			// On reconnection, restart pagination from scratch per MCP spec.
			hasReconnected = true
			cursor = ""
			tools = nil
			continue
		}

		tools = append(tools, resp.Tools...)

		if resp.NextCursor == "" {
			break
		}
		cursor = resp.NextCursor
	}

	return tools, nil
}

// withRetry executes fn with the current session, and if it fails, attempts to refresh
// the connection and retry once. Returns the result, whether a reconnection occurred, and any error.
func withRetry[T any](ctx context.Context, c *connectionRefresher, fn func(*mcp.ClientSession) (T, error)) (T, bool, error) {
	var zero T

	session, err := c.getSession(ctx)
	if err != nil {
		return zero, false, err
	}

	result, err := fn(session)
	if err != nil {
		// On any error, attempt to refresh the connection.
		// refreshConnection uses ping to verify if reconnection is actually needed.
		session, reconnected, refreshErr := c.refreshConnection(ctx)
		if refreshErr != nil {
			return zero, false, fmt.Errorf("%w (reconnection also failed: %v)", err, refreshErr)
		}
		if !reconnected {
			// The connection was alive, so the original error was not a connection issue.
			// Do not retry, just return the original error.
			return zero, false, err
		}
		result, err = fn(session)
		return result, reconnected, err
	}
	return result, false, err
}

func (c *connectionRefresher) getSession(ctx context.Context) (*mcp.ClientSession, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.session != nil {
		return c.session, nil
	}

	session, err := c.client.Connect(ctx, c.transport, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to init MCP session: %w", err)
	}

	c.session = session
	return c.session, nil
}

func (c *connectionRefresher) refreshConnection(ctx context.Context) (*mcp.ClientSession, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// First, try ping to confirm connection is dead
	if c.session != nil {
		if err := c.session.Ping(ctx, &mcp.PingParams{}); err == nil {
			// Connection is actually alive, don't refresh
			return c.session, false, nil
		}
		c.session.Close()
		c.session = nil
	}

	session, err := c.client.Connect(ctx, c.transport, nil)
	if err != nil {
		return nil, false, fmt.Errorf("failed to refresh MCP session: %w", err)
	}

	c.session = session
	return c.session, true, nil
}

var _ MCPClient = (*connectionRefresher)(nil)
