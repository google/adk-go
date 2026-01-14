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
	"errors"
	"fmt"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// MCPClient defines the interface for interacting with an MCP server.
// It abstracts the session management and reconnection logic.
type MCPClient interface {
	ListTools(ctx context.Context, params *mcp.ListToolsParams) (*mcp.ListToolsResult, error)
	CallTool(ctx context.Context, params *mcp.CallToolParams) (*mcp.CallToolResult, error)
}

// reconnectingClient implements MCPClient with auto-reconnection logic.
type reconnectingClient struct {
	client    *mcp.Client
	transport mcp.Transport

	mu      sync.Mutex
	session *mcp.ClientSession
}

func (c *reconnectingClient) ListTools(ctx context.Context, params *mcp.ListToolsParams) (*mcp.ListToolsResult, error) {
	session, err := c.getSession(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get MCP session: %w", err)
	}

	result, err := session.ListTools(ctx, params)
	if errors.Is(err, mcp.ErrConnectionClosed) {
		// Attempt to refresh the connection and retry
		session, refreshErr := c.refreshConnection(ctx)
		if refreshErr != nil {
			return nil, fmt.Errorf("failed to refresh MCP session after connection closed: %w", refreshErr)
		}
		// Retry with the new session
		// Note from original logic: "Per MCP spec, cursors should not persist across sessions."
		// So we reset cursor if it was present, but params is passed by pointer.
		// If we are just listing, we might need to be careful about pagination.
		// However, the original code reset cursor to "" inside the loop.
		// Since this is a single call, the caller (set.Tools) manages the loop and cursor.
		// If we reconnect here, the caller's cursor might be invalid for the new session if it wasn't empty.
		// But ListTools is stateless regarding session if cursor is empty.
		// If cursor is NOT empty, retrying might fail or return bad data if session state matters,
		// but typically cursor is opaque token.
		// For safety/strict correctness matching previous logic, if we reconnect,
		// we return the error or handle it.
		// The previous logic in `Tools` loop handled the cursor reset.
		// Here we just retry the call. If the cursor is invalid for new session, server return error.
		return session.ListTools(ctx, params)
	}
	return result, err
}

func (c *reconnectingClient) CallTool(ctx context.Context, params *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	session, err := c.getSession(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get MCP session: %w", err)
	}

	result, err := session.CallTool(ctx, params)
	if errors.Is(err, mcp.ErrConnectionClosed) {
		// Attempt to refresh the connection and retry
		session, refreshErr := c.refreshConnection(ctx)
		if refreshErr != nil {
			return nil, fmt.Errorf("failed to refresh MCP session after connection closed: %w", refreshErr)
		}
		return session.CallTool(ctx, params)
	}
	return result, err
}

func (c *reconnectingClient) getSession(ctx context.Context) (*mcp.ClientSession, error) {
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

func (c *reconnectingClient) refreshConnection(ctx context.Context) (*mcp.ClientSession, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// First, try ping to confirm connection is dead
	// This double check is useful if another goroutine already refreshed it,
	// OR if the error was transient/misleading (though ErrConnectionClosed shouldn't be).
	if c.session != nil {
		if err := c.session.Ping(ctx, &mcp.PingParams{}); err == nil {
			// Connection is actually alive, don't refresh
			return c.session, nil
		}
		// Ensure previous session is closed if it wasn't already
		// We ignore error here as we know it's likely dead
		_ = c.session.Close()
		c.session = nil
	}

	session, err := c.client.Connect(ctx, c.transport, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to refresh MCP session: %w", err)
	}

	c.session = session
	return c.session, nil
}
