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
	"io"
	"log"
	"sync"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/internal/version"
)

// MCPClient abstracts MCP session operations for easier connection management.
type MCPClient interface {
	CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error)
	ListTools(context.Context) ([]*mcp.Tool, error)
}

// connectionRefresher wraps an MCP client/transport and handles automatic reconnection.
// It implements MCPClient. After a session failure it reconnects, and it retries
// an operation only when repeating it is safe: ListTools always, and CallTool
// only when its request was never sent.
type connectionRefresher struct {
	client    *mcp.Client
	transport mcp.Transport

	mu      sync.Mutex
	session *mcp.ClientSession
}

// refreshableErrors is a list of errors that should trigger a connection refresh.
var refreshableErrors = []error{
	mcp.ErrConnectionClosed,
	mcp.ErrSessionMissing,
	io.ErrClosedPipe,
	io.EOF,
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

// CallTool calls a tool on the MCP server. After a connection failure it
// repairs the connection, but it resends the call only when the failure proves
// the request was never sent. Any other failure leaves the outcome unknown,
// because the server may already have run the call, and running a mutating
// tool twice could duplicate a side effect. In that case CallTool returns the
// original error, and the repaired connection serves the next call.
func (c *connectionRefresher) CallTool(ctx context.Context, params *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	session, err := c.getSession(ctx)
	if err != nil {
		return nil, err
	}
	result, err := session.CallTool(ctx, params)
	if err == nil {
		return result, nil
	}
	session, repairErr := c.repairConnection(ctx, err)
	if repairErr != nil {
		return nil, repairErr
	}
	if !isUnsentCallError(err) {
		return nil, err
	}
	return session.CallTool(ctx, params)
}

// isUnsentCallError reports whether err shows that a call was refused before
// its request was written. The MCP SDK documents ErrConnectionClosed as the
// error for sending on a connection that is closed or closing, which is what a
// call returns when the connection broke while the client was idle.
func isUnsentCallError(err error) bool {
	return errors.Is(err, mcp.ErrConnectionClosed)
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
// Use it only for operations that are safe to repeat.
func withRetry[T any](ctx context.Context, c *connectionRefresher, fn func(*mcp.ClientSession) (T, error)) (T, bool, error) {
	var zero T

	session, err := c.getSession(ctx)
	if err != nil {
		return zero, false, err
	}

	result, err := fn(session)
	if err != nil {
		session, err = c.repairConnection(ctx, err)
		if err != nil {
			return zero, false, err
		}
		result, err = fn(session)
		return result, true, err
	}
	return result, false, err
}

// repairConnection refreshes the connection after opErr, an operation's
// failure, when opErr indicates the connection is broken. It returns the new
// session. It returns opErr itself when opErr does not call for a refresh, and
// an error wrapping opErr when the refresh fails.
func (c *connectionRefresher) repairConnection(ctx context.Context, opErr error) (*mcp.ClientSession, error) {
	if !shouldRefreshConnection(opErr) {
		return nil, opErr
	}
	session, err := c.refreshConnection(ctx)
	if err != nil {
		return nil, fmt.Errorf("%w (reconnection also failed: %v)", opErr, err)
	}
	return session, nil
}

// shouldRefreshConnection returns true if the error indicates we should
// attempt to refresh the MCP connection.
func shouldRefreshConnection(err error) bool {
	for _, target := range refreshableErrors {
		if errors.Is(err, target) {
			return true
		}
	}
	return false
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

func (c *connectionRefresher) refreshConnection(ctx context.Context) (*mcp.ClientSession, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Ping to verify the connection is actually dead before reconnecting.
	// This handles the case where another goroutine already reconnected.
	if c.session != nil {
		if err := c.session.Ping(ctx, &mcp.PingParams{}); err == nil {
			return c.session, nil
		}
		if err := c.session.Close(); err != nil {
			log.Printf("failed to close MCP session: %v", err)
		}
		c.session = nil
	}

	session, err := c.client.Connect(ctx, c.transport, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to refresh MCP session: %w", err)
	}

	c.session = session
	return c.session, nil
}

var _ MCPClient = (*connectionRefresher)(nil)
