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
	"golang.org/x/sync/semaphore"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/version"
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

	// Session initialization and refresh perform network I/O. Waiters must be
	// able to leave when their own context ends without interrupting the owner.
	mu      *semaphore.Weighted
	session *mcp.ClientSession
	closed  bool

	closeOnce sync.Once
	closeErr  error
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
	return &connectionRefresher{
		client:    orDefaultClient(client),
		transport: transport,
		mu:        semaphore.NewWeighted(1),
	}
}

// orDefaultClient returns client, or a default MCP client if it is nil.
func orDefaultClient(client *mcp.Client) *mcp.Client {
	if client == nil {
		return mcp.NewClient(&mcp.Implementation{Name: "adk-mcp-client", Version: version.Version}, nil)
	}
	return client
}

func (c *connectionRefresher) Close() error {
	c.closeOnce.Do(func() {
		// Close has no caller context, so it waits for any connection attempt
		// in progress. Acquire cannot fail with a context that is never done.
		_ = c.mu.Acquire(context.Background(), 1)
		// Closing is final so failed in-flight calls cannot reconnect.
		c.closed = true
		session := c.session
		c.session = nil
		c.mu.Release(1)

		if session != nil {
			c.closeErr = session.Close()
		}
	})
	return c.closeErr
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
		if !shouldRefreshConnection(err) {
			return zero, false, err
		}
		session, refreshErr := c.refreshConnection(ctx)
		if refreshErr != nil {
			return zero, false, fmt.Errorf("%w (reconnection also failed: %v)", err, refreshErr)
		}
		result, err = fn(session)
		return result, true, err
	}
	return result, false, err
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
	if err := c.mu.Acquire(ctx, 1); err != nil {
		return nil, err
	}
	defer c.mu.Release(1)

	if c.closed {
		return nil, mcp.ErrConnectionClosed
	}
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
	if err := c.mu.Acquire(ctx, 1); err != nil {
		return nil, err
	}
	defer c.mu.Release(1)

	if c.closed {
		return nil, mcp.ErrConnectionClosed
	}

	// Ping to verify the connection is actually dead before reconnecting.
	// This handles the case where another goroutine already reconnected.
	if c.session != nil {
		if err := c.session.Ping(ctx, &mcp.PingParams{}); err == nil {
			return c.session, nil
		}
		if err := c.session.Close(); err != nil {
			log.Printf("failed to close MCP session: %v", err) //nolint:forbidigo // pre-slog call site
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

// userKey identifies the acting user an MCP session belongs to.
type userKey struct {
	appName string
	userID  string
}

// perUserClients gives each acting user an MCP session of its own, keyed by
// the app and user from [agent.IdentityFromContext].
//
// It is used when Config.Auth is set. The credential then differs per user,
// and an MCP server that binds a session to the user who created it (as the
// MCP spec recommends, and as go-sdk servers do whenever their token verifier
// reports a UserID) rejects requests on another user's session. A single
// shared session would work only for whichever user opened it, and a rejected
// request from another user would also close it for that user.
//
// A context without an identity, or with an empty UserID, shares one session,
// which is what every caller got before.
//
// Sessions are kept until Close, one per user that has called the toolset.
type perUserClients struct {
	client    *mcp.Client
	transport mcp.Transport

	mu       sync.Mutex
	closed   bool
	sessions map[userKey]*connectionRefresher
}

func newPerUserClients(client *mcp.Client, transport mcp.Transport) *perUserClients {
	return &perUserClients{
		client:    orDefaultClient(client),
		transport: transport,
		sessions:  make(map[userKey]*connectionRefresher),
	}
}

// forUser returns the connection of the user acting in ctx, creating it on
// first use. The connection itself connects lazily.
func (p *perUserClients) forUser(ctx context.Context) (*connectionRefresher, error) {
	var key userKey
	if id, ok := agent.IdentityFromContext(ctx); ok && id.UserID != "" {
		key = userKey{appName: id.AppName, userID: id.UserID}
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil, mcp.ErrConnectionClosed
	}
	c, ok := p.sessions[key]
	if !ok {
		c = newConnectionRefresher(p.client, p.transport)
		p.sessions[key] = c
	}
	return c, nil
}

// CallTool calls a tool on the acting user's MCP session.
func (p *perUserClients) CallTool(ctx context.Context, params *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	c, err := p.forUser(ctx)
	if err != nil {
		return nil, err
	}
	return c.CallTool(ctx, params)
}

// ListTools lists tools on the acting user's MCP session.
func (p *perUserClients) ListTools(ctx context.Context) ([]*mcp.Tool, error) {
	c, err := p.forUser(ctx)
	if err != nil {
		return nil, err
	}
	return c.ListTools(ctx)
}

// Close closes every user's session. Later calls fail with
// [mcp.ErrConnectionClosed].
func (p *perUserClients) Close() error {
	p.mu.Lock()
	p.closed = true
	sessions := p.sessions
	p.sessions = nil
	p.mu.Unlock()

	var errs []error
	for _, c := range sessions {
		if err := c.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

var _ MCPClient = (*perUserClients)(nil)
