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

package mcptoolset

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type countInput struct{}

type countOutput struct{}

// newCountingServer returns a server with a single tool, "count", which
// records each execution in calls.
func newCountingServer(calls *atomic.Int32) *mcp.Server {
	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "count"},
		func(context.Context, *mcp.CallToolRequest, countInput) (*mcp.CallToolResult, countOutput, error) {
			calls.Add(1)
			return nil, countOutput{}, nil
		})
	return server
}

// serverSessionTransport connects each client to server over a fresh in-memory
// transport and records the server side of every connection. When
// maxConnects is positive, connections beyond that many fail.
type serverSessionTransport struct {
	server      *mcp.Server
	maxConnects int
	sessions    []*mcp.ServerSession
}

func (t *serverSessionTransport) Connect(ctx context.Context) (mcp.Connection, error) {
	if t.maxConnects > 0 && len(t.sessions) >= t.maxConnects {
		return nil, errors.New("server unavailable")
	}
	ct, st := mcp.NewInMemoryTransports()
	ss, err := t.server.Connect(ctx, st, nil)
	if err != nil {
		return nil, err
	}
	t.sessions = append(t.sessions, ss)
	return ct.Connect(ctx)
}

// closeIdleSession ends the server side of the current connection while the
// client is idle, as when an MCP server process exits between tool calls, and
// waits until the client session has observed it.
func closeIdleSession(t *testing.T, c *connectionRefresher, transport *serverSessionTransport) {
	t.Helper()
	session, err := c.getSession(t.Context())
	if err != nil {
		t.Fatalf("getSession() failed: %v", err)
	}
	if err := transport.sessions[len(transport.sessions)-1].Close(); err != nil {
		t.Fatalf("Closing server session failed: %v", err)
	}
	// Wait reports why the connection ended. Any reason will do here.
	_ = session.Wait()
}

func TestCallToolReplaysUnsentCall(t *testing.T) {
	var calls atomic.Int32
	transport := &serverSessionTransport{server: newCountingServer(&calls)}
	c := newConnectionRefresher(nil, transport)
	closeIdleSession(t, c, transport)

	// The connection is already broken, so the call is refused before its
	// request is written, and it is safe to send it on a new connection.
	if _, err := c.CallTool(t.Context(), &mcp.CallToolParams{Name: "count"}); err != nil {
		t.Fatalf("CallTool() after idle disconnect failed: %v", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("Server tool executions = %d, want 1", got)
	}
	if got := len(transport.sessions); got != 2 {
		t.Errorf("Connections = %d, want 2", got)
	}
}

func TestCallToolReconnectionFailure(t *testing.T) {
	var calls atomic.Int32
	transport := &serverSessionTransport{server: newCountingServer(&calls), maxConnects: 1}
	c := newConnectionRefresher(nil, transport)
	closeIdleSession(t, c, transport)

	_, err := c.CallTool(t.Context(), &mcp.CallToolParams{Name: "count"})
	if !errors.Is(err, mcp.ErrConnectionClosed) {
		t.Errorf("CallTool() error = %v, want one wrapping %v", err, mcp.ErrConnectionClosed)
	}
	if err == nil || !strings.Contains(err.Error(), "reconnection also failed") {
		t.Errorf("CallTool() error = %v, want it to report the failed reconnection", err)
	}
	if got := calls.Load(); got != 0 {
		t.Errorf("Server tool executions = %d, want 0", got)
	}
}
