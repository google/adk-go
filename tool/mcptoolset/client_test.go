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

func TestClosePreventsRetry(t *testing.T) {
	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "1"}, nil)
	connects := 0
	transport := connectFunc(func(ctx context.Context) (mcp.Connection, error) {
		connects++
		ct, st := mcp.NewInMemoryTransports()
		session, err := server.Connect(ctx, st, nil)
		if err != nil {
			return nil, err
		}
		t.Cleanup(func() { _ = session.Close() })
		return ct.Connect(ctx)
	})
	client := newConnectionRefresher(nil, transport)
	t.Cleanup(func() { _ = client.Close() })
	calls := 0
	_, _, err := withRetry(t.Context(), client, func(*mcp.ClientSession) (*mcp.CallToolResult, error) {
		calls++
		// The toolset closes after an operation starts, before its error is handled.
		if err := client.Close(); err != nil {
			t.Fatal(err)
		}
		return nil, mcp.ErrConnectionClosed
	})
	if !errors.Is(err, mcp.ErrConnectionClosed) {
		t.Errorf("withRetry() error = %v, want ErrConnectionClosed", err)
	}
	if calls != 1 || connects != 1 {
		t.Errorf("after Close(): %d calls and %d connections, want 1 each", calls, connects)
	}
}

type connectFunc func(context.Context) (mcp.Connection, error)

func (f connectFunc) Connect(ctx context.Context) (mcp.Connection, error) {
	return f(ctx)
}

func TestWithRetryDoesNotRefreshNonRefreshableError(t *testing.T) {
	server := mcp.NewServer(&mcp.Implementation{Name: "test_server", Version: "1"}, nil)
	connects := 0
	transport := connectFunc(func(ctx context.Context) (mcp.Connection, error) {
		connects++
		ct, st := mcp.NewInMemoryTransports()
		session, err := server.Connect(ctx, st, nil)
		if err != nil {
			return nil, err
		}
		t.Cleanup(func() { _ = session.Close() })
		return ct.Connect(ctx)
	})
	client := newConnectionRefresher(nil, transport)
	t.Cleanup(func() { _ = client.Close() })
	wantErr := errors.New("operation failed")
	calls := 0
	_, reconnected, err := withRetry(t.Context(), client, func(*mcp.ClientSession) (*mcp.CallToolResult, error) {
		calls++
		return nil, wantErr
	})
	if !errors.Is(err, wantErr) {
		t.Errorf("withRetry() error = %v, want %v", err, wantErr)
	}
	if reconnected {
		t.Error("withRetry() reconnected for a non-refreshable error")
	}
	if calls != 1 || connects != 1 {
		t.Errorf("after a non-refreshable error: %d calls and %d connections, want 1 each", calls, connects)
	}
}

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

func TestCallToolDoesNotReplayAfterIdleDisconnect(t *testing.T) {
	var calls atomic.Int32
	transport := &serverSessionTransport{server: newCountingServer(&calls)}
	c := newConnectionRefresher(nil, transport)
	t.Cleanup(func() { _ = c.Close() })
	closeIdleSession(t, c, transport)

	// Even though this in-memory transport refuses the call before writing it,
	// CallTool cannot reliably distinguish that error from one returned by a
	// server after execution, so it repairs the connection without replaying.
	if _, err := c.CallTool(t.Context(), &mcp.CallToolParams{Name: "count"}); !errors.Is(err, mcp.ErrConnectionClosed) {
		t.Fatalf("CallTool() after idle disconnect error = %v, want %v", err, mcp.ErrConnectionClosed)
	}
	if got := calls.Load(); got != 0 {
		t.Errorf("Server tool executions after failed call = %d, want 0", got)
	}
	if got := len(transport.sessions); got != 2 {
		t.Errorf("Connections = %d, want 2", got)
	}

	if _, err := c.CallTool(t.Context(), &mcp.CallToolParams{Name: "count"}); err != nil {
		t.Fatalf("CallTool() on repaired connection failed: %v", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("Server tool executions after next call = %d, want 1", got)
	}
}

func TestCallToolReconnectionFailure(t *testing.T) {
	var calls atomic.Int32
	transport := &serverSessionTransport{server: newCountingServer(&calls), maxConnects: 1}
	c := newConnectionRefresher(nil, transport)
	t.Cleanup(func() { _ = c.Close() })
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
