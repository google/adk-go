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
	"io"
	"log/slog"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/internal/adkcontext"
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

type testUserKey struct{}

// newTestCredentialSessions returns a pool keyed on the user in testUserKey,
// whose sessions never connect.
func newTestCredentialSessions(t *testing.T) *credentialSessions {
	t.Helper()
	provider := auth.ProviderFunc(func(ctx context.Context) (auth.Credential, error) {
		return auth.BearerCredential{Token: ctx.Value(testUserKey{}).(string)}, nil
	})
	transport := connectFunc(func(context.Context) (mcp.Connection, error) {
		return nil, errors.New("not used")
	})
	p := newCredentialSessions(nil, transport, provider)
	t.Cleanup(func() { _ = p.Close() })
	return p
}

// TestCredentialSessionsReplaceOwnIdleSession checks that a credential whose
// own session went idle past the timeout gets a new session, not the evicted
// one.
func TestCredentialSessionsReplaceOwnIdleSession(t *testing.T) {
	SetSessionBoundsForTest(t, 10*time.Millisecond, 64)
	p := newTestCredentialSessions(t)
	ctx := context.WithValue(t.Context(), testUserKey{}, "alice")

	first, err := p.acquire(ctx)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	p.release(first, nil)
	time.Sleep(20 * time.Millisecond)

	second, err := p.acquire(ctx)
	if err != nil {
		t.Fatalf("acquire after idle timeout: %v", err)
	}
	p.release(second, nil)
	if second == first {
		t.Error("acquire after idle timeout returned the evicted session")
	}
	if got := p.sessions[second.key]; got != second {
		t.Error("session returned after idle timeout is not in the pool")
	}
}

// TestCredentialSessionsShrinkToCapAfterBurst checks that once a burst of
// concurrent users has pushed the pool over maxSessions, the next new
// credential evicts idle sessions until the pool is back within the cap.
func TestCredentialSessionsShrinkToCapAfterBurst(t *testing.T) {
	SetSessionBoundsForTest(t, time.Hour, 2)
	p := newTestCredentialSessions(t)

	var held []*pooledSession
	for _, user := range []string{"alice", "bob", "carol"} {
		s, err := p.acquire(context.WithValue(t.Context(), testUserKey{}, user))
		if err != nil {
			t.Fatalf("acquire(%s): %v", user, err)
		}
		held = append(held, s)
	}
	for _, s := range held {
		p.release(s, nil)
	}

	s, err := p.acquire(context.WithValue(t.Context(), testUserKey{}, "dave"))
	if err != nil {
		t.Fatalf("acquire(dave): %v", err)
	}
	p.release(s, nil)
	if n := len(p.sessions); n != 2 {
		t.Errorf("pool keeps %d sessions, want 2", n)
	}
}

type otherKey struct{}

// TestIdentityOnlyContext checks that the connection context keeps only the
// caller's identity, follows the caller's cancellation, and has no deadline.
func TestIdentityOnlyContext(t *testing.T) {
	want := agent.Identity{AppName: "app", UserID: "alice"}
	parent, cancelParent := context.WithTimeout(t.Context(), time.Hour)
	parent = context.WithValue(parent, adkcontext.IdentityKey, want)
	parent = context.WithValue(parent, otherKey{}, "secret")

	connCtx, stop := identityOnlyContext(parent)
	defer stop()

	if got, ok := agent.IdentityFromContext(connCtx); !ok || got != want {
		t.Errorf("IdentityFromContext = %+v, %v; want %+v, true", got, ok, want)
	}
	if v := connCtx.Value(otherKey{}); v != nil {
		t.Errorf("Value(otherKey) = %v, want nil", v)
	}
	if _, ok := connCtx.Deadline(); ok {
		t.Error("connection context has the caller's deadline")
	}
	cancelParent()
	<-connCtx.Done()
	if !errors.Is(context.Cause(connCtx), context.Canceled) {
		t.Errorf("Cause = %v, want context.Canceled", context.Cause(connCtx))
	}
}

// TestIdentityOnlyContextWithoutIdentity checks that a caller without an ADK
// identity yields a context without one, which stop cancels.
func TestIdentityOnlyContextWithoutIdentity(t *testing.T) {
	connCtx, stop := identityOnlyContext(t.Context())
	if _, ok := agent.IdentityFromContext(connCtx); ok {
		t.Error("IdentityFromContext found an identity, want none")
	}
	stop()
	<-connCtx.Done()
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
// transport and records the server side of every connection.
type serverSessionTransport struct {
	server   *mcp.Server
	sessions []*mcp.ServerSession
}

func (t *serverSessionTransport) Connect(ctx context.Context) (mcp.Connection, error) {
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
	_, err := c.getSession(t.Context())
	if err != nil {
		t.Fatalf("getSession() failed: %v", err)
	}
	done := c.sessionDone
	if err := transport.sessions[len(transport.sessions)-1].Close(); err != nil {
		t.Fatalf("Closing server session failed: %v", err)
	}
	<-done
}

func TestCallToolReconnectsBeforeSendingAfterIdleDisconnect(t *testing.T) {
	var calls atomic.Int32
	transport := &serverSessionTransport{server: newCountingServer(&calls)}
	c := newConnectionRefresher(nil, transport)
	t.Cleanup(func() { _ = c.Close() })
	closeIdleSession(t, c, transport)

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

type countingLogHandler struct{ count atomic.Int32 }

func (*countingLogHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *countingLogHandler) Handle(context.Context, slog.Record) error {
	h.count.Add(1)
	return nil
}

func (h *countingLogHandler) WithAttrs([]slog.Attr) slog.Handler { return h }

func (h *countingLogHandler) WithGroup(string) slog.Handler { return h }

func TestIdleDisconnectClosesClientSession(t *testing.T) {
	logs := &countingLogHandler{}
	client := mcp.NewClient(
		&mcp.Implementation{Name: "test_client", Version: "v1.0.0"},
		&mcp.ClientOptions{
			Logger:                    slog.New(logs),
			KeepAlive:                 5 * time.Millisecond,
			KeepAliveFailureThreshold: 100,
		},
	)
	server := mcp.NewServer(
		&mcp.Implementation{Name: "test_server", Version: "v1.0.0"},
		&mcp.ServerOptions{SupportedProtocolVersions: []string{"2025-11-25"}},
	)
	transport := &serverSessionTransport{server: server}
	c := newConnectionRefresher(client, transport)
	t.Cleanup(func() { _ = c.Close() })
	closeIdleSession(t, c, transport)

	before := logs.count.Load()
	time.Sleep(30 * time.Millisecond)
	if got := logs.count.Load(); got != before {
		t.Errorf("custom client emitted %d logs after ended-session cleanup, want 0", got-before)
	}
}

func TestCallToolReconnectionFailure(t *testing.T) {
	var calls atomic.Int32
	server := newCountingServer(&calls)
	connects := 0
	failed := &atomic.Bool{}
	transport := connectFunc(func(ctx context.Context) (mcp.Connection, error) {
		connects++
		if connects > 1 {
			return nil, errors.New("server unavailable")
		}
		ct, st := mcp.NewInMemoryTransports()
		ss, err := server.Connect(ctx, st, nil)
		if err != nil {
			return nil, err
		}
		t.Cleanup(func() { _ = ss.Close() })
		conn, err := ct.Connect(ctx)
		if err != nil {
			return nil, err
		}
		return &failCallResponseConnection{Connection: conn, failed: failed}, nil
	})
	c := newConnectionRefresher(nil, transport)
	t.Cleanup(func() { _ = c.Close() })

	_, err := c.CallTool(t.Context(), &mcp.CallToolParams{Name: "count"})
	if !errors.Is(err, io.EOF) {
		t.Errorf("CallTool() error = %v, want one wrapping %v", err, io.EOF)
	}
	if err == nil || !strings.Contains(err.Error(), "reconnection also failed") {
		t.Errorf("CallTool() error = %v, want it to report the failed reconnection", err)
	}
	if got := calls.Load(); got != 1 {
		t.Errorf("Server tool executions = %d, want 1", got)
	}
	if connects != 2 {
		t.Errorf("Connections = %d, want 2", connects)
	}
}

// failCallResponseConnection lets the first tools/call reach the server, then
// drops its response to simulate a connection failure after execution.
type failCallResponseConnection struct {
	mcp.Connection
	failed       *atomic.Bool
	failNextRead atomic.Bool
}

func (c *failCallResponseConnection) Write(ctx context.Context, msg jsonrpc.Message) error {
	if req, ok := msg.(*jsonrpc.Request); ok && req.Method == "tools/call" && c.failed.CompareAndSwap(false, true) {
		c.failNextRead.Store(true)
	}
	return c.Connection.Write(ctx, msg)
}

func (c *failCallResponseConnection) Read(ctx context.Context) (jsonrpc.Message, error) {
	msg, err := c.Connection.Read(ctx)
	if err == nil && c.failNextRead.Swap(false) {
		if closeErr := c.Connection.Close(); closeErr != nil {
			return nil, closeErr
		}
		return nil, io.EOF
	}
	return msg, err
}
