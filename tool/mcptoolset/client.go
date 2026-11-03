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
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"slices"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	"google.golang.org/adk/v2/internal/adkcontext"
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

	// mu is a one-slot channel rather than a mutex: session initialization and
	// refresh perform network I/O, and waiters must be able to leave when their
	// own context ends without interrupting the owner. See lock.
	mu      chan struct{}
	session *mcp.ClientSession
	closed  bool

	// identityOnlyConnect connects sessions on identityOnlyContext(ctx)
	// rather than ctx.
	identityOnlyConnect bool

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
		mu:        make(chan struct{}, 1),
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
		// in progress.
		c.mu <- struct{}{}
		// Closing is final so failed in-flight calls cannot reconnect.
		c.closed = true
		session := c.session
		c.session = nil
		c.unlock()

		if session != nil {
			c.closeErr = session.Close()
		}
	})
	return c.closeErr
}

// lock takes mu, or returns ctx's error if ctx ends first. A ctx that is done
// by the time mu is taken fails too, even when mu was free, as
// semaphore.Weighted.Acquire does.
func (c *connectionRefresher) lock(ctx context.Context) error {
	select {
	case c.mu <- struct{}{}:
		// ctx may have ended while this call waited for mu to be handed over.
		if err := ctx.Err(); err != nil {
			c.unlock()
			return err
		}
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (c *connectionRefresher) unlock() {
	<-c.mu
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
	if err := c.lock(ctx); err != nil {
		return nil, err
	}
	defer c.unlock()

	if c.closed {
		return nil, mcp.ErrConnectionClosed
	}
	if c.session != nil {
		return c.session, nil
	}

	session, err := c.connect(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to init MCP session: %w", err)
	}

	c.session = session
	return c.session, nil
}

func (c *connectionRefresher) refreshConnection(ctx context.Context) (*mcp.ClientSession, error) {
	if err := c.lock(ctx); err != nil {
		return nil, err
	}
	defer c.unlock()

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

	session, err := c.connect(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to refresh MCP session: %w", err)
	}

	c.session = session
	return c.session, nil
}

// connect opens a new MCP session. The caller holds c.mu.
func (c *connectionRefresher) connect(ctx context.Context) (*mcp.ClientSession, error) {
	if !c.identityOnlyConnect {
		return c.client.Connect(ctx, c.transport, nil)
	}
	connCtx, stop := identityOnlyContext(ctx)
	defer stop()
	return c.client.Connect(connCtx, c.transport, nil)
}

// disconnected reports whether c has no session and no connection attempt
// in progress.
func (c *connectionRefresher) disconnected() bool {
	select {
	case c.mu <- struct{}{}:
	default:
		return false
	}
	defer c.unlock()
	return c.session == nil
}

var _ MCPClient = (*connectionRefresher)(nil)

// Bounds on the sessions a credentialSessions keeps. Variables so tests can
// shrink them.
var (
	// sessionIdleTimeout is how long a session may go unused before it is
	// closed.
	sessionIdleTimeout = 10 * time.Minute
	// maxSessions caps the sessions kept open. Sessions in use are never
	// evicted, so the cap is exceeded only while more than that many are busy.
	maxSessions = 64
)

// credentialSessions gives each distinct credential an MCP session of its own.
//
// It is used when Config.Auth is set. An MCP server that binds a session to
// the user who created it (as the MCP spec recommends, and as go-sdk servers do
// whenever their token verifier reports a UserID) rejects requests made on that
// session with another user's credential, and the rejection also closes the
// session for its owner. So a provider that mints a credential per user needs a
// session per user, while a user-independent one such as [auth.StaticToken]
// serves every caller on one session. Keying sessions on the credential the
// provider resolves covers both, as adk-python does with its header-based
// session key.
//
// A session is closed once it has been idle for sessionIdleTimeout, or when
// the pool is full and it is the least recently used idle one.
type credentialSessions struct {
	client    *mcp.Client
	transport mcp.Transport
	provider  auth.CredentialProvider

	mu       sync.Mutex
	closed   bool
	sessions map[string]*pooledSession
	// evicting tracks evicted sessions that are still closing, so Close can
	// wait for them.
	evicting sync.WaitGroup

	closeOnce sync.Once
	closeErr  error
}

// pooledSession is one credential's connection and its use. inUse and lastUsed
// are guarded by credentialSessions.mu.
type pooledSession struct {
	key      string
	conn     *connectionRefresher
	inUse    int
	lastUsed time.Time
}

func newCredentialSessions(client *mcp.Client, transport mcp.Transport, provider auth.CredentialProvider) *credentialSessions {
	return &credentialSessions{
		client:    orDefaultClient(client),
		transport: transport,
		provider:  provider,
		sessions:  make(map[string]*pooledSession),
	}
}

// sessionKey returns a digest of the headers the provider's credential sets
// for the caller in ctx.
func (p *credentialSessions) sessionKey(ctx context.Context) (string, error) {
	cred, err := p.provider.Credential(ctx)
	if err != nil {
		return "", fmt.Errorf("auth: resolve credential: %w", err)
	}
	if cred == nil {
		return "", fmt.Errorf("auth: provider returned nil credential")
	}
	h := http.Header{}
	if err := cred.Apply(h); err != nil {
		return "", fmt.Errorf("auth: apply credential: %w", err)
	}
	names := make([]string, 0, len(h))
	for name := range h {
		names = append(names, name)
	}
	slices.Sort(names)
	// Length-prefix every field so that no two header sets encode alike.
	var b []byte
	for _, name := range names {
		b = fmt.Appendf(b, "%d:%s%d:", len(name), name, len(h[name]))
		for _, v := range h[name] {
			b = fmt.Appendf(b, "%d:%s", len(v), v)
		}
	}
	sum := sha256.Sum256(b)
	return string(sum[:]), nil
}

// acquire returns the session for the credential of the caller in ctx,
// creating it on first use, and marks it in use until release. The session
// itself connects lazily.
func (p *credentialSessions) acquire(ctx context.Context) (*pooledSession, error) {
	if p.isClosed() {
		return nil, mcp.ErrConnectionClosed
	}
	key, err := p.sessionKey(ctx)
	if err != nil {
		return nil, err
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil, mcp.ErrConnectionClosed
	}
	now := time.Now()
	// Sweep before the lookup so an expired session of this key is replaced
	// rather than handed out after it was evicted.
	p.evictIdleLocked(now)
	s, ok := p.sessions[key]
	if !ok {
		p.makeRoomLocked()
		conn := newConnectionRefresher(p.client, p.transport)
		conn.identityOnlyConnect = true
		s = &pooledSession{key: key, conn: conn}
		p.sessions[key] = s
	}
	s.inUse++
	s.lastUsed = now
	return s, nil
}

func (p *credentialSessions) isClosed() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.closed
}

// release ends a use of s that finished with err. A session that never
// connected, or lost its connection and could not reconnect, is dropped so the
// pool does not keep entries for credentials the server refuses.
func (p *credentialSessions) release(s *pooledSession, err error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	s.inUse--
	s.lastUsed = time.Now()
	// disconnected does not block, so checking it under mu means the last
	// release of a failed session always sees it.
	if err != nil && s.inUse == 0 && p.sessions[s.key] == s && s.conn.disconnected() {
		p.evictSessionLocked(s)
	}
}

// evictIdleLocked closes sessions unused for sessionIdleTimeout.
func (p *credentialSessions) evictIdleLocked(now time.Time) {
	for _, s := range p.sessions {
		if s.inUse == 0 && now.Sub(s.lastUsed) >= sessionIdleTimeout {
			p.evictSessionLocked(s)
		}
	}
}

// makeRoomLocked closes least recently used idle sessions until there is room
// for one more, or no idle session is left. A burst of concurrent callers can
// leave the pool over maxSessions, so this may close more than one.
func (p *credentialSessions) makeRoomLocked() {
	for len(p.sessions) >= maxSessions {
		var lru *pooledSession
		for _, s := range p.sessions {
			if s.inUse == 0 && (lru == nil || s.lastUsed.Before(lru.lastUsed)) {
				lru = s
			}
		}
		if lru == nil {
			return
		}
		p.evictSessionLocked(lru)
	}
}

// evictSessionLocked removes s from the pool and closes it in the background,
// since closing sends a request to the server.
func (p *credentialSessions) evictSessionLocked(s *pooledSession) {
	delete(p.sessions, s.key)
	p.evicting.Go(func() {
		// No caller is left to report a failure to, and the session is gone
		// from the pool either way.
		_ = s.conn.Close()
	})
}

// CallTool calls a tool on the session for the caller's credential.
func (p *credentialSessions) CallTool(ctx context.Context, params *mcp.CallToolParams) (result *mcp.CallToolResult, err error) {
	s, err := p.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { p.release(s, err) }()
	return s.conn.CallTool(ctx, params)
}

// ListTools lists tools on the session for the caller's credential.
func (p *credentialSessions) ListTools(ctx context.Context) (tools []*mcp.Tool, err error) {
	s, err := p.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { p.release(s, err) }()
	return s.conn.ListTools(ctx)
}

// Close closes every session, waiting for connection setup and MCP operations
// in progress, as [connectionRefresher.Close] does. It is safe to call
// concurrently or more than once, and every call returns the first one's
// result. Later operations fail with [mcp.ErrConnectionClosed].
func (p *credentialSessions) Close() error {
	p.closeOnce.Do(func() {
		p.mu.Lock()
		p.closed = true
		sessions := p.sessions
		p.sessions = nil
		p.mu.Unlock()

		var errs []error
		for _, s := range sessions {
			if err := s.conn.Close(); err != nil {
				errs = append(errs, err)
			}
		}
		p.evicting.Wait()
		p.closeErr = errors.Join(errs...)
	})
	return p.closeErr
}

var _ MCPClient = (*credentialSessions)(nil)

// identityOnlyContext returns a context that carries the ADK identity of ctx
// and no other value, and is canceled when ctx is until stop is called.
//
// go-sdk keeps the context a streamable session is connected on, minus its
// cancellation, for the life of the session, so that the standalone stream and
// the closing request carry the same values. Connecting on an invocation's
// context would keep that whole invocation and its session in memory until the
// MCP session closes. The identity is all the credential provider needs.
func identityOnlyContext(ctx context.Context) (connCtx context.Context, stop func()) {
	var base identityContext
	if id, ok := agent.IdentityFromContext(ctx); ok {
		base.id = &id
	}
	connCtx, cancel := context.WithCancelCause(base)
	stopCancel := context.AfterFunc(ctx, func() { cancel(context.Cause(ctx)) })
	return connCtx, func() {
		stopCancel()
		cancel(nil)
	}
}

// identityContext is a root context, never canceled and without a deadline,
// that carries an ADK identity and no other value. It stands in for
// context.Background, which library code must not use.
type identityContext struct {
	id *agent.Identity
}

func (identityContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (identityContext) Done() <-chan struct{}       { return nil }
func (identityContext) Err() error                  { return nil }

func (c identityContext) Value(key any) any {
	if key == adkcontext.IdentityKey && c.id != nil {
		return *c.id
	}
	return nil
}
