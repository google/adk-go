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

package mcptoolset_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
	"weak"

	sdkauth "github.com/modelcontextprotocol/go-sdk/auth"
	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/mcptoolset"
)

// userBoundServer is an MCP server that authenticates every request with a
// bearer token "token-of-<user>" and, like any go-sdk server whose verifier
// reports a UserID, binds each MCP session to the user that created it:
// a request on another user's session gets 403 "session user mismatch".
//
// It records the MCP sessions each user's requests arrived on, and those each
// user closed with DELETE.
type userBoundServer struct {
	*httptest.Server

	mu       sync.Mutex
	sessions map[string]map[string]bool // user -> MCP session IDs used
	deleted  map[string]map[string]bool // user -> MCP session IDs closed
}

func newUserBoundServer(t *testing.T) *userBoundServer {
	t.Helper()
	server := mcp.NewServer(&mcp.Implementation{Name: "user_bound_server", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "whoami", Description: "returns the caller"},
		func(ctx context.Context, req *mcp.CallToolRequest, _ struct{}) (*mcp.CallToolResult, any, error) {
			return &mcp.CallToolResult{Content: []mcp.Content{
				&mcp.TextContent{Text: req.Extra.TokenInfo.UserID},
			}}, nil, nil
		})
	verify := func(_ context.Context, token string, _ *http.Request) (*sdkauth.TokenInfo, error) {
		user, ok := strings.CutPrefix(token, "token-of-")
		if !ok {
			return nil, sdkauth.ErrInvalidToken
		}
		return &sdkauth.TokenInfo{UserID: user, Expiration: time.Now().Add(time.Hour)}, nil
	}
	handler := sdkauth.RequireBearerToken(verify, nil)(
		mcp.NewStreamableHTTPHandler(func(*http.Request) *mcp.Server { return server }, nil))

	s := &userBoundServer{sessions: map[string]map[string]bool{}, deleted: map[string]map[string]bool{}}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		user := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer token-of-")
		if id := r.Header.Get("Mcp-Session-Id"); id != "" {
			s.mu.Lock()
			record(s.sessions, user, id)
			if r.Method == http.MethodDelete {
				record(s.deleted, user, id)
			}
			s.mu.Unlock()
		}
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(s.Close)
	return s
}

func record(m map[string]map[string]bool, user, id string) {
	if m[user] == nil {
		m[user] = map[string]bool{}
	}
	m[user][id] = true
}

func (s *userBoundServer) sessionsOf(user string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.sessions[user])
}

func (s *userBoundServer) deletedOf(user string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.deleted[user])
}

// perUserToken mints "token-of-<user>" for the acting user.
var perUserToken = auth.ProviderFunc(func(ctx context.Context) (auth.Credential, error) {
	id, ok := agent.IdentityFromContext(ctx)
	if !ok || id.UserID == "" {
		return nil, errors.New("no acting user")
	}
	return auth.BearerCredential{Token: "token-of-" + id.UserID}, nil
})

func newAuthToolset(t *testing.T, server *userBoundServer, provider auth.CredentialProvider) tool.Toolset {
	t.Helper()
	ts, err := mcptoolset.New(mcptoolset.Config{Endpoint: server.URL, Auth: provider})
	if err != nil {
		t.Fatalf("mcptoolset.New() failed: %v", err)
	}
	t.Cleanup(func() { _ = ts.(io.Closer).Close() })
	return ts
}

// invocationAs returns an invocation context whose session belongs to user of
// app.
func invocationAs(t *testing.T, app, user string) agent.InvocationContext {
	t.Helper()
	created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: app, UserID: user})
	if err != nil {
		t.Fatalf("Create session for %q: %v", user, err)
	}
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: created.Session})
}

// callWhoami lists the toolset's tools and runs whoami as user of app,
// returning the user the server saw.
func callWhoami(ts tool.Toolset, invCtx agent.InvocationContext) (string, error) {
	tools, err := ts.Tools(icontext.NewReadonlyContext(invCtx))
	if err != nil {
		return "", fmt.Errorf("Tools(): %w", err)
	}
	if len(tools) != 1 {
		return "", fmt.Errorf("Tools() returned %d tools, want 1", len(tools))
	}
	got, err := tools[0].(toolinternal.FunctionTool).Run(agent.NewToolContext(invCtx, "", nil, nil), map[string]any{})
	if err != nil {
		return "", fmt.Errorf("Run(): %w", err)
	}
	return fmt.Sprint(got["output"]), nil
}

func callAs(t *testing.T, ts tool.Toolset, app, user, want string) {
	t.Helper()
	got, err := callWhoami(ts, invocationAs(t, app, user))
	if err != nil {
		t.Fatalf("as %s/%s: %v", app, user, err)
	}
	if got != want {
		t.Errorf("as %s/%s: server saw %q, want %q", app, user, got, want)
	}
}

// TestAuthGivesEachUserItsOwnSession checks that with Config.Auth, users
// sharing one toolset each get an MCP session of their own. A server that binds
// sessions to users rejects a request on another user's session, so a single
// shared session works only for whichever user opened it.
func TestAuthGivesEachUserItsOwnSession(t *testing.T) {
	server := newUserBoundServer(t)
	ts := newAuthToolset(t, server, perUserToken)

	for _, user := range []string{"alice", "bob", "alice", "bob"} {
		callAs(t, ts, "app", user, user)
	}

	// Each user stays on one session: bob's calls did not tear down alice's.
	for _, user := range []string{"alice", "bob"} {
		if n := server.sessionsOf(user); n != 1 {
			t.Errorf("%s used %d MCP sessions, want 1", user, n)
		}
	}
}

// TestAuthConcurrentUsers checks that many users calling one toolset at once
// each stay on a session of their own.
func TestAuthConcurrentUsers(t *testing.T) {
	server := newUserBoundServer(t)
	ts := newAuthToolset(t, server, perUserToken)

	const users, callsPerUser = 8, 4
	var wg sync.WaitGroup
	errs := make(chan error, users*callsPerUser)
	for u := range users {
		user := fmt.Sprintf("user%d", u)
		for range callsPerUser {
			invCtx := invocationAs(t, "app", user)
			wg.Go(func() {
				got, err := callWhoami(ts, invCtx)
				if err == nil && got != user {
					err = fmt.Errorf("server saw %q, want %q", got, user)
				}
				if err != nil {
					errs <- fmt.Errorf("as %s: %w", user, err)
				}
			})
		}
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	for u := range users {
		if n := server.sessionsOf(fmt.Sprintf("user%d", u)); n != 1 {
			t.Errorf("user%d used %d MCP sessions, want 1", u, n)
		}
	}
}

// TestAuthSameUserAcrossApps checks that sessions follow the credential, not
// the user ID: the same user ID in two apps gets two sessions when the
// provider mints a credential per app.
func TestAuthSameUserAcrossApps(t *testing.T) {
	server := newUserBoundServer(t)
	perAppToken := auth.ProviderFunc(func(ctx context.Context) (auth.Credential, error) {
		id, _ := agent.IdentityFromContext(ctx)
		return auth.BearerCredential{Token: "token-of-" + id.AppName + "." + id.UserID}, nil
	})
	ts := newAuthToolset(t, server, perAppToken)

	for _, app := range []string{"app1", "app2", "app1", "app2"} {
		callAs(t, ts, app, "alice", app+".alice")
	}
	for _, user := range []string{"app1.alice", "app2.alice"} {
		if n := server.sessionsOf(user); n != 1 {
			t.Errorf("%s used %d MCP sessions, want 1", user, n)
		}
	}
}

// TestAuthUserIndependentCredentialSharesOneSession checks that a credential
// that is the same for every user, such as auth.StaticToken, keeps one MCP
// session for all of them.
func TestAuthUserIndependentCredentialSharesOneSession(t *testing.T) {
	server := newUserBoundServer(t)
	ts := newAuthToolset(t, server, auth.StaticToken("token-of-svc"))

	for _, user := range []string{"alice", "bob", "carol"} {
		callAs(t, ts, "app", user, "svc")
	}
	if n := server.sessionsOf("svc"); n != 1 {
		t.Errorf("three users with one static token used %d MCP sessions, want 1", n)
	}
}

// TestAuthWithoutActingUserSharesOneSession checks that callers with no ADK
// identity, such as a service-account credential used outside an invocation,
// keep sharing one session as before.
func TestAuthWithoutActingUserSharesOneSession(t *testing.T) {
	server := newUserBoundServer(t)
	ts := newAuthToolset(t, server, auth.StaticToken("token-of-svc"))

	for range 2 {
		ctx := icontext.NewReadonlyContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}))
		if _, err := ts.Tools(ctx); err != nil {
			t.Fatalf("Tools() failed: %v", err)
		}
	}
	if n := server.sessionsOf("svc"); n != 1 {
		t.Errorf("callers without a user used %d MCP sessions, want 1", n)
	}
}

// TestAuthEvictsIdleSessions checks that a session unused for the idle timeout
// is closed and that its user gets a new one on the next call.
func TestAuthEvictsIdleSessions(t *testing.T) {
	mcptoolset.SetSessionBoundsForTest(t, 50*time.Millisecond, 64)
	server := newUserBoundServer(t)
	ts := newAuthToolset(t, server, perUserToken)

	callAs(t, ts, "app", "alice", "alice")
	time.Sleep(100 * time.Millisecond)
	callAs(t, ts, "app", "bob", "bob") // evicts alice's idle session
	callAs(t, ts, "app", "alice", "alice")

	if err := ts.(io.Closer).Close(); err != nil {
		t.Fatalf("Close() failed: %v", err)
	}
	if n := server.sessionsOf("alice"); n != 2 {
		t.Errorf("alice used %d MCP sessions, want 2 (one evicted, one new)", n)
	}
	if n := server.deletedOf("alice"); n != 2 {
		t.Errorf("alice's sessions closed with DELETE: %d, want 2", n)
	}
}

// TestAuthIdleSessionSameUserReturns checks that a user whose own session
// went idle past the timeout gets a new session rather than the evicted one.
func TestAuthIdleSessionSameUserReturns(t *testing.T) {
	mcptoolset.SetSessionBoundsForTest(t, 50*time.Millisecond, 64)
	server := newUserBoundServer(t)
	ts := newAuthToolset(t, server, perUserToken)

	callAs(t, ts, "app", "alice", "alice")
	time.Sleep(100 * time.Millisecond)
	callAs(t, ts, "app", "alice", "alice")

	if n := mcptoolset.SessionCountForTest(ts); n != 1 {
		t.Errorf("toolset keeps %d sessions, want 1", n)
	}
	if n := server.sessionsOf("alice"); n != 2 {
		t.Errorf("alice used %d MCP sessions, want 2 (one evicted, one new)", n)
	}
}

// TestAuthCapsOpenSessions checks that a full pool closes its least recently
// used idle session to make room for a new one.
func TestAuthCapsOpenSessions(t *testing.T) {
	mcptoolset.SetSessionBoundsForTest(t, time.Hour, 2)
	server := newUserBoundServer(t)
	ts := newAuthToolset(t, server, perUserToken)

	for _, user := range []string{"alice", "bob", "carol"} {
		callAs(t, ts, "app", user, user)
	}
	if n := mcptoolset.SessionCountForTest(ts); n != 2 {
		t.Errorf("toolset keeps %d sessions, want 2", n)
	}
	callAs(t, ts, "app", "alice", "alice")
	if n := server.sessionsOf("alice"); n != 2 {
		t.Errorf("alice used %d MCP sessions, want 2 (least recently used one evicted)", n)
	}
	if n := server.sessionsOf("carol"); n != 1 {
		t.Errorf("carol used %d MCP sessions, want 1", n)
	}
}

// TestAuthDropsSessionThatFailedToConnect checks that a credential the server
// refuses does not leave a session behind.
func TestAuthDropsSessionThatFailedToConnect(t *testing.T) {
	server := newUserBoundServer(t)
	rejected := auth.ProviderFunc(func(ctx context.Context) (auth.Credential, error) {
		id, _ := agent.IdentityFromContext(ctx)
		return auth.BearerCredential{Token: "revoked-" + id.UserID}, nil
	})
	ts := newAuthToolset(t, server, rejected)

	for _, user := range []string{"alice", "bob"} {
		if _, err := ts.Tools(icontext.NewReadonlyContext(invocationAs(t, "app", user))); err == nil {
			t.Fatalf("Tools() as %s with a refused credential succeeded, want error", user)
		}
	}
	if n := mcptoolset.SessionCountForTest(ts); n != 0 {
		t.Errorf("toolset keeps %d sessions after failed connects, want 0", n)
	}
}

// TestAuthDoesNotRetainInvocation checks that an open session does not keep
// the invocation that opened it in memory. go-sdk keeps the context a session
// was connected on for the session's lifetime.
func TestAuthDoesNotRetainInvocation(t *testing.T) {
	server := newUserBoundServer(t)
	ts := newAuthToolset(t, server, perUserToken)

	type marker struct{ _ [1 << 10]byte }
	type markerKey struct{}
	ref := func() weak.Pointer[marker] {
		m := &marker{}
		created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "alice"})
		if err != nil {
			t.Fatalf("Create session: %v", err)
		}
		base := context.WithValue(t.Context(), markerKey{}, m)
		invCtx := icontext.NewInvocationContext(base, icontext.InvocationContextParams{Session: created.Session})
		if _, err := ts.Tools(icontext.NewReadonlyContext(invCtx)); err != nil {
			t.Fatalf("Tools() failed: %v", err)
		}
		return weak.Make(m)
	}()

	for range 5 {
		runtime.GC()
		if ref.Value() == nil {
			return
		}
	}
	t.Error("the open MCP session keeps the invocation that opened it in memory")
}

// TestAuthCloseClosesEveryUsersSession checks that Close ends the sessions of
// all users on the server and that later calls fail.
func TestAuthCloseClosesEveryUsersSession(t *testing.T) {
	server := newUserBoundServer(t)
	ts := newAuthToolset(t, server, perUserToken)

	alice, bob := invocationAs(t, "app", "alice"), invocationAs(t, "app", "bob")
	var tools []tool.Tool
	for _, invCtx := range []agent.InvocationContext{alice, bob} {
		var err error
		if tools, err = ts.Tools(icontext.NewReadonlyContext(invCtx)); err != nil {
			t.Fatalf("Tools() failed: %v", err)
		}
	}

	if err := ts.(io.Closer).Close(); err != nil {
		t.Fatalf("Close() failed: %v", err)
	}
	for _, user := range []string{"alice", "bob"} {
		if n := server.deletedOf(user); n != 1 {
			t.Errorf("%s's sessions closed with DELETE: %d, want 1", user, n)
		}
	}
	for _, invCtx := range []agent.InvocationContext{alice, bob} {
		if _, err := ts.Tools(icontext.NewReadonlyContext(invCtx)); !errors.Is(err, mcp.ErrConnectionClosed) {
			t.Errorf("Tools() after Close() error = %v, want %v", err, mcp.ErrConnectionClosed)
		}
	}
	_, err := tools[0].(toolinternal.FunctionTool).Run(agent.NewToolContext(alice, "", nil, nil), map[string]any{})
	if !errors.Is(err, mcp.ErrConnectionClosed) {
		t.Errorf("Run() after Close() error = %v, want %v", err, mcp.ErrConnectionClosed)
	}
}

// TestAuthConcurrentCloseWaits checks that a second Close waits for the first
// to finish closing sessions, including a connection still being set up.
func TestAuthConcurrentCloseWaits(t *testing.T) {
	inner := newUserBoundServer(t)
	release := make(chan struct{})
	initializing := make(chan struct{})
	front := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.Header.Get("Authorization"), "token-of-slow") && r.Method == http.MethodPost {
			b, _ := io.ReadAll(r.Body)
			if strings.Contains(string(b), `"initialize"`) {
				close(initializing)
				<-release // hold the slow user's connection setup open
			}
			r.Body = io.NopCloser(strings.NewReader(string(b)))
		}
		inner.Config.Handler.ServeHTTP(w, r)
	}))
	t.Cleanup(front.Close)
	ts, err := mcptoolset.New(mcptoolset.Config{Endpoint: front.URL, Auth: perUserToken})
	if err != nil {
		t.Fatalf("mcptoolset.New() failed: %v", err)
	}
	closer := ts.(io.Closer)

	invCtx := invocationAs(t, "app", "slow")
	go func() { _, _ = ts.Tools(icontext.NewReadonlyContext(invCtx)) }()
	<-initializing

	first := make(chan struct{})
	go func() { _ = closer.Close(); close(first) }()
	time.Sleep(100 * time.Millisecond) // let the first Close start waiting
	second := make(chan struct{})
	go func() { _ = closer.Close(); close(second) }()

	select {
	case <-second:
		t.Error("second Close returned while the first was still closing sessions")
	case <-time.After(100 * time.Millisecond):
	}
	close(release)
	<-first
	<-second
}
