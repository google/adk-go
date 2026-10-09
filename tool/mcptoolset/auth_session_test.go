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
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

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
// It records the MCP session each user's requests arrived on.
type userBoundServer struct {
	*httptest.Server

	mu       sync.Mutex
	sessions map[string]map[string]bool // user -> MCP session IDs used
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

	s := &userBoundServer{sessions: map[string]map[string]bool{}}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		user := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer token-of-")
		if id := r.Header.Get("Mcp-Session-Id"); id != "" {
			s.mu.Lock()
			if s.sessions[user] == nil {
				s.sessions[user] = map[string]bool{}
			}
			s.sessions[user][id] = true
			s.mu.Unlock()
		}
		handler.ServeHTTP(w, r)
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *userBoundServer) sessionsOf(user string) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.sessions[user])
}

// perUserToken mints "token-of-<user>" for the acting user.
var perUserToken = auth.ProviderFunc(func(ctx context.Context) (auth.Credential, error) {
	id, ok := agent.IdentityFromContext(ctx)
	if !ok || id.UserID == "" {
		return nil, errors.New("no acting user")
	}
	return auth.BearerCredential{Token: "token-of-" + id.UserID}, nil
})

// TestAuthGivesEachUserItsOwnSession checks that with Config.Auth, users
// sharing one toolset each get an MCP session of their own. A server that binds
// sessions to users rejects a request on another user's session, so a single
// shared session works only for whichever user opened it.
func TestAuthGivesEachUserItsOwnSession(t *testing.T) {
	server := newUserBoundServer(t)
	ts, err := mcptoolset.New(mcptoolset.Config{Endpoint: server.URL, Auth: perUserToken})
	if err != nil {
		t.Fatalf("mcptoolset.New() failed: %v", err)
	}
	t.Cleanup(func() { _ = ts.(interface{ Close() error }).Close() })

	sessionService := session.InMemoryService()
	callAs := func(user string) {
		t.Helper()
		created, err := sessionService.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: user})
		if err != nil {
			t.Fatalf("Create session for %q: %v", user, err)
		}
		invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: created.Session})

		tools, err := ts.Tools(icontext.NewReadonlyContext(invCtx))
		if err != nil {
			t.Fatalf("Tools() as %q: %v", user, err)
		}
		if len(tools) != 1 {
			t.Fatalf("Tools() as %q returned %d tools, want 1", user, len(tools))
		}
		got, err := tools[0].(toolinternal.FunctionTool).Run(agent.NewToolContext(invCtx, "", nil, nil), map[string]any{})
		if err != nil {
			t.Fatalf("Run() as %q: %v", user, err)
		}
		if got["output"] != user {
			t.Errorf("Run() as %q: server saw %v, want %q", user, got["output"], user)
		}
	}

	callAs("alice")
	callAs("bob")
	callAs("alice")
	callAs("bob")

	// Each user stays on one session: bob's calls did not tear down alice's.
	for _, user := range []string{"alice", "bob"} {
		if n := server.sessionsOf(user); n != 1 {
			t.Errorf("%s used %d MCP sessions, want 1", user, n)
		}
	}
}

// TestAuthWithoutActingUserSharesOneSession checks that callers with no ADK
// identity, such as a service-account credential used outside an invocation,
// keep sharing one session as before.
func TestAuthWithoutActingUserSharesOneSession(t *testing.T) {
	server := newUserBoundServer(t)
	ts, err := mcptoolset.New(mcptoolset.Config{Endpoint: server.URL, Auth: auth.StaticToken("token-of-svc")})
	if err != nil {
		t.Fatalf("mcptoolset.New() failed: %v", err)
	}
	t.Cleanup(func() { _ = ts.(interface{ Close() error }).Close() })

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

// TestAuthCloseClosesEveryUsersSession checks that Close ends the sessions of
// all users and that later calls fail.
func TestAuthCloseClosesEveryUsersSession(t *testing.T) {
	server := newUserBoundServer(t)
	ts, err := mcptoolset.New(mcptoolset.Config{Endpoint: server.URL, Auth: perUserToken})
	if err != nil {
		t.Fatalf("mcptoolset.New() failed: %v", err)
	}

	sessionService := session.InMemoryService()
	ctxFor := func(user string) agent.ReadonlyContext {
		created, err := sessionService.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: user})
		if err != nil {
			t.Fatalf("Create session for %q: %v", user, err)
		}
		return icontext.NewReadonlyContext(icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: created.Session}))
	}
	alice, bob := ctxFor("alice"), ctxFor("bob")
	var tools []tool.Tool
	for _, ctx := range []agent.ReadonlyContext{alice, bob} {
		if tools, err = ts.Tools(ctx); err != nil {
			t.Fatalf("Tools() failed: %v", err)
		}
	}

	if err := ts.(interface{ Close() error }).Close(); err != nil {
		t.Fatalf("Close() failed: %v", err)
	}
	for _, ctx := range []agent.ReadonlyContext{alice, bob} {
		if _, err := ts.Tools(ctx); !errors.Is(err, mcp.ErrConnectionClosed) {
			t.Errorf("Tools() after Close() error = %v, want %v", err, mcp.ErrConnectionClosed)
		}
	}
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	_, err = tools[0].(toolinternal.FunctionTool).Run(agent.NewToolContext(invCtx, "", nil, nil), map[string]any{})
	if !errors.Is(err, mcp.ErrConnectionClosed) {
		t.Errorf("Run() after Close() error = %v, want %v", err, mcp.ErrConnectionClosed)
	}
}
