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
	"testing"
	"time"

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
