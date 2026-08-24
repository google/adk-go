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

package context_test

import (
	"context"
	"testing"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

type wrapKey struct{}

// TestIdentityFromContextRecoversIdentity verifies that agent.IdentityFromContext
// recovers the ADK identity from a context that has been wrapped by non-ADK
// intermediaries (as jsonrpc2 / net/http do), across the base invocation context,
// a promoted common context, and a tool context.
func TestIdentityFromContextRecoversIdentity(t *testing.T) {
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app-1", UserID: "user-42"})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	sessionID := resp.Session.ID()
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: resp.Session})

	cases := []struct {
		name string
		ctx  context.Context
	}{
		{"invocation context", ic},
		{"promoted common context", agent.Promote(ic)},
		{"tool context", agent.NewToolContext(ic, "fc-1", nil, nil)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Wrap in non-ADK children so a plain type-assert is erased but the
			// Value lookup still resolves up the chain.
			wrapped := context.WithValue(tc.ctx, wrapKey{}, "x")
			wrapped, cancel := context.WithCancel(wrapped)
			defer cancel()

			id, ok := agent.IdentityFromContext(wrapped)
			if !ok {
				t.Fatal("IdentityFromContext() ok = false, want true")
			}
			want := agent.Identity{UserID: "user-42", AppName: "app-1", SessionID: sessionID}
			if id != want {
				t.Errorf("IdentityFromContext() = %+v, want %+v", id, want)
			}
		})
	}
}

func TestIdentityFromContextAbsent(t *testing.T) {
	if _, ok := agent.IdentityFromContext(t.Context()); ok {
		t.Error("IdentityFromContext() ok = true for a plain context, want false")
	}
}

// TestIdentityFromContextSessionShapes covers the session shapes a
// [session.Session] implementation can legally take. Several of them used to
// panic inside Value — a struct value tripped reflect.Value.IsNil, a typed-nil
// pointer passed an interface-nil check and then dereferenced, and a session
// wrapping a nil one (llmagent.newWrappedSession's shape for a nil original)
// panicked in the accessor. Value runs inside an http.RoundTripper, where
// net/http does not recover. Every context implementation must also agree: two
// of them answering the identity key differently is its own bug.
func TestIdentityFromContextSessionShapes(t *testing.T) {
	cases := []struct {
		name    string
		session session.Session
		want    agent.Identity
		wantOK  bool
	}{
		{
			name:    "pointer",
			session: &ptrSession{id: "sid-1", app: "app-1", user: "user-42"},
			want:    agent.Identity{UserID: "user-42", AppName: "app-1", SessionID: "sid-1"},
			wantOK:  true,
		},
		{
			name:    "struct value",
			session: valueSession{id: "sid-1", app: "app-1", user: "user-42"},
			want:    agent.Identity{UserID: "user-42", AppName: "app-1", SessionID: "sid-1"},
			wantOK:  true,
		},
		{
			// A typed-nil pointer whose accessors do not dereference is usable.
			name:    "typed-nil pointer with safe accessors",
			session: (*safeNilSession)(nil),
			want:    agent.Identity{UserID: "user-nil", AppName: "app-nil", SessionID: "sid-nil"},
			wantOK:  true,
		},
		{name: "nil", session: nil},
		{name: "typed-nil pointer", session: (*ptrSession)(nil)},
		{name: "wrapper over a nil session", session: &wrapperSession{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: tc.session})
			for _, ctx := range []struct {
				name string
				ctx  context.Context
			}{
				{"invocation context", ic},
				{"promoted common context", agent.Promote(ic)},
				{"tool context", agent.NewToolContext(ic, "fc-1", nil, nil)},
			} {
				id, ok := agent.IdentityFromContext(ctx.ctx)
				if ok != tc.wantOK || id != tc.want {
					t.Errorf("IdentityFromContext(%s) = %+v, %v; want %+v, %v", ctx.name, id, ok, tc.want, tc.wantOK)
				}
			}
		})
	}
}

// TestIdentityAfterWithContext pins the promoted context's own identity branch.
// WithContext replaces the embedded parent with a non-ADK context while keeping
// the invocation — the one shape where delegating to the parent cannot recover
// the identity, and the reason the branch exists. agent.go does exactly this
// around a tracing span.
func TestIdentityAfterWithContext(t *testing.T) {
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app-1", UserID: "user-42"})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: resp.Session})

	detached := agent.Promote(ic).WithContext(context.Background())
	id, ok := agent.IdentityFromContext(detached)
	want := agent.Identity{UserID: "user-42", AppName: "app-1", SessionID: resp.Session.ID()}
	if !ok || id != want {
		t.Errorf("IdentityFromContext(WithContext) = %+v, %v; want %+v, true", id, ok, want)
	}
}

// TestValueWithNilEmbeddedContext pins the nil-parent guard:
// NewCleanToolContextTestOnly builds a context with no embedded parent, so
// without it every non-identity key is a nil-interface method call.
func TestValueWithNilEmbeddedContext(t *testing.T) {
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	clean, err := agent.NewCleanToolContextTestOnly(agent.Promote(ic), "fc-1", nil, nil)
	if err != nil {
		t.Fatalf("NewCleanToolContextTestOnly() error = %v", err)
	}
	if got := clean.Value(wrapKey{}); got != nil {
		t.Errorf("Value(wrapKey{}) = %v, want nil", got)
	}
}

// TestIdentityDoesNotInheritEnclosingInvocation pins that identity resolution
// fails closed. A nested invocation with no session of its own must not report
// the enclosing invocation's user: that user's credential would then be minted
// for a call they never made.
func TestIdentityDoesNotInheritEnclosingInvocation(t *testing.T) {
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app-1", UserID: "alice"})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	outer := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: resp.Session})
	if id, ok := agent.IdentityFromContext(outer); !ok || id.UserID != "alice" {
		t.Fatalf("outer IdentityFromContext() = %+v, %v; want alice", id, ok)
	}

	for _, tc := range []struct {
		name string
		ctx  context.Context
	}{
		{"nested invocation", icontext.NewInvocationContext(outer, icontext.InvocationContextParams{})},
		{"nested and promoted", agent.Promote(icontext.NewInvocationContext(outer, icontext.InvocationContextParams{}))},
	} {
		if id, ok := agent.IdentityFromContext(tc.ctx); ok {
			t.Errorf("%s IdentityFromContext() = %+v, true; want no identity, not the enclosing user", tc.name, id)
		}
	}
}

// TestValueDelegatesUnknownKeys pins that a session shape that stops the
// identity lookup does not stop every other key: Value is on the hot path for
// net/http, tracing and logging keys that have nothing to do with the session.
func TestValueDelegatesUnknownKeys(t *testing.T) {
	parent := context.WithValue(t.Context(), wrapKey{}, "x")
	ic := icontext.NewInvocationContext(parent, icontext.InvocationContextParams{Session: (*ptrSession)(nil)})
	for _, tc := range []struct {
		name string
		ctx  context.Context
	}{
		{"invocation context", ic},
		{"promoted common context", agent.Promote(ic)},
	} {
		if got := tc.ctx.Value(wrapKey{}); got != "x" {
			t.Errorf("%s Value(wrapKey{}) = %v, want %q", tc.name, got, "x")
		}
	}
}

// valueSession and ptrSession embed a nil session.Session for the accessors the
// identity path never reaches, and read their own fields for the ones it does —
// so a typed-nil *ptrSession panics on use, as a real session would.
type valueSession struct {
	session.Session
	id, app, user string
}

func (s valueSession) ID() string      { return s.id }
func (s valueSession) AppName() string { return s.app }
func (s valueSession) UserID() string  { return s.user }

type ptrSession struct {
	session.Session
	id, app, user string
}

// wrapperSession is the shape llmagent.newWrappedSession produces for a nil
// original: non-nil, but every identity accessor is promoted from the nil
// embedded interface and panics on the first call.
type wrapperSession struct{ session.Session }

// safeNilSession answers without touching its receiver, so a typed-nil one is
// still a working session.
type safeNilSession struct{ session.Session }

func (*safeNilSession) ID() string      { return "sid-nil" }
func (*safeNilSession) AppName() string { return "app-nil" }
func (*safeNilSession) UserID() string  { return "user-nil" }

func (s *ptrSession) ID() string      { return s.id }
func (s *ptrSession) AppName() string { return s.app }
func (s *ptrSession) UserID() string  { return s.user }
