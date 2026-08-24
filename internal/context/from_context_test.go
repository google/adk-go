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
// [session.Session] implementation can legally take. Two of them used to panic
// inside Value — a struct value tripped reflect.Value.IsNil, and a typed-nil
// pointer passed an interface-nil check and then dereferenced — which lands
// inside an http.RoundTripper, where net/http does not recover. The two Value
// implementations must also agree: a promoted context and its parent answering
// the identity key differently is its own bug.
func TestIdentityFromContextSessionShapes(t *testing.T) {
	// A session whose own accessors panic is not covered: no nil test can save
	// that, and it panics anywhere else in ADK too.
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
		{name: "nil", session: nil},
		{name: "typed-nil pointer", session: (*ptrSession)(nil)},
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
			} {
				id, ok := agent.IdentityFromContext(ctx.ctx)
				if ok != tc.wantOK || id != tc.want {
					t.Errorf("IdentityFromContext(%s) = %+v, %v; want %+v, %v", ctx.name, id, ok, tc.want, tc.wantOK)
				}
			}
		})
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

func (s *ptrSession) ID() string      { return s.id }
func (s *ptrSession) AppName() string { return s.app }
func (s *ptrSession) UserID() string  { return s.user }
