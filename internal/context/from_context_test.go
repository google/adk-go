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

// TestIdentityFromContextNoSession pins that an ADK context with no session
// yields (zero, false) instead of panicking — Value is a context.Context method
// and must never panic — across both the invocation and promoted common context
// Value paths.
func TestIdentityFromContextNoSession(t *testing.T) {
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}) // Session nil
	cases := []struct {
		name string
		ctx  context.Context
	}{
		{"invocation context", ic},
		{"promoted common context", agent.Promote(ic)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if id, ok := agent.IdentityFromContext(tc.ctx); ok {
				t.Errorf("IdentityFromContext() = %+v, ok = true; want zero, false", id)
			}
		})
	}
}
