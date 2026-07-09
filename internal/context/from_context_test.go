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
	"reflect"
	"testing"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
)

type wrapKey struct{}

// TestFromContextRecoversIdentity verifies that agent.FromContext recovers the
// ADK identity from a context that has been wrapped by non-ADK intermediaries
// (as jsonrpc2 / net/http do), across the base invocation context, a promoted
// common context, and a tool context.
func TestFromContextRecoversIdentity(t *testing.T) {
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app-1", UserID: "user-42"})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
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

			if _, ok := wrapped.(agent.ReadonlyContext); ok {
				t.Fatal("wrapped context unexpectedly type-asserts to ReadonlyContext")
			}

			rc, ok := agent.FromContext(wrapped)
			if !ok {
				t.Fatal("FromContext() ok = false, want true")
			}
			if got := rc.UserID(); got != "user-42" {
				t.Errorf("UserID() = %q, want %q", got, "user-42")
			}
			if got := rc.AppName(); got != "app-1" {
				t.Errorf("AppName() = %q, want %q", got, "app-1")
			}
			// The recovered context must stay read-only: it must not widen back
			// to a mutable Context/InvocationContext via a type assertion.
			if _, ok := rc.(agent.Context); ok {
				t.Error("recovered context widened to agent.Context, want read-only only")
			}
			if _, ok := rc.(agent.InvocationContext); ok {
				t.Error("recovered context widened to agent.InvocationContext, want read-only only")
			}
		})
	}
}

func TestFromContextAbsent(t *testing.T) {
	if _, ok := agent.FromContext(t.Context()); ok {
		t.Error("FromContext() ok = true for a plain context, want false")
	}
}

// TestFromContextNoReflectionWiden pins that the recovered read-only view does
// not expose the underlying mutable context through any exported struct field,
// i.e. it cannot be widened back to an agent.InvocationContext by ordinary
// reflection (reflect.Value.Interface panics on unexported fields).
func TestFromContextNoReflectionWiden(t *testing.T) {
	svc := session.InMemoryService()
	resp, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatalf("session Create() error = %v", err)
	}
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: resp.Session})
	rc, ok := agent.FromContext(agent.Promote(ic))
	if !ok {
		t.Fatal("FromContext() ok = false, want true")
	}

	v := reflect.ValueOf(rc)
	if v.Kind() == reflect.Pointer {
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		return
	}
	for i := range v.NumField() {
		f := v.Field(i)
		if !f.CanInterface() {
			continue // unexported: reflection cannot extract it
		}
		if _, ok := f.Interface().(agent.InvocationContext); ok {
			t.Errorf("field %q exposes a widenable agent.InvocationContext", v.Type().Field(i).Name)
		}
	}
}
