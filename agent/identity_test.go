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

package agent

import (
	"context"
	"testing"

	"google.golang.org/adk/v2/internal/adkcontext"
	"google.golang.org/adk/v2/session"
)

// TestInvocationContextIdentityIsOwned pins the guard on the third invocation
// implementation, the one in this package: it owns its session, so no session
// means no identity — never the enclosing invocation's, whose user made no such
// call and whose credential would otherwise be minted for it.
func TestInvocationContextIdentityIsOwned(t *testing.T) {
	outer := &invocationContext{Context: t.Context(), session: &identityTestSession{}}
	if got, ok := IdentityFromContext(outer); !ok || got.UserID != "alice" {
		t.Fatalf("outer IdentityFromContext() = %+v, %v; want alice", got, ok)
	}

	nested := &invocationContext{Context: outer} // no session of its own
	if got, ok := IdentityFromContext(nested); ok {
		t.Errorf("nested IdentityFromContext() = %+v, true; want no identity", got)
	}
	if got := nested.Value(wrapKey{}); got != nil {
		t.Errorf("nested Value(wrapKey{}) = %v, want nil (unrelated keys still delegate)", got)
	}
	// A nil embedded parent must not panic either: Value is a context.Context
	// method and runs inside an http.RoundTripper.
	if got := (&invocationContext{}).Value(wrapKey{}); got != nil {
		t.Errorf("Value(wrapKey{}) with no parent = %v, want nil", got)
	}
}

// TestCommonContextWithoutInvocation pins the guard for a commonContext that
// speaks for no invocation: it has nothing to answer the identity key with, so it
// delegates, and a nil parent on top of that must not panic — Value is a
// context.Context method and runs inside an http.RoundTripper.
func TestCommonContextWithoutInvocation(t *testing.T) {
	owner := &invocationContext{Context: t.Context(), session: &identityTestSession{}}
	c := &commonContext{Context: owner} // no invocationContext
	if got, ok := IdentityFromContext(c); !ok || got.UserID != "alice" {
		t.Errorf("IdentityFromContext() = %+v, %v; want the parent's identity", got, ok)
	}
	if got := (&commonContext{}).Value(adkcontext.IdentityKey); got != nil {
		t.Errorf("Value(IdentityKey) with no invocation and no parent = %v, want nil", got)
	}
	if got := (&commonContext{}).Value(wrapKey{}); got != nil {
		t.Errorf("Value(wrapKey{}) with no invocation and no parent = %v, want nil", got)
	}
}

// TestReadIdentityRecoversPanickingAccessor pins that a session accessor which
// panics costs the identity and not the process.
func TestReadIdentityRecoversPanickingAccessor(t *testing.T) {
	c := &invocationContext{Context: t.Context(), session: panickingSession{}}
	if got := c.Value(adkcontext.IdentityKey); got != nil {
		t.Errorf("Value(IdentityKey) = %v, want nil for a session that panics", got)
	}
}

// identityTestSession answers the three identity accessors; panickingSession
// panics on the first one, the shape a broken third-party session takes.
type identityTestSession struct{ session.Session }

func (identityTestSession) ID() string      { return "sid-1" }
func (identityTestSession) AppName() string { return "app-1" }
func (identityTestSession) UserID() string  { return "alice" }

type panickingSession struct{ session.Session }

func (panickingSession) UserID() string { panic("UserID is not available") }

type wrapKey struct{}

var _ context.Context = (*invocationContext)(nil)
