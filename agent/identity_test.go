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
	"bytes"
	"context"
	"log"
	"strings"
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

// TestIdentityFromPermissiveInvocation pins that an invocation answering every
// key with something that is not an [Identity] does not swallow the fallback: a
// decorator or test double that returns a placeholder for any key would
// otherwise cost the identity on every outbound request.
func TestIdentityFromPermissiveInvocation(t *testing.T) {
	owner := &invocationContext{Context: t.Context(), session: &identityTestSession{}}
	c := &commonContext{Context: t.Context(), invocationContext: permissiveInvocation{InvocationContext: owner}}
	got, ok := IdentityFromContext(c)
	if !ok || got.UserID != "alice" {
		t.Errorf("IdentityFromContext() = %+v, %v; want the session read to be reached", got, ok)
	}
}

// TestIdentityFromDecoratedInvocation pins that an invocation reports its OWN
// user, not the one it inherited. An InvocationContext written outside this
// module embeds the context it was derived from, to inherit cancellation, and
// cannot override a key it cannot name — so its Value answers with the enclosing
// invocation's identity. Reading its session first is what stops one user's
// credential being minted for another's call.
func TestIdentityFromDecoratedInvocation(t *testing.T) {
	enclosing := &invocationContext{Context: t.Context(), session: &identityTestSession{}} // alice
	decorated := decoratedInvocation{
		InvocationContext: enclosing,
		own:               &otherUserSession{},
	}
	for _, tc := range []struct {
		name string
		ctx  context.Context
	}{
		{"promoted", Promote(decorated)},
		{"tool context", NewToolContext(decorated, "fc-1", nil, nil)},
		{"callback context", NewCallbackContext(decorated, nil)},
	} {
		id, ok := IdentityFromContext(tc.ctx)
		if !ok || id.UserID != "bob" {
			t.Errorf("%s IdentityFromContext() = %+v, %v; want bob, the decorated invocation's own user", tc.name, id, ok)
		}
	}
}

// decoratedInvocation is how an invocation is wrapped outside this module: embed
// the enclosing one, override the accessors that differ.
type decoratedInvocation struct {
	InvocationContext
	own session.Session
}

func (d decoratedInvocation) Session() session.Session { return d.own }

type otherUserSession struct{ session.Session }

func (otherUserSession) ID() string      { return "sid-2" }
func (otherUserSession) AppName() string { return "app-1" }
func (otherUserSession) UserID() string  { return "bob" }

// permissiveInvocation answers every key, as a decorator or a test double might.
type permissiveInvocation struct{ InvocationContext }

func (permissiveInvocation) Value(any) any { return "something that is not an Identity" }

// TestIdentityFromBrokenWrapper pins the recover inside identityFrom. The tool and callback wrappers are this package's own types, so
// the identity procedure trusts them to answer — but a hand-built one can hold a
// nil inner context, and Value runs inside http.RoundTripper on the caller's
// goroutine, where net/http does not recover. Losing the identity is the
// intended cost; losing the process is not.
func TestIdentityFromBrokenWrapper(t *testing.T) {
	for _, tc := range []struct {
		name string
		ic   InvocationContext
	}{
		{"tool context wrapper with no inner context", &toolContextWrapper{}},
		{"callback context wrapper with no inner context", &callbackContextWrapper{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("IdentityFromContext panicked: %v", p)
				}
			}()
			if id, ok := IdentityFromContext(Promote(tc.ic)); ok {
				t.Errorf("IdentityFromContext() = %+v, true; want no identity", id)
			}
		})
	}
}

// TestIdentityThroughWrapperDoesNotLogPerLookup pins that resolving an identity
// through a tool or callback context does not call the wrapper's Session().
// auth.Transport resolves a credential per outbound request, and those wrappers
// log on every Session() call, so reading the session to find the identity would
// put one log line on every authenticated request.
func TestIdentityThroughWrapperDoesNotLogPerLookup(t *testing.T) {
	var buf bytes.Buffer
	out := log.Writer()
	flags := log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0)
	t.Cleanup(func() { log.SetOutput(out); log.SetFlags(flags) })

	ic := &invocationContext{Context: t.Context(), session: matrixOwner("u")}
	// A tool context re-derived from a tool context: the outer one's invocation
	// is the inner wrapper, which is the shape that reads a wrapper's session.
	toolCtx := NewToolContext(NewToolContext(ic, "a", nil, nil), "b", nil, nil)

	if _, ok := IdentityFromContext(toolCtx); !ok {
		t.Fatal("IdentityFromContext() ok = false, want the invocation's identity")
	}
	if got := buf.String(); strings.Contains(got, "Session()") {
		t.Errorf("resolving the identity logged %q; it must not read a wrapper's session", got)
	}
}

// TestIdentityFromNilContexts pins that a typed-nil receiver costs the identity
// and not the process. The dispatch trusts a context of ours to answer for
// itself, and a typed-nil pointer satisfies that interface as readily as a live
// one — so the guard has to be on the answering side. Value runs inside
// http.RoundTripper on the caller's goroutine, where net/http does not recover.
func TestIdentityFromNilContexts(t *testing.T) {
	for _, tc := range []struct {
		name string
		ctx  context.Context
	}{
		{"a typed-nil commonContext", (*commonContext)(nil)},
		{"one wrapping a typed-nil commonContext", &commonContext{invocationContext: (*commonContext)(nil)}},
		{"one wrapping a typed-nil invocationContext", &commonContext{invocationContext: (*invocationContext)(nil)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("Value panicked: %v", p)
				}
			}()
			if id, ok := IdentityFromContext(tc.ctx); ok {
				t.Errorf("IdentityFromContext() = %+v, true; want no identity", id)
			}
		})
	}
}

// TestIdentityFromUserlessSession pins the case the decision matrix cannot
// express: a session that reads fine and simply carries no user. That must
// report ok with an empty UserID, not "no identity" — the two reach the
// credential path as different errors, and only the second means "this is not an
// agent invocation".
func TestIdentityFromUserlessSession(t *testing.T) {
	ic := &invocationContext{Context: t.Context(), session: &matrixSession{id: "sid", app: "app"}}
	for _, tc := range []struct {
		name string
		ctx  context.Context
	}{
		{"the invocation itself", ic},
		{"Promote", Promote(ic)},
		{"NewToolContext", NewToolContext(ic, "fc", nil, nil)},
		{"NewCallbackContext", NewCallbackContext(ic, nil)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			id, ok := IdentityFromContext(tc.ctx)
			if !ok {
				t.Fatal("IdentityFromContext() ok = false; a readable session with no user still has an identity")
			}
			if want := (Identity{AppName: "app", SessionID: "sid"}); id != want {
				t.Errorf("IdentityFromContext() = %+v, want %+v", id, want)
			}
		})
	}
}

// nilDeltaInvocation is a partial implementation of the kind an embedder writes:
// it returns nil from WithICDelta rather than a derived invocation. ADK's own
// ContextMock returns nil from WithContext, WithBranch and WithAgentContext, so
// the shape is not exotic.
type nilDeltaInvocation struct {
	InvocationContext
	own session.Session
}

func (d nilDeltaInvocation) Session() session.Session                            { return d.own }
func (nilDeltaInvocation) WithICDelta(*InvocationContextDelta) InvocationContext { return nil }

// TestDeltaOnInvocationThatReturnsNil pins that losing the invocation to a delta
// does not fail open. A commonContext with no invocation asks its parent, and the
// parent here is the enclosing call — so adopting the nil would report a user who
// made no such call, with ok true.
func TestDeltaOnInvocationThatReturnsNil(t *testing.T) {
	enclosing := &invocationContext{Context: t.Context(), session: matrixOwner("enclosing")}
	ic := nilDeltaInvocation{InvocationContext: enclosing, own: matrixOwner("u")}
	branch := "br"
	for _, tc := range []struct {
		name string
		ctx  context.Context
	}{
		{"PromoteWithDelta", PromoteWithDelta(ic, &CommonContextDelta{InvocationContextDelta: &InvocationContextDelta{Branch: &branch}})},
		{"WithICDelta", Promote(ic).WithICDelta(&InvocationContextDelta{Branch: &branch})},
	} {
		t.Run(tc.name, func(t *testing.T) {
			id, ok := IdentityFromContext(tc.ctx)
			if !ok || id.UserID != "u" {
				t.Errorf("IdentityFromContext() = %+v, %v; want the invocation's own user %q", id, ok, "u")
			}
		})
	}
}

// TestIdentityThroughNestedSessionlessContexts pins the marker on commonContext
// itself. A promoted tool context is a commonContext whose own invocation is a
// session-less wrapper, so if it could not answer for itself the outer context
// would read that wrapper's nil session and report no user at all.
func TestIdentityThroughNestedSessionlessContexts(t *testing.T) {
	ic := &invocationContext{Context: t.Context(), session: matrixOwner("u")}
	promotedTool := Promote(NewToolContext(ic, "fc", nil, nil))
	for _, tc := range []struct {
		name string
		ctx  context.Context
	}{
		{"callback context over a promoted tool context", NewCallbackContext(promotedTool, nil)},
		{"tool context over a promoted tool context", NewToolContext(promotedTool, "fc2", nil, nil)},
		{"context over a promoted tool context", NewContext(promotedTool)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			id, ok := IdentityFromContext(tc.ctx)
			if !ok || id.UserID != "u" {
				t.Errorf("IdentityFromContext() = %+v, %v; want %q", id, ok, "u")
			}
		})
	}
}

// TestWrapperValueFailsClosed pins that the wrappers' own Value survives a
// hand-built receiver. Reached through the identity procedure these are already
// recovered, but a wrapper is a context.Context and anything may call Value on
// it directly, where nothing recovers.
func TestWrapperValueFailsClosed(t *testing.T) {
	type probeKey struct{}
	for _, tc := range []struct {
		name string
		ctx  context.Context
	}{
		{"tool wrapper with no inner context", &toolContextWrapper{}},
		{"callback wrapper with no inner context", &callbackContextWrapper{}},
		{"typed-nil tool wrapper", (*toolContextWrapper)(nil)},
		{"typed-nil callback wrapper", (*callbackContextWrapper)(nil)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("Value panicked: %v", p)
				}
			}()
			if v := tc.ctx.Value(probeKey{}); v != nil {
				t.Errorf("Value(probeKey{}) = %v, want nil", v)
			}
			if _, ok := IdentityFromContext(tc.ctx); ok {
				t.Error("IdentityFromContext() ok = true, want no identity")
			}
		})
	}
}

// TestDeltaReachesTheInvocation pins that a delta actually lands on the
// invocation, for a decorated one as much as an ADK one.
//
// It exists because an attempt to stop a delta dropping an out-of-module
// decorator did stop it — by discarding the delta wholesale, so agent.Run read
// Agent, Branch and IsolationScope from the enclosing invocation and nil-panicked
// where there was no enclosing agent. The identity tests could not see that: they
// assert who the context speaks for, never what the delta was for. Any future
// attempt on that problem has to keep this green.
func TestDeltaReachesTheInvocation(t *testing.T) {
	enclosing := &invocationContext{
		Context: t.Context(),
		session: matrixOwner("enclosing"),
		agent:   &agent{name: "parent"},
	}
	var child Agent = &agent{name: "child"}
	branch := "child-branch"
	delta := func() *CommonContextDelta {
		return &CommonContextDelta{InvocationContextDelta: &InvocationContextDelta{Agent: &child, Branch: &branch}}
	}
	for _, tc := range []struct {
		name string
		ic   InvocationContext
	}{
		{"an ADK invocation", &invocationContext{Context: enclosing, session: matrixOwner("u"), agent: &agent{name: "parent"}}},
		{"one decorated outside the module", decoratedInvocationValue{InvocationContext: enclosing, own: matrixOwner("u")}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := PromoteWithDelta(tc.ic, delta())
			if got := ctx.(InvocationContext).Agent(); got == nil || got.Name() != "child" {
				t.Errorf("Agent() = %v, want the agent the delta named", got)
			}
			if got := ctx.Branch(); got != branch {
				t.Errorf("Branch() = %q, want %q", got, branch)
			}
		})
	}
}
