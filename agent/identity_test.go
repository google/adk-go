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
	"time"

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

// TestCommonContextWithoutInvocation pins that a commonContext speaking for no
// invocation reports no identity rather than its parent's. The parent is a
// different call, so passing its user through would be the same fail-open every
// other arm of this procedure refuses. A nil parent on top of that must not panic
// — Value is a context.Context method and runs inside an http.RoundTripper.
func TestCommonContextWithoutInvocation(t *testing.T) {
	owner := &invocationContext{Context: t.Context(), session: &identityTestSession{}}
	c := &commonContext{Context: owner} // no invocationContext
	if got, ok := IdentityFromContext(c); ok {
		t.Errorf("IdentityFromContext() = %+v, true; want no identity, not the parent's", got)
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

// identityTestSession answers the three identity accessors, and
// panickingSession panics on the first one, the shape a broken third-party
// session takes.
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
// intended cost. Losing the process is not.
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
		// Every other type in this package that gained a nil guard. Each guard's
		// comment says it exists for a DIRECT call, where nothing above recovers,
		// so a direct call is what has to exercise it — going through
		// IdentityFromContext would prove nothing now that the entry point
		// recovers too. Guard count and pin count stay equal.
		{"typed-nil common context", (*commonContext)(nil)},
		{"typed-nil invocation context", (*invocationContext)(nil)},
		{"common context with no embedded context", &commonContext{}},
		{"invocation context with no embedded context", &invocationContext{}},
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

// TestNilDeltaKeepsTheContextUsable pins that a delta carrying nothing about the
// invocation leaves a context ADK built exactly as usable as before.
//
// It exists because skipping the call for everyone broke that: the tool and
// callback wrappers do real work for a nil delta — they forward to the
// commonContext they hold, which hands back that inner context — so keeping the
// wrapper left a context whose Session() is nil by design, and UserID() panicked.
func TestNilDeltaKeepsTheContextUsable(t *testing.T) {
	ic := &invocationContext{Context: t.Context(), session: matrixOwner("u"), agent: &agent{name: "a"}}
	path := "n@1"
	for _, tc := range []struct {
		name string
		ic   InvocationContext
	}{
		{"a tool context re-derived from a tool context", NewToolContext(NewToolContext(ic, "a", nil, nil), "b", nil, nil)},
		{"a callback context re-derived from a tool context", NewCallbackContext(NewToolContext(ic, "a", nil, nil), nil)},
		{"a plain invocation", Promote(ic)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("panicked after a nil delta: %v", p)
				}
			}()
			c := tc.ic.(Context).WithDelta(&CommonContextDelta{Path: &path})
			if got := c.UserID(); got != "u" {
				t.Errorf("UserID() = %q, want %q", got, "u")
			}
			if id, ok := IdentityFromContext(c); !ok || id.UserID != "u" {
				t.Errorf("IdentityFromContext() = %+v, %v; want %q", id, ok, "u")
			}
		})
	}
}

// panickingIntermediary is a wrapper between the transport and the ADK context
// whose Value panics — a jsonrpc2 or net/http layer, or anything else the caller
// does not control. Seeing past exactly this is what IdentityFromContext is for.
type panickingIntermediary struct{ context.Context }

func (panickingIntermediary) Value(any) any { panic("intermediary Value is not available") }

// TestIdentityFromContextContainsAPanickingIntermediary pins the containment at
// the entry point rather than one hop inside it.
//
// Every inner read is already recovered, and the matrix reaches the panicking
// shapes through Promote — so identityFrom's recover absorbs them and the
// direct-call route is never exercised. That route is the one an
// http.RoundTripper actually enters through, on the caller's goroutine, where
// net/http does not recover: a panic here is the process rather than the
// identity. The inner recovers cannot help, because the panic happens before the
// chain reaches one.
func TestIdentityFromContextContainsAPanickingIntermediary(t *testing.T) {
	ic := &invocationContext{Context: t.Context(), session: matrixOwner("u")}
	for _, tc := range []struct {
		name string
		ctx  context.Context
	}{
		{"an intermediary whose Value panics", panickingIntermediary{Context: ic}},
		// Nested, so the panic is not merely the outermost frame.
		{"one behind a plain wrapper", context.WithValue(panickingIntermediary{Context: ic}, wrapKey{}, "x")},
		// A nil context is a caller error rather than a hostile wrapper, but it
		// reaches the same bare ctx.Value and the same recover covers it.
		{"a nil context", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("panic escaped IdentityFromContext: %v", p)
				}
			}()
			if id, ok := IdentityFromContext(tc.ctx); ok || id != (Identity{}) {
				t.Errorf("IdentityFromContext() = %+v, %v; want the zero Identity and false: a "+
					"context that cannot answer must cost the identity, not the process", id, ok)
			}
		})
	}
}

// sessionlessForwarder is the shape agent writes as toolContextWrapper, written
// OUTSIDE the module instead: no session of its own, and WithICDelta forwarded to
// an inner ADK context that does have one.
type sessionlessForwarder struct {
	InvocationContext
	inner Context
}

func (w sessionlessForwarder) Deadline() (time.Time, bool) { return w.inner.Deadline() }
func (w sessionlessForwarder) Done() <-chan struct{}       { return w.inner.Done() }
func (w sessionlessForwarder) Err() error                  { return w.inner.Err() }
func (w sessionlessForwarder) Value(k any) any             { return w.inner.Value(k) }
func (w sessionlessForwarder) Session() session.Session    { return nil }
func (w sessionlessForwarder) WithICDelta(d *InvocationContextDelta) InvocationContext {
	return w.inner.WithICDelta(d)
}

// TestEmptyDeltaCostsASessionlessForwarderItsUnwrap pins what the empty-delta
// shortcut costs, so the trade is asserted rather than only described.
//
// The shortcut keeps an out-of-module invocation instead of asking it, which is
// what stops a decorator being dropped onto the enclosing user. A forwarding view
// pays for that: it is kept too, so the session it does not have is the one the
// caller gets. withICDelta cannot tell the two apart — both are "not ours, session
// unreadable" before the call, and diverge only after it — so this is accepted
// rather than fixed, and an out-of-module InvocationContext must carry its own
// session.
//
// Asserted because it is a real change against forwarding the delta through, and
// an accepted cost that nothing pins is indistinguishable from one nobody noticed.
func TestEmptyDeltaCostsASessionlessForwarderItsUnwrap(t *testing.T) {
	enclosing := &invocationContext{Context: t.Context(), session: matrixOwner("enclosing")}
	inner := &invocationContext{Context: enclosing, session: matrixOwner("u")}
	w := sessionlessForwarder{inner: Promote(inner)}

	// A non-zero delta reaches WithICDelta, which forwards, so the user survives.
	branch := "br"
	withDelta := PromoteWithDelta(InvocationContext(w),
		&CommonContextDelta{InvocationContextDelta: &InvocationContextDelta{Branch: &branch}})
	if id, ok := IdentityFromContext(withDelta); !ok || id.UserID != "u" {
		t.Errorf("non-zero delta: IdentityFromContext() = %+v, %v; want %q — the forward should "+
			"still unwrap to the inner invocation", id, ok, "u")
	}

	// An empty one does not, and the cost lands here. Never the enclosing user:
	// reporting a live user who made no such call is the one outcome that is not
	// an acceptable price.
	empty := PromoteWithDelta(InvocationContext(w), &CommonContextDelta{})
	id, ok := IdentityFromContext(empty)
	if ok {
		t.Errorf("empty delta: IdentityFromContext() = %+v, %v; want no identity — the wrapper "+
			"is kept and reports no session of its own", id, ok)
	}
	if id.UserID == "enclosing" {
		t.Fatalf("empty delta reported the ENCLOSING user, which is the outcome the whole "+
			"procedure exists to prevent: %+v", id)
	}
}
