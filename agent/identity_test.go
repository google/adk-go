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
	"iter"
	"log"
	"reflect"
	"slices"
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
	withDelta := PromoteWithDelta(InvocationContext(w),
		&CommonContextDelta{InvocationContextDelta: &InvocationContextDelta{Branch: new("br")}})
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

type appObserveMethodKey struct{}

func (appObserveMethodKey) ObserveADKContext(any) bool {
	panic("application context key method must not be called by ADK Value")
}

func TestLineageWalksWithoutExposingContextOrInterceptingAppKeys(t *testing.T) {
	base := context.WithValue(t.Context(), appObserveMethodKey{}, "stored-app-value")
	outer := &invocationContext{Context: base, session: &identityTestSession{}}
	promoted := Promote(outer)
	detached := context.WithoutCancel(promoted)
	child := Promote(outer).WithContext(detached)

	// Application keys defining ObserveADKContext(any) bool must pass through untouched.
	for _, tc := range []struct {
		name string
		ctx  context.Context
	}{
		{"invocationContext", outer},
		{"commonContext", promoted},
		{"childWithDetachedParent", child},
	} {
		if got := tc.ctx.Value(appObserveMethodKey{}); got != "stored-app-value" {
			t.Fatalf("%s.Value(appObserveMethodKey{}) = %v, want %q", tc.name, got, "stored-app-value")
		}
	}

	tokens := adkcontext.WalkLineage(child)
	if len(tokens) != 3 {
		t.Fatalf("len(tokens) = %d, want 3 (child, promoted, outer)", len(tokens))
	}
	if tokens[0] == tokens[1] || tokens[1] == tokens[2] || tokens[0] == tokens[2] {
		t.Fatalf("expected distinct tokens along lineage, got %#v", tokens)
	}
	for i, tok := range tokens {
		if _, ok := tok.(context.Context); ok {
			t.Fatalf("token[%d] implements context.Context (%T); must remain opaque", i, tok)
		}
		if _, ok := tok.(InvocationContext); ok {
			t.Fatalf("token[%d] implements InvocationContext (%T); must remain opaque", i, tok)
		}
		rv := reflect.ValueOf(tok)
		if rv.NumMethod() != 0 || rv.MethodByName("Value").IsValid() {
			t.Fatalf("token[%d] exposes methods (NumMethod=%d); weak.Pointer.Value must not be reachable", i, rv.NumMethod())
		}
		if rv.Kind() != reflect.Struct || rv.NumField() != 1 || rv.Field(0).CanInterface() {
			t.Fatalf("token[%d] must be a single-field unexported struct (Kind=%v, NumField=%d)", i, rv.Kind(), rv.NumField())
		}
		if _, ok := adkcontext.Recovered(func() any {
			return rv.Field(0).Elem().MethodByName("Value").Call(nil)[0].Interface()
		}); ok {
			t.Fatalf("token[%d] allowed reflection recovery of underlying context", i)
		}
	}

	if got := promoted.(interface{ ADKContextToken() any }).ADKContextToken(); got != tokens[1] {
		t.Fatalf("promoted.ADKContextToken() = %#v, want %#v", got, tokens[1])
	}

	var nilInv *invocationContext
	var nilCommon *commonContext
	if nilInv.Value(adkcontext.LineageKey) != nil || nilCommon.Value(adkcontext.LineageKey) != nil {
		t.Fatal("typed-nil Value returned non-nil")
	}
	if nilInv.ADKContextToken() != nil || nilCommon.ADKContextToken() != nil {
		t.Fatal("typed-nil ADKContextToken returned non-nil")
	}
	panickingMid := Promote(outer).WithContext(panickingIntermediary{Context: promoted})
	if got := adkcontext.WalkLineage(panickingMid); len(got) != 1 || got[0] != panickingMid.(interface{ ADKContextToken() any }).ADKContextToken() {
		t.Fatalf("WalkLineage across panicking intermediary = %#v, want single outer token", got)
	}

	pathA := "wf/left"
	pmVal := map[string]string{"leaf": "worker"}
	pmCtx := adkcontext.WithParentMap(context.Background(), pmVal)
	withPath := Promote(outer).WithAgentContext(pmCtx).WithDelta(&CommonContextDelta{Path: &pathA})
	cbCtx := NewCallbackContext(withPath, nil)
	pathReader := Promote(nil).(interface{ ADKContextPath(context.Context) string })
	pmReader := Promote(nil).(interface{ ADKParentMap(context.Context) any })
	if got := pathReader.ADKContextPath(cbCtx); got != "wf/left" {
		t.Fatalf("ADKContextPath(cbCtx) = %q, want wf/left", got)
	}
	if got, ok := pmReader.ADKParentMap(cbCtx).(map[string]string); !ok || got["leaf"] != "worker" {
		t.Fatalf("ADKParentMap(cbCtx) = %#v, want leaf->worker", got)
	}
}

func TestPromoteADKParentMap_ReturnsSnapshotCopy(t *testing.T) {
	sub, err := New(Config{Name: "sub_agent"})
	if err != nil {
		t.Fatalf("New(sub): %v", err)
	}
	root, err := New(Config{Name: "root_agent", SubAgents: []Agent{sub}})
	if err != nil {
		t.Fatalf("New(root): %v", err)
	}
	other, err := New(Config{Name: "other_agent"})
	if err != nil {
		t.Fatalf("New(other): %v", err)
	}

	type runnerParentMap map[string]Agent
	src := runnerParentMap{"sub_agent": root, "nil_entry": nil}
	ctx := context.WithValue(context.Background(), adkcontext.ParentMapKey, src)

	carrier, ok := Promote(nil).(interface{ ADKParentMap(context.Context) any })
	if !ok {
		t.Fatal("Promote(nil) does not expose ADKParentMap")
	}
	got, ok := carrier.ADKParentMap(ctx).(runnerParentMap)
	if !ok {
		t.Fatalf("ADKParentMap() type = %T, want runnerParentMap", carrier.ADKParentMap(ctx))
	}
	if _, hasNil := got["nil_entry"]; hasNil {
		t.Fatalf("ADKParentMap() kept nil entry: %v", got)
	}
	if got["sub_agent"] != root {
		t.Fatalf("ADKParentMap()[sub_agent] = %v, want %v", got["sub_agent"], root)
	}
	got["sub_agent"] = other
	got["injected"] = other
	if src["sub_agent"] != root {
		t.Fatalf("mutating ADKParentMap() overwrote source map: %v", src["sub_agent"])
	}
	if _, leaked := src["injected"]; leaked {
		t.Fatalf("mutating ADKParentMap() injected key into source map: %v", src)
	}

	nilMapCtx := context.WithValue(context.Background(), adkcontext.ParentMapKey, runnerParentMap(nil))
	if gotNil := carrier.ADKParentMap(nilMapCtx); gotNil != nil {
		t.Fatalf("ADKParentMap(typed-nil map) = %#v, want nil", gotNil)
	}
}

type runTokenUint8Blocker struct {
	context.Context
}

func (c *runTokenUint8Blocker) Value(key any) any {
	if key != nil && reflect.TypeOf(key).Kind() == reflect.Uint8 {
		return nil
	}
	return c.Context.Value(key)
}

func TestRunToken_StableWithinRun_FreshPerChild(t *testing.T) {
	type runTestSpanKey struct{}
	outer := &invocationContext{Context: t.Context(), session: &identityTestSession{}}
	rtReader, ok := any(Promote(nil)).(interface{ ADKRunToken(context.Context) any })
	if !ok {
		t.Fatal("Promote(nil) does not expose ADKRunToken")
	}

	var nilCtx context.Context
	if got := rtReader.ADKRunToken(nilCtx); got != nil {
		t.Fatalf("ADKRunToken(nil) = %#v, want nil", got)
	}
	if got := rtReader.ADKRunToken(Promote(outer)); got != nil {
		t.Fatalf("ADKRunToken(unpromoted outer) = %#v, want nil", got)
	}

	var parentBeforeTok, parentAfterDeltaTok, parentAfterCtxTok, cbTok, toolTok, childTok any
	var resetBgTok, resetBlockedTok, reboundTok any
	var childLineage []any
	var childCtx InvocationContext

	child, err := New(Config{
		Name: "child",
		Run: func(ctx InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				childCtx = ctx
				childTok = rtReader.ADKRunToken(ctx)
				childLineage = adkcontext.WalkLineage(ctx)
			}
		},
	})
	if err != nil {
		t.Fatalf("New(child): %v", err)
	}

	parent, err := New(Config{
		Name:      "parent",
		SubAgents: []Agent{child},
		Run: func(ctx InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				parentBeforeTok = rtReader.ADKRunToken(ctx)
				icCopy := ctx.WithICDelta(&InvocationContextDelta{Branch: new("b1")})
				parentAfterDeltaTok = rtReader.ADKRunToken(icCopy)
				withSpan := icCopy.WithContext(context.WithValue(icCopy, runTestSpanKey{}, "span"))
				parentAfterCtxTok = rtReader.ADKRunToken(withSpan)
				cbTok = rtReader.ADKRunToken(NewCallbackContext(icCopy, nil))
				tCtx := NewToolContext(withSpan, "call-1", nil, nil)
				toolTok = rtReader.ADKRunToken(tCtx)

				// Synthetic child InvocationContext directly wrapping nodeCtx stops RunTokenKey.
				synthetic := &invocationContext{Context: ctx, session: &identityTestSession{}}
				if got := rtReader.ADKRunToken(synthetic); got != nil {
					t.Fatalf("synthetic invocationContext stole parent runToken: %#v", got)
				}

				for ev, runErr := range child.Run(icCopy) {
					if !yield(ev, runErr) {
						return
					}
				}

				// WithAgentContext resets runToken on non-InvocationContext contexts
				// rather than retaining a stale parent token, and picks up another
				// run's token when wrapping a context derived from that run.
				withSpanCtx, ok := withSpan.(Context)
				if !ok {
					t.Fatalf("withSpan %T does not implement agent.Context", withSpan)
				}
				resetBgTok = rtReader.ADKRunToken(withSpanCtx.WithAgentContext(context.Background()))
				resetBlockedTok = rtReader.ADKRunToken(withSpanCtx.WithAgentContext(&runTokenUint8Blocker{Context: withSpan}))
				reboundTok = rtReader.ADKRunToken(withSpanCtx.WithAgentContext(context.WithValue(childCtx, runTestSpanKey{}, "other")))
			}
		},
	})
	if err != nil {
		t.Fatalf("New(parent): %v", err)
	}

	for _, runErr := range parent.Run(outer) {
		if runErr != nil {
			t.Fatalf("parent.Run error: %v", runErr)
		}
	}

	if parentBeforeTok == nil {
		t.Fatal("parentBeforeTok = nil, want non-nil per-run token")
	}
	for name, got := range map[string]any{
		"WithICDelta":        parentAfterDeltaTok,
		"WithContext":        parentAfterCtxTok,
		"NewCallbackContext": cbTok,
		"NewToolContext":     toolTok,
	} {
		if got != parentBeforeTok {
			t.Fatalf("%s runToken = %#v, want parent token %#v", name, got, parentBeforeTok)
		}
	}
	if childTok == nil || childTok == parentBeforeTok {
		t.Fatalf("child runToken = %#v, want fresh token != %#v", childTok, parentBeforeTok)
	}
	if resetBgTok != nil {
		t.Fatalf("WithAgentContext(Background) retained stale runToken %#v, want nil", resetBgTok)
	}
	if resetBlockedTok != nil {
		t.Fatalf("WithAgentContext(uint8Blocker) retained stale runToken %#v, want nil", resetBlockedTok)
	}
	if reboundTok != childTok {
		t.Fatalf("WithAgentContext(WithValue(childCtx)) runToken = %#v, want child token %#v", reboundTok, childTok)
	}
	if !slices.Contains(childLineage, childTok) || !slices.Contains(childLineage, parentBeforeTok) {
		t.Fatalf("child WalkLineage = %#v, want both child (%#v) and parent (%#v) runTokens", childLineage, childTok, parentBeforeTok)
	}
	if rtReader.ADKRunToken(outer) != nil {
		t.Fatal("parent outer invocationContext mutated by child run")
	}
}
