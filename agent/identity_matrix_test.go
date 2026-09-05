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
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/artifact"
	"google.golang.org/adk/v2/internal/adkcontext"
	"google.golang.org/adk/v2/session"
)

// The identity key is answered by a decision procedure, so it is pinned by a
// table rather than by cases. Rows are the shapes a session or an invocation can
// legally take, columns the ways a context is derived from one. Reviewing this
// procedure a diff at a time found one failing shape per round over five rounds,
// each fix moving the failure to a neighbouring shape.
//
// The policy the table encodes:
//   - An invocation reports the user of its OWN session, never one it inherited.
//   - An invocation with no session of its own delegates, which is what a tool or
//     callback context needs — theirs is nil by design.
//   - An invocation that cannot answer at all reports nothing. Broken is not
//     absent, and delegating would hand back a different call's user.
//   - No shape panics out of Value, and no shape disturbs an unrelated key.

type matrixSession struct {
	session.Session
	id, app, user string
}

func (s *matrixSession) ID() string                { return s.id }
func (s *matrixSession) AppName() string           { return s.app }
func (s *matrixSession) UserID() string            { return s.user }
func (s *matrixSession) State() session.State      { return nil }
func (s *matrixSession) Events() session.Events    { return nil }
func (s *matrixSession) LastUpdateTime() time.Time { return time.Time{} }

func matrixOwner(user string) session.Session {
	return &matrixSession{id: "sid-" + user, app: "app", user: user}
}

// structSession is a value type, which a six-accessor interface invites and which
// a reflect-based nil check cannot inspect.
type structSession struct{ session.Session }

func (structSession) ID() string      { return "sid" }
func (structSession) AppName() string { return "app" }
func (structSession) UserID() string  { return "owner" }

// safeNilSession answers without touching its receiver, so a typed-nil one works.
type safeNilPtrSession struct{ session.Session }

func (*safeNilPtrSession) ID() string      { return "sid" }
func (*safeNilPtrSession) AppName() string { return "app" }
func (*safeNilPtrSession) UserID() string  { return "owner" }

// nilWrappingSession promotes its accessors from a nil embedded session, the
// shape llmagent.newWrappedSession produces for a nil original.
type nilWrappingSession struct{ session.Session }

// panickingAccessorSession is broken in its own code, not in its nil-ness.
type panickingAccessorSession struct{ session.Session }

func (panickingAccessorSession) ID() string      { return "sid" }
func (panickingAccessorSession) AppName() string { return "app" }
func (panickingAccessorSession) UserID() string  { panic("accessor is not available") }

// panickingSessionInvocation declines to hand over a session at all, as the
// exported StrictContextMock does.
type panickingSessionInvocation struct{ InvocationContext }

func (panickingSessionInvocation) Session() session.Session { panic("Session is not available") }

// permissiveInvocationValue answers every key, as a decorator or double might.
type permissiveInvocationValue struct{ InvocationContext }

func (permissiveInvocationValue) Value(any) any { return "something that is not an Identity" }

// decoratedInvocationValue is the shape written outside this module: embed the
// invocation you were derived from to inherit cancellation, carry your own
// session. It cannot override the identity key, because it cannot name it.
type decoratedInvocationValue struct {
	InvocationContext
	own session.Session
}

func (d decoratedInvocationValue) Session() session.Session { return d.own }

// contextDecorator decorates agent.Context, which is the interface a tool or a
// workflow node is handed — so it is the shape an author actually writes. It
// overrides both methods on InvocationContext, which is everything the rule used
// to name, and is still dropped by the four that Context adds.
type contextDecorator struct {
	Context
	own session.Session
}

func (d contextDecorator) Session() session.Session                              { return d.own }
func (d contextDecorator) WithContext(context.Context) InvocationContext         { return d }
func (d contextDecorator) WithICDelta(*InvocationContextDelta) InvocationContext { return d }

// Two columns deliberately put a value on the chain, so the probe for "an
// unrelated key is undisturbed" must use a key nothing injects.
type unrelatedKey struct{}

type probeKey struct{}

// identityColumn is one way of deriving a context from an invocation.
type identityColumn struct {
	name string
	// method is the interface method this column calls on the invocation, empty
	// for a package function or a plain derivation. TestEveryContextProducingMethodIsTabulated
	// builds its list of covered methods from this field, so deleting a column
	// takes the coverage claim with it instead of leaving the guard green.
	method string
	// promoted marks a column that calls a context-producing method ON the
	// invocation. An out-of-module decorator is dropped by its own promoted
	// method, either before the derivation or inside it.
	promoted bool
	// contextOnly marks a column whose method is on Context rather than
	// InvocationContext. A row that is only the latter reaches it through
	// Promote, which holds the row rather than replacing it — the path a
	// workflow scheduler takes on every node — so those rows are exercised
	// too, and expect their own user rather than wantContextPromoted.
	contextOnly bool
	// reachesInvocation marks a contextOnly column that gets at the held
	// invocation rather than only re-parenting the context.Context above it. Of
	// the four methods Context adds, only WithDelta does: it forwards to the
	// invocation's WithICDelta, so promoting first does not shelter a decorator
	// from being dropped, where for the other three it does.
	reachesInvocation bool
	// wantAll overrides every row's expectation, for a column that reparents
	// onto a fixed invocation and so answers the same whatever the row is.
	wantAll string
	of      func(InvocationContext) context.Context
}

// identityColumns lists the derivations the matrix walks.
//
// It is a function so that TestEveryContextProducingMethodIsTabulated can read
// the method names off the same slice the matrix runs, rather than keeping a
// second copy that drifts from it. That guard passes a nil enclosing: it reads
// only the names and never calls of.
func identityColumns(t *testing.T, enclosing InvocationContext) []identityColumn {
	t.Helper()
	return []identityColumn{
		{name: "the invocation itself", of: func(ic InvocationContext) context.Context { return ic }},
		{name: "Promote", of: func(ic InvocationContext) context.Context { return Promote(ic) }},
		{name: "NewContext", of: func(ic InvocationContext) context.Context { return NewContext(ic) }},
		{name: "NewToolContext", of: func(ic InvocationContext) context.Context { return NewToolContext(ic, "fc", nil, nil) }},
		{name: "NewCallbackContext", of: func(ic InvocationContext) context.Context { return NewCallbackContext(ic, nil) }},
		{name: "NewCallbackContextWithArtifactTracking", of: func(ic InvocationContext) context.Context {
			return NewCallbackContextWithArtifactTracking(ic, nil)
		}},
		{name: "reparented onto a carrier of the enclosing invocation", method: "WithContext", of: func(ic InvocationContext) context.Context {
			return Promote(ic).WithContext(context.WithValue(context.Context(enclosing), unrelatedKey{}, "x"))
		}},
		// The same method given an InvocationContext rather than a carrier of
		// one. WithAgentContext, which WithContext delegates to, then rebinds the
		// invocation to the argument, so this column answers for the argument
		// whatever the row is. It is the one derivation that legitimately changes
		// who the context speaks for, which is why it is pinned rather than left
		// to the branch coverage of the column above.
		{name: "reparented onto the enclosing invocation itself", method: "WithContext", wantAll: "enclosing", of: func(ic InvocationContext) context.Context {
			return Promote(ic).WithContext(enclosing)
		}},
		{name: "tool context of a tool context", of: func(ic InvocationContext) context.Context {
			return NewToolContext(NewToolContext(ic, "a", nil, nil), "b", nil, nil)
		}},
		{name: "behind a non-ADK wrapper", of: func(ic InvocationContext) context.Context {
			return context.WithValue(Promote(ic), unrelatedKey{}, "x")
		}},
		// The delta derivations. WithICDelta is on the public InvocationContext
		// interface, so these columns are as much a way of deriving a context as
		// the constructors above, and agent.Run takes this exact path on every run.
		{name: "PromoteWithDelta", method: "WithICDelta", promoted: true, of: func(ic InvocationContext) context.Context {
			branch := "br"
			return PromoteWithDelta(ic, &CommonContextDelta{InvocationContextDelta: &InvocationContextDelta{Branch: &branch}})
		}},
		// A delta that says nothing about the invocation does not touch it, so this
		// column is safe where the two below are not.
		{name: "PromoteWithDelta, nil InvocationContextDelta", of: func(ic InvocationContext) context.Context {
			return PromoteWithDelta(ic, &CommonContextDelta{})
		}},
		// The one-token neighbour of the column above, and it used to answer
		// differently: keying the shortcut on a nil pointer rather than on
		// emptiness meant an allocated-but-unset delta reached the invocation and
		// dropped an out-of-module decorator. A caller that allocates the delta
		// and fills it conditionally produces exactly this whenever no condition
		// fires, so the two must agree.
		{name: "PromoteWithDelta, empty InvocationContextDelta", of: func(ic InvocationContext) context.Context {
			return PromoteWithDelta(ic, &CommonContextDelta{InvocationContextDelta: &InvocationContextDelta{}})
		}},
		{name: "WithICDelta on a promoted context", method: "WithICDelta", promoted: true, of: func(ic InvocationContext) context.Context {
			branch := "br"
			return Promote(ic).WithICDelta(&InvocationContextDelta{Branch: &branch})
		}},
		// The same method called directly on the invocation rather than on a
		// commonContext holding it. That is the shape the rule warns about, and
		// the column above does not reach it: Promote holds the row, so the call
		// lands on the commonContext.
		{name: "WithICDelta called on the invocation", method: "WithICDelta", promoted: true, of: func(ic InvocationContext) context.Context {
			branch := "br"
			return Promote(ic.WithICDelta(&InvocationContextDelta{Branch: &branch}))
		}},
		// WithContext is inherited exactly as WithICDelta is.
		{name: "WithContext called on the invocation", method: "WithContext", promoted: true, of: func(ic InvocationContext) context.Context {
			return Promote(ic.WithContext(t.Context()))
		}},
		// The four that agent.Context adds.
		{name: "WithAgentCancel called on the context", method: "WithAgentCancel", promoted: true, contextOnly: true, of: func(ic InvocationContext) context.Context {
			c, cancel := asContext(ic).WithAgentCancel()
			t.Cleanup(cancel)
			return Promote(c)
		}},
		{name: "WithAgentTimeout called on the context", method: "WithAgentTimeout", promoted: true, contextOnly: true, of: func(ic InvocationContext) context.Context {
			c, cancel := asContext(ic).WithAgentTimeout(time.Minute)
			t.Cleanup(cancel)
			return Promote(c)
		}},
		{name: "WithAgentContext called on the context", method: "WithAgentContext", promoted: true, contextOnly: true, of: func(ic InvocationContext) context.Context {
			return Promote(asContext(ic).WithAgentContext(t.Context()))
		}},
		{name: "WithDelta called on the context", method: "WithDelta", promoted: true, contextOnly: true, reachesInvocation: true, of: func(ic InvocationContext) context.Context {
			branch := "br"
			return Promote(asContext(ic).WithDelta(&CommonContextDelta{
				InvocationContextDelta: &InvocationContextDelta{Branch: &branch},
			}))
		}},
	}
}

// asContext reaches the Context methods from a row that may only be an
// InvocationContext. Promoting first holds the row in a commonContext instead of
// replacing it, so the row still answers for itself — that is what a workflow
// scheduler does to the context a node was handed. A row that is already a
// Context is used directly, so its own promoted method runs and a decorator
// that fails to override it is dropped.
func asContext(ic InvocationContext) Context {
	if c, ok := ic.(Context); ok {
		return c
	}
	return Promote(ic)
}

func TestIdentityDecisionMatrix(t *testing.T) {
	// Every row is nested under this one, so any cell that reports "enclosing" is
	// serving a user who made no such call.
	enclosing := &invocationContext{Context: t.Context(), session: matrixOwner("enclosing")}

	rows := []struct {
		name string
		ic   func() InvocationContext
		want string // "" means no identity, and ok must be false
		// outsideModule marks an invocation that cannot override the identity key.
		// The derivations answer for it correctly; asking it *directly* cannot be
		// made to, because the parent it embeds serves a key it cannot name. That
		// is the documented limit of the mechanism, so wantDirect records what the
		// limit costs rather than skipping the cell and hiding it.
		outsideModule bool
		wantDirect    string
		// wantPromoted is what a context-producing method called ON the invocation
		// answers. Those methods are inherited by promotion, so for one written
		// outside the module the promoted method hands back the invocation it
		// embeds and the decorator is dropped — session and all, not just the
		// identity. It cannot be repaired from inside the derivation, so it is
		// asserted rather than assumed. Defaults to want.
		wantPromoted    string
		hasWantPromoted bool
		// wantContextPromoted is the same for the methods Context adds on top of
		// InvocationContext. A decorator can override one interface's worth and not
		// the other's, and that is exactly the gap this row set exists to catch.
		wantContextPromoted    string
		hasWantContextPromoted bool
	}{
		{name: "pointer session", want: "u", ic: func() InvocationContext {
			return &invocationContext{Context: enclosing, session: matrixOwner("u")}
		}},
		{name: "struct-value session", want: "owner", ic: func() InvocationContext {
			return &invocationContext{Context: enclosing, session: structSession{}}
		}},
		{name: "typed-nil with safe accessors", want: "owner", ic: func() InvocationContext {
			return &invocationContext{Context: enclosing, session: (*safeNilPtrSession)(nil)}
		}},
		{name: "no session", ic: func() InvocationContext {
			return &invocationContext{Context: enclosing}
		}},
		{name: "typed-nil session", ic: func() InvocationContext {
			return &invocationContext{Context: enclosing, session: (*matrixSession)(nil)}
		}},
		{name: "session wrapping a nil session", ic: func() InvocationContext {
			return &invocationContext{Context: enclosing, session: &nilWrappingSession{}}
		}},
		{name: "session accessor panics", ic: func() InvocationContext {
			return &invocationContext{Context: enclosing, session: panickingAccessorSession{}}
		}},
		// Read the direct cell here for what it is: "enclosing" because the
		// promoted Value answers off the embedded invocation and Session() is
		// never called, not because a panic was contained. Containment is what
		// every other column on this row asserts — those do reach Session() — and
		// what the two "accessor panics" rows assert on an in-module invocation.
		{name: "Session() panics", outsideModule: true, wantDirect: "enclosing", wantPromoted: "enclosing", hasWantPromoted: true, ic: func() InvocationContext {
			return panickingSessionInvocation{InvocationContext: enclosing}
		}},
		// A permissive Value hands back something that is not an Identity, so the
		// direct cell yields nothing rather than the parent's user.
		{name: "permissive Value, own session", want: "u", outsideModule: true, ic: func() InvocationContext {
			return permissiveInvocationValue{InvocationContext: &invocationContext{Context: enclosing, session: matrixOwner("u")}}
		}},
		{name: "permissive Value, no session", outsideModule: true, ic: func() InvocationContext {
			return permissiveInvocationValue{InvocationContext: &invocationContext{Context: enclosing}}
		}},
		{name: "decorated outside the module", want: "u", outsideModule: true, wantDirect: "enclosing", wantPromoted: "enclosing", hasWantPromoted: true, ic: func() InvocationContext {
			return decoratedInvocationValue{InvocationContext: enclosing, own: matrixOwner("u")}
		}},
		// The two axes have to be crossed, not just walked. An invocation that owns
		// a session fails closed on its own, so an unreadable session only reaches
		// the delegation through a decorator — where inheriting is a live user's
		// credential minted for someone else's call.
		{name: "decorated, typed-nil session", outsideModule: true, wantDirect: "enclosing", wantPromoted: "enclosing", hasWantPromoted: true, ic: func() InvocationContext {
			return decoratedInvocationValue{InvocationContext: enclosing, own: (*matrixSession)(nil)}
		}},
		{name: "decorated, session accessor panics", outsideModule: true, wantDirect: "enclosing", wantPromoted: "enclosing", hasWantPromoted: true, ic: func() InvocationContext {
			return decoratedInvocationValue{InvocationContext: enclosing, own: panickingAccessorSession{}}
		}},
		// A nil session field is what a decorator author gets by default, and it is
		// the shape that used to fail OPEN: a nil interface reads exactly like the
		// session-less view a tool context is, so the procedure delegated to the
		// decorator and its parent answered with a live user who made no such call.
		// Decorating agent.Context rather than agent.InvocationContext. This row
		// overrides both methods the rule used to name and is still dropped by the
		// four that Context adds, which is what the promoted columns assert.
		{
			name: "decorated agent.Context outside the module", want: "u", outsideModule: true, wantDirect: "enclosing",
			wantContextPromoted: "enclosing", hasWantContextPromoted: true, ic: func() InvocationContext {
				return contextDecorator{Context: Promote(enclosing), own: matrixOwner("u")}
			},
		},
		{name: "decorated, nil session", outsideModule: true, wantDirect: "enclosing", wantPromoted: "enclosing", hasWantPromoted: true, ic: func() InvocationContext {
			return decoratedInvocationValue{InvocationContext: enclosing, own: nil}
		}},
	}

	cols := identityColumns(t, enclosing)

	// The four columns on Context reach two different shapes: a row that is
	// already a Context has the method called on it directly, and a row that is
	// only an InvocationContext reaches it through Promote. Exactly one row is of
	// the first shape, so deleting it would silently degrade those columns to
	// testing the second — no compile error, no failing assertion, eighteen cells
	// quietly gone. Both shapes must stay represented.
	var direct, viaPromote int
	for _, r := range rows {
		ic := r.ic()
		_, isCtx := ic.(Context)
		_, ours := ic.(adkcontext.Source)
		// Being a Context is not the property that matters; being one written
		// OUTSIDE the module is. An in-module Context satisfies the first and
		// tests nothing, which is how this check first passed while the coverage
		// it stands for was gone.
		if isCtx && !ours {
			direct++
		} else {
			viaPromote++
		}
	}
	if direct == 0 || viaPromote == 0 {
		t.Fatalf("rows that are an out-of-module Context = %d, rows reaching the Context "+
			"columns through Promote = %d; both shapes are needed, and a decorator over "+
			"agent.Context is the one the promoted columns exist to catch", direct, viaPromote)
	}

	// The full Identity is compared, not just the user: reading two of the three
	// fields off the wrong session, or leaving them empty, would otherwise pass
	// every cell. matrixOwner builds "sid-<user>"; structSession and
	// safeNilPtrSession answer for a fixed "owner".
	full := map[string]Identity{
		"":          {},
		"u":         {UserID: "u", AppName: "app", SessionID: "sid-u"},
		"enclosing": {UserID: "enclosing", AppName: "app", SessionID: "sid-enclosing"},
		"owner":     {UserID: "owner", AppName: "app", SessionID: "sid"},
	}

	for _, r := range rows {
		for _, c := range cols {
			_, isContext := r.ic().(Context)
			want := r.want
			switch {
			case c.wantAll != "":
				want = c.wantAll
			case c.name == "the invocation itself" && r.outsideModule:
				want = r.wantDirect
			// A contextOnly column calls its method directly on a row that is
			// already a Context, so a decorator not overriding it is dropped.
			case c.contextOnly && isContext && r.hasWantContextPromoted:
				want = r.wantContextPromoted
			// A row that is only an InvocationContext reaches the same method
			// through Promote, which holds it rather than replacing it. For three
			// of the four that shelters the row, which still answers for itself;
			// for WithDelta it does not, because that one forwards to the held
			// invocation's WithICDelta and drops a decorator there. Exercising
			// these rows rather than skipping them is the difference between the
			// four Context methods being covered by one row and by all of them,
			// and it is what surfaced that asymmetry.
			case c.contextOnly && !isContext:
				if c.reachesInvocation && r.hasWantPromoted {
					want = r.wantPromoted
				} else {
					want = r.want
				}
			case c.promoted && r.hasWantPromoted:
				want = r.wantPromoted
			}
			t.Run(r.name+" / "+c.name, func(t *testing.T) {
				defer func() {
					if p := recover(); p != nil {
						t.Fatalf("Value panicked: %v", p)
					}
				}()
				ctx := c.of(r.ic())
				var got string
				id, ok := IdentityFromContext(ctx)
				if ok {
					got = id.UserID
				}
				if got != want {
					t.Errorf("IdentityFromContext() user = %q, want %q", got, want)
				}
				if id != full[want] {
					t.Errorf("IdentityFromContext() = %+v, want %+v", id, full[want])
				}
				// Reported separately from the user: returning a zero Identity where
				// none exists would satisfy every want:"" row while telling the
				// provider the invocation has an identity that merely carries no user.
				if ok != (want != "") {
					t.Errorf("IdentityFromContext() ok = %v, want %v", ok, want != "")
				}
				// An invocation that hijacks every key is answering for itself.
				if _, hijacks := r.ic().(permissiveInvocationValue); hijacks {
					return
				}
				if v := ctx.Value(probeKey{}); v != nil {
					t.Errorf("Value(probeKey{}) = %v, want nil: only the identity key reads the session", v)
				}
			})
		}
	}
}

// TestEveryContextProducingMethodIsTabulated is the guard on the table itself.
//
// Four rounds of review found the identity rule short by a method, and the guard
// written to stop that carried the same defect one level up: it kept its own copy
// of the method list, so deleting a matrix column left it green, and it looked
// only for a method returning a context — which is not where the last one was.
// Both are fixed here. The covered set is read off the columns, and a method
// returning any interface must be classified rather than assumed harmless.
func TestEveryContextProducingMethodIsTabulated(t *testing.T) {
	cols := identityColumns(t, nil) // nil enclosing: only names are read, of is not called.

	// Covered means a column calls the method ON the invocation. Keying on the
	// name alone was not enough: three columns name WithContext and only one
	// calls it on the row, so deleting that one — the path agent.Run takes on
	// every run — left this green. promoted is exactly the flag for "called on
	// the row", which is the only shape that exercises the promotion hazard.
	calledOnTheRow := map[string]bool{}
	for _, c := range cols {
		if c.method != "" && c.promoted {
			calledOnTheRow[c.method] = true
		}
	}

	// Deriving from the columns cannot notice a column that is simply gone: nine
	// of them derive a context through a package function and name no method at
	// all, and two more share a name with a column that stays. So the set is also
	// pinned. Unlike the list this test used to keep, this one is not pretending
	// to be derived — it is a snapshot, and changing the table is meant to fail
	// here and be updated deliberately.
	wantColumns := []string{
		"the invocation itself",
		"Promote",
		"NewContext",
		"NewToolContext",
		"NewCallbackContext",
		"NewCallbackContextWithArtifactTracking",
		"reparented onto a carrier of the enclosing invocation",
		"reparented onto the enclosing invocation itself",
		"tool context of a tool context",
		"behind a non-ADK wrapper",
		"PromoteWithDelta",
		"PromoteWithDelta, nil InvocationContextDelta",
		"PromoteWithDelta, empty InvocationContextDelta",
		"WithICDelta on a promoted context",
		"WithICDelta called on the invocation",
		"WithContext called on the invocation",
		"WithAgentCancel called on the context",
		"WithAgentTimeout called on the context",
		"WithAgentContext called on the context",
		"WithDelta called on the context",
	}
	gotColumns := make([]string, 0, len(cols))
	for _, c := range cols {
		gotColumns = append(gotColumns, c.name)
	}
	if !slices.Equal(gotColumns, wantColumns) {
		t.Errorf("the derivation columns changed.\n got: %q\nwant: %q\nA removed column takes "+
			"real coverage with it and the checks above cannot see it. If the change is "+
			"deliberate, update wantColumns.", gotColumns, wantColumns)
	}

	// What a method HANDS BACK can hold a context, or an acting user, just as the
	// receiver does — and a promoted one hands back the embedded invocation's,
	// not the decorator's. So every method returning anything but an inert scalar
	// is classified with the reason. Restricting this to interface returns was
	// too narrow: a pointer or a slice can carry a context just as well.
	holdsNothing := map[string]string{
		"Agent":               "the agent this invocation is running, which carries no user of its own",
		"Err":                 "cancellation cause",
		"OutputForAncestors":  "node names",
		"RequestConfirmation": "reports an error, returns no object",
		"ResumedInput":        "caller-supplied resume payload",
		"RunConfig":           "run configuration, shared across the whole run",
		"ToolConfirmation":    "the confirmation for this one call",
		"UserContent":         "the prompt content, which carries no identity a credential is minted from",
		"Value":               "the context value, which is how identity is answered in the first place",
	}
	// Returns something scoped to the invocation, but which cannot be used to
	// reach a context or act as another user.
	holdsSessionData := map[string]string{
		"Session": "the session itself, which is what identity is read FROM",
	}
	// Returns something that carries the invocation's ACTING USER, so a promoted
	// one addresses the enclosing user's data even where the identity procedure
	// answers for the decorator. Not a context drop path, but the same
	// wrong-user harm by another route, so each must be named in
	// IdentityFromContext's doc.
	holdsActingUser := map[string]string{
		"Artifacts": "an artifact handle carrying AppName/UserID/SessionID, sent as the storage key on every Save, Load and List",
		"Memory":    "a memory handle carrying AppName/UserID, sent as the search key",
		// Not merely "a response and an error": it reaches the result by calling
		// Memory() on the invocation, so a promoted one searches the enclosing
		// user's memories. Same route as Memory, and it was in holdsSessionData
		// under a reason that described the return type and not the path to it.
		"SearchMemory": "searches through the invocation's Memory handle, so it carries that handle's UserID",
		// Reclassified: the reason strings here used to name the return TYPE, which
		// is what got Artifacts, Memory and SearchMemory wrong in turn. These three
		// route through the invocation the context holds, so a promoted one reads
		// and writes the enclosing user's session state.
		"State":         "reaches the invocation's Session().State(), so a promoted one writes the enclosing user's state",
		"ReadonlyState": "reaches the invocation's Session().State(), so a promoted one reads the enclosing user's state",
		"Actions":       "the event actions of the invocation the context holds, whose StateDelta commits to that user's session",
	}
	// Returns something holding a context outright.
	holdsContext := map[string]string{
		"SubScheduler": "the scheduler captures the context it was built from and derives every child from it",
	}

	classified := func(name string) int {
		var n int
		for _, m := range []map[string]string{holdsNothing, holdsSessionData, holdsActingUser, holdsContext} {
			if m[name] != "" {
				n++
			}
		}
		return n
	}

	// The bucket has to have a consequence, or this is a spell-check: until now
	// any of the four satisfied the check, so moving a method between them was
	// green and only the reason string changed. Both hazardous buckets say a
	// promoted call reaches the enclosing invocation's data, which is exactly
	// what IdentityFromContext's rule must warn a decorator author about — so
	// every name in them has to appear there. Five methods reached that bucket
	// only because a reviewer caught each one by hand.
	rule := identityRuleDoc(t)
	for _, m := range []map[string]string{holdsActingUser, holdsContext} {
		for name := range m {
			if !strings.Contains(rule, name) {
				t.Errorf("%s is classified as reaching the enclosing invocation's data, and "+
					"IdentityFromContext's doc comment never mentions it. That comment is what "+
					"a decorator author reads to decide what they must override.", name)
			}
		}
	}

	ctxType := reflect.TypeOf((*Context)(nil)).Elem()
	icType := reflect.TypeOf((*InvocationContext)(nil)).Elem()
	produces := func(m reflect.Method) bool {
		for i := range m.Type.NumOut() {
			if out := m.Type.Out(i); out == ctxType || out == icType {
				return true
			}
		}
		return false
	}
	// Kinds that cannot carry a reference to anything. Everything else — a
	// pointer, a slice, a map, a struct, an interface — must be classified.
	inert := func(k reflect.Kind) bool {
		switch k {
		case reflect.Bool, reflect.String, reflect.Chan, reflect.Func,
			reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
			reflect.Float32, reflect.Float64:
			return true
		}
		return false
	}

	for _, iface := range []reflect.Type{ctxType, icType} {
		for i := range iface.NumMethod() {
			m := iface.Method(i)
			if produces(m) {
				if !calledOnTheRow[m.Name] {
					t.Errorf("%s.%s returns a context and no matrix column CALLS IT on the "+
						"invocation: an invocation written outside the module inherits it by "+
						"promotion, so it is a way to drop the decorator. Add a column with "+
						"method: %q and promoted: true, and name it in IdentityFromContext's doc.",
						iface.Name(), m.Name, m.Name)
				}
				continue
			}
			if n := classified(m.Name); n > 1 {
				t.Errorf("%s.%s is classified %d ways; it must be exactly one", iface.Name(), m.Name, n)
				continue
			} else if n == 1 {
				continue
			}
			for j := range m.Type.NumOut() {
				out := m.Type.Out(j)
				// time.Time is a struct but holds nothing reachable.
				if inert(out.Kind()) || out == reflect.TypeOf(time.Time{}) {
					continue
				}
				t.Errorf("%s.%s returns %s and is classified nowhere. Decide which it is: "+
					"holdsContext (a drop path — name it in IdentityFromContext's doc), "+
					"holdsActingUser (addresses the enclosing user's data — also name it there), "+
					"holdsSessionData, or holdsNothing. Each entry needs a reason.",
					iface.Name(), m.Name, out)
				break
			}
		}
	}
}

// TestPromotedColumnsActuallyDropADecorator checks what the columns DO, which is
// the thing four rounds of guards did not.
//
// Every guard before this one read the table's own account of itself — a method
// name, then a name plus a promoted flag, then the list of column names. All of
// them are labels, and a label is only worth what verifies it: a column whose
// body was gutted, or one whose promoted flag was simply wrong, satisfied every
// version while testing nothing. The single row that would have caught it could
// itself be swapped for an in-module context with the suite green.
//
// So this builds its own decorator, owing nothing to the row table, and requires
// each column marked promoted to actually drop it. A body that no longer calls
// its method on the invocation now fails here, and so does a mislabelled flag.
func TestPromotedColumnsActuallyDropADecorator(t *testing.T) {
	enclosing := &invocationContext{Context: t.Context(), session: matrixOwner("enclosing")}

	var promoted int
	for _, c := range identityColumns(t, enclosing) {
		if !c.promoted {
			continue
		}
		promoted++
		t.Run(c.name, func(t *testing.T) {
			// Overrides nothing but Session, so every context-producing method on
			// it is promoted and hands back the invocation it embeds.
			d := bareContextDecorator{Context: Promote(enclosing), own: matrixOwner("u")}
			id, ok := IdentityFromContext(c.of(d))
			if !ok || id.UserID != "enclosing" {
				t.Errorf("IdentityFromContext() = %q, %v; want \"enclosing\", true.\n"+
					"This column is marked promoted, which asserts it calls its method ON the "+
					"invocation, so a decorator overriding nothing must be dropped by it. Either "+
					"the column body no longer makes that call, or promoted is set wrongly.",
					id.UserID, ok)
			}
		})
	}
	// Pinned because "no promoted columns" would make every check above vacuous
	// and still report success. Eight, not six: three columns reach WithICDelta,
	// through PromoteWithDelta, on a held invocation, and on one directly.
	if want := 8; promoted != want {
		t.Errorf("promoted columns = %d, want %d", promoted, want)
	}
}

// TestPromotedColumnsCallTheMethodTheyName closes the half the test above leaves
// open.
//
// Dropping a bare decorator proves a column calls SOME context-producing method
// on it, not the one its method field names — any two of them are
// interchangeable there. So the column is run a second time against a decorator
// overriding exactly the named method and nothing else: if the column really
// calls it, the override keeps the decorator, and if it calls a different one
// that method is promoted and the decorator goes. Swapping a column body from
// WithContext to WithICDelta now fails, where before it left the covered-set
// claim standing with nothing behind it.
func TestPromotedColumnsCallTheMethodTheyName(t *testing.T) {
	enclosing := &invocationContext{Context: t.Context(), session: matrixOwner("enclosing")}
	var checked int
	for _, c := range identityColumns(t, enclosing) {
		if !c.promoted {
			continue
		}
		checked++
		t.Run(c.name, func(t *testing.T) {
			d := selectiveDecorator{
				Context:  Promote(enclosing),
				own:      matrixOwner("u"),
				override: c.method,
			}
			id, ok := IdentityFromContext(c.of(d))
			if !ok || id.UserID != "u" {
				t.Errorf("IdentityFromContext() = %q, %v; want \"u\", true.\n"+
					"The decorator overrides %s and nothing else, so a column that calls %s "+
					"must keep it. Getting the enclosing user means this column calls some "+
					"OTHER context-producing method, and the coverage it is credited with for "+
					"%s is not real.", id.UserID, ok, c.method, c.method, c.method)
			}
		})
	}
	// Pinned here too rather than relying on the sibling test's count: a guard
	// whose non-emptiness lives in another function is one deletion from being
	// vacuous, which is the failure this file catalogues.
	if want := 8; checked != want {
		t.Errorf("promoted columns checked = %d, want %d", checked, want)
	}
}

// TestDecoratorAnswersForItselfWhileReadingTheEnclosingUsersData pins the claim
// the rule makes about Artifacts, Memory and SearchMemory.
//
// It was asserted in the doc and in a reason string in the classification map,
// and nothing checked it. That is the shape of every defect this file exists to
// catch, so the claim gets a test: one context, answering "u" to the credential
// path and addressing "enclosing" on the artifact path at the same time.
func TestDecoratorAnswersForItselfWhileReadingTheEnclosingUsersData(t *testing.T) {
	enclosing := &invocationContext{
		Context:   t.Context(),
		session:   matrixOwner("enclosing"),
		artifacts: artifactsOf("enclosing"),
	}
	// Overrides Session, as the rule instructs, and not Artifacts.
	d := bareContextDecorator{Context: Promote(enclosing), own: matrixOwner("u")}
	tc := NewToolContext(d, "fc", nil, nil)

	if id, ok := IdentityFromContext(tc); !ok || id.UserID != "u" {
		t.Fatalf("IdentityFromContext() = %q, %v; want \"u\", true", id.UserID, ok)
	}
	// A tool context wraps the handle for save tracking; the handle underneath is
	// the one carrying the user.
	a := tc.Artifacts()
	if tracked, ok := a.(*trackedArtifacts); ok {
		a = tracked.Artifacts
	}
	got, ok := a.(interface{ owner() string })
	if !ok {
		t.Fatalf("Artifacts() = %T, want the test handle", a)
	}
	if got.owner() != "enclosing" {
		t.Errorf("Artifacts() addresses %q, want \"enclosing\"", got.owner())
	}
	// Stated as a test rather than left implied: the two disagree, deliberately
	// and by promotion, and that is what the rule warns a decorator author about.
}

// TestRebindDoesNotCarryTheArtifactsHandle pins the other half of that split.
//
// WithContext given an invocation rebinds who the context speaks for, but an
// artifacts handle already cached on the receiver is copied across untouched. So
// the result answers for the argument while Artifacts() still addresses the
// receiver's user — the same disagreement as the promotion case, reached from
// the opposite direction, and the rule claims it in prose.
// TestRebindIsTotalWithNoCachedHandle is the other half, and it only became
// reachable when the constructors started returning a nil handle for an
// invocation with no artifact service rather than a wrapper around nil. Before
// that, artifacts was always non-nil and the split below was unconditional.
func TestRebindIsTotalWithNoCachedHandle(t *testing.T) {
	alice := &invocationContext{Context: t.Context(), session: matrixOwner("alice"), artifacts: artifactsOf("alice")}
	bob := &invocationContext{Context: t.Context(), session: matrixOwner("bob")} // no artifact service

	wrapper, ok := NewToolContext(bob, "fc", nil, nil).(*toolContextWrapper)
	if !ok {
		t.Fatalf("NewToolContext() = %T, want *toolContextWrapper", NewToolContext(bob, "fc", nil, nil))
	}
	held, ok := wrapper.context.(*commonContext)
	if !ok {
		t.Fatalf("the wrapper holds %T, want *commonContext", wrapper.context)
	}
	if held.artifacts != nil {
		t.Fatalf("artifacts = %T, want nil; this test covers the uncached half and the "+
			"constructors no longer produce it", held.artifacts)
	}

	rebound, ok := held.WithContext(alice).(*commonContext)
	if !ok {
		t.Fatal("WithContext() did not return a *commonContext")
	}
	if id, ok := IdentityFromContext(rebound); !ok || id.UserID != "alice" {
		t.Errorf("IdentityFromContext() = %q, %v; want \"alice\", true", id.UserID, ok)
	}
	owner, ok := rebound.Artifacts().(interface{ owner() string })
	if !ok {
		t.Fatalf("Artifacts() = %T, want the test handle", rebound.Artifacts())
	}
	if got := owner.owner(); got != "alice" {
		t.Errorf("Artifacts() addresses %q, want \"alice\": with nothing cached the fallback "+
			"resolves through the rebound invocation, so the rebind is total here", got)
	}
}

func TestRebindDoesNotCarryTheArtifactsHandle(t *testing.T) {
	alice := &invocationContext{Context: t.Context(), session: matrixOwner("alice"), artifacts: artifactsOf("alice")}
	bob := &invocationContext{Context: t.Context(), session: matrixOwner("bob"), artifacts: artifactsOf("bob")}

	// A callback context is one of the shapes that caches the handle.
	wrapper, ok := NewCallbackContext(bob, nil).(*callbackContextWrapper)
	if !ok {
		t.Fatalf("NewCallbackContext() = %T, want *callbackContextWrapper", NewCallbackContext(bob, nil))
	}
	rebound, ok := wrapper.context.WithContext(alice).(*commonContext)
	if !ok {
		t.Fatalf("WithContext() did not return a *commonContext")
	}

	if id, ok := IdentityFromContext(rebound); !ok || id.UserID != "alice" {
		t.Errorf("IdentityFromContext() = %q, %v; want \"alice\", true", id.UserID, ok)
	}
	owner, ok := rebound.Artifacts().(interface{ owner() string })
	if !ok {
		t.Fatalf("Artifacts() = %T, want the test handle", rebound.Artifacts())
	}
	if got := owner.owner(); got != "bob" {
		t.Errorf("Artifacts() addresses %q, want \"bob\": the cached handle is expected NOT to "+
			"follow the rebind, and the rule says so", got)
	}
}

// TestAnUnmarkedContextCanAnswerTheKeyItself pins the limit of the mechanism,
// which the rule used to state backwards.
//
// It said a decorator intercepting the key "reports nothing". That is true only
// when it answers with some other type, which is the only variant the matrix
// covered. Identity is an ordinary exported struct, so a wrapper can build one
// and have it reported — the key is unforgeable, the value is not. Pinned in
// both directions, because the false half was the half that mattered: a reader
// of the old sentence would have taken interception for a defence.
func TestAnUnmarkedContextCanAnswerTheKeyItself(t *testing.T) {
	enclosing := &invocationContext{Context: t.Context(), session: matrixOwner("enclosing")}

	if id, ok := IdentityFromContext(permissiveInvocationValue{InvocationContext: enclosing}); ok || id != (Identity{}) {
		t.Errorf("answering with another type: IdentityFromContext() = %+v, %v; "+
			"want the zero Identity and false", id, ok)
	}
	forged := Identity{UserID: "someone else", AppName: "app", SessionID: "s"}
	id, ok := IdentityFromContext(forgingInvocationValue{InvocationContext: enclosing, id: forged})
	if !ok || id != forged {
		t.Errorf("answering with an Identity: IdentityFromContext() = %+v, %v; want %+v, true. "+
			"The mechanism does not authenticate the value, and the rule must not imply it does.",
			id, ok, forged)
	}
}

// TestTheKeyValueEscapesToAnyContext pins the limit the rule now states, because
// the rule used to state the opposite.
//
// IdentityFromContext hands the key to whatever Value implementation it is given,
// so one call is enough to capture it — after which an Identity can be planted
// under it from a context descending from no invocation at all. The type being
// unnameable does not prevent that: the value is all anyone needs.
//
// Pinned as a limit rather than fixed. Value is the only mechanism that reaches
// through the intermediaries this function exists to see past, so there is no
// version of it that both works and withholds the key.
func TestTheKeyValueEscapesToAnyContext(t *testing.T) {
	var captured any
	IdentityFromContext(keySniffer{Context: t.Context(), seen: func(k any) { captured = k }})
	if captured == nil {
		t.Fatal("the key never reached the context's own Value; if that is now true by " +
			"design, replace this test with one asserting the stronger property")
	}

	forged := Identity{UserID: "someone else", AppName: "app", SessionID: "s"}
	id, ok := IdentityFromContext(context.WithValue(t.Context(), captured, forged))
	if !ok || id != forged {
		t.Errorf("IdentityFromContext() = %+v, %v; want %+v, true. The captured key can be "+
			"replanted, and the rule must keep saying so rather than promising it cannot.",
			id, ok, forged)
	}
}

// keySniffer records the key it is handed and answers nothing.
type keySniffer struct {
	context.Context
	seen func(any)
}

func (s keySniffer) Value(k any) any {
	s.seen(k)
	return nil
}

// forgingInvocationValue substitutes the identity on its way past without naming
// the key, which is the shape that actually matters: it forwards keys and swaps
// only what comes back an Identity. A wrapper answering EVERY key with an
// Identity would be a far cruder thing and would break its own context, so
// modelling that instead would suggest forging costs more than it does. Matching
// on the type is what lets this fixture stay short — it does also catch an
// Identity stored under some other key, which a real one aiming to be quiet
// would avoid, and which is beside the point being pinned here.
type forgingInvocationValue struct {
	InvocationContext
	id Identity
}

func (d forgingInvocationValue) Value(key any) any {
	v := d.InvocationContext.Value(key)
	if _, isIdentity := v.(Identity); isIdentity {
		return d.id
	}
	return v
}

func artifactsOf(user string) Artifacts { return userArtifacts(user) }

// userArtifacts stands in for internal/artifact.Artifacts, which carries
// AppName/UserID/SessionID and sends them as the storage key.
type userArtifacts string

func (a userArtifacts) owner() string { return string(a) }

func (a userArtifacts) Save(context.Context, string, *genai.Part) (*artifact.SaveResponse, error) {
	return nil, nil
}
func (a userArtifacts) List(context.Context) (*artifact.ListResponse, error) { return nil, nil }
func (a userArtifacts) Load(context.Context, string) (*artifact.LoadResponse, error) {
	return nil, nil
}

func (a userArtifacts) LoadVersion(context.Context, string, int) (*artifact.LoadResponse, error) {
	return nil, nil
}

// selectiveDecorator intercepts exactly one context-producing method and lets
// every other one promote, so a column can be checked for calling the method it
// says it calls.
type selectiveDecorator struct {
	Context
	own      session.Session
	override string
}

func (d selectiveDecorator) Session() session.Session { return d.own }

func (d selectiveDecorator) WithContext(ctx context.Context) InvocationContext {
	if d.override == "WithContext" {
		return d
	}
	return d.Context.WithContext(ctx)
}

func (d selectiveDecorator) WithICDelta(delta *InvocationContextDelta) InvocationContext {
	if d.override == "WithICDelta" {
		return d
	}
	return d.Context.WithICDelta(delta)
}

func (d selectiveDecorator) WithDelta(delta *CommonContextDelta) Context {
	if d.override == "WithDelta" {
		return d
	}
	return d.Context.WithDelta(delta)
}

func (d selectiveDecorator) WithAgentContext(ctx context.Context) Context {
	if d.override == "WithAgentContext" {
		return d
	}
	return d.Context.WithAgentContext(ctx)
}

func (d selectiveDecorator) WithAgentTimeout(timeout time.Duration) (Context, context.CancelFunc) {
	c, cancel := d.Context.WithAgentTimeout(timeout)
	if d.override == "WithAgentTimeout" {
		return d, cancel
	}
	return c, cancel
}

func (d selectiveDecorator) WithAgentCancel() (Context, context.CancelFunc) {
	c, cancel := d.Context.WithAgentCancel()
	if d.override == "WithAgentCancel" {
		return d, cancel
	}
	return c, cancel
}

// bareContextDecorator is the minimum an author writes: embed the context you
// were handed, carry your own session, override nothing else.
type bareContextDecorator struct {
	Context
	own session.Session
}

func (d bareContextDecorator) Session() session.Session { return d.own }

// TestIdentityFromAHostileSource pins what a marked context can do to the
// procedure that asks it.
//
// A marked type is asked for the identity rather than read, which is more trust
// than any other arm extends, so the two shapes that abuse it are the ones worth
// pinning: answering with the wrong type, and panicking on the way.
//
// This catches the recover and NOT the inner type assertion, deliberately and as
// identityFrom's own doc says — IdentityFromContext asserts again, so removing
// the inner one changes what Value returns and nothing a caller can observe.
// Claiming otherwise here would be one more unverified assertion in a file whose
// whole subject is unverified assertions.
func TestIdentityFromAHostileSource(t *testing.T) {
	enclosing := &invocationContext{Context: t.Context(), session: matrixOwner("enclosing")}

	for _, tc := range []struct {
		name string
		ic   func() InvocationContext
	}{{
		// Answers the identity key with something that is not an Identity.
		"a source answering with the wrong type",
		func() InvocationContext { return &junkSource{InvocationContext: enclosing} },
	}, {
		// Panics on the way, which for an unmarked context Recovered already
		// contains; this is the marked path, which asks rather than reads.
		"a source that panics",
		func() InvocationContext { return &panickingSource{InvocationContext: enclosing} },
	}} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("identity read panicked out: %v", p)
				}
			}()
			id, ok := IdentityFromContext(Promote(tc.ic()))
			if ok || id != (Identity{}) {
				t.Errorf("IdentityFromContext() = %+v, %v; want the zero Identity and false. "+
					"A context that cannot answer must report nothing, not something taken for "+
					"an Identity and not the enclosing call's user.", id, ok)
			}
		})
	}
}

// junkSource carries the marker, so the procedure asks it, and then answers with
// something that is not an Identity.
type junkSource struct {
	InvocationContext
	adkcontext.Marker
}

func (*junkSource) Value(any) any { return "not an Identity" }

// panickingSource carries the marker and then fails on the way.
type panickingSource struct {
	InvocationContext
	adkcontext.Marker
}

func (*panickingSource) Value(any) any { panic("Value is not available") }

// TestPromotedSubSchedulerDropsTheDecorator pins the drop path the guard
// classifies but cannot tabulate.
//
// SubScheduler returns a scheduler, not a context, so no matrix column can call
// it. The hazard is the same as every other promoted method: a decorator that
// does not override it hands back the scheduler belonging to what it embeds, and
// that scheduler derives every child from the context it captured. In
// workflow.RunNode that is one call — ctx.SubScheduler() — and the whole child
// subtree then runs as the enclosing invocation.
func TestPromotedSubSchedulerDropsTheDecorator(t *testing.T) {
	enclosing := &invocationContext{Context: t.Context(), session: matrixOwner("enclosing")}
	var theirs DynamicSubScheduler = stubSubScheduler{owner: "enclosing"}
	base := Promote(enclosing).WithDelta(&CommonContextDelta{SubScheduler: &theirs})

	decorated := contextDecorator{Context: base, own: matrixOwner("u")}
	if got, ok := decorated.SubScheduler().(stubSubScheduler); !ok || got.owner != "enclosing" {
		t.Fatalf("SubScheduler() = %#v, want the embedded context's scheduler", decorated.SubScheduler())
	}
	// The decorator answers for its own user, and still hands out a scheduler
	// bound to someone else's invocation. That gap is what the doc must state.
	if id, ok := IdentityFromContext(Promote(decorated)); !ok || id.UserID != "u" {
		t.Fatalf("IdentityFromContext() = %q, %v, want \"u\", true", id.UserID, ok)
	}
}

type stubSubScheduler struct {
	DynamicSubScheduler
	owner string
}

// TestDeltaDerivationsReachWithICDeltaOnly pins which method the two delta
// derivations actually call on the invocation.
//
// The doc used to claim PromoteWithDelta reaches WithDelta for a Context, so
// that overriding WithICDelta alone was not enough. It does not: both it and
// commonContext.WithDelta go through withICDelta, which calls WithICDelta. A
// decorator overriding only WithICDelta therefore survives both, and the claim
// sent an author looking for a bug that was not there. Pinned because a
// statement about which method runs is exactly the kind that drifts unchecked.
func TestDeltaDerivationsReachWithICDeltaOnly(t *testing.T) {
	enclosing := &invocationContext{Context: t.Context(), session: matrixOwner("enclosing")}
	branch := "br"
	delta := &CommonContextDelta{InvocationContextDelta: &InvocationContextDelta{Branch: &branch}}

	for _, tc := range []struct {
		name string
		of   func(Context) context.Context
	}{
		{"PromoteWithDelta", func(c Context) context.Context { return PromoteWithDelta(c, delta) }},
		{"WithDelta on a promoted context", func(c Context) context.Context { return Promote(c).WithDelta(delta) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Overrides WithICDelta and deliberately not WithDelta.
			d := icDeltaOnlyDecorator{Context: Promote(enclosing), own: matrixOwner("u")}
			if id, ok := IdentityFromContext(tc.of(d)); !ok || id.UserID != "u" {
				t.Errorf("IdentityFromContext() = %q, %v; want \"u\", true: overriding "+
					"WithICDelta alone should survive this derivation", id.UserID, ok)
			}
		})
	}

	// The counterpart: calling WithDelta directly on the decorator does drop it,
	// which is why WithDelta is still in the rule.
	d := icDeltaOnlyDecorator{Context: Promote(enclosing), own: matrixOwner("u")}
	if id, ok := IdentityFromContext(Promote(d.WithDelta(delta))); !ok || id.UserID != "enclosing" {
		t.Errorf("IdentityFromContext() = %q, %v; want \"enclosing\", true: a direct "+
			"WithDelta is promoted from the embedded context and drops the decorator", id.UserID, ok)
	}
}

// TestNilInvocationDeltaIsSafeOnlyOnceHeld pins where the nil-delta exception
// actually holds.
//
// The rule used to offer it as a general exception, in the bullet about calling
// a method directly on the decorator — where it is false. A promoted method
// never reaches the decorator, so it cannot consult the delta at all, and the
// decorator is gone whatever the delta says. Held inside a commonContext it is
// true, because there the delta is inspected before the invocation is touched.
// An author reading the old wording would have concluded a delta carrying only
// Path or RunID needed no interception, which is exactly how entering a workflow
// drops one.
func TestNilInvocationDeltaIsSafeOnlyOnceHeld(t *testing.T) {
	enclosing := &invocationContext{Context: t.Context(), session: matrixOwner("enclosing")}
	path := "n@1"
	noICDelta := func() *CommonContextDelta { return &CommonContextDelta{Path: &path} }

	d := icDeltaOnlyDecorator{Context: Promote(enclosing), own: matrixOwner("u")}
	if id, ok := IdentityFromContext(d.WithDelta(noICDelta())); !ok || id.UserID != "enclosing" {
		t.Errorf("direct WithDelta: IdentityFromContext() = %q, %v; want \"enclosing\", true. "+
			"A promoted method cannot consult the delta, so an empty one is no shelter.", id.UserID, ok)
	}
	if id, ok := IdentityFromContext(Promote(d).WithDelta(noICDelta())); !ok || id.UserID != "u" {
		t.Errorf("held WithDelta: IdentityFromContext() = %q, %v; want \"u\", true. "+
			"Held in a commonContext the delta is inspected first, so an empty one "+
			"leaves the invocation alone.", id.UserID, ok)
	}
}

type icDeltaOnlyDecorator struct {
	Context
	own session.Session
}

func (d icDeltaOnlyDecorator) Session() session.Session { return d.own }

func (d icDeltaOnlyDecorator) WithICDelta(*InvocationContextDelta) InvocationContext { return d }

// identityRuleDoc returns the doc comment on IdentityFromContext, read from
// source.
//
// Read rather than duplicated: a copy here would be one more list to drift, and
// drift between the rule and what enforces it is the failure this file exists to
// stop.
func identityRuleDoc(t *testing.T) string {
	t.Helper()
	src, err := os.ReadFile("common_context.go")
	if err != nil {
		t.Fatalf("read common_context.go: %v", err)
	}
	const start = "// IdentityFromContext returns"
	i := bytes.Index(src, []byte(start))
	if i < 0 {
		t.Fatal("IdentityFromContext's doc comment not found; this guard reads it by prefix")
	}
	j := bytes.Index(src[i:], []byte("\nfunc IdentityFromContext"))
	if j < 0 {
		t.Fatal("IdentityFromContext's declaration not found after its doc comment")
	}
	return string(src[i : i+j])
}
