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
	"time"

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
		// wantAfterDelta is what a delta derivation answers. WithICDelta is
		// inherited by promotion, so for an invocation from outside the module the
		// promoted method hands back the one it embeds and the decorator is dropped
		// — session and all, not just the identity. That cannot be repaired from
		// inside the derivation, so it is asserted rather than assumed. Defaults to
		// want.
		wantAfterDelta string
		hasAfterDelta  bool
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
		{name: "Session() panics", outsideModule: true, wantDirect: "enclosing", wantAfterDelta: "enclosing", hasAfterDelta: true, ic: func() InvocationContext {
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
		{name: "decorated outside the module", want: "u", outsideModule: true, wantDirect: "enclosing", wantAfterDelta: "enclosing", hasAfterDelta: true, ic: func() InvocationContext {
			return decoratedInvocationValue{InvocationContext: enclosing, own: matrixOwner("u")}
		}},
		// The two axes have to be crossed, not just walked. An invocation that owns
		// a session fails closed on its own, so an unreadable session only reaches
		// the delegation through a decorator — where inheriting is a live user's
		// credential minted for someone else's call.
		{name: "decorated, typed-nil session", outsideModule: true, wantDirect: "enclosing", wantAfterDelta: "enclosing", hasAfterDelta: true, ic: func() InvocationContext {
			return decoratedInvocationValue{InvocationContext: enclosing, own: (*matrixSession)(nil)}
		}},
		{name: "decorated, session accessor panics", outsideModule: true, wantDirect: "enclosing", wantAfterDelta: "enclosing", hasAfterDelta: true, ic: func() InvocationContext {
			return decoratedInvocationValue{InvocationContext: enclosing, own: panickingAccessorSession{}}
		}},
		// A nil session field is what a decorator author gets by default, and it is
		// the shape that used to fail OPEN: a nil interface reads exactly like the
		// session-less view a tool context is, so the procedure delegated to the
		// decorator and its parent answered with a live user who made no such call.
		{name: "decorated, nil session", outsideModule: true, wantDirect: "enclosing", wantAfterDelta: "enclosing", hasAfterDelta: true, ic: func() InvocationContext {
			return decoratedInvocationValue{InvocationContext: enclosing, own: nil}
		}},
	}

	// Two columns deliberately put a value on the chain, so the probe for "an
	// unrelated key is undisturbed" must use a key nothing injects.
	type unrelatedKey struct{}
	type probeKey struct{}
	cols := []struct {
		name    string
		isDelta bool
		of      func(InvocationContext) context.Context
	}{
		{"the invocation itself", false, func(ic InvocationContext) context.Context { return ic }},
		{"Promote", false, func(ic InvocationContext) context.Context { return Promote(ic) }},
		{"NewContext", false, func(ic InvocationContext) context.Context { return NewContext(ic) }},
		{"NewToolContext", false, func(ic InvocationContext) context.Context { return NewToolContext(ic, "fc", nil, nil) }},
		{"NewCallbackContext", false, func(ic InvocationContext) context.Context { return NewCallbackContext(ic, nil) }},
		{"NewCallbackContextWithArtifactTracking", false, func(ic InvocationContext) context.Context {
			return NewCallbackContextWithArtifactTracking(ic, nil)
		}},
		{"reparented onto a carrier of the enclosing invocation", false, func(ic InvocationContext) context.Context {
			return Promote(ic).WithContext(context.WithValue(context.Context(enclosing), unrelatedKey{}, "x"))
		}},
		{"tool context of a tool context", false, func(ic InvocationContext) context.Context {
			return NewToolContext(NewToolContext(ic, "a", nil, nil), "b", nil, nil)
		}},
		{"behind a non-ADK wrapper", false, func(ic InvocationContext) context.Context {
			return context.WithValue(Promote(ic), unrelatedKey{}, "x")
		}},
		// The delta derivations. WithICDelta is on the public InvocationContext
		// interface, so these columns are as much a way of deriving a context as
		// the constructors above, and agent.Run takes this exact path on every run.
		{"PromoteWithDelta", true, func(ic InvocationContext) context.Context {
			branch := "br"
			return PromoteWithDelta(ic, &CommonContextDelta{InvocationContextDelta: &InvocationContextDelta{Branch: &branch}})
		}},
		// A delta that says nothing about the invocation does not touch it, so this
		// column is safe where the two below are not.
		{"PromoteWithDelta, nil InvocationContextDelta", false, func(ic InvocationContext) context.Context {
			return PromoteWithDelta(ic, &CommonContextDelta{})
		}},
		{"WithICDelta on a promoted context", true, func(ic InvocationContext) context.Context {
			branch := "br"
			return Promote(ic).WithICDelta(&InvocationContextDelta{Branch: &branch})
		}},
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
			want := r.want
			switch {
			case c.name == "the invocation itself" && r.outsideModule:
				want = r.wantDirect
			case c.isDelta && r.hasAfterDelta:
				want = r.wantAfterDelta
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
