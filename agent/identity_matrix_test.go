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
		want string // "" means no identity
		// outsideModule marks an invocation that cannot override the identity key,
		// so asking it directly answers with whatever it embeds. That is the
		// documented limit of the mechanism; the derivations are what fix it.
		outsideModule bool
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
		{name: "Session() panics", outsideModule: true, ic: func() InvocationContext {
			return panickingSessionInvocation{InvocationContext: enclosing}
		}},
		{name: "permissive Value, own session", want: "u", outsideModule: true, ic: func() InvocationContext {
			return permissiveInvocationValue{InvocationContext: &invocationContext{Context: enclosing, session: matrixOwner("u")}}
		}},
		{name: "decorated outside the module", want: "u", outsideModule: true, ic: func() InvocationContext {
			return decoratedInvocationValue{InvocationContext: enclosing, own: matrixOwner("u")}
		}},
		// The two axes have to be crossed, not just walked. An invocation that owns
		// a session fails closed on its own, so an unreadable session only reaches
		// the delegation through a decorator — where inheriting is a live user's
		// credential minted for someone else's call.
		{name: "decorated, typed-nil session", outsideModule: true, ic: func() InvocationContext {
			return decoratedInvocationValue{InvocationContext: enclosing, own: (*matrixSession)(nil)}
		}},
		{name: "decorated, session accessor panics", outsideModule: true, ic: func() InvocationContext {
			return decoratedInvocationValue{InvocationContext: enclosing, own: panickingAccessorSession{}}
		}},
	}

	// Two columns deliberately put a value on the chain, so the probe for "an
	// unrelated key is undisturbed" must use a key nothing injects.
	type unrelatedKey struct{}
	type probeKey struct{}
	cols := []struct {
		name string
		of   func(InvocationContext) context.Context
	}{
		{"the invocation itself", func(ic InvocationContext) context.Context { return ic }},
		{"Promote", func(ic InvocationContext) context.Context { return Promote(ic) }},
		{"NewContext", func(ic InvocationContext) context.Context { return NewContext(ic) }},
		{"NewToolContext", func(ic InvocationContext) context.Context { return NewToolContext(ic, "fc", nil, nil) }},
		{"NewCallbackContext", func(ic InvocationContext) context.Context { return NewCallbackContext(ic, nil) }},
		{"NewCallbackContextWithArtifactTracking", func(ic InvocationContext) context.Context {
			return NewCallbackContextWithArtifactTracking(ic, nil)
		}},
		{"reparented onto a carrier of the enclosing invocation", func(ic InvocationContext) context.Context {
			return Promote(ic).WithContext(context.WithValue(context.Context(enclosing), unrelatedKey{}, "x"))
		}},
		{"tool context of a tool context", func(ic InvocationContext) context.Context {
			return NewToolContext(NewToolContext(ic, "a", nil, nil), "b", nil, nil)
		}},
		{"behind a non-ADK wrapper", func(ic InvocationContext) context.Context {
			return context.WithValue(Promote(ic), unrelatedKey{}, "x")
		}},
	}

	for _, r := range rows {
		for _, c := range cols {
			if c.name == "the invocation itself" && r.outsideModule {
				continue
			}
			t.Run(r.name+" / "+c.name, func(t *testing.T) {
				defer func() {
					if p := recover(); p != nil {
						t.Fatalf("Value panicked: %v", p)
					}
				}()
				ctx := c.of(r.ic())
				var got string
				if id, ok := IdentityFromContext(ctx); ok {
					got = id.UserID
				}
				if got != r.want {
					t.Errorf("IdentityFromContext() user = %q, want %q", got, r.want)
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
