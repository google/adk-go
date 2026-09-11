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
	"log"
	"strings"
	"testing"

	"google.golang.org/adk/v2/session"
)

// nilDeltaInvocation returns nil from WithICDelta rather than a derived
// invocation, the shape a partial implementation takes. ADK's own ContextMock
// returns nil from WithContext, WithBranch and WithAgentContext.
type nilDeltaInvocation struct {
	InvocationContext
	own session.Session
}

func (d nilDeltaInvocation) Session() session.Session                            { return d.own }
func (nilDeltaInvocation) WithICDelta(*InvocationContextDelta) InvocationContext { return nil }

// TestDeltaOnInvocationThatReturnsNil pins that a nil from WithICDelta costs the
// delta and not the context. Storing the nil leaves a commonContext with no
// invocation, and Agent() and Branch() dereference it — on the merge base this
// same input panics.
func TestDeltaOnInvocationThatReturnsNil(t *testing.T) {
	enclosing := &invocationContext{
		Context: t.Context(),
		agent:   &agent{name: "parent"},
		branch:  "parent-branch",
		session: matrixOwner("enclosing"),
	}
	ic := nilDeltaInvocation{InvocationContext: enclosing, own: matrixOwner("u")}

	var child Agent = &agent{name: "child"}
	branch := "child-branch"
	delta := func() *CommonContextDelta {
		return &CommonContextDelta{
			InvocationContextDelta: &InvocationContextDelta{Agent: &child, Branch: &branch},
		}
	}

	for _, tc := range []struct {
		name string
		ctx  func() Context
	}{
		{"PromoteWithDelta", func() Context { return PromoteWithDelta(ic, delta()) }},
		{"WithDelta", func() Context { return Promote(ic).WithDelta(delta()) }},
		{"WithICDelta", func() Context {
			return Promote(ic).WithICDelta(delta().InvocationContextDelta).(Context)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if p := recover(); p != nil {
					t.Fatalf("panicked after a nil WithICDelta: %v", p)
				}
			}()
			c := tc.ctx()
			// The previous invocation stands, so its agent and branch are what the
			// caller sees. Asserted rather than merely surviving the call: "does not
			// panic" would also pass if the accessors started returning zero values.
			if got := c.(InvocationContext).Agent(); got == nil || got.Name() != "parent" {
				t.Errorf("Agent() = %v, want the previous invocation's agent %q", got, "parent")
			}
			if got := c.Branch(); got != "parent-branch" {
				t.Errorf("Branch() = %q, want the previous invocation's branch %q", got, "parent-branch")
			}
			// Keeping the invocation must not fail open either. A commonContext that
			// lost its invocation reports no identity, but one that silently adopted
			// the enclosing invocation would report a user who made no such call.
			if id, ok := IdentityFromContext(c); !ok || id.UserID != "u" {
				t.Errorf("IdentityFromContext() = %+v, %v; want the invocation's own user %q", id, ok, "u")
			}
		})
	}

	// The discard is the cost of keeping the invocation, and nothing in the
	// assertions above separates it from the delta having been applied. Pinned on
	// the log, which is the only thing that does.
	t.Run("the discard is reported", func(t *testing.T) {
		var buf bytes.Buffer
		out := log.Writer()
		log.SetOutput(&buf)
		t.Cleanup(func() { log.SetOutput(out) })

		c := PromoteWithDelta(ic, delta())
		if got := c.Branch(); got == branch {
			t.Fatalf("Branch() = %q, so the delta was applied after all and this test no "+
				"longer covers what it is named for", got)
		}
		if got := buf.String(); !strings.Contains(got, "discarding the delta") {
			t.Errorf("log = %q, want the discard reported", got)
		}
	})
}

// TestDeltaReachesTheInvocation pins that a delta actually lands on the
// invocation, for a decorated one as much as an ADK one. A guard that kept the
// original unconditionally would pass every assertion in the test above.
//
// It also exists because an attempt to stop a delta dropping an out-of-module
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
