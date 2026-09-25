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
	"strings"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/adkcontext"

	"google.golang.org/adk/v2/session"
)

// nilDeltaInvocation returns nil from WithICDelta rather than a derived
// invocation — the shape an implementation written outside this repository
// takes. Nothing in the repository does it, which is why the fixture has to.
type nilDeltaInvocation struct {
	InvocationContext
	own session.Session
}

func (d nilDeltaInvocation) Session() session.Session                            { return d.own }
func (nilDeltaInvocation) WithICDelta(*InvocationContextDelta) InvocationContext { return nil }

// markedNilDeltaInvocation is the same shape carrying the identity marker, which
// the exported InvocationContext interface permits and which ADK's own wrappers
// are. The marker is what keeps the two guards below reachable: withICDelta skips
// the call entirely for a zero delta on an UNMARKED invocation, so after that
// shortcut landed neither guard was entered by any test and both could be deleted
// with the suite staying green. A marked receiver is still asked, so it still
// returns nil, and the report path still runs.
type markedNilDeltaInvocation struct {
	InvocationContext
	adkcontext.Marker
}

func (markedNilDeltaInvocation) WithICDelta(*InvocationContextDelta) InvocationContext { return nil }

// freshReports makes the once-per-type report available again. The set has
// process lifetime, so without this a test that asserts on the report passes
// only on the first run of the binary and fails under -count=2.
func freshReports(t *testing.T) {
	t.Helper()
	reportedNilICDelta.Clear()
	t.Cleanup(reportedNilICDelta.Clear)
}

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
		freshReports(t)
		var c Context
		got := captureLog(t, func() { c = PromoteWithDelta(ic, delta()) })
		if b := c.Branch(); b == branch {
			t.Fatalf("Branch() = %q, so the delta was applied after all and this test no "+
				"longer covers what it is named for", b)
		}
		if !strings.Contains(got, "did not reach it") {
			t.Errorf("log = %q, want the discard reported", got)
		}
	})
}

// TestDiscardReportNamesWhatWasLost pins the field list. Without it the whole
// enumeration is unasserted: swapping two labels, or pointing %T at the delta
// instead of the implementation, passes every other test in this file.
func TestDiscardReportNamesWhatWasLost(t *testing.T) {
	enclosing := &invocationContext{Context: t.Context(), agent: &agent{name: "parent"}}
	var child Agent = &agent{name: "child"}
	branch, scope := "child-branch", "child-scope"
	content := &genai.Content{}
	newCtx := context.Background()

	for _, tc := range []struct {
		name  string
		delta *InvocationContextDelta
		want  string
	}{
		{"agent", &InvocationContextDelta{Agent: &child}, "Agent"},
		{"branch", &InvocationContextDelta{Branch: &branch}, `Branch="child-branch"`},
		{"isolation scope", &InvocationContextDelta{IsolationScope: &scope}, `IsolationScope="child-scope"`},
		{"user content", &InvocationContextDelta{UserContent: &content}, "UserContent"},
		{
			"every reported field at once",
			&InvocationContextDelta{Agent: &child, Branch: &branch, IsolationScope: &scope, UserContent: &content},
			`Agent, Branch="child-branch", IsolationScope="child-scope", UserContent`,
		},
		{
			// Named on both entry points: neither installs it on the invocation.
			// WithDelta puts it on the context it returns, which is a different
			// object and is why the message says "did not reach it".
			"context never reaches the invocation, so it is named",
			&InvocationContextDelta{Agent: &child, Context: &newCtx},
			"Agent, Context",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			freshReports(t)
			got := captureLog(t, func() {
				_ = PromoteWithDelta(nilDeltaInvocation{InvocationContext: enclosing},
					&CommonContextDelta{InvocationContextDelta: tc.delta})
			})
			if !strings.Contains(got, tc.want) {
				t.Errorf("log = %q, want it to name %s", got, tc.want)
			}
			if !strings.Contains(got, "agent.nilDeltaInvocation.WithICDelta") {
				t.Errorf("log = %q, want the implementation's type named", got)
			}
		})
	}
}

// TestDiscardReportIsEmittedOnce pins the deduplication. Deriving from two
// distinct values of the same type is what makes it a test of the per-type key
// rather than a per-instance one: keying on ic itself passes a single-value
// version of this, and in production would hold every invocation object a
// non-conforming type ever produced.
func TestDiscardReportIsEmittedOnce(t *testing.T) {
	freshReports(t)
	branch := "child-branch"
	derive := func(ic InvocationContext) {
		_ = PromoteWithDelta(ic, &CommonContextDelta{
			InvocationContextDelta: &InvocationContextDelta{Branch: &branch},
		})
	}
	out := captureLog(t, func() {
		for range 3 {
			// A fresh enclosing invocation each time, so each derivation runs on a
			// different value of the same type.
			derive(nilDeltaInvocation{InvocationContext: &invocationContext{
				Context: t.Context(), agent: &agent{name: "parent"},
			}})
		}
	})
	if got := strings.Count(out, "returned nil"); got != 1 {
		t.Errorf("three discards of the same shape produced %d report(s), want 1:\n%s", got, out)
	}
}

// TestDiscardReportsEachDistinctLoss pins the other half of the key. A later
// discard that loses different fields carries information the first line did
// not, so suppressing it would make the comment on reportKey untrue.
func TestDiscardReportsEachDistinctLoss(t *testing.T) {
	freshReports(t)
	enclosing := &invocationContext{Context: t.Context(), agent: &agent{name: "parent"}}
	ic := nilDeltaInvocation{InvocationContext: enclosing}
	branch := "child-branch"
	var child Agent = &agent{name: "child"}

	out := captureLog(t, func() {
		// The shape the scheduler derives per node, twice, then the one that
		// actually swaps the agent.
		for range 2 {
			_ = PromoteWithDelta(ic, &CommonContextDelta{
				InvocationContextDelta: &InvocationContextDelta{Branch: &branch},
			})
		}
		_ = PromoteWithDelta(ic, &CommonContextDelta{
			InvocationContextDelta: &InvocationContextDelta{Agent: &child},
		})
	})
	if got := strings.Count(out, "returned nil"); got != 2 {
		t.Errorf("two distinct losses produced %d report(s), want 2:\n%s", got, out)
	}
	if !strings.Contains(out, "Agent") {
		t.Errorf("log = %q, want the agent loss reported and not swallowed by the branch one", out)
	}
}

// TestDiscardRepeatDoesNotAllocate pins the ordering the mask exists for: a
// repeat must not pay the slice or the formatting.
func TestDiscardRepeatDoesNotAllocate(t *testing.T) {
	freshReports(t)
	enclosing := &invocationContext{Context: t.Context(), agent: &agent{name: "parent"}}
	ic := nilDeltaInvocation{InvocationContext: enclosing}
	branch, scope := "child-branch", "child-scope"
	d := &InvocationContextDelta{Branch: &branch, IsolationScope: &scope}

	// Bound to the interface once: converting the fixture struct at each call
	// would allocate here in the test and be counted against the function.
	var boxed InvocationContext = ic
	_ = captureLog(t, func() {
		reportDiscardedDelta(boxed, d) // claim the bit
		if got := testing.AllocsPerRun(100, func() { reportDiscardedDelta(boxed, d) }); got != 0 {
			t.Errorf("a repeat allocated %v times, want 0 — the field list is being built before the dedup check", got)
		}
	})
}

// TestDiscardWithNothingToReport pins that a delta which asked for nothing
// neither reports nor spends the type's one report. Claiming the type before
// building the field list made an empty delta silence the next real loss.
func TestDiscardWithNothingToReport(t *testing.T) {
	freshReports(t)
	enclosing := &invocationContext{Context: t.Context(), agent: &agent{name: "parent"}}
	// Marked, so the empty-delta shortcut does not fire and the fields == 0 guard
	// is actually entered. With an unmarked fixture the call never happens and
	// deleting that guard leaves this test green.
	marked := markedNilDeltaInvocation{InvocationContext: enclosing}
	empty := captureLog(t, func() {
		_ = PromoteWithDelta(marked, &CommonContextDelta{InvocationContextDelta: &InvocationContextDelta{}})
	})
	if empty != "" {
		t.Errorf("a delta that asked for nothing logged %q, want silence", empty)
	}

	branch := "child-branch"
	real := captureLog(t, func() {
		_ = PromoteWithDelta(marked, &CommonContextDelta{
			InvocationContextDelta: &InvocationContextDelta{Branch: &branch},
		})
	})
	if !strings.Contains(real, `Branch="child-branch"`) {
		t.Errorf("log = %q, want the empty delta to have left the report unspent", real)
	}
}

// TestNilDeltaStillReachesTheInvocation pins that a delta carrying no
// InvocationContextDelta is still handed to the implementation. Both wrappers in
// this package delegate to the inner commonContext, which answers a nil delta by
// returning itself — skipping the call leaves the wrapper in place, and its
// Agent() returns nil.
func TestNilDeltaStillReachesTheInvocation(t *testing.T) {
	inner := &invocationContext{Context: t.Context(), agent: &agent{name: "parent"}}
	wrapped := NewToolContext(inner, "fc-1", nil, nil).(InvocationContext)
	path := "wf/n@1"

	c := Promote(wrapped).WithDelta(&CommonContextDelta{Path: &path})

	got := c.(InvocationContext).Agent()
	if got == nil || got.Name() != "parent" {
		t.Fatalf("Agent() = %v, want the wrapper to have been unwrapped to %q", got, "parent")
	}
	if name := c.AgentName(); name != "parent" {
		t.Errorf("AgentName() = %q, want %q", name, "parent")
	}
}

// TestDeltaReachesTheInvocation pins that a delta actually lands on the
// invocation, for a decorated one as much as an ADK one, and that accepting it
// stays silent. A guard that kept the original unconditionally would pass every
// assertion in the test above.
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
		branch:  "parent-branch",
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
			freshReports(t)
			var c Context
			// Silence matters as much as the values: a helper that reported on every
			// derivation, not only on a discard, would satisfy every other assertion here.
			if out := captureLog(t, func() { c = PromoteWithDelta(tc.ic, delta()) }); out != "" {
				t.Errorf("a delta the invocation accepted logged %q, want silence", out)
			}
			if got := c.(InvocationContext).Agent(); got == nil || got.Name() != "child" {
				t.Errorf("Agent() = %v, want the agent the delta named", got)
			}
			if got := c.Branch(); got != branch {
				t.Errorf("Branch() = %q, want %q", got, branch)
			}
		})
	}
}

// TestDiscardKeepsTheRestOfTheDelta pins the CommonContextDelta fields that
// WithDelta applies alongside the invocation delta. RunID and SubScheduler are
// asserted nowhere else in this package, so dropping them was invisible.
func TestDiscardKeepsTheRestOfTheDelta(t *testing.T) {
	freshReports(t)
	enclosing := &invocationContext{Context: t.Context(), agent: &agent{name: "parent"}}
	ic := nilDeltaInvocation{InvocationContext: enclosing}

	runID, path := "run-7", "wf/n@1"
	ancestors := []string{"a", "b"}
	var sub DynamicSubScheduler = stubSubScheduler{}
	branch := "child-branch"

	var c Context
	_ = captureLog(t, func() {
		c = PromoteWithDelta(ic, &CommonContextDelta{
			InvocationContextDelta: &InvocationContextDelta{Branch: &branch},
			RunID:                  &runID,
			Path:                   &path,
			OutputForAncestors:     &ancestors,
			SubScheduler:           &sub,
		})
	})

	if got := c.RunID(); got != runID {
		t.Errorf("RunID() = %q, want %q — a discarded invocation delta must not cost the rest", got, runID)
	}
	if got := c.SubScheduler(); got == nil {
		t.Error("SubScheduler() = nil, want the one the delta carried")
	}
	if got := c.Path(); got != path {
		t.Errorf("Path() = %q, want %q", got, path)
	}
}

// stubSubScheduler is a non-nil DynamicSubScheduler for the assertion above.
type stubSubScheduler struct{ DynamicSubScheduler }

// TestDiscardNamesContextOnBothEntryPoints pins that Context is reported
// whichever method was called. Neither installs it on the invocation, so it
// never reaches it — WithDelta separately puts it on the context it returns,
// which is a different object and is what the message's wording is careful about.
func TestDiscardNamesContextOnBothEntryPoints(t *testing.T) {
	enclosing := &invocationContext{Context: t.Context(), agent: &agent{name: "parent"}}
	ic := nilDeltaInvocation{InvocationContext: enclosing}
	type key struct{}
	newCtx := context.WithValue(context.Background(), key{}, "from-the-delta")

	t.Run("WithDelta", func(t *testing.T) {
		freshReports(t)
		var c Context
		got := captureLog(t, func() {
			c = PromoteWithDelta(ic, &CommonContextDelta{
				InvocationContextDelta: &InvocationContextDelta{Context: &newCtx},
			})
		})
		if !strings.Contains(got, "Context") {
			t.Errorf("log = %q, want Context named — it never reached the invocation", got)
		}
		if v := c.Value(key{}); v != "from-the-delta" {
			t.Errorf("Value(key) = %v, want the delta's context on the returned context", v)
		}
	})

	t.Run("WithICDelta", func(t *testing.T) {
		freshReports(t)
		got := captureLog(t, func() {
			_ = Promote(ic).WithICDelta(&InvocationContextDelta{Context: &newCtx})
		})
		if !strings.Contains(got, "Context") {
			t.Errorf("log = %q, want Context named", got)
		}
	})
}

// TestDiscardWithNoInvocationDelta pins the nil guard in withICDelta. The
// reporter dereferences d, so without the guard this shape panics — and it is
// the shape workflow.Run and dynamic nodes pass, a CommonContextDelta carrying
// no InvocationContextDelta at all.
func TestDiscardWithNoInvocationDelta(t *testing.T) {
	freshReports(t)
	enclosing := &invocationContext{Context: t.Context(), agent: &agent{name: "parent"}}
	runID := "run-7"

	// Marked for the same reason as the test above: the shortcut would otherwise
	// return before withICDelta ever reaches the nil guard this test is named for.
	var c Context
	got := captureLog(t, func() {
		c = PromoteWithDelta(markedNilDeltaInvocation{InvocationContext: enclosing},
			&CommonContextDelta{RunID: &runID})
	})
	if got != "" {
		t.Errorf("log = %q, want silence — no invocation delta was asked for", got)
	}
	if c.RunID() != runID {
		t.Errorf("RunID() = %q, want %q", c.RunID(), runID)
	}
}
