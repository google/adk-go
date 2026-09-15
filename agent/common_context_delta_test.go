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
)

// nilDeltaInvocation returns nil from WithICDelta rather than a derived
// invocation — the shape an implementation written outside this repository
// takes. Nothing in the repository does it, which is why the fixture has to.
type nilDeltaInvocation struct {
	InvocationContext
}

func (nilDeltaInvocation) WithICDelta(*InvocationContextDelta) InvocationContext { return nil }

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
	}
	ic := nilDeltaInvocation{InvocationContext: enclosing}

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
		if !strings.Contains(got, "this delta is discarded") {
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
		{"agent", &InvocationContextDelta{Agent: &child}, "(Agent)"},
		{"branch", &InvocationContextDelta{Branch: &branch}, `(Branch="child-branch")`},
		{"isolation scope", &InvocationContextDelta{IsolationScope: &scope}, `(IsolationScope="child-scope")`},
		{"user content", &InvocationContextDelta{UserContent: &content}, "(UserContent)"},
		{
			"every reported field at once",
			&InvocationContextDelta{Agent: &child, Branch: &branch, IsolationScope: &scope, UserContent: &content},
			`(Agent, Branch="child-branch", IsolationScope="child-scope", UserContent)`,
		},
		{
			// WithDelta installs Context on the context it returns whether or not
			// the invocation took the delta, so it is not lost on this path.
			"context applied by WithDelta is not named",
			&InvocationContextDelta{Agent: &child, Context: &newCtx},
			"(Agent)",
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

// TestDiscardReportIsEmittedOnce pins the deduplication itself. Deleting the
// LoadOrStore guard leaves every other test in this file green.
func TestDiscardReportIsEmittedOnce(t *testing.T) {
	freshReports(t)
	enclosing := &invocationContext{Context: t.Context(), agent: &agent{name: "parent"}}
	ic := nilDeltaInvocation{InvocationContext: enclosing}
	branch := "child-branch"
	out := captureLog(t, func() {
		for range 3 {
			_ = PromoteWithDelta(ic, &CommonContextDelta{
				InvocationContextDelta: &InvocationContextDelta{Branch: &branch},
			})
		}
	})
	if got := strings.Count(out, "returned nil"); got != 1 {
		t.Errorf("three discards produced %d report(s), want 1:\n%s", got, out)
	}
}

// TestDiscardWithNothingToReport pins that a delta which asked for nothing
// neither reports nor spends the type's one report. Claiming the type before
// building the field list made an empty delta silence the next real loss.
func TestDiscardWithNothingToReport(t *testing.T) {
	freshReports(t)
	enclosing := &invocationContext{Context: t.Context(), agent: &agent{name: "parent"}}
	ic := nilDeltaInvocation{InvocationContext: enclosing}

	empty := captureLog(t, func() {
		_ = PromoteWithDelta(ic, &CommonContextDelta{InvocationContextDelta: &InvocationContextDelta{}})
	})
	if empty != "" {
		t.Errorf("a delta that asked for nothing logged %q, want silence", empty)
	}

	branch := "child-branch"
	real := captureLog(t, func() {
		_ = PromoteWithDelta(ic, &CommonContextDelta{
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

// TestDeltaReachesTheInvocation pins that the guard above does not cost a
// working invocation its delta. A guard that kept the original unconditionally
// would pass every assertion in the test above.
func TestDeltaReachesTheInvocation(t *testing.T) {
	ic := &invocationContext{
		Context: t.Context(),
		agent:   &agent{name: "parent"},
		branch:  "parent-branch",
	}
	var child Agent = &agent{name: "child"}
	branch := "child-branch"

	var c Context
	// Silence matters as much as the values: a helper that reported on every
	// derivation, not only on a discard, would satisfy every other assertion here.
	if out := captureLog(t, func() {
		c = PromoteWithDelta(ic, &CommonContextDelta{
			InvocationContextDelta: &InvocationContextDelta{Agent: &child, Branch: &branch},
		})
	}); out != "" {
		t.Errorf("a delta the invocation accepted logged %q, want silence", out)
	}
	if got := c.(InvocationContext).Agent(); got == nil || got.Name() != "child" {
		t.Errorf("Agent() = %v, want the agent the delta named", got)
	}
	if got := c.Branch(); got != branch {
		t.Errorf("Branch() = %q, want %q", got, branch)
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

// TestDiscardNamesContextOnlyWhereItIsLost pins the asymmetry between the two
// entry points. WithDelta installs d.Context on the context it returns, so the
// caller keeps it and it is not lost. WithICDelta installs nothing, so a
// discarded delta costs it and the report has to say so.
func TestDiscardNamesContextOnlyWhereItIsLost(t *testing.T) {
	enclosing := &invocationContext{Context: t.Context(), agent: &agent{name: "parent"}}
	ic := nilDeltaInvocation{InvocationContext: enclosing}
	type key struct{}
	newCtx := context.WithValue(context.Background(), key{}, "from-the-delta")

	t.Run("WithDelta applies it, so it is not named", func(t *testing.T) {
		freshReports(t)
		var c Context
		got := captureLog(t, func() {
			c = PromoteWithDelta(ic, &CommonContextDelta{
				InvocationContextDelta: &InvocationContextDelta{Context: &newCtx},
			})
		})
		if v := c.Value(key{}); v != "from-the-delta" {
			t.Fatalf("Value(key) = %v, want the delta's context to have been applied", v)
		}
		if got != "" {
			t.Errorf("log = %q, want silence — Context was applied, so nothing was lost", got)
		}
	})

	t.Run("WithICDelta does not, so it is named", func(t *testing.T) {
		freshReports(t)
		got := captureLog(t, func() {
			_ = Promote(ic).WithICDelta(&InvocationContextDelta{Context: &newCtx})
		})
		if !strings.Contains(got, "(Context)") {
			t.Errorf("log = %q, want Context named as lost on this path", got)
		}
	})
}
