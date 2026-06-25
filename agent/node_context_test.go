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
	"testing"

	"google.golang.org/adk/agent"
)

// stubSubScheduler is a no-op DynamicSubScheduler used to verify that
// NewDynamicNodeContext stores and returns the exact scheduler it is given.
type stubSubScheduler struct{}

func (*stubSubScheduler) RunNode(_, _, _ any) (any, error)           { return nil, nil }
func (*stubSubScheduler) ParentPath() string                         { return "" }
func (*stubSubScheduler) OutputForAncestors() []string               { return nil }
func (*stubSubScheduler) DelegatedOutput() (any, bool)               { return nil, false }
func (*stubSubScheduler) ResolveByRunID(_, _ string) (string, error) { return "", nil }

// newNodeBaseCtx returns a minimal InvocationContext suitable as the parent for
// the node-context constructors. These construction tests only read fields that
// live on the node context itself, never on this base.
func newNodeBaseCtx(t *testing.T) InvocationContext {
	t.Helper()
	return &invocationContext{Context: t.Context()}
}

func TestNodeContext_ResumedInput(t *testing.T) {
	parent := newNodeBaseCtx(t)

	t.Run("nil resumeInputs returns (nil, false)", func(t *testing.T) {
		c := agent.NewContext(parent)
		v, ok := c.ResumedInput("any_id")
		if v != nil || ok {
			t.Errorf("ResumedInput() = (%v, %v), want (nil, false)", v, ok)
		}
	})

	t.Run("populated resumeInputs returns matched payload", func(t *testing.T) {
		c := NewNodeContext(parent, map[string]any{
			"approval": "yes",
			"comment":  "looks good",
		})
		if v, ok := c.ResumedInput("approval"); !ok || v != "yes" {
			t.Errorf("ResumedInput(%q) = (%v, %v), want (%q, true)", "approval", v, ok, "yes")
		}
	})

	t.Run("unmatched InterruptID returns (nil, false)", func(t *testing.T) {
		c := NewNodeContext(parent, map[string]any{"approval": "yes"})
		if v, ok := c.ResumedInput("missing"); v != nil || ok {
			t.Errorf("ResumedInput(%q) = (%v, %v), want (nil, false)", "missing", v, ok)
		}
	})
}

func TestNodeContext_PathAndRunID(t *testing.T) {
	t.Run("top-level static returns empty", func(t *testing.T) {
		c := agent.NewContext(newMockCtx(t))
		if got := c.Path(); got != "" {
			t.Errorf("Path() = %q, want empty", got)
		}
		if got := c.RunID(); got != "" {
			t.Errorf("RunID() = %q, want empty", got)
		}
	})

	t.Run("child populated from constructor", func(t *testing.T) {
		parent := agent.NewContext(newMockCtx(t))
		child := agent.NewDynamicNodeContext(parent, "wf/fixer@2", "2", nil, nil)
		if got, want := child.Path(), "wf/fixer@2"; got != want {
			t.Errorf("Path() = %q, want %q", got, want)
		}
		if got, want := child.RunID(), "2"; got != want {
			t.Errorf("RunID() = %q, want %q", got, want)
		}
	})

	t.Run("activation populated with empty runID", func(t *testing.T) {
		parent := agent.NewContext(newMockCtx(t))
		act := agent.NewDynamicNodeContext(parent, "city_workflow", "", nil, nil)
		if got, want := act.Path(), "city_workflow"; got != want {
			t.Errorf("Path() = %q, want %q", got, want)
		}
		if got := act.RunID(); got != "" {
			t.Errorf("RunID() = %q, want empty for dynamic-node activation", got)
		}
	})
}

func TestNodeContext_DynamicInheritsResumeInputs(t *testing.T) {
	parent := NewNodeContext(newNodeBaseCtx(t), map[string]any{"approval": "yes"})
	sub := &stubSubScheduler{}

	t.Run("child", func(t *testing.T) {
		child := NewDynamicNodeContext(parent, "wf/asker@1", "1", sub, nil)
		if v, ok := child.ResumedInput("approval"); !ok || v != "yes" {
			t.Errorf("child.ResumedInput(%q) = (%v, %v), want (%q, true)", "approval", v, ok, "yes")
		}
		if child.SubScheduler() != sub {
			t.Errorf("SubScheduler() differs from the supplied scheduler")
		}
	})

	t.Run("activation", func(t *testing.T) {
		act := NewDynamicNodeContext(parent, "city_workflow", "", sub, nil)
		if v, ok := act.ResumedInput("approval"); !ok || v != "yes" {
			t.Errorf("act.ResumedInput(%q) = (%v, %v), want (%q, true)", "approval", v, ok, "yes")
		}
		if act.SubScheduler() != sub {
			t.Errorf("SubScheduler() differs from the supplied scheduler")
		}
	})
}
