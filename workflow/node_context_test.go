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

package workflow

import (
	"testing"

	"google.golang.org/adk/agent"
)

// These tests pin the workflow-node surface of the unified
// agent.Context (built via agent.NewNodeContext): ResumedInput, Path,
// and RunID. They replace the former tests of the removed
// workflow.nodeContext type.

func TestNodeContext_ResumedInput(t *testing.T) {
	parent := newMockCtx(t)

	t.Run("nil resumeInputs returns (nil, false)", func(t *testing.T) {
		c := agent.NewNodeContext(parent, "", "", nil, nil, nil)
		v, ok := c.ResumedInput("any_id")
		if v != nil || ok {
			t.Errorf("ResumedInput() = (%v, %v), want (nil, false)", v, ok)
		}
	})

	t.Run("populated resumeInputs returns matched payload", func(t *testing.T) {
		c := agent.NewNodeContext(parent, "", "", map[string]any{
			"approval": "yes",
			"comment":  "looks good",
		}, nil, nil)
		if v, ok := c.ResumedInput("approval"); !ok || v != "yes" {
			t.Errorf("ResumedInput(\"approval\") = (%v, %v), want (\"yes\", true)", v, ok)
		}
	})

	t.Run("unmatched InterruptID returns (nil, false)", func(t *testing.T) {
		c := agent.NewNodeContext(parent, "", "", map[string]any{"approval": "yes"}, nil, nil)
		if v, ok := c.ResumedInput("missing"); v != nil || ok {
			t.Errorf("ResumedInput(\"missing\") = (%v, %v), want (nil, false)", v, ok)
		}
	})
}

func TestNodeContext_PathAndRunID(t *testing.T) {
	t.Run("top-level static returns empty", func(t *testing.T) {
		c := agent.NewNodeContext(newMockCtx(t), "", "", nil, nil, nil)
		if got := c.Path(); got != "" {
			t.Errorf("Path() = %q, want empty", got)
		}
		if got := c.RunID(); got != "" {
			t.Errorf("RunID() = %q, want empty", got)
		}
	})

	t.Run("child populated from constructor", func(t *testing.T) {
		child := agent.NewNodeContext(newMockCtx(t), "wf/fixer@2", "2", nil, nil, nil)
		if got, want := child.Path(), "wf/fixer@2"; got != want {
			t.Errorf("Path() = %q, want %q", got, want)
		}
		if got, want := child.RunID(), "2"; got != want {
			t.Errorf("RunID() = %q, want %q", got, want)
		}
	})

	t.Run("activation populated with empty runID", func(t *testing.T) {
		act := agent.NewNodeContext(newMockCtx(t), "city_workflow", "", nil, nil, nil)
		if got, want := act.Path(), "city_workflow"; got != want {
			t.Errorf("Path() = %q, want %q", got, want)
		}
		if got := act.RunID(); got != "" {
			t.Errorf("RunID() = %q, want empty for dynamic-node activation", got)
		}
	})
}

func TestNodeContext_SchedulerStored(t *testing.T) {
	sub := &dynamicSubScheduler{}
	c := agent.NewNodeContext(newMockCtx(t), "wf/asker@1", "1",
		map[string]any{"approval": "yes"}, sub, nil)

	if v, ok := c.ResumedInput("approval"); !ok || v != "yes" {
		t.Errorf("ResumedInput(\"approval\") = (%v, %v), want (\"yes\", true)", v, ok)
	}
	// The scheduler is carried as an opaque token, recovered via the
	// concrete NodeScheduler() any accessor (not on the Context
	// interface) — the same way RunNode reaches it.
	ns, ok := c.(interface{ NodeScheduler() any })
	if !ok {
		t.Fatalf("node context does not expose NodeScheduler() any")
	}
	if ns.NodeScheduler() != any(sub) {
		t.Errorf("NodeScheduler() token differs from supplied")
	}
}
