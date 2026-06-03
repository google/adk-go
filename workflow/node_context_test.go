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
)

func TestNodeContext_ResumedInput(t *testing.T) {
	parent := newMockCtx(t)

	t.Run("nil resumeInputs returns (nil, false)", func(t *testing.T) {
		c := newNodeContext(parent, nil)
		v, ok := c.ResumedInput("any_id")
		if v != nil || ok {
			t.Errorf("ResumedInput() = (%v, %v), want (nil, false)", v, ok)
		}
	})

	t.Run("populated resumeInputs returns matched payload", func(t *testing.T) {
		c := newNodeContext(parent, map[string]any{
			"approval": "yes",
			"comment":  "looks good",
		})
		if v, ok := c.ResumedInput("approval"); !ok || v != "yes" {
			t.Errorf("ResumedInput(\"approval\") = (%v, %v), want (\"yes\", true)", v, ok)
		}
	})

	t.Run("unmatched InterruptID returns (nil, false)", func(t *testing.T) {
		c := newNodeContext(parent, map[string]any{"approval": "yes"})
		if v, ok := c.ResumedInput("missing"); v != nil || ok {
			t.Errorf("ResumedInput(\"missing\") = (%v, %v), want (nil, false)", v, ok)
		}
	})
}

func TestNodeContext_PathAndRunID(t *testing.T) {
	t.Run("top-level static returns empty", func(t *testing.T) {
		c := newNodeContext(newMockCtx(t), nil)
		if got := c.Path(); got != "" {
			t.Errorf("Path() = %q, want empty", got)
		}
		if got := c.RunID(); got != "" {
			t.Errorf("RunID() = %q, want empty", got)
		}
	})

	t.Run("child populated from constructor", func(t *testing.T) {
		parent := newNodeContext(newMockCtx(t), nil)
		child := newDynamicNodeContext(parent, "wf/fixer@2", "2", nil)
		if got, want := child.Path(), "wf/fixer@2"; got != want {
			t.Errorf("Path() = %q, want %q", got, want)
		}
		if got, want := child.RunID(), "2"; got != want {
			t.Errorf("RunID() = %q, want %q", got, want)
		}
	})

	t.Run("activation populated with empty runID", func(t *testing.T) {
		parent := newNodeContext(newMockCtx(t), nil)
		act := newDynamicNodeContext(parent, "city_workflow", "", nil)
		if got, want := act.Path(), "city_workflow"; got != want {
			t.Errorf("Path() = %q, want %q", got, want)
		}
		if got := act.RunID(); got != "" {
			t.Errorf("RunID() = %q, want empty for dynamic-node activation", got)
		}
	})
}

func TestNodeContext_DynamicInheritsResumeInputs(t *testing.T) {
	parent := newNodeContext(newMockCtx(t), map[string]any{"approval": "yes"})
	sub := &dynamicSubScheduler{}

	t.Run("child", func(t *testing.T) {
		child := newDynamicNodeContext(parent, "wf/asker@1", "1", sub)
		if v, ok := child.ResumedInput("approval"); !ok || v != "yes" {
			t.Errorf("child.ResumedInput(\"approval\") = (%v, %v), want (\"yes\", true)", v, ok)
		}
		if child.subScheduler != sub {
			t.Errorf("subScheduler pointer differs from supplied")
		}
	})

	t.Run("activation", func(t *testing.T) {
		act := newDynamicNodeContext(parent, "city_workflow", "", sub)
		if v, ok := act.ResumedInput("approval"); !ok || v != "yes" {
			t.Errorf("act.ResumedInput(\"approval\") = (%v, %v), want (\"yes\", true)", v, ok)
		}
		if act.subScheduler != sub {
			t.Errorf("subScheduler pointer differs from supplied")
		}
	})
}
