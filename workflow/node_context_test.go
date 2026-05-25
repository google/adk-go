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
	"context"
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
	// Today Path and RunID always return "" — the dynamic sub-scheduler
	// (future change) will populate them for dynamic children. These
	// tests pin the current contract so a regression that accidentally
	// changes the default is caught early.
	c := newNodeContext(newMockCtx(t), nil)
	if got := c.Path(); got != "" {
		t.Errorf("Path() = %q, want empty for top-level static activation", got)
	}
	if got := c.RunID(); got != "" {
		t.Errorf("RunID() = %q, want empty for top-level static activation", got)
	}
}

func TestNodeContext_WithGoCtx(t *testing.T) {
	parent := newMockCtx(t)
	resume := map[string]any{"approval": "yes"}
	c := newNodeContext(parent, resume)

	// Swap the underlying context.Context.
	swappedCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	swapped := c.WithGoCtx(swappedCtx)

	t.Run("returns a new instance", func(t *testing.T) {
		// Must not return the same pointer — WithGoCtx is documented
		// as returning a new NodeContext.
		if any(swapped) == any(c) {
			t.Errorf("WithGoCtx returned the same NodeContext; want a new instance")
		}
	})

	t.Run("underlying ctx is swapped", func(t *testing.T) {
		// InvocationContext embeds context.Context directly. We can't
		// pointer-compare interfaces, so we verify the swap via
		// behavior: cancelling swappedCtx must propagate to the new
		// NodeContext's Done channel; the parent's must remain open.
		cancel()
		select {
		case <-swapped.Done():
			// expected
		default:
			t.Errorf("WithGoCtx-returned NodeContext did not observe ctx cancellation; ctx was not swapped")
		}
		// Parent's ctx (t.Context) must NOT have been cancelled.
		select {
		case <-c.Done():
			t.Errorf("original NodeContext was cancelled; WithGoCtx must return a new instance, not mutate the original")
		default:
			// expected
		}
	})

	t.Run("resume inputs preserved", func(t *testing.T) {
		if v, ok := swapped.ResumedInput("approval"); !ok || v != "yes" {
			t.Errorf("after WithGoCtx, ResumedInput(\"approval\") = (%v, %v), want (\"yes\", true)", v, ok)
		}
	})

	t.Run("invocation metadata preserved", func(t *testing.T) {
		if got := swapped.InvocationID(); got != parent.InvocationID() {
			t.Errorf("InvocationID after WithGoCtx = %q, want %q", got, parent.InvocationID())
		}
	})

	t.Run("Path and RunID still empty after swap", func(t *testing.T) {
		if got := swapped.Path(); got != "" {
			t.Errorf("Path after WithGoCtx = %q, want empty", got)
		}
		if got := swapped.RunID(); got != "" {
			t.Errorf("RunID after WithGoCtx = %q, want empty", got)
		}
	})
}
