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

package llminternal

import (
	"context"
	"testing"
)

func TestResolveMode(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		declared    Mode
		byPlacement Mode
		want        Mode
	}{
		{name: "undeclared takes the placement", declared: ModeUnset, byPlacement: ModeSingleTurn, want: ModeSingleTurn},
		{name: "declaration beats the placement", declared: ModeChat, byPlacement: ModeSingleTurn, want: ModeChat},
		{name: "declared task beats a chat placement", declared: ModeTask, byPlacement: ModeChat, want: ModeTask},
		{name: "no placement leaves it unset", declared: ModeUnset, byPlacement: ModeUnset, want: ModeUnset},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := ResolveMode(tc.declared, tc.byPlacement); got != tc.want {
				t.Errorf("ResolveMode(%q, %q) = %q, want %q", tc.declared, tc.byPlacement, got, tc.want)
			}
		})
	}
}

// The binding is keyed by agent, and the two properties that follow from that
// are what the whole design rests on. Neither is otherwise reachable from a
// test: no in-tree path binds two different values for one name, and no in-tree
// path reads a binding for an agent that is not the one running.
func TestBoundMode_Scoping(t *testing.T) {
	t.Parallel()

	t.Run("a binding is invisible to another agent", func(t *testing.T) {
		ctx := WithBoundMode(t.Context(), "worker", ModeSingleTurn)
		if _, ok := BoundMode(ctx, "other"); ok {
			t.Error("a binding made for \"worker\" was visible to \"other\"")
		}
		if got := ModeFor(ctx, "other", ModeChat); got != ModeChat {
			t.Errorf("ModeFor(other) = %q, want the declaration %q", got, ModeChat)
		}
	})

	t.Run("binding one agent leaves another's alone", func(t *testing.T) {
		ctx := WithBoundMode(t.Context(), "worker", ModeSingleTurn)
		ctx = WithBoundMode(ctx, "helper", ModeChat)

		// This is the transfer round trip in miniature: helper's binding must
		// not erase worker's, or worker loses its placement on re-entry.
		if got, ok := BoundMode(ctx, "worker"); !ok || got != ModeSingleTurn {
			t.Errorf("BoundMode(worker) = (%q, %v) after binding helper, want (%q, true)", got, ok, ModeSingleTurn)
		}
		if got, ok := BoundMode(ctx, "helper"); !ok || got != ModeChat {
			t.Errorf("BoundMode(helper) = (%q, %v), want (%q, true)", got, ok, ModeChat)
		}
	})

	t.Run("re-binding the same agent shadows the outer binding", func(t *testing.T) {
		outer := WithBoundMode(t.Context(), "worker", ModeChat)
		inner := WithBoundMode(outer, "worker", ModeSingleTurn)

		if got, ok := BoundMode(inner, "worker"); !ok || got != ModeSingleTurn {
			t.Errorf("BoundMode on the inner context = (%q, %v), want (%q, true) — "+
				"a re-binding placement must win over the one it nests inside", got, ok, ModeSingleTurn)
		}
		// Shadowing, not mutation: the outer context is a separate value and
		// still answers with what it was given.
		if got, ok := BoundMode(outer, "worker"); !ok || got != ModeChat {
			t.Errorf("BoundMode on the outer context = (%q, %v), want (%q, true) — "+
				"an inner bind must not reach back into the context it derived from", got, ok, ModeChat)
		}
	})

	t.Run("an unset mode does not bind", func(t *testing.T) {
		ctx := WithBoundMode(t.Context(), "worker", ModeUnset)
		if _, ok := BoundMode(ctx, "worker"); ok {
			t.Error("ModeUnset produced a binding; BoundMode would then report a placement that resolved nothing")
		}
		if got := ModeFor(ctx, "worker", ModeTask); got != ModeTask {
			t.Errorf("ModeFor = %q, want the declaration %q", got, ModeTask)
		}
	})

	t.Run("an empty name binds like any other", func(t *testing.T) {
		ctx := WithBoundMode(t.Context(), "", ModeSingleTurn)
		if got, ok := BoundMode(ctx, ""); !ok || got != ModeSingleTurn {
			t.Errorf("BoundMode(\"\") = (%q, %v), want (%q, true)", got, ok, ModeSingleTurn)
		}
		if _, ok := BoundMode(ctx, "worker"); ok {
			t.Error("a binding for the empty name was visible to a named agent")
		}
	})
}

func TestModeFor(t *testing.T) {
	t.Parallel()

	t.Run("the binding wins over the declaration", func(t *testing.T) {
		ctx := WithBoundMode(t.Context(), "worker", ModeSingleTurn)
		if got := ModeFor(ctx, "worker", ModeChat); got != ModeSingleTurn {
			t.Errorf("ModeFor = %q, want %q", got, ModeSingleTurn)
		}
	})

	t.Run("the declaration is the fallback", func(t *testing.T) {
		if got := ModeFor(context.Background(), "worker", ModeChat); got != ModeChat {
			t.Errorf("ModeFor with no binding = %q, want %q", got, ModeChat)
		}
	})
}
