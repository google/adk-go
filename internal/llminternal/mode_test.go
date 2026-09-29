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

// The binding is keyed by agent name AND agent identity, and the properties
// that follow are what the whole design rests on. Two of them are reachable
// only here: no in-tree path binds two different values for one name, and no
// in-tree path reads a binding for an agent that is not the one running.
// Nesting is the exception — TestAgentNode_PlacementSurvivesATransferRoundTrip
// drives it end to end — and the nesting subtest below is that round trip in
// miniature, kept because it isolates the property from everything else that
// test needs to be true.
func TestBoundMode_Scoping(t *testing.T) {
	t.Parallel()

	t.Run("a binding is invisible to another agent", func(t *testing.T) {
		worker, other := &State{}, &State{Mode: ModeChat}
		ctx := WithBoundMode(t.Context(), "worker", worker, ModeSingleTurn)
		if _, ok := BoundMode(ctx, "other", other); ok {
			t.Error("a binding made for \"worker\" was visible to \"other\"")
		}
		if got := ModeFor(ctx, "other", other); got != ModeChat {
			t.Errorf("ModeFor(other) = %q, want the declaration %q", got, ModeChat)
		}
	})

	t.Run("binding one agent leaves another's alone", func(t *testing.T) {
		worker, helper := &State{}, &State{}
		ctx := WithBoundMode(t.Context(), "worker", worker, ModeSingleTurn)
		ctx = WithBoundMode(ctx, "helper", helper, ModeChat)

		// This is the transfer round trip in miniature: helper's binding must
		// not erase worker's, or worker loses its placement on re-entry.
		if got, ok := BoundMode(ctx, "worker", worker); !ok || got != ModeSingleTurn {
			t.Errorf("BoundMode(worker) = (%q, %v) after binding helper, want (%q, true)", got, ok, ModeSingleTurn)
		}
		if got, ok := BoundMode(ctx, "helper", helper); !ok || got != ModeChat {
			t.Errorf("BoundMode(helper) = (%q, %v), want (%q, true)", got, ok, ModeChat)
		}
	})

	t.Run("re-binding the same agent shadows the outer binding", func(t *testing.T) {
		worker := &State{}
		outer := WithBoundMode(t.Context(), "worker", worker, ModeChat)
		inner := WithBoundMode(outer, "worker", worker, ModeSingleTurn)

		if got, ok := BoundMode(inner, "worker", worker); !ok || got != ModeSingleTurn {
			t.Errorf("BoundMode on the inner context = (%q, %v), want (%q, true) — "+
				"a re-binding placement must win over the one it nests inside", got, ok, ModeSingleTurn)
		}
		// Shadowing, not mutation: the outer context is a separate value and
		// still answers with what it was given.
		if got, ok := BoundMode(outer, "worker", worker); !ok || got != ModeChat {
			t.Errorf("BoundMode on the outer context = (%q, %v), want (%q, true) — "+
				"an inner bind must not reach back into the context it derived from", got, ok, ModeChat)
		}
	})

	// The collision the name key alone cannot resolve, and the reason the
	// binding carries a state pointer. Two DISTINCT agents share one name —
	// runner.New accepts that when one of them sits behind a graph node — and
	// the one that was never placed must keep its own declaration.
	t.Run("a binding does not reach a different agent of the same name", func(t *testing.T) {
		placed, nested := &State{}, &State{}
		ctx := WithBoundMode(t.Context(), "worker", placed, ModeSingleTurn)

		if _, ok := BoundMode(ctx, "worker", nested); ok {
			t.Error("a placement resolved for one agent was reported as governing a different agent of the same name")
		}
		if got := ModeFor(ctx, "worker", nested); got != ModeUnset {
			t.Errorf("ModeFor(nested) = %q, want its own (unset) declaration", got)
		}
		if got, ok := BoundMode(ctx, "worker", placed); !ok || got != ModeSingleTurn {
			t.Errorf("BoundMode(placed) = (%q, %v), want (%q, true) — the agent it WAS resolved for still sees it", got, ok, ModeSingleTurn)
		}
	})

	t.Run("an unset mode does not bind", func(t *testing.T) {
		worker := &State{Mode: ModeTask}
		ctx := WithBoundMode(t.Context(), "worker", worker, ModeUnset)
		if _, ok := BoundMode(ctx, "worker", worker); ok {
			t.Error("ModeUnset produced a binding; BoundMode would then report a placement that resolved nothing")
		}
		if got := ModeFor(ctx, "worker", worker); got != ModeTask {
			t.Errorf("ModeFor = %q, want the declaration %q", got, ModeTask)
		}
	})

	t.Run("an empty name binds like any other", func(t *testing.T) {
		nameless, worker := &State{}, &State{}
		ctx := WithBoundMode(t.Context(), "", nameless, ModeSingleTurn)
		if got, ok := BoundMode(ctx, "", nameless); !ok || got != ModeSingleTurn {
			t.Errorf("BoundMode(\"\") = (%q, %v), want (%q, true)", got, ok, ModeSingleTurn)
		}
		if _, ok := BoundMode(ctx, "worker", worker); ok {
			t.Error("a binding for the empty name was visible to a named agent")
		}
	})

	// Two NAMELESS agents are the case identity rescues and a name key cannot:
	// they share the empty-string slot entirely.
	t.Run("two nameless agents do not share a binding", func(t *testing.T) {
		placed, nested := &State{}, &State{}
		ctx := WithBoundMode(t.Context(), "", placed, ModeSingleTurn)
		if _, ok := BoundMode(ctx, "", nested); ok {
			t.Error("a nameless agent inherited a placement resolved for a different nameless agent")
		}
	})
}

func TestModeFor(t *testing.T) {
	t.Parallel()

	t.Run("the binding supplies a mode the agent did not declare", func(t *testing.T) {
		worker := &State{}
		ctx := WithBoundMode(t.Context(), "worker", worker, ModeSingleTurn)
		if got := ModeFor(ctx, "worker", worker); got != ModeSingleTurn {
			t.Errorf("ModeFor = %q, want %q", got, ModeSingleTurn)
		}
	})

	t.Run("the declaration is the fallback", func(t *testing.T) {
		if got := ModeFor(context.Background(), "worker", &State{Mode: ModeChat}); got != ModeChat {
			t.Errorf("ModeFor with no binding = %q, want %q", got, ModeChat)
		}
	})

	// A placement is authoritative for the agent it was resolved for, including
	// over that agent's own declaration — they agree in practice, since a binder
	// binds ResolveMode(declared, placementDefault), but the binding is what
	// knows where the agent was put.
	t.Run("the binding wins for the agent it was resolved for", func(t *testing.T) {
		worker := &State{Mode: ModeChat}
		ctx := WithBoundMode(t.Context(), "worker", worker, ModeSingleTurn)
		if got := ModeFor(ctx, "worker", worker); got != ModeSingleTurn {
			t.Errorf("ModeFor = %q, want the bound %q", got, ModeSingleTurn)
		}
	})

	// ...and carries no weight for anyone else, whatever they declare. Without
	// this, a chat root placed a same-named nested single_turn agent into chat,
	// and a single_turn graph node placed a same-named nested undeclared agent
	// into single_turn.
	t.Run("another agent's binding loses to this agent's declaration", func(t *testing.T) {
		placed := &State{}
		// Each arm binds a mode the nested agent did NOT declare, so an arm
		// cannot pass merely because the two happen to coincide.
		for _, tc := range []struct{ declared, bound Mode }{
			{ModeUnset, ModeSingleTurn},
			{ModeChat, ModeSingleTurn},
			{ModeSingleTurn, ModeTask},
			{ModeTask, ModeSingleTurn},
		} {
			nested := &State{Mode: tc.declared}
			ctx := WithBoundMode(t.Context(), "worker", placed, tc.bound)
			if got := ModeFor(ctx, "worker", nested); got != tc.declared {
				t.Errorf("ModeFor with declared %q and another agent bound %q = %q, want %q",
					tc.declared, tc.bound, got, tc.declared)
			}
		}
	})

	// A placement resolved for one agent must not be DESTROYED by a placement
	// for a different agent of the same name either. Identity in the context
	// key gives them separate slots; with identity only in the value, the
	// second bind replaced the first and its owner silently fell back to its
	// declaration.
	t.Run("a same-named agent's binding does not displace ours", func(t *testing.T) {
		ours, theirs := &State{}, &State{}
		ctx := WithBoundMode(t.Context(), "worker", ours, ModeSingleTurn)
		ctx = WithBoundMode(ctx, "worker", theirs, ModeTask)

		if got, ok := BoundMode(ctx, "worker", ours); !ok || got != ModeSingleTurn {
			t.Errorf("BoundMode(ours) = (%q, %v), want (%q, true)", got, ok, ModeSingleTurn)
		}
		if got, ok := BoundMode(ctx, "worker", theirs); !ok || got != ModeTask {
			t.Errorf("BoundMode(theirs) = (%q, %v), want (%q, true)", got, ok, ModeTask)
		}
	})
}
