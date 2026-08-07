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

import "context"

// An agent that declares no Mode takes one from where it is placed: chat
// as a runner root, single_turn as a workflow node. Placement belongs to
// the invocation and one agent instance serves many concurrent
// invocations, so both the placement default and the mode it resolves to
// travel on the context. State.Mode stays the agent's own declaration and
// is never written to after construction.

// ResolveMode returns declared when set, else byPlacement.
func ResolveMode(declared, byPlacement Mode) Mode {
	if declared == ModeUnset {
		return byPlacement
	}
	return declared
}

type (
	placementModeKey struct{}
	resolvedModeKey  struct{}
)

// WithPlacementMode returns ctx carrying the mode agents run under here
// unless they declare otherwise. Set by whatever places an agent: the
// runner for a root agent, an AgentNode for a graph node.
func WithPlacementMode(ctx context.Context, mode Mode) context.Context {
	return context.WithValue(ctx, placementModeKey{}, mode)
}

// PlacementMode returns the default imposed by ctx's placement, or
// fallback when it imposes none.
func PlacementMode(ctx context.Context, fallback Mode) Mode {
	if m, ok := ctx.Value(placementModeKey{}).(Mode); ok && m != ModeUnset {
		return m
	}
	return fallback
}

// WithResolvedMode returns ctx carrying the mode now in effect, so the
// request processors downstream agree with the agent runner on it.
func WithResolvedMode(ctx context.Context, mode Mode) context.Context {
	return context.WithValue(ctx, resolvedModeKey{}, mode)
}

// ResolvedMode returns the mode in effect on ctx, or fallback when the
// agent runs outside a path that resolves one.
func ResolvedMode(ctx context.Context, fallback Mode) Mode {
	if m, ok := ctx.Value(resolvedModeKey{}).(Mode); ok && m != ModeUnset {
		return m
	}
	return fallback
}
