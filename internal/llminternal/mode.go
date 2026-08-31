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

// Mode resolution.
//
// An agent that declares no Mode takes one from where it is placed: chat at a
// runner root, single_turn at a workflow node. State.Mode is the agent's own
// immutable declaration, so the resolved mode lives on the invocation instead —
// one agent instance serves many concurrent invocations, and the same instance
// may sit at two different placements.
//
// Readers that only test for task mode (basic_processor, outputschema_processor)
// deliberately stay on the declaration: no placement ever defaults to task, so
// there is nothing for them to resolve.
//
// The binding names the agent it describes. A bare context value would be
// inherited by every nested activation and would then govern an agent it was
// never resolved for: a peer reached by transfer, or a child that declares its
// own mode. Callers therefore bind per agent and read per agent.

// ResolveMode returns declared when set, else byPlacement.
func ResolveMode(declared, byPlacement Mode) Mode {
	if declared == ModeUnset {
		return byPlacement
	}
	return declared
}

type boundModeKey struct{}

type boundMode struct {
	agent string
	mode  Mode
}

// WithBoundMode returns ctx carrying the mode the named agent runs under for
// this invocation. Set by whatever places the agent: the runner for a root
// agent, an AgentNode for a graph node.
//
// An empty agentName is bound like any other. Config.Name is documented as
// required and nothing here can supply a missing one, but skipping the binding
// would silently turn a nameless agent at a graph node into a chat agent
// carrying the whole transcript. Binding "" leaves such an agent at the
// placement it was given, and leaves the missing name to whatever validates
// names.
func WithBoundMode(ctx context.Context, agentName string, mode Mode) context.Context {
	if mode == ModeUnset {
		return ctx
	}
	return context.WithValue(ctx, boundModeKey{}, boundMode{agent: agentName, mode: mode})
}

// BoundMode reports the mode this invocation bound to agentName, and whether it
// bound one at all. A binding made for a different agent does not count.
//
// Use this only to ask "did a placement put THIS agent in that mode" — a
// declared mode is deliberately not consulted. Callers wanting the mode an
// agent actually runs under want [ModeFor].
func BoundMode(ctx context.Context, agentName string) (Mode, bool) {
	b, ok := ctx.Value(boundModeKey{}).(boundMode)
	if !ok || b.agent != agentName {
		return ModeUnset, false
	}
	return b.mode, true
}

// ModeFor returns the mode agentName runs under: the mode this invocation bound
// to it, else its own declaration.
func ModeFor(ctx context.Context, agentName string, declared Mode) Mode {
	if m, ok := BoundMode(ctx, agentName); ok {
		return m
	}
	return declared
}
