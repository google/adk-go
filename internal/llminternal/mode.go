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
// Three kinds of reader deliberately stay on the declaration.
//
// Those that only test for task mode — basic_processor, outputschema_processor,
// AgentNode's synthesizeMode, the runner's task-sub-agent scan — have nothing to
// resolve, because no placement ever defaults to task.
//
// isUntransferableMode and installTaskTools both branch on single_turn, and
// both are right to read the declaration, for the same reason: each is deciding
// something about the shape of the agent TREE — whether a peer is reachable by
// transfer, which tool a parent installs for a sub-agent — not about how one
// agent behaves on one run. A placement is per-run and cannot answer either.
//
// Note that "there is no binding to consult" would be the wrong reason for
// isUntransferableMode. Bindings nest, so when a callee computes its transfer
// targets the caller's binding is still live in that context. It is looked past
// deliberately.
//
// Every other reader resolves. A reader that tests for single_turn and skips
// the resolution disagrees with the request the flow actually builds.
//
// The binding names the agent it describes, and the name is part of the context
// KEY rather than of the value. Two properties follow, and both are needed.
//
// A binding never governs an agent it was not resolved for. A bare context value
// would be inherited by every nested activation — a peer reached by transfer, a
// child that declares its own mode — and would govern all of them.
//
// A binding also survives a nested activation that binds a different agent.
// Storing one pair under one key gave only the first property: a single_turn
// graph node that transferred to a chat peer had its slot overwritten, and when
// the peer transferred back the node's own agent was re-entered on the peer's
// context and fell back to chat, taking the identity preamble, the transfer
// instructions and the whole conversation with it. Placements nest, so the
// bindings have to nest too.

// The two values agent/llmagent's IncludeContents constants carry. Duplicated
// as untyped constants because llminternal cannot import llmagent, which
// depends on it.
const (
	includeContentsNone    = "none"
	includeContentsDefault = "default"
)

// ResolveMode returns declared when set, else byPlacement.
func ResolveMode(declared, byPlacement Mode) Mode {
	if declared == ModeUnset {
		return byPlacement
	}
	return declared
}

// boundModeKey is the context key for one agent's binding. The agent name is
// part of the key, so each agent gets its own slot and binding one cannot
// disturb another's.
type boundModeKey struct{ agent string }

// WithBoundMode returns ctx carrying the mode the named agent runs under for
// this invocation. Set by whatever places the agent: the runner for a root
// agent, an AgentNode for a graph node.
//
// Binding an agent that already has one shadows it for the rest of that
// context, which is what a re-entrant placement of the same agent should do.
// Binding a different agent leaves the first alone.
//
// The key is the name alone, so the binding is only as well-scoped as names are
// unique within one context chain. runner.New rejects a duplicate name anywhere
// in its agent tree and workflow.New rejects two nodes sharing one, but neither
// covers a node agent's own descendants, so two distinct same-named agents
// nested inside one placement would share a slot. Every other name-keyed lookup
// in the framework — FindAgent, findAgentToRun, transfer targets — has the same
// precondition.
//
// An empty agentName is bound like any other. Config.Name is documented as
// required and nothing here can supply a missing one, but skipping the binding
// would silently turn a nameless agent at a graph node into a chat agent
// carrying the whole transcript. Binding "" leaves such an agent at the
// placement it was given, and leaves the missing name to whatever validates
// names.
//
// ctx must be non-nil, as for any context helper.
//
// A ModeUnset mode returns ctx untouched. That is a guard against recording a
// placement that resolved nothing, not a way to decline to bind: every binder
// passes a ResolveMode whose fallback is concrete, so it is unreachable today.
// There is deliberately no way to CLEAR a binding — a nested placement shadows
// an outer one by binding its own value, and nothing needs to un-place an agent.
func WithBoundMode(ctx context.Context, agentName string, mode Mode) context.Context {
	if mode == ModeUnset {
		return ctx
	}
	return context.WithValue(ctx, boundModeKey{agent: agentName}, mode)
}

// BoundMode reports the mode this invocation bound to agentName, and whether it
// bound one at all. A binding made for a different agent does not count.
//
// Use this only to ask "did a placement put THIS agent in that mode" — a
// declared mode is deliberately not consulted. Callers wanting the mode an
// agent actually runs under want [ModeFor].
//
// ctx must be non-nil. Passing nil panics in ctx.Value, as it would for any
// context helper, so callers holding a context that may be nil check it
// themselves rather than relying on this.
func BoundMode(ctx context.Context, agentName string) (Mode, bool) {
	mode, ok := ctx.Value(boundModeKey{agent: agentName}).(Mode)
	if !ok {
		return ModeUnset, false
	}
	return mode, true
}

// ModeFor returns the mode agentName runs under: the mode this invocation bound
// to it, else its own declaration.
func ModeFor(ctx context.Context, agentName string, declared Mode) Mode {
	if m, ok := BoundMode(ctx, agentName); ok {
		return m
	}
	return declared
}
