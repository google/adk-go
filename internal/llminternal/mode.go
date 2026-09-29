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
// targets the caller's binding is still live in that context, and it is looked
// past deliberately, and it has to be. A transfer up to a parent has the caller
// filtered as one of the callee's sub-agents, and a transfer to a peer has it
// filtered as a peer, so a resolving isUntransferableMode would drop a placed
// undeclared agent off a callee's target list on the strength of a placement
// made for somebody else's activation.
//
// Every other reader resolves, with one deliberate exception. A reader that
// tests for single_turn and skips the resolution disagrees with the request the
// flow actually builds.
//
// The exception is the history half of the contents processor, which asks
// BoundMode directly rather than going through ModeFor. It is the one reader
// that has to tell a single_turn PLACEMENT from a single_turn DECLARATION —
// only the placement hides the conversation — and resolving flattens exactly
// that distinction. Its other half, the single-turn nudge, resolves like
// everything else.
//
// The binding names the agent it describes, and the name is part of the context
// KEY rather than of the value. Two properties follow, and both are needed.
//
// A binding never governs an agent it was not resolved for. A bare context value
// would be inherited by every nested activation — a peer reached by transfer, a
// child that declares its own mode — and would govern all of them.
//
// A binding also survives a nested activation that binds a different agent.
// Placements nest — a single_turn graph node transfers to a chat peer, which
// transfers back — so the bindings have to nest too, which one pair under one
// shared key does not do.
//
// The name alone is not enough to say WHOSE binding one is. Names are unique
// only across SubAgents(), and a graph node's agent is not in SubAgents(), so
// two distinct same-named agents can share one context chain and runner.New
// will not reject the tree. The agent's identity is therefore part of the
// context key, so each gets a slot of its own: a reader never sees another
// agent's placement, and never loses its own to one.
//
// Identity has to be the *State rather than the agent.Agent: Reveal hands every
// binder and every reader the same pointer for one agent, a pointer is always
// comparable, and an agent.Agent in a context key would panic on an
// implementation whose dynamic type is not.

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

// boundModeKey is the context key for one agent's binding. BOTH the name and
// the agent's identity are part of the key, and each does a different job.
//
// Identity keeps a binding from reaching an agent it was not resolved for, and
// keeps a placement resolved for one agent from being destroyed by a placement
// for a same-named other: the two occupy different slots rather than one
// overwriting the other. Putting identity only in the VALUE gave the first
// property and not the second — the second bind still replaced the first in the
// lookup chain, and the reader then rejected it, so the agent that owned it
// silently lost its placement.
//
// The name adds nothing to a lookup: every binder and reader pairs an agent's
// name with that same agent's state, so it is a function of the identity. It is
// kept because a future call site that pairs them inconsistently would be a
// wrong HIT against a state-only key — one agent governed by another's resolved
// mode — and the name turns that into a miss, which is the safer failure.
type boundModeKey struct {
	agent string
	state *State
}

// WithBoundMode returns ctx carrying the mode the named agent runs under for
// this invocation. Set by whatever places the agent: the runner for a root
// agent, an AgentNode for a graph node.
//
// Binding an agent that already has one shadows it for the rest of that
// context, which is what a re-entrant placement of the same agent should do.
// Binding a different agent leaves the first alone.
//
// state identifies the agent the mode was resolved for, and is what makes a
// same-named agent nested inside this placement fall back to its own
// declaration instead of inheriting this one. runner.New rejects a duplicate
// name anywhere in its agent tree and workflow.New rejects two nodes sharing
// one, but neither covers a node agent's own descendants — so the collision is
// constructible, and the name in the key cannot resolve it on its own.
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
func WithBoundMode(ctx context.Context, agentName string, state *State, mode Mode) context.Context {
	if mode == ModeUnset {
		return ctx
	}
	return context.WithValue(ctx, boundModeKey{agent: agentName, state: state}, mode)
}

// BoundMode reports the mode this invocation bound for the agent identified by
// state, and whether it bound one at all. A binding made for a different agent
// does not count, including one made for a DIFFERENT agent of the same name.
//
// Use this only to ask "did a placement put THIS agent in that mode" — a
// declared mode is deliberately not consulted. Callers wanting the mode an
// agent actually runs under want [ModeFor].
//
// ctx must be non-nil. Passing nil panics in ctx.Value, as it would for any
// context helper, so callers holding a context that may be nil check it
// themselves rather than relying on this.
func BoundMode(ctx context.Context, agentName string, state *State) (Mode, bool) {
	mode, ok := ctx.Value(boundModeKey{agent: agentName, state: state}).(Mode)
	if !ok {
		return ModeUnset, false
	}
	return mode, true
}

// ModeFor returns the mode agentName runs under: the mode this invocation bound
// for it, else its own declaration.
//
// state must be non-nil — the declaration is read off it.
//
// The binding is consulted first and is authoritative, because it is the only
// thing that knows where the agent was placed. It is safe to prefer because
// [BoundMode] has already established the binding was resolved for this exact
// agent — for any other, including a same-named one, it reports nothing and the
// declaration stands.
func ModeFor(ctx context.Context, agentName string, state *State) Mode {
	if m, ok := BoundMode(ctx, agentName, state); ok {
		return m
	}
	return state.Mode
}
