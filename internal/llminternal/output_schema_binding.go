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

	"google.golang.org/genai"
)

// boundOutputSchemaKey is the context key for one agent's placement-derived
// output schema, mirroring boundModeKey (mode.go): identity is part of the
// key so a binding never reaches an agent, or a same-named other, it was not
// resolved for, and a nested placement can shadow it without disturbing an
// outer one.
type boundOutputSchemaKey struct {
	agent string
	state *State
}

// WithBoundOutputSchema returns ctx carrying the model-facing schema a
// placement derived for the named agent's Output type, for this invocation
// only.
//
// State.OutputSchema cannot hold this directly: one agent.Agent instance can
// back more than one workflow node (the same agent wrapped by two
// NewAgentNodeTyped calls with different Output types, or wrapped
// concurrently by several goroutines at once, see
// TestNewAgentNode_ConcurrentWrappingIsRaceFree), so State is shared across
// placements and invocations. Binding it into the context the way
// WithBoundMode already does keeps the derived schema scoped to the one
// invocation that resolved it, with no shared write at construction time.
//
// A nil schema returns ctx untouched, mirroring WithBoundMode's ModeUnset
// guard: every binder passes whatever newAgentNodeWithSchemasTyped derived
// for that node, which is nil for an untyped (any) Output.
func WithBoundOutputSchema(ctx context.Context, agentName string, state *State, schema *genai.Schema) context.Context {
	if schema == nil {
		return ctx
	}
	return context.WithValue(ctx, boundOutputSchemaKey{agent: agentName, state: state}, schema)
}

// BoundOutputSchema reports the schema this invocation bound for the named
// agent, and whether it bound one at all. A binding made for a different
// agent does not count, including one made for a different agent of the same
// name (see boundModeKey's identity discussion in mode.go).
func BoundOutputSchema(ctx context.Context, agentName string, state *State) (*genai.Schema, bool) {
	schema, ok := ctx.Value(boundOutputSchemaKey{agent: agentName, state: state}).(*genai.Schema)
	if !ok {
		return nil, false
	}
	return schema, true
}

// OutputSchemaFor returns the schema agentName's LLM request should be
// constrained to: its own declared State.OutputSchema if set, else the
// schema (if any) a placement bound for this invocation.
//
// Precedence is the mirror of ModeFor's: there the binding wins over the
// declaration, because Mode is about where an otherwise-undeclared agent
// runs. Here the declaration wins, because an explicit
// llmagent.Config.OutputSchema is a choice the agent's own author made, and a
// node's derived schema exists only to fill the gap when the agent declares
// none: it must never displace an explicit one.
func OutputSchemaFor(ctx context.Context, agentName string, state *State) *genai.Schema {
	if state.OutputSchema != nil {
		return state.OutputSchema
	}
	if schema, ok := BoundOutputSchema(ctx, agentName, state); ok {
		return schema
	}
	return nil
}
