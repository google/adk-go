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

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/parentmap"
)

type boundOutputSchemaKey struct {
	agent string
	state *State
}

// WithBoundOutputSchema returns a copy of ctx in which the agent identified by
// agentName and state is asked to answer with schema. A typed workflow node
// uses it to pass its output shape to an LLM agent that declares none; nil
// binds no schema, so an inner placement can shadow an outer one.
//
// The schema is bound to the invocation rather than written to State because
// one agent can back several nodes with different output types, and State is
// shared by all of them. Like [WithBoundMode], the key is the agent's name and
// identity, so the binding does not reach a peer the agent transfers to or a
// same-named agent nested beneath it.
func WithBoundOutputSchema(ctx context.Context, agentName string, state *State, schema *genai.Schema) context.Context {
	return context.WithValue(ctx, boundOutputSchemaKey{agent: agentName, state: state}, schema)
}

// outputSchemaFor returns the output schema for the current agent's next
// request: a copy of its declared OutputSchema, else a copy of the schema bound
// by [WithBoundOutputSchema], else nil. state is the current agent's. A schema
// that cannot be copied returns an error.
//
// A bound schema is a placement's request, not the agent's own, so it gives way
// whenever applying it would change a request that works without it:
//   - the agent does not run single_turn here. A chat or task agent talks to
//     the user between replies. Reading the mode per request also covers a
//     nested placement that binds a mode of its own for this agent, such as
//     an agent tool over it. A placement that binds no mode for it, such as
//     an agent tool over a composite that contains it, inherits the outer
//     mode and with it the schema;
//   - the agent sets a response schema or MIME type in GenerateContentConfig,
//     which would otherwise be replaced, or sent alongside a second schema that
//     some models reject;
//   - the agent can transfer, because the transfer_to_agent declaration is a
//     function tool and needOutputSchemaProcessor counts only the agent's own
//     tools, so the schema would go out next to it as a response schema.
//     adk-python counts transfer targets there (_has_function_tools in
//     flows/llm_flows/prompt/_schema.py) and offers set_model_response instead.
//
// The copy keeps a request callback that edits the schema from reaching the
// node's other runs, including concurrent ones.
func outputSchemaFor(ctx agent.InvocationContext, state *State) (*genai.Schema, error) {
	if state.OutputSchema != nil {
		// Unlike Python, isolate schema mutations made by model callbacks.
		return CloneSchema(state.OutputSchema)
	}
	cur := ctx.Agent()
	schema, _ := ctx.Value(boundOutputSchemaKey{agent: cur.Name(), state: state}).(*genai.Schema)
	if schema == nil || ModeFor(ctx, cur.Name(), state) != ModeSingleTurn {
		return nil, nil
	}
	if cfg := state.GenerateContentConfig; cfg != nil && (cfg.ResponseSchema != nil || cfg.ResponseJsonSchema != nil || cfg.ResponseMIMEType != "") {
		return nil, nil
	}
	if shouldUseAutoFlow(cur) && len(transferTargets(cur, parentmap.FromContext(ctx)[cur.Name()])) > 0 {
		return nil, nil
	}
	return CloneSchema(schema)
}
