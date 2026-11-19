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
	"slices"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/typeutil"
)

// modelOutputSchema returns the schema an agent node asks the model to answer
// with, derived from the node's output schema, or nil to leave the model
// unconstrained as it was before typed nodes passed their schema on.
//
// Only a closed, non-null object with at least one property qualifies. A
// string Output sent as a response schema makes the model answer with a
// quoted JSON string, and the set_model_response tool needs object parameters,
// so primitive, slice, map and nullable (pointer) outputs stay unconstrained.
// Every object must also set additionalProperties to false, as a Go struct
// does: set_model_response drops keys its schema does not list, so an open
// object would lose keys the node accepts.
//
// A nullable property (a pointer or slice field) keeps Nullable. The
// set_model_response tool accepts a null argument only when its schema is
// nullable, so dropping it would turn a null the node accepts into an error.
// The tool keeps a null for a required key, so a required pointer field that
// the model sets to null reaches the node as null.
//
// The result is a hint, not the contract: BaseNode.ValidateOutput still checks
// the reply against the full schema. So a schema that genai.Schema cannot
// carry yields nil rather than an error, and constructing a node never fails
// because of it.
//
// adk-python has no typed node: an LLM agent node there is sent the agent's
// own output_schema, whatever its shape (workflow/_llm_agent_wrapper.py). The
// limits here exist because the schema is derived from a Go type the agent's
// author never declared, so it must not break a node that works without it.
func modelOutputSchema(resolved *jsonschema.Resolved) *genai.Schema {
	if resolved == nil {
		return nil
	}
	root := resolved.Schema()
	if root == nil || slices.Contains(root.Types, "null") {
		return nil
	}
	s, ok := typeutil.ToGenaiSchema(root)
	if !ok || s.Type != genai.TypeObject {
		return nil
	}
	return s
}
