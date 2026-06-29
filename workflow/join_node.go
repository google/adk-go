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
	"fmt"
	"iter"

	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/session"
)

// JoinNode is a fan-in barrier. It is activated exactly once,
// after every predecessor declared by the graph edges has
// completed, and receives those predecessors' outputs as a single
// map[string]any keyed by predecessor name. Its own output is
// that map, emitted verbatim.
//
// All incoming edges feed the barrier; conditional routing into a
// JoinNode is a configuration error, because the barrier waits
// for every declared predecessor and a route-skipped predecessor
// never fires.
type JoinNode struct {
	BaseNode
}

// NewJoinNode returns a JoinNode with the given name.
func NewJoinNode(name string) *JoinNode {
	return &JoinNode{BaseNode: NewBaseNode(name, "", NodeConfig{})}
}

// NewJoinNodeWithSchema returns a JoinNode with the given name and input schema.
//
// The input schema is applied to each predecessor's output individually when validation is performed,
// rather than being applied to the combined map structure itself.
//
// If a predecessor's output is nil, validation is bypassed and the nil value is preserved.
func NewJoinNodeWithSchema(name string, schema *jsonschema.Resolved) *JoinNode {
	return &JoinNode{
		BaseNode: NewBaseNodeWithSchemas(name, "", NodeConfig{}, schema, nil),
	}
}

// Run satisfies the Node interface. See JoinNode for the
// aggregation contract.
func (n *JoinNode) Run(ctx agent.Context, input any) iter.Seq2[*session.Event, error] {
	return func(yield func(*session.Event, error) bool) {
		event := session.NewEvent(ctx, ctx.InvocationID())
		event.Output = input
		yield(event, nil)
	}
}

// ValidateInput overrides BaseNode.ValidateInput. Instead of validating the
// aggregated map as a single value, it validates each predecessor's output
// individually against the node's InputSchema.
func (n *JoinNode) ValidateInput(input any) (any, error) {
	schema := n.InputSchema()
	if schema == nil {
		return input, nil
	}
	if m, ok := input.(map[string]any); ok {
		out := make(map[string]any, len(m))
		for predName, v := range m {
			validated, err := defaultValidateInput(v, schema)
			if err != nil {
				return nil, fmt.Errorf("predecessor %q: %w", predName, err)
			}
			out[predName] = validated
		}
		return out, nil
	}
	// Fallback — 1-predecessor case or unexpected shape — fall back to default.
	return n.BaseNode.ValidateInput(input)
}
