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
	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/internal/typeutil"
)

// BaseNode provides identity and a default Config implementation for
// types that satisfy the Node interface. Custom node types embed
// BaseNode by value and supply only Run.
type BaseNode struct {
	name         string
	desc         string
	config       NodeConfig
	inputSchema  *jsonschema.Resolved
	outputSchema *jsonschema.Resolved
}

// NewBaseNodeWithSchemas returns a BaseNode with the given identity, configuration, and schemas.
func NewBaseNodeWithSchemas(
	name, description string,
	cfg NodeConfig,
	inputSchema, outputSchema *jsonschema.Resolved,
) BaseNode {
	return BaseNode{
		name:         name,
		desc:         description,
		config:       cfg,
		inputSchema:  inputSchema,
		outputSchema: outputSchema,
	}
}

// NewBaseNode returns a BaseNode with the given identity and
// configuration. Embedders typically call it from their own
// constructor:
//
//	type CustomNode struct {
//	    BaseNode
//	    // ...
//	}
//
//	func NewCustomNode(name string, cfg NodeConfig) *CustomNode {
//	    return &CustomNode{BaseNode: NewBaseNode(name, "", cfg)}
//	}
func NewBaseNode(name, description string, cfg NodeConfig) BaseNode {
	return NewBaseNodeWithSchemas(name, description, cfg, nil, nil)
}

// Name returns the node's name.
func (b BaseNode) Name() string { return b.name }

// Description returns the node's human-readable description.
func (b BaseNode) Description() string { return b.desc }

// Config returns the node's configuration.
func (b BaseNode) Config() NodeConfig { return b.config }

// InputSchema returns the node's input validation schema.
func (b BaseNode) InputSchema() *jsonschema.Resolved { return b.inputSchema }

// OutputSchema returns the node's output validation schema.
func (b BaseNode) OutputSchema() *jsonschema.Resolved { return b.outputSchema }

// ValidateInput validates and coerces the input using the node's input schema.
func (b BaseNode) ValidateInput(in any) (any, error) {
	return defaultValidateInput(in, b.inputSchema)
}

func defaultValidateInput(in any, schema *jsonschema.Resolved) (any, error) {
	if schema == nil {
		return in, nil
	}
	return typeutil.ConvertToWithJSONSchema[any, any](in, schema)
}
