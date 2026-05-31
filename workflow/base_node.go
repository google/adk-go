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
	"encoding/json"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/typeutil"
	"google.golang.org/adk/session"
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

// ValidateOutput validates the output against the node's output schema.
// See defaultValidateOutput for the validation strategy.
func (b BaseNode) ValidateOutput(out any) (any, error) {
	return defaultValidateOutput(out, b.outputSchema)
}

// defaultValidateOutput is the shared output-validation helper used
// by BaseNode.ValidateOutput. It applies the following strategy:
//
//  1. When schema is nil, the output is returned unchanged.
//  2. Framework control values (*session.Event, *session.RequestInput)
//     are returned unchanged: they are routed through Event.Output by
//     some nodes but they are not user-defined output payloads and
//     should never be schema-validated.
//  3. The output is validated against schema. On success, it is
//     returned unchanged.
//  4. On validation failure, when the output is a *genai.Content, the
//     helper falls back to extracting the concatenated text from the
//     content's parts and:
//     - returns the text directly when the schema's root type is
//     "string",
//     - otherwise parses the text as JSON and re-validates the parsed
//     value, returning the parsed value on success.
//  5. When neither the standard nor the fallback path succeeds, the
//     original validation error is returned.
//
// The Content fallback mirrors ADK Python's _validate_output_data and
// is intended for nodes (notably LlmAgent) that yield raw model output
// as *genai.Content without first projecting it onto the node's
// declared output schema.
func defaultValidateOutput(out any, schema *jsonschema.Resolved) (any, error) {
	if schema == nil {
		return out, nil
	}
	switch out.(type) {
	case *session.Event, *session.RequestInput:
		return out, nil
	}
	err := schema.Validate(out)
	if err == nil {
		return out, nil
	}
	// Try the Content fallback before giving up.
	if content, ok := out.(*genai.Content); ok {
		if v, ok := validateContentFallback(content, schema); ok {
			return v, nil
		}
	}
	return nil, err
}

// validateContentFallback implements the *genai.Content path of
// defaultValidateOutput: extract concatenated text, return it as a
// string when the schema expects a string, otherwise JSON-parse and
// re-validate. Returns ok=false when the fallback cannot produce a
// valid value, leaving error reporting to the caller.
func validateContentFallback(content *genai.Content, schema *jsonschema.Resolved) (any, bool) {
	var text strings.Builder
	for _, part := range content.Parts {
		if part != nil && part.Text != "" {
			text.WriteString(part.Text)
		}
	}
	s := text.String()
	if rootSchemaIsString(schema) {
		return s, true
	}
	if strings.TrimSpace(s) == "" {
		return nil, false
	}
	var parsed any
	if err := json.Unmarshal([]byte(s), &parsed); err != nil {
		return nil, false
	}
	if err := schema.Validate(parsed); err != nil {
		return nil, false
	}
	return parsed, true
}

// rootSchemaIsString reports whether schema's root type is "string".
// Used by the Content fallback to short-circuit JSON parsing for
// string-typed schemas.
func rootSchemaIsString(schema *jsonschema.Resolved) bool {
	root := schema.Schema()
	if root == nil {
		return false
	}
	return root.Type == "string"
}
