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
	"fmt"
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
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

// ValidateOutput validates the output against the node's output schema.
func (b BaseNode) ValidateOutput(out any) (any, error) {
	return defaultValidateOutput(out, b.outputSchema)
}

// defaultValidateOutput backs BaseNode.ValidateOutput. Framework
// control values (*session.Event, *session.RequestInput) bypass the
// schema — they ride through Event.Output but are not user payloads.
// When direct validation fails on model text (a string, or a
// *genai.Content built by synthesizeAgentOutput), projectTextOntoSchema
// retries via the schema; on total failure the original error wins so
// downstream parse details don't mask the real mismatch. Mirrors
// adk-python _validate_output_data.
func defaultValidateOutput(out any, schema *jsonschema.Resolved) (any, error) {
	if schema == nil || out == nil {
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
	if text, ok := modelText(out); ok {
		if v, ok := projectTextOntoSchema(text, schema); ok {
			return v, nil
		}
	}
	return nil, err
}

// modelText returns the model text carried by an output value: the
// string itself, or the non-thought text parts of a *genai.Content
// concatenated. Skipping thought parts mirrors messageText so
// validation sees the same text the agent surfaces as output.
func modelText(out any) (string, bool) {
	switch v := out.(type) {
	case string:
		return v, true
	case *genai.Content:
		var text strings.Builder
		for _, part := range v.Parts {
			if part != nil && part.Text != "" && !part.Thought {
				text.WriteString(part.Text)
			}
		}
		return text.String(), true
	default:
		return "", false
	}
}

// projectTextOntoSchema projects model text onto schema: return it
// directly for a string schema (after re-validating so per-string
// constraints like minLength or pattern still apply), otherwise
// JSON-parse and re-validate. ok is false when no valid value can be
// produced, leaving error reporting to the caller.
func projectTextOntoSchema(s string, schema *jsonschema.Resolved) (any, bool) {
	if rootSchemaIsString(schema) {
		if err := schema.Validate(s); err != nil {
			return nil, false
		}
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

// rootSchemaIsString reports whether schema's root permits the "string"
// type, via either the single Type field or the Types list. The two are
// mutually exclusive in jsonschema.Schema, so checking both covers a root
// like {"type": ["string", "null"]}.
func rootSchemaIsString(schema *jsonschema.Resolved) bool {
	root := schema.Schema()
	if root == nil {
		return false
	}
	return root.Type == "string" || slices.Contains(root.Types, "string")
}

// validateAndStampOutput runs n.ValidateOutput on out and, on success,
// stamps the validated value back onto ev.Output when it rode there
// (MessageAsOutput-derived values stay off the event). Errors wrap
// both ErrNodeFailed and the underlying validation error so callers
// can match either sentinel via errors.Is/errors.As.
func validateAndStampOutput(n Node, out any, ev *session.Event) (any, error) {
	validated, err := n.ValidateOutput(out)
	if err != nil {
		return nil, fmt.Errorf("%w: output validation failed for node %q: %w", ErrNodeFailed, n.Name(), err)
	}
	if ev.Output != nil {
		ev.Output = validated
	}
	return validated, nil
}
