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
	"reflect"
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

// ValidateOutput validates out against the node's output schema and
// returns the value the node's successors receive.
//
// A nil schema or a nil out returns out unchanged, as do the framework
// control values *session.Event and *session.RequestInput. A
// *genai.Content or a genai.Content is unwrapped: an empty schema ({} or
// true) keeps the Content, as a *genai.Content; a string schema validates
// its non-thought text; any other schema validates the text parsed as
// JSON, or the text itself when it is not JSON. When that text fails and
// the schema is one generated from the genai.Content type (its root
// declares the property "parts"), the Content is kept if it satisfies
// the schema. Every other value is validated in its JSON form after the
// schema's defaults are filled in for absent properties, at any depth; a
// null in a property the schema lists as optional is then dropped. A string that fails the schema is accepted
// when it holds a JSON number or boolean that passes, so "42" satisfies an
// integer schema as float64(42); a string holding a JSON object, array or
// null is not parsed.
//
// The result is out itself when out is already made of JSON values
// (map[string]any, []any, strings, booleans, numbers) and validation
// changed nothing. Otherwise it is the JSON form: a struct, a pointer or a
// typed map or slice validated against a schema arrives as map[string]any
// or []any. out is never modified. On failure ValidateOutput returns nil
// and the schema error.
func (b BaseNode) ValidateOutput(out any) (any, error) {
	return strictValidateOutput(out, b.outputSchema)
}

// stringOutput selects how validateOutput treats a string output that
// fails the schema.
type stringOutput int

const (
	// scalarStrings accepts the string's JSON parse only when it is a
	// number or a boolean.
	scalarStrings stringOutput = iota
	// modelTextStrings accepts any JSON document the string holds, for
	// model text that answers a structured output request.
	modelTextStrings
)

// strictValidateOutput is the output validation every node kind gets by
// default; see BaseNode.ValidateOutput.
func strictValidateOutput(out any, schema *jsonschema.Resolved) (any, error) {
	return validateOutput(out, schema, scalarStrings)
}

// defaultValidateOutput is the output validation for model text, used by
// AgentNode.ValidateOutput: as strictValidateOutput, except that a string
// that fails the schema is parsed as any JSON document and validated
// again. Mirrors adk-python _validate_output_data.
func defaultValidateOutput(out any, schema *jsonschema.Resolved) (any, error) {
	return validateOutput(out, schema, modelTextStrings)
}

// validateOutput implements strictValidateOutput and
// defaultValidateOutput; mode is the only difference between them. When
// a string output fails the schema and its parse fails too, the error of
// the string itself is returned, so a parse detail does not mask the real
// mismatch.
func validateOutput(out any, schema *jsonschema.Resolved, mode stringOutput) (any, error) {
	if schema == nil || out == nil {
		return out, nil
	}
	switch out.(type) {
	case *session.Event, *session.RequestInput:
		return out, nil
	}
	if c, ok := asContent(out); ok {
		return validateContentOutput(c, schema)
	}
	form, unchanged, err := conform(out, schema)
	if err == nil {
		if unchanged {
			return out, nil
		}
		return form, nil
	}
	if text, ok := out.(string); ok {
		if parsed, ok := parseString(text, mode); ok {
			if form, perr := conformed(parsed, schema); perr == nil {
				return form, nil
			}
		}
	}
	return nil, err
}

// validateContentOutput validates the payload of c, as adk-python's
// _unwrap_content_payload and validate_node_data do
// (utils/_schema_utils.py): a schema that accepts any value keeps c; a
// string schema validates c's non-thought text; any other schema validates
// the text parsed as JSON, or, when the text is not JSON (empty text
// included), the text itself. When that payload fails, the schema
// describes genai.Content (see schemaDescribesContent) and c satisfies it
// in its JSON form, c is kept: adk-python keeps a Content only for an
// annotation that accepts Content (annotation_accepts_content). It
// returns the conformed payload or c, or nil and the payload's error. A
// nil c returns nil.
func validateContentOutput(c *genai.Content, schema *jsonschema.Resolved) (any, error) {
	if c == nil {
		return nil, nil
	}
	if schemaAcceptsAnyValue(schema) {
		return c, nil
	}
	payload, err := conformed(contentPayload(c, schema), schema)
	if err == nil {
		return payload, nil
	}
	if !schemaDescribesContent(schema) {
		return nil, err
	}
	if form, _, ferr := jsonForm(c); ferr == nil && schema.Validate(form) == nil {
		return c, nil
	}
	return nil, err
}

// schemaDescribesContent reports whether schema's root declares the parts
// property of genai.Content, as the schema generated from the
// genai.Content type does. It keeps a Content whose JSON form satisfies a
// schema for other data by accident, such as an open object schema, from
// passing as that data.
func schemaDescribesContent(schema *jsonschema.Resolved) bool {
	root := schema.Schema()
	return root != nil && root.Properties["parts"] != nil
}

// contentPayload returns the value of c that schema validates: its
// non-thought text for a string schema, otherwise the text parsed as JSON,
// or the text itself when it is not JSON.
func contentPayload(c *genai.Content, schema *jsonschema.Resolved) any {
	text, _ := modelText(c)
	if rootSchemaIsString(schema) {
		return text
	}
	var parsed any
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		return text
	}
	return parsed
}

// parseString returns the JSON value s holds when mode accepts it:
// modelTextStrings accepts any JSON document, scalarStrings only a number
// or a boolean.
func parseString(s string, mode stringOutput) (any, bool) {
	var parsed any
	if err := json.Unmarshal([]byte(s), &parsed); err != nil {
		return nil, false
	}
	if mode == modelTextStrings {
		return parsed, true
	}
	switch parsed.(type) {
	case float64, bool:
		return parsed, true
	}
	return nil, false
}

// schemaAcceptsAnyValue reports whether schema's root is the empty schema
// ({} or true), which every value satisfies. jsonschema-go refuses to
// validate a Go struct even against the empty schema, so a caller that
// must keep a struct such as *genai.Content checks this first.
func schemaAcceptsAnyValue(schema *jsonschema.Resolved) bool {
	root := schema.Schema()
	return root != nil && reflect.ValueOf(*root).IsZero()
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
		return nil, outputValidationError(n.Name(), err)
	}
	if ev.Output != nil {
		ev.Output = validated
	}
	return validated, nil
}

// outputValidationError reports that the output of the node or workflow
// called name failed its schema with err. It wraps both ErrNodeFailed and
// err.
func outputValidationError(name string, err error) error {
	return fmt.Errorf("%w: output validation failed for node %q: %w", ErrNodeFailed, name, err)
}
