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
	"net/url"
	"slices"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
)

// StateSchemaReason says why a state write was refused.
type StateSchemaReason int

const (
	// StateKeyUndeclared means the key has no colon and is not a property of
	// the schema.
	StateKeyUndeclared StateSchemaReason = iota + 1
	// StateValueInvalid means the key is declared and the value fails its
	// property's schema.
	StateValueInvalid
)

// StateSchemaError is returned when a state write violates the workflow's
// state schema (see [WithStateSchema]). Match it with errors.As. It carries
// the key and never the value.
type StateSchemaError struct {
	// Key is the state key of the refused write.
	Key string
	// Schema is the title of the state schema, empty when the schema has none.
	Schema string
	// Reason is the class of violation. It is never the zero value in an
	// error ADK returns.
	Reason StateSchemaReason
	// Declared lists the schema's property names in sorted order, for
	// StateKeyUndeclared; nil for StateValueInvalid.
	Declared []string
	// Type describes the property's declared JSON type ("integer",
	// "string|null"), for StateValueInvalid; empty for StateKeyUndeclared and
	// for a property whose schema declares no type (an enum, a $ref, an empty
	// schema).
	Type string
}

// Error names the key, the schema title when there is one, the class of
// violation, and the declared names or the property's type. It never contains
// the rejected value.
func (e *StateSchemaError) Error() string {
	var in string
	if e.Schema != "" {
		in = fmt.Sprintf(" %q", e.Schema)
	}
	switch e.Reason {
	case StateKeyUndeclared:
		return fmt.Sprintf("workflow: state key %q is not declared in the state schema%s; declared: %v", e.Key, in, e.Declared)
	case StateValueInvalid:
		if e.Type != "" {
			return fmt.Sprintf("workflow: value for state key %q does not satisfy type %s in the state schema%s", e.Key, e.Type, in)
		}
		return fmt.Sprintf("workflow: value for state key %q does not satisfy its property in the state schema%s", e.Key, in)
	default:
		return fmt.Sprintf("workflow: state key %q refused by the state schema", e.Key)
	}
}

// stateSchemaURI names the clone of the author's schema document that the
// derived schema of a stateValidator refers to.
const stateSchemaURI = "urn:adk:state-schema"

// stateValidator implements agent.StateValidator over one resolved state
// schema. It is immutable after construction and safe for concurrent use.
type stateValidator struct {
	title    string
	declared []string             // sorted property names
	types    map[string]string    // declared property -> StateSchemaError.Type
	resolved *jsonschema.Resolved // nil for a schema with no root
}

// newStateValidator derives the validator of schema, which must be non-nil.
//
// The validator checks one entry at a time against the property the root
// declares for it, as adk-python validates one field per write
// (sessions/state.py _validate_state_entry): no root keyword such as required
// or additionalProperties takes part. It validates against a derived object
// schema whose only keyword is properties, each a $ref into a clone of the
// author's root, so references inside the author's document resolve where the
// author wrote them. A reference to another document cannot be resolved here
// and is returned as an error. The author's schema is never modified. A zero
// schema has no root and declares no property.
func newStateValidator(schema *jsonschema.Resolved) (*stateValidator, error) {
	v := &stateValidator{declared: extractFieldNames(schema), types: map[string]string{}}
	root := schema.Schema()
	if root == nil {
		return v, nil
	}
	v.title = root.Title
	// Resolving writes into the loaded schema, so it must not be the
	// author's own, which other goroutines may be validating with.
	doc := root.CloneSchemas()
	props := make(map[string]*jsonschema.Schema, len(root.Properties))
	for key, p := range root.Properties {
		props[key] = &jsonschema.Schema{Ref: stateSchemaURI + "#/properties/" + pointerToken(key)}
		v.types[key] = typeOf(p)
	}
	derived := &jsonschema.Schema{Properties: props}
	resolved, err := derived.Resolve(&jsonschema.ResolveOptions{
		BaseURI: "https://adk.invalid/derived",
		Loader: func(uri *url.URL) (*jsonschema.Schema, error) {
			if uri.String() == stateSchemaURI {
				return doc, nil
			}
			return nil, fmt.Errorf("reference to %q is outside the state schema document", uri)
		},
	})
	if err != nil {
		return nil, fmt.Errorf("workflow: state schema cannot be validated at run time: %w", err)
	}
	v.resolved = resolved
	return v, nil
}

// pointerToken escapes key as one JSON Pointer reference token inside a URI
// fragment.
func pointerToken(key string) string {
	return url.PathEscape(strings.NewReplacer("~", "~0", "/", "~1").Replace(key))
}

// typeOf describes the JSON type a property declares, or "" when it declares
// none.
func typeOf(p *jsonschema.Schema) string {
	if p == nil {
		return ""
	}
	if p.Type != "" {
		return p.Type
	}
	return strings.Join(p.Types, "|")
}

// ValidateStateEntry implements agent.StateValidator. A key that contains a
// colon is never refused: besides the app:, user: and temp: scopes, ADK keeps
// its own state under "<owner>:<key>" names, as in adk-python. value is
// validated in its JSON form and never kept.
func (v *stateValidator) ValidateStateEntry(key string, value any) error {
	if strings.Contains(key, ":") {
		return nil
	}
	typ, declared := v.types[key]
	if !declared {
		return &StateSchemaError{Key: key, Schema: v.title, Reason: StateKeyUndeclared, Declared: slices.Clone(v.declared)}
	}
	invalid := &StateSchemaError{Key: key, Schema: v.title, Reason: StateValueInvalid, Type: typ}
	// A value with no JSON form (a channel, a function) is refused even for
	// a property with no constraints.
	form, _, err := jsonForm(value)
	if err != nil {
		return invalid
	}
	// The library's error embeds the value, so it is not kept as a cause.
	if v.resolved.Validate(map[string]any{key: form}) != nil {
		return invalid
	}
	return nil
}
