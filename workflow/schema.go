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

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/mitchellh/mapstructure"
)

// Schema validates and coerces a value against a typed contract. Three
// concrete forms ship in this package:
//
//   - JSONSchemaFor[T]() builds a Schema from a Go type via reflection.
//     Validation accepts map[string]any (decoded to T) or T directly.
//   - JSONSchemaRaw wraps a *jsonschema.Schema and performs raw validation
//     without coercion to a Go type.
//   - Implementations may live elsewhere; the interface is the contract.
//
// nil Schema means no validation (passthrough).
type Schema interface {
	// Validate validates data and returns its coerced form. data may be a
	// typed Go value, a map[string]any, or any JSON-marshalable shape.
	Validate(data any) (any, error)

	// JSONSchema returns the underlying JSON Schema for telemetry, prompt
	// derivation, and edge-time compatibility checks.
	JSONSchema() *jsonschema.Schema
}

// JSONSchemaFor builds a Schema from a Go type. T may be a struct, slice,
// map, or primitive — anything jsonschema.For accepts.
//
// On Validate(data):
//   - If data is already of type *T or T, the schema is checked and the
//     value is returned as-is (no needless re-decode).
//   - If data is map[string]any or a JSON-shaped value, it is JSON-marshaled
//     then decoded into T using mapstructure (which is lenient with field
//     casing and missing fields).
//
// Mirrors adk-python's TypeAdapter[T].validate_python coercion path.
func JSONSchemaFor[T any]() Schema {
	var zero T
	js, err := jsonschema.For[T](nil)
	if err != nil {
		panic(fmt.Sprintf("workflow: JSONSchemaFor[%T]: %v", zero, err))
	}
	return &reflectSchema[T]{schema: js}
}

// JSONSchemaRaw wraps a pre-built *jsonschema.Schema. Validation runs
// against the schema; data is returned unchanged.
//
// Use this when the caller already has a JSON Schema (e.g. from an
// external API spec) and doesn't want to declare a Go type.
func JSONSchemaRaw(s *jsonschema.Schema) Schema {
	return &rawSchema{schema: s}
}

// reflectSchema is the concrete implementation backing JSONSchemaFor.
type reflectSchema[T any] struct {
	schema *jsonschema.Schema
}

func (r *reflectSchema[T]) JSONSchema() *jsonschema.Schema { return r.schema }

func (r *reflectSchema[T]) Validate(data any) (any, error) {
	// Fast path: already the right Go type.
	if v, ok := data.(T); ok {
		return v, nil
	}
	if p, ok := data.(*T); ok && p != nil {
		return *p, nil
	}

	// JSON round-trip + mapstructure: handles map[string]any, []any, etc.
	var out T
	if err := decodeInto(data, &out); err != nil {
		return nil, fmt.Errorf("workflow: input validation: %w", err)
	}

	// Validate the coerced value's JSON form against the schema.
	rs, err := r.schema.Resolve(nil)
	if err != nil {
		return nil, fmt.Errorf("workflow: schema resolve: %w", err)
	}
	jsonBytes, err := json.Marshal(out)
	if err != nil {
		return nil, fmt.Errorf("workflow: marshal coerced value: %w", err)
	}
	var parsed any
	if err := json.Unmarshal(jsonBytes, &parsed); err != nil {
		return nil, fmt.Errorf("workflow: unmarshal coerced value: %w", err)
	}
	if err := rs.Validate(parsed); err != nil {
		return nil, fmt.Errorf("workflow: validation failed: %w", err)
	}
	return out, nil
}

// rawSchema is the concrete implementation backing JSONSchemaRaw.
type rawSchema struct {
	schema *jsonschema.Schema
}

func (r *rawSchema) JSONSchema() *jsonschema.Schema { return r.schema }

func (r *rawSchema) Validate(data any) (any, error) {
	rs, err := r.schema.Resolve(nil)
	if err != nil {
		return nil, fmt.Errorf("workflow: schema resolve: %w", err)
	}
	// JSON round-trip so jsonschema sees the canonical form regardless of
	// whether the caller passed a struct or a map.
	jsonBytes, err := json.Marshal(data)
	if err != nil {
		return nil, fmt.Errorf("workflow: marshal value: %w", err)
	}
	var parsed any
	if err := json.Unmarshal(jsonBytes, &parsed); err != nil {
		return nil, fmt.Errorf("workflow: unmarshal value: %w", err)
	}
	if err := rs.Validate(parsed); err != nil {
		return nil, fmt.Errorf("workflow: validation failed: %w", err)
	}
	return data, nil
}

// decodeInto fills out from data using mapstructure (lenient) when the
// shapes diverge, falling back to a JSON marshal/unmarshal trip for
// non-map inputs.
func decodeInto(data any, out any) error {
	// mapstructure handles maps directly; for other inputs, JSON-bridge first.
	if _, ok := data.(map[string]any); ok {
		dec, err := mapstructure.NewDecoder(&mapstructure.DecoderConfig{
			TagName: "json",
			Result:  out,
		})
		if err != nil {
			return err
		}
		return dec.Decode(data)
	}
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, out)
}
