// Copyright 2025 Google LLC
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

// Package typeutil is a collection of type handling utility functions.
package typeutil

import (
	"encoding/json"

	"github.com/google/jsonschema-go/jsonschema"
)

// ConvertToWithJSONSchema converts the given value to another type using json marshal/unmarshal.
// If non-nil resolvedSchema is provided, validation against the resolvedSchema will run
// during the conversion.
func ConvertToWithJSONSchema[From, To any](v From, resolvedSchema *jsonschema.Resolved) (To, error) {
	var zero To
	rawArgs, err := json.Marshal(v)
	if err != nil {
		return zero, err
	}
	if resolvedSchema != nil {
		// Validate the JSON-decoded form rather than the Go value:
		// struct validation can't account for `omitempty` or custom
		// marshaling (see
		// https://github.com/google/jsonschema-go/issues/23).
		var decoded any
		if err := json.Unmarshal(rawArgs, &decoded); err != nil {
			return zero, err
		}
		validated := decoded
		// An absent input (e.g. a tool invoked with no arguments)
		// should satisfy an object schema.
		if validated == nil && schemaExpectsObject(resolvedSchema) {
			validated = map[string]any{}
		}
		if err := resolvedSchema.Validate(validated); err != nil {
			return zero, err
		}
		if decoded == nil {
			return zero, nil
		}
		if typed, ok := decoded.(To); ok {
			return typed, nil
		}
	}
	var typed To
	if err := json.Unmarshal(rawArgs, &typed); err != nil {
		return zero, err
	}
	return typed, nil
}

// ConvertArgsWithJSONSchema validates a decoded JSON object map against resolvedSchema
// and unmarshals it once into To.
func ConvertArgsWithJSONSchema[To any](m map[string]any, resolvedSchema *jsonschema.Resolved) (To, error) {
	var zero To
	if resolvedSchema != nil {
		var validated any = m
		if m == nil && schemaExpectsObject(resolvedSchema) {
			validated = map[string]any{}
		}
		if err := resolvedSchema.Validate(validated); err != nil {
			return zero, err
		}
	}
	rawArgs, err := json.Marshal(m)
	if err != nil {
		return zero, err
	}
	var typed To
	if err := json.Unmarshal(rawArgs, &typed); err != nil {
		return zero, err
	}
	return typed, nil
}

// ConvertResultWithJSONSchema marshals output once, unmarshals it once into any,
// returns the decoded map when it satisfies resolvedSchema, or wraps non-map
// outputs as {"result": output} (validating the decoded value only when
// validateWrapped is true).
func ConvertResultWithJSONSchema[From any](output From, resolvedSchema *jsonschema.Resolved, validateWrapped bool) (map[string]any, error) {
	rawOutput, err := json.Marshal(output)
	if err != nil {
		return nil, err
	}
	var decoded any
	if err := json.Unmarshal(rawOutput, &decoded); err != nil {
		return nil, err
	}
	switch m := decoded.(type) {
	case map[string]any:
		if resolvedSchema != nil {
			if err := resolvedSchema.Validate(m); err != nil {
				if validateWrapped {
					return nil, err
				}
				return map[string]any{"result": output}, nil
			}
		}
		return m, nil
	case nil:
		if resolvedSchema != nil {
			var validated any
			if schemaExpectsObject(resolvedSchema) {
				validated = map[string]any{}
			}
			if err := resolvedSchema.Validate(validated); err != nil {
				if validateWrapped {
					return nil, err
				}
				return map[string]any{"result": output}, nil
			}
		}
		return nil, nil
	default:
		if validateWrapped && resolvedSchema != nil {
			if err := resolvedSchema.Validate(decoded); err != nil {
				return nil, err
			}
		}
		return map[string]any{"result": output}, nil
	}
}

// ValidateWithJSONSchema validates a Go value against a resolved schema by
// first converting it to its JSON-decoded form to avoid struct validation issues.
func ValidateWithJSONSchema(v any, resolvedSchema *jsonschema.Resolved) error {
	if resolvedSchema == nil {
		return nil
	}
	rawArgs, err := json.Marshal(v)
	if err != nil {
		return err
	}
	var decoded any
	if err := json.Unmarshal(rawArgs, &decoded); err != nil {
		return err
	}
	if decoded == nil && schemaExpectsObject(resolvedSchema) {
		decoded = map[string]any{}
	}
	return resolvedSchema.Validate(decoded)
}

// schemaExpectsObject reports whether the resolved schema's root type
// is (or includes) "object". Used to decide whether a JSON `null`
// input should be treated as an empty object for validation.
func schemaExpectsObject(resolved *jsonschema.Resolved) bool {
	if resolved == nil {
		return false
	}
	root := resolved.Schema()
	if root == nil {
		return false
	}
	if root.Type == "object" {
		return true
	}
	for _, t := range root.Types {
		if t == "object" {
			return true
		}
	}
	return false
}
