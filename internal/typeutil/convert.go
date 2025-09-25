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
		// See https://github.com/google/jsonschema-go/issues/23: in order to
		// validate, we must validate against a map[string]any. Struct validation
		// does not work as it cannot account for `omitempty` or custom marshalling.
		var m map[string]any
		if err := json.Unmarshal(rawArgs, &m); err != nil {
			return zero, err
		}
		if err := resolvedSchema.Validate(m); err != nil {
			return zero, err
		}
	}
	var typed To
	if err := json.Unmarshal(rawArgs, &typed); err != nil {
		return zero, err
	}
	return typed, nil
}
