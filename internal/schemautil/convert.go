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

// Package schemautil provides utilities for schema conversion and validation.
package schemautil

import (
	"encoding/json"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"
)

// GenaiToJSONSchema converts a genai.Schema to a jsonschema.Schema.
func GenaiToJSONSchema(gs *genai.Schema) (*jsonschema.Schema, error) {
	if gs == nil {
		return nil, nil
	}

	// Marshal to intermediate map
	data, err := json.Marshal(gs)
	if err != nil {
		return nil, err
	}

	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}

	// Normalize type to lowercase (genai uses "STRING", jsonschema expects "string")
	normalizeTypes(m)

	// Marshal back and unmarshal to jsonschema.Schema
	data, err = json.Marshal(m)
	if err != nil {
		return nil, err
	}

	var js jsonschema.Schema
	if err := json.Unmarshal(data, &js); err != nil {
		return nil, err
	}

	return &js, nil
}

// normalizeTypes recursively lowercases type fields in the schema map.
func normalizeTypes(m map[string]any) {
	if t, ok := m["type"].(string); ok {
		m["type"] = strings.ToLower(t)
	}

	// Recurse into properties
	if props, ok := m["properties"].(map[string]any); ok {
		for _, v := range props {
			if prop, ok := v.(map[string]any); ok {
				normalizeTypes(prop)
			}
		}
	}

	// Recurse into items
	if items, ok := m["items"].(map[string]any); ok {
		normalizeTypes(items)
	}

	// Recurse into anyOf
	if anyOf, ok := m["anyOf"].([]any); ok {
		for _, v := range anyOf {
			if s, ok := v.(map[string]any); ok {
				normalizeTypes(s)
			}
		}
	}
}

// GenaiToResolvedJSONSchema converts a genai.Schema to a resolved jsonschema.
func GenaiToResolvedJSONSchema(gs *genai.Schema) (*jsonschema.Resolved, error) {
	if gs == nil {
		return nil, nil
	}
	js, err := GenaiToJSONSchema(gs)
	if err != nil {
		return nil, err
	}
	return js.Resolve(nil)
}
