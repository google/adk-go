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

package mcptoolset

import (
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// TestConvertTool_SanitizesMapInputSchemaForVertex verifies that an MCP tool
// whose InputSchema arrives as map[string]any (the client-side form) with an
// anyOf-plus-siblings parameter is normalized before reaching the model — the
// pydantic/MCP optional-field shape Vertex rejects.
func TestConvertTool_SanitizesMapInputSchemaForVertex(t *testing.T) {
	in := &mcp.Tool{
		Name:        "create_person",
		Description: "creates a person",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"age": map[string]any{
					"description": "optional age",
					"anyOf": []any{
						map[string]any{"type": "integer"},
						map[string]any{"type": "null"},
					},
				},
			},
		},
	}

	got, err := convertTool(in, nil, false, nil)
	if err != nil {
		t.Fatalf("convertTool: %v", err)
	}

	schema, ok := got.(*mcpTool).Declaration().ParametersJsonSchema.(map[string]any)
	if !ok {
		t.Fatalf("ParametersJsonSchema is %T, want map[string]any", got.(*mcpTool).Declaration().ParametersJsonSchema)
	}
	age, ok := schema["properties"].(map[string]any)["age"].(map[string]any)
	if !ok {
		t.Fatalf("age property missing: %+v", schema)
	}
	if _, has := age["anyOf"]; has {
		t.Errorf("anyOf not sanitized for client-form MCP schema: %+v", age)
	}
	if age["description"] != "optional age" {
		t.Errorf("description dropped: %+v", age)
	}
}
