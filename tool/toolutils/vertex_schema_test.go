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

package toolutils

import (
	"encoding/json"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
)

func TestSanitizeSchemaForVertex_MovesSiblingsIntoAnyOfBranches(t *testing.T) {
	in := map[string]any{
		"title":       "Age",
		"description": "the person's age, if known",
		"anyOf": []any{
			map[string]any{"type": "integer"},
			map[string]any{"type": "null"},
		},
	}
	out := SanitizeSchemaForVertex(in).(map[string]any)
	if _, ok := out["description"]; ok {
		t.Fatalf("description should not remain alongside anyOf: %#v", out)
	}
	if _, ok := out["title"]; ok {
		t.Fatalf("title should not remain alongside anyOf: %#v", out)
	}
	if _, ok := out["anyOf"]; !ok {
		t.Fatalf("combination schema = %#v", out)
	}
}

func TestSanitizeSchemaForVertex_NestedProperties(t *testing.T) {
	raw := `{
	  "type": "object",
	  "properties": {
	    "age": {
	      "title": "Age",
	      "description": "the person's age, if known",
	      "anyOf": [{"type": "integer"}, {"type": "null"}]
	    }
	  }
	}`
	var in any
	if err := json.Unmarshal([]byte(raw), &in); err != nil {
		t.Fatal(err)
	}
	out := SanitizeSchemaForVertex(in).(map[string]any)
	age := out["properties"].(map[string]any)["age"].(map[string]any)
	if _, ok := age["anyOf"]; !ok {
		t.Fatalf("age = %#v", age)
	}
	if _, ok := age["description"]; ok {
		t.Fatalf("description still alongside anyOf: %#v", age)
	}
}

func TestSanitizeSchemaForVertex_RewritesOneOf(t *testing.T) {
	out := SanitizeSchemaForVertex(map[string]any{
		"description": "a value",
		"oneOf":       []any{map[string]any{"type": "string"}, map[string]any{"type": "integer"}},
	}).(map[string]any)
	if _, ok := out["oneOf"]; !ok {
		t.Fatalf("combination schema = %#v", out)
	}
}

func TestSanitizeSchemaForVertex_RewritesTypedSchemaOnlyWhenNeeded(t *testing.T) {
	in := &jsonschema.Schema{}
	if err := json.Unmarshal([]byte(`{"description":"a value","anyOf":[{"type":"string"},{"type":"null"}]}`), in); err != nil {
		t.Fatal(err)
	}
	out, ok := SanitizeSchemaForVertex(in).(map[string]any)
	if !ok {
		t.Fatalf("sanitized schema type = %T, want map[string]any", out)
	}
	if _, ok := out["anyOf"]; !ok {
		t.Fatalf("sanitized schema = %#v", out)
	}

	unchanged := &jsonschema.Schema{Type: "string"}
	if got := SanitizeSchemaForVertex(unchanged); got != unchanged {
		t.Fatalf("unchanged schema = %T, want original pointer", got)
	}
}

func TestSanitizeSchemaForVertex_PreservesScalarsAndDoesNotDuplicateSiblings(t *testing.T) {
	schema := any(map[string]any{"description": "outer", "anyOf": []any{map[string]any{"type": "string"}, map[string]any{"type": "null"}}})
	for range 18 {
		schema = map[string]any{"description": "outer", "anyOf": []any{schema, map[string]any{"type": "null"}}}
	}
	encoded, err := json.Marshal(SanitizeSchemaForVertex(schema))
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) > 20_000 {
		t.Fatalf("sanitized schema is unexpectedly large: %d bytes", len(encoded))
	}
}

func TestSanitizeSchemaForVertex_PreservesValidationSiblings(t *testing.T) {
	in := map[string]any{"minimum": 1, "anyOf": []any{map[string]any{"type": "number"}, map[string]any{"type": "null"}}}
	out := SanitizeSchemaForVertex(in).(map[string]any)
	if out["minimum"] != 1 {
		t.Fatalf("validation sibling was removed: %#v", out)
	}
}
