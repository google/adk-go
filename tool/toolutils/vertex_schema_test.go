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
	branches, ok := out["anyOf"].([]any)
	if !ok || len(branches) != 2 {
		t.Fatalf("anyOf = %#v", out["anyOf"])
	}
	first := branches[0].(map[string]any)
	if first["type"] != "integer" || first["description"] != "the person's age, if known" {
		t.Fatalf("first branch = %#v", first)
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
