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

package anthropicmodel

import (
	"strings"
	"testing"

	"google.golang.org/genai"
)

func testPtr[T any](v T) *T { return &v }

func TestBuildOutputFormatSchemaMovesUnsupportedKeywordsToDescription(t *testing.T) {
	cfg := &genai.GenerateContentConfig{
		ResponseSchema: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"age": {
					Type:    "INTEGER",
					Minimum: testPtr(float64(0)),
					Maximum: testPtr(float64(150)),
				},
				"name": {
					Type:   "STRING",
					Format: "uri-reference",
				},
				"website": {
					Type:   "STRING",
					Format: "uri",
				},
				"tags": {
					Type:     "ARRAY",
					Items:    &genai.Schema{Type: "STRING"},
					MinItems: testPtr(int64(2)),
					MaxItems: testPtr(int64(5)),
				},
			},
			Required: []string{"name"},
		},
	}

	got, unwrap, err := buildOutputFormatSchema(cfg)
	if err != nil {
		t.Fatalf("buildOutputFormatSchema() error = %v", err)
	}
	if unwrap {
		t.Error("unwrap = true, want false for an object root")
	}
	if gotType, _ := got["type"].(string); gotType != "object" {
		t.Errorf("type = %q, want %q", gotType, "object")
	}
	if got["additionalProperties"] != false {
		t.Errorf("additionalProperties = %v, want false", got["additionalProperties"])
	}

	props, ok := got["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties has type %T, want map[string]any", got["properties"])
	}

	age, ok := props["age"].(map[string]any)
	if !ok {
		t.Fatalf("properties[age] has type %T, want map[string]any", props["age"])
	}
	for _, key := range []string{"minimum", "maximum"} {
		if _, ok := age[key]; ok {
			t.Errorf("age still has unsupported key %q", key)
		}
	}
	assertDescriptionContains(t, age, "minimum: 0", "maximum: 150")

	name, ok := props["name"].(map[string]any)
	if !ok {
		t.Fatalf("properties[name] has type %T, want map[string]any", props["name"])
	}
	if _, ok := name["format"]; ok {
		t.Error("name still has unsupported format uri-reference")
	}
	assertDescriptionContains(t, name, "format: uri-reference")

	website, ok := props["website"].(map[string]any)
	if !ok {
		t.Fatalf("properties[website] has type %T, want map[string]any", props["website"])
	}
	if got, _ := website["format"].(string); got != "uri" {
		t.Errorf("website format = %q, want %q", got, "uri")
	}

	tags, ok := props["tags"].(map[string]any)
	if !ok {
		t.Fatalf("properties[tags] has type %T, want map[string]any", props["tags"])
	}
	for _, key := range []string{"minItems", "maxItems"} {
		if _, ok := tags[key]; ok {
			t.Errorf("tags still has unsupported key %q", key)
		}
	}
	assertDescriptionContains(t, tags, "minItems: 2", "maxItems: 5")

	required, ok := got["required"].([]any)
	if !ok {
		t.Fatalf("required has type %T, want []any", got["required"])
	}
	if len(required) != 1 || required[0] != "name" {
		t.Errorf("required = %v, want [name]", required)
	}
}

func assertDescriptionContains(t *testing.T, schema map[string]any, want ...string) {
	t.Helper()
	description, _ := schema["description"].(string)
	for _, s := range want {
		if !strings.Contains(description, s) {
			t.Errorf("description %q does not contain %q", description, s)
		}
	}
}
