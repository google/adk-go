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

package schemautil

import (
	"testing"

	"google.golang.org/genai"
)

func TestGenaiToJSONSchema_Nil(t *testing.T) {
	js, err := GenaiToJSONSchema(nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if js != nil {
		t.Errorf("expected nil, got %v", js)
	}
}

func TestGenaiToJSONSchema_BasicTypes(t *testing.T) {
	tests := []struct {
		name      string
		genaiType genai.Type
		wantType  string
	}{
		{"string", genai.TypeString, "string"},
		{"integer", genai.TypeInteger, "integer"},
		{"number", genai.TypeNumber, "number"},
		{"boolean", genai.TypeBoolean, "boolean"},
		{"array", genai.TypeArray, "array"},
		{"object", genai.TypeObject, "object"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gs := &genai.Schema{Type: tt.genaiType}
			js, err := GenaiToJSONSchema(gs)
			if err != nil {
				t.Fatalf("GenaiToJSONSchema error: %v", err)
			}
			if js.Type != tt.wantType {
				t.Errorf("Type = %q, want %q", js.Type, tt.wantType)
			}
		})
	}
}

func TestGenaiToJSONSchema_Enum(t *testing.T) {
	gs := &genai.Schema{
		Type: genai.TypeString,
		Enum: []string{"red", "green", "blue"},
	}
	js, err := GenaiToJSONSchema(gs)
	if err != nil {
		t.Fatalf("GenaiToJSONSchema error: %v", err)
	}

	if len(js.Enum) != 3 {
		t.Fatalf("Enum length = %d, want 3", len(js.Enum))
	}
	for i, want := range []string{"red", "green", "blue"} {
		if js.Enum[i] != want {
			t.Errorf("Enum[%d] = %v, want %q", i, js.Enum[i], want)
		}
	}
}

func TestGenaiToJSONSchema_EnumValidation(t *testing.T) {
	gs := &genai.Schema{
		Type: genai.TypeString,
		Enum: []string{"red", "green", "blue"},
	}
	resolved, err := GenaiToResolvedJSONSchema(gs)
	if err != nil {
		t.Fatalf("GenaiToResolvedJSONSchema error: %v", err)
	}

	if err := resolved.Validate("red"); err != nil {
		t.Errorf("Validate('red') error = %v, want nil", err)
	}

	if err := resolved.Validate("purple"); err == nil {
		t.Error("Validate('purple') error = nil, want error")
	}
}

func TestGenaiToJSONSchema_Properties(t *testing.T) {
	gs := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"name": {Type: genai.TypeString, Description: "The name"},
			"age":  {Type: genai.TypeInteger},
		},
		Required: []string{"name"},
	}
	js, err := GenaiToJSONSchema(gs)
	if err != nil {
		t.Fatalf("GenaiToJSONSchema error: %v", err)
	}

	if len(js.Properties) != 2 {
		t.Fatalf("Properties length = %d, want 2", len(js.Properties))
	}
	if js.Properties["name"].Type != "string" {
		t.Errorf("Properties[name].Type = %q, want string", js.Properties["name"].Type)
	}
	if js.Properties["name"].Description != "The name" {
		t.Errorf("Properties[name].Description = %q, want 'The name'", js.Properties["name"].Description)
	}
	if js.Properties["age"].Type != "integer" {
		t.Errorf("Properties[age].Type = %q, want integer", js.Properties["age"].Type)
	}
	if len(js.Required) != 1 || js.Required[0] != "name" {
		t.Errorf("Required = %v, want [name]", js.Required)
	}
}

func TestGenaiToJSONSchema_ObjectValidation(t *testing.T) {
	gs := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"color": {
				Type: genai.TypeString,
				Enum: []string{"red", "green", "blue"},
			},
		},
		Required: []string{"color"},
	}
	resolved, err := GenaiToResolvedJSONSchema(gs)
	if err != nil {
		t.Fatalf("GenaiToResolvedJSONSchema error: %v", err)
	}

	valid := map[string]any{"color": "red"}
	if err := resolved.Validate(valid); err != nil {
		t.Errorf("Validate(valid) error = %v, want nil", err)
	}

	invalid := map[string]any{"color": "purple"}
	if err := resolved.Validate(invalid); err == nil {
		t.Error("Validate(invalid) error = nil, want error for invalid enum")
	}

	missing := map[string]any{}
	if err := resolved.Validate(missing); err == nil {
		t.Error("Validate(missing) error = nil, want error for missing required")
	}
}

func TestGenaiToJSONSchema_Array(t *testing.T) {
	gs := &genai.Schema{
		Type: genai.TypeArray,
		Items: &genai.Schema{
			Type: genai.TypeString,
			Enum: []string{"a", "b", "c"},
		},
	}
	resolved, err := GenaiToResolvedJSONSchema(gs)
	if err != nil {
		t.Fatalf("GenaiToResolvedJSONSchema error: %v", err)
	}

	valid := []any{"a", "b"}
	if err := resolved.Validate(valid); err != nil {
		t.Errorf("Validate(valid) error = %v, want nil", err)
	}

	invalid := []any{"a", "d"}
	if err := resolved.Validate(invalid); err == nil {
		t.Error("Validate(invalid) error = nil, want error for invalid enum in array item")
	}
}

func TestGenaiToJSONSchema_NumericConstraints(t *testing.T) {
	min := 0.0
	max := 100.0
	gs := &genai.Schema{
		Type:    genai.TypeNumber,
		Minimum: &min,
		Maximum: &max,
	}
	resolved, err := GenaiToResolvedJSONSchema(gs)
	if err != nil {
		t.Fatalf("GenaiToResolvedJSONSchema error: %v", err)
	}

	if err := resolved.Validate(50.0); err != nil {
		t.Errorf("Validate(50) error = %v, want nil", err)
	}

	if err := resolved.Validate(-1.0); err == nil {
		t.Error("Validate(-1) error = nil, want error")
	}

	if err := resolved.Validate(101.0); err == nil {
		t.Error("Validate(101) error = nil, want error")
	}
}

func TestGenaiToResolvedJSONSchema_Nil(t *testing.T) {
	resolved, err := GenaiToResolvedJSONSchema(nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if resolved != nil {
		t.Error("expected nil resolved schema for nil input")
	}
}
