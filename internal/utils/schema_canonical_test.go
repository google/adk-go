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

package utils

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
)

func TestCanonicalSchemaJSON_PropertyOrder(t *testing.T) {
	// Two schemas representing {"foo": string, "bar": integer}
	// schema1 has PropertyOrder set to ["foo", "bar"]
	schema1 := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"foo": {Type: "string"},
			"bar": {Type: "integer"},
		},
		PropertyOrder: []string{"foo", "bar"},
	}

	// schema2 has PropertyOrder set to ["bar", "foo"]
	schema2 := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"foo": {Type: "string"},
			"bar": {Type: "integer"},
		},
		PropertyOrder: []string{"bar", "foo"},
	}

	// schema3 has no PropertyOrder set
	schema3 := &jsonschema.Schema{
		Type: "object",
		Properties: map[string]*jsonschema.Schema{
			"foo": {Type: "string"},
			"bar": {Type: "integer"},
		},
	}

	out1, err := CanonicalSchemaJSON(schema1)
	if err != nil {
		t.Fatalf("failed to canonicalize schema1: %v", err)
	}

	out2, err := CanonicalSchemaJSON(schema2)
	if err != nil {
		t.Fatalf("failed to canonicalize schema2: %v", err)
	}

	out3, err := CanonicalSchemaJSON(schema3)
	if err != nil {
		t.Fatalf("failed to canonicalize schema3: %v", err)
	}

	if diff := cmp.Diff(string(out1), string(out2)); diff != "" {
		t.Errorf("canonicalSchemaJSON mismatch between schema1 and schema2 (-want +got):\n%s", diff)
	}

	if diff := cmp.Diff(string(out1), string(out3)); diff != "" {
		t.Errorf("canonicalSchemaJSON mismatch between schema1 and schema3 (-want +got):\n%s", diff)
	}
}

func TestCanonicalize_DeeplyNested(t *testing.T) {
	// Deeply nested object with keys unsorted (>=3 levels)
	input := map[string]any{
		"z": map[string]any{
			"y": map[string]any{
				"x": "val",
				"a": 123.0,
			},
			"b": true,
		},
		"a": "top",
	}

	out, err := canonicalize(input)
	if err != nil {
		t.Fatalf("failed to canonicalize nested map: %v", err)
	}

	// Canonical sorted key order:
	// "a" then "z" at level 1
	// "b" then "y" at level 2
	// "a" then "x" at level 3
	expected := `{"a":"top","z":{"b":true,"y":{"a":123,"x":"val"}}}`
	if string(out) != expected {
		t.Errorf("expected canonicalized JSON:\n%s\ngot:\n%s", expected, string(out))
	}
}

func TestCanonicalize_ArraysPreserved(t *testing.T) {
	input := []any{
		map[string]any{"b": 2, "a": 1},
		[]any{4, 3, 5},
		"string",
	}

	out, err := canonicalize(input)
	if err != nil {
		t.Fatalf("failed to canonicalize array input: %v", err)
	}

	// Keys inside array elements should be sorted, but array itself retains order.
	expected := `[{"a":1,"b":2},[4,3,5],"string"]`
	if string(out) != expected {
		t.Errorf("expected canonicalized JSON:\n%s\ngot:\n%s", expected, string(out))
	}
}

func TestCanonicalize_Primitives(t *testing.T) {
	tests := []struct {
		name  string
		input any
		want  string
	}{
		{
			name:  "string",
			input: "hello world",
			want:  `"hello world"`,
		},
		{
			name:  "number",
			input: 42.5,
			want:  "42.5",
		},
		{
			name:  "bool",
			input: true,
			want:  "true",
		},
		{
			name:  "nil",
			input: nil,
			want:  "null",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out, err := canonicalize(tc.input)
			if err != nil {
				t.Fatalf("failed to canonicalize: %v", err)
			}
			if string(out) != tc.want {
				t.Errorf("canonicalize() = %s, want %s", string(out), tc.want)
			}
		})
	}
}
