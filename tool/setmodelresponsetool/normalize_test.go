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

package setmodelresponsetool_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

func TestRun_RecordNormalization_Valid(t *testing.T) {
	schema := &genai.Schema{
		Type:  genai.TypeObject,
		Title: "Profile",
		Properties: map[string]*genai.Schema{
			"name":     {Type: genai.TypeString},
			"opt_null": {Type: genai.TypeString, Nullable: new(true)},
			"req_null": {Type: genai.TypeString, Nullable: new(true)},
			"opt_def":  {Type: genai.TypeInteger, Default: float64(7)},
			"score":    {Type: genai.TypeInteger, Nullable: new(true), Default: float64(7)},
			"tags": {
				Type:    genai.TypeArray,
				Items:   &genai.Schema{Type: genai.TypeString, Nullable: new(true)},
				Default: []any{"def"},
			},
			"meta": {Type: genai.TypeObject},
			"nested": {
				Type:  genai.TypeObject,
				Title: "Nested",
				Properties: map[string]*genai.Schema{
					"k":    {Type: genai.TypeString},
					"drop": {Type: genai.TypeString, Nullable: new(true)},
				},
				Required: []string{"k"},
			},
		},
		Required: []string{"name", "req_null", "nested"},
	}

	t.Run("defaults_nulls_and_unknown_keys", func(t *testing.T) {
		ctx := newToolContext(t)
		args := map[string]any{
			"name":        "Alice",
			"opt_null":    nil,
			"req_null":    nil,
			"score":       nil,
			"meta":        map[string]any{"keep_nil": nil, "x": "y"},
			"nested":      map[string]any{"k": "v", "drop": nil, "unknown_nested": 1},
			"unknown_top": "ignored",
		}
		want := map[string]any{
			"name":     "Alice",
			"req_null": nil,
			"opt_def":  float64(7),
			"tags":     []any{"def"},
			"meta":     map[string]any{"keep_nil": nil, "x": "y"},
			"nested":   map[string]any{"k": "v"},
		}
		wantStored := map[string]any{
			"name":     "Alice",
			"req_null": nil,
			"opt_def":  float64(7),
			"tags":     []any{"def"},
			"meta":     map[string]any{"keep_nil": nil, "x": "y"},
			"nested":   map[string]any{"k": "v"},
		}

		got, err := newTool(t, schema).Run(ctx, args)
		if err != nil {
			t.Fatalf("Run() error: %v", err)
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(wantStored, ctx.Actions().SetModelResponse); diff != "" {
			t.Fatalf("SetModelResponse mismatch (-want +got):\n%s", diff)
		}

		got["name"] = "Mutated"
		got["tags"].([]any)[0] = "changed"

		if diff := cmp.Diff(wantStored, ctx.Actions().SetModelResponse); diff != "" {
			t.Errorf("SetModelResponse shares memory with the result (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff([]any{"def"}, schema.Properties["tags"].Default); diff != "" {
			t.Errorf("schema default shares memory with the result (-want +got):\n%s", diff)
		}
	})

	t.Run("absent_nullable_with_default_is_filled_and_null_items_kept", func(t *testing.T) {
		ctx := newToolContext(t)
		args := map[string]any{
			"name":     "Bob",
			"req_null": "r",
			"tags":     []any{"a", nil},
			"nested":   map[string]any{"k": "v"},
		}
		want := map[string]any{
			"name":     "Bob",
			"req_null": "r",
			"opt_def":  float64(7),
			"score":    float64(7),
			"tags":     []any{"a", nil},
			"nested":   map[string]any{"k": "v"},
		}

		got, err := newTool(t, schema).Run(ctx, args)
		if err != nil {
			t.Fatalf("Run() error: %v", err)
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("nil_property_schema_is_untyped", func(t *testing.T) {
		ctx := newToolContext(t)
		withNil := &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"a": nil,
				"b": {Type: genai.TypeString},
			},
		}

		got, err := newTool(t, withNil).Run(ctx, map[string]any{"b": "x"})
		if err != nil {
			t.Fatalf("Run() error: %v", err)
		}
		if diff := cmp.Diff(map[string]any{"b": "x"}, got); diff != "" {
			t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
		}

		got, err = newTool(t, withNil).Run(ctx, map[string]any{"a": nil, "b": "x"})
		if err != nil {
			t.Fatalf("Run() with a null for the untyped key: %v", err)
		}
		if diff := cmp.Diff(map[string]any{"a": nil, "b": "x"}, got); diff != "" {
			t.Errorf("Run() with a null for the untyped key mismatch (-want +got):\n%s", diff)
		}

		got, err = newTool(t, withNil).Run(ctx, map[string]any{"a": []any{1}, "b": "x"})
		if err != nil {
			t.Fatalf("Run() with a value for the untyped key: %v", err)
		}
		if diff := cmp.Diff(map[string]any{"a": []any{1}, "b": "x"}, got); diff != "" {
			t.Errorf("Run() with a value for the untyped key mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestRun_LaxCoercion_Valid(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"s":       {Type: genai.TypeString},
			"i":       {Type: genai.TypeInteger},
			"n":       {Type: genai.TypeNumber},
			"b":       {Type: genai.TypeBoolean},
			"ints":    {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeInteger}},
			"anylist": {Type: genai.TypeArray},
			"dict":    {Type: genai.TypeObject},
			"untyped": {},
			"enum":    {Type: genai.TypeString, Enum: []string{"a", "b", "c"}},
		},
	}
	tests := []struct {
		field string
		in    any
		want  any
	}{
		{"s", "hello", "hello"},
		{"s", "", ""},
		{"i", float64(25), float64(25)},
		{"i", float32(12), float32(12)},
		{"i", int(7), int(7)},
		{"i", int64(-9), int64(-9)},
		{"i", uint(3), uint(3)},
		{"i", "25", float64(25)},
		{"i", " 25 ", float64(25)},
		{"i", "+25", float64(25)},
		{"i", "-25", float64(-25)},
		{"i", "25.0", float64(25)},
		{"i", "25.00", float64(25)},
		{"i", " 25.0 ", float64(25)},
		{"i", "1_000", float64(1000)},
		{"i", "1_0.0", float64(10)},
		{"i", true, float64(1)},
		{"i", false, float64(0)},
		{"n", float64(1.5), float64(1.5)},
		{"n", float32(2.5), float32(2.5)},
		{"n", int(4), int(4)},
		{"n", "1.5", float64(1.5)},
		{"n", " 1.5 ", float64(1.5)},
		{"n", "1e5", float64(100000)},
		{"n", ".5", float64(0.5)},
		{"n", "5.", float64(5)},
		{"n", "1e-5", float64(0.00001)},
		{"n", true, float64(1)},
		{"n", false, float64(0)},
		{"b", true, true},
		{"b", false, false},
		{"b", int(1), true},
		{"b", int(0), false},
		{"b", float64(1.0), true},
		{"b", float64(0.0), false},
		{"b", "true", true},
		{"b", "t", true},
		{"b", "yes", true},
		{"b", "y", true},
		{"b", "on", true},
		{"b", "1", true},
		{"b", "TRUE", true},
		{"b", "Yes", true},
		{"b", "ON", true},
		{"b", "false", false},
		{"b", "f", false},
		{"b", "no", false},
		{"b", "n", false},
		{"b", "off", false},
		{"b", "0", false},
		{"b", "FALSE", false},
		{"b", "No", false},
		{"b", "OFF", false},
		{"ints", []any{"1", 2.0, true}, []any{float64(1), float64(2), float64(1)}},
		{"ints", []string{"10", "20"}, []any{float64(10), float64(20)}},
		{"anylist", []any{"a", 1, nil}, []any{"a", 1, nil}},
		{"dict", map[string]any{"k": 1}, map[string]any{"k": 1}},
		{"untyped", "anything", "anything"},
		{"untyped", float64(9), float64(9)},
		{"enum", "b", "b"},
	}
	tl := newTool(t, schema)
	for _, tc := range tests {
		t.Run(tc.field, func(t *testing.T) {
			ctx := newToolContext(t)

			got, err := tl.Run(ctx, map[string]any{tc.field: tc.in})
			if err != nil {
				t.Fatalf("Run({%q: %#v}) error: %v", tc.field, tc.in, err)
			}
			if diff := cmp.Diff(map[string]any{tc.field: tc.want}, got); diff != "" {
				t.Errorf("Run({%q: %#v}) mismatch (-want +got):\n%s", tc.field, tc.in, diff)
			}
		})
	}
}
