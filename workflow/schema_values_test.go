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

package workflow

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/session"
)

// resolveSchemaJSON resolves a hand-written JSON Schema document.
func resolveSchemaJSON(t *testing.T, src string) *jsonschema.Resolved {
	t.Helper()
	var s jsonschema.Schema
	if err := json.Unmarshal([]byte(src), &s); err != nil {
		t.Fatalf("unmarshal schema: %v", err)
	}
	r, err := s.Resolve(nil)
	if err != nil {
		t.Fatalf("resolve schema: %v", err)
	}
	return r
}

// sameMap reports whether a and b are the same map value, not just equal.
func sameMap(a, b any) bool {
	va, vb := reflect.ValueOf(a), reflect.ValueOf(b)
	return va.Kind() == reflect.Map && vb.Kind() == reflect.Map && va.UnsafePointer() == vb.UnsafePointer()
}

func TestValidateOutput_DefaultFilling_ArrayItemsAndNested(t *testing.T) {
	schema := resolveSchemaJSON(t, `{
		"type": "object",
		"properties": {
			"items": {
				"type": "array",
				"items": {
					"type": "object",
					"required": ["name"],
					"properties": {
						"name": {"type": "string"},
						"tag": {"type": "string", "default": "v1"},
						"note": {"type": ["string", "null"], "default": "fallback"}
					}
				}
			},
			"matrix": {
				"type": "array",
				"items": {
					"type": "array",
					"items": {
						"type": "object",
						"properties": {"k": {"type": "integer", "default": 9}}
					}
				}
			}
		}
	}`)
	node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)
	orig := map[string]any{
		"items": []any{
			map[string]any{"name": "a"},
			map[string]any{"name": "b", "tag": "custom", "note": nil},
		},
		"matrix": []any{[]any{map[string]any{}}},
	}

	got, err := node.ValidateOutput(orig)
	if err != nil {
		t.Fatalf("ValidateOutput: %v", err)
	}

	want := map[string]any{
		"items": []any{
			map[string]any{"name": "a", "tag": "v1", "note": "fallback"},
			map[string]any{"name": "b", "tag": "custom"},
		},
		"matrix": []any{[]any{map[string]any{"k": float64(9)}}},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
	}
	wantOrig := map[string]any{
		"items": []any{
			map[string]any{"name": "a"},
			map[string]any{"name": "b", "tag": "custom", "note": nil},
		},
		"matrix": []any{[]any{map[string]any{}}},
	}
	if diff := cmp.Diff(wantOrig, orig); diff != "" {
		t.Errorf("caller's value was mutated (-want +got):\n%s", diff)
	}
}

func TestValidateOutput_NullFieldStripping(t *testing.T) {
	schema := &jsonschema.Schema{
		Type:     "object",
		Required: []string{"id", "req_note"},
		Properties: map[string]*jsonschema.Schema{
			"id":       {Type: "string"},
			"req_note": {Types: []string{"string", "null"}},
			"opt_note": {Types: []string{"string", "null"}},
		},
	}
	producer, err := NewFunctionNodeWithSchema("producer",
		func(_ agent.Context, _ string) (map[string]any, error) {
			return map[string]any{"id": "1", "req_note": nil, "opt_note": nil}, nil
		}, nil, schema, NodeConfig{})
	if err != nil {
		t.Fatalf("producer: %v", err)
	}
	consumer, err := NewFunctionNodeWithSchema("consumer",
		func(_ agent.Context, in map[string]any) (map[string]any, error) {
			return in, nil
		}, schema, nil, NodeConfig{})
	if err != nil {
		t.Fatalf("consumer: %v", err)
	}
	w := mustNew(t, Chain(Start, producer, consumer))

	outputs := outputsByNode(drain(t, w.Run(newMockCtx(t))))

	want := map[string]any{"id": "1", "req_note": nil}
	for _, name := range []string{"producer", "consumer"} {
		if diff := cmp.Diff(want, outputs[name]); diff != "" {
			t.Errorf("%s output mismatch (-want +got):\n%s", name, diff)
		}
	}
}

// ioschemaReview is a struct output whose reviewer field is omitted from
// its JSON form when empty, so a schema default can fill it.
type ioschemaReview struct {
	Result   string `json:"result"`
	Score    int    `json:"score"`
	Reviewer string `json:"reviewer,omitempty"`
}

func TestValidateOutput_StructOutputIsAlwaysAMap(t *testing.T) {
	schema := resolveSchemaJSON(t, `{
		"type": "object",
		"required": ["result", "score"],
		"properties": {
			"result": {"type": "string"},
			"score": {"type": "integer"},
			"reviewer": {"type": "string", "default": "auto"}
		}
	}`)
	node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

	tests := []struct {
		name string
		out  any
		want any
	}{
		{
			name: "struct_with_every_field_set",
			out:  ioschemaReview{Result: "ok", Score: 90, Reviewer: "bob"},
			want: map[string]any{"result": "ok", "score": float64(90), "reviewer": "bob"},
		},
		{
			name: "struct_with_default_to_fill",
			out:  ioschemaReview{Result: "ok", Score: 90},
			want: map[string]any{"result": "ok", "score": float64(90), "reviewer": "auto"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := node.ValidateOutput(tc.out)
			if err != nil {
				t.Fatalf("ValidateOutput: %v", err)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
			}
		})
	}

	// A container holding a struct beside plain values is converted
	// whole, whichever element comes last.
	t.Run("struct_beside_plain_values", func(t *testing.T) {
		review := ioschemaReview{Result: "ok", Score: 1}
		reviewForm := map[string]any{"result": "ok", "score": float64(1)}
		containers := []struct {
			name   string
			schema string
			out    any
			want   any
		}{
			{
				name:   "slice",
				schema: `{"type": "array"}`,
				out:    []any{review, "x"},
				want:   []any{reviewForm, "x"},
			},
			{
				name:   "map",
				schema: `{"type": "object"}`,
				out:    map[string]any{"a": "x", "b": "y", "review": review, "c": "z"},
				want:   map[string]any{"a": "x", "b": "y", "review": reviewForm, "c": "z"},
			},
		}
		for _, tc := range containers {
			t.Run(tc.name, func(t *testing.T) {
				got, err := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, resolveSchemaJSON(t, tc.schema)).ValidateOutput(tc.out)
				if err != nil {
					t.Fatalf("ValidateOutput: %v", err)
				}
				if diff := cmp.Diff(tc.want, got); diff != "" {
					t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})

	t.Run("json_shaped_map_is_returned_as_is", func(t *testing.T) {
		intSchema := resolveSchemaJSON(t, `{"type": "object", "properties": {"result": {"type": "integer"}}}`)
		intNode := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, intSchema)
		out := map[string]any{"result": 1}

		got, err := intNode.ValidateOutput(out)
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		if diff := cmp.Diff(map[string]any{"result": 1}, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
		if !sameMap(out, got) {
			t.Error("ValidateOutput returned a copy, want the caller's own map")
		}
	})
}

func TestValidateOutput_NullDefaultNotFilled(t *testing.T) {
	schema := resolveSchemaJSON(t, `{
		"type": "object",
		"properties": {"note": {"type": "string", "default": null}}
	}`)
	node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)
	out := map[string]any{}

	got, err := node.ValidateOutput(out)
	if err != nil {
		t.Fatalf("ValidateOutput: %v", err)
	}
	if diff := cmp.Diff(map[string]any{}, got); diff != "" {
		t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
	}
	if !sameMap(out, got) {
		t.Error("ValidateOutput returned a copy, want the caller's own map")
	}
}

func TestValidateOutput_DefaultsThroughRefAndAdditionalProperties(t *testing.T) {
	t.Run("ref_in_array_items", func(t *testing.T) {
		schema := resolveSchemaJSON(t, `{
			"type": "object",
			"properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
			"$defs": {"Item": {"type": "object", "properties": {"tag": {"type": "string", "default": "v1"}}}}
		}`)
		node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

		got, err := node.ValidateOutput(map[string]any{"items": []any{map[string]any{}}})
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		want := map[string]any{"items": []any{map[string]any{"tag": "v1"}}}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("recursive_ref", func(t *testing.T) {
		schema := resolveSchemaJSON(t, `{
			"$ref": "#/$defs/Node",
			"$defs": {"Node": {
				"type": "object",
				"properties": {
					"depth": {"type": "integer", "default": 0},
					"child": {"$ref": "#/$defs/Node"}
				}
			}}
		}`)
		node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

		got, err := node.ValidateOutput(map[string]any{"child": map[string]any{"child": map[string]any{}}})
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		want := map[string]any{
			"depth": float64(0),
			"child": map[string]any{
				"depth": float64(0),
				"child": map[string]any{"depth": float64(0)},
			},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("ref_to_root", func(t *testing.T) {
		schema := resolveSchemaJSON(t, `{
			"type": "object",
			"properties": {
				"label": {"type": "string", "default": "node"},
				"next": {"$ref": "#"}
			}
		}`)
		node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

		got, err := node.ValidateOutput(map[string]any{"next": map[string]any{}})
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		want := map[string]any{"label": "node", "next": map[string]any{"label": "node"}}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("ref_to_definitions", func(t *testing.T) {
		schema := resolveSchemaJSON(t, `{
			"type": "object",
			"properties": {"item": {"$ref": "#/definitions/Entry"}},
			"definitions": {"Entry": {"type": "object", "properties": {"status": {"type": "string", "default": "pending"}}}}
		}`)
		node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

		got, err := node.ValidateOutput(map[string]any{"item": map[string]any{}})
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		want := map[string]any{"item": map[string]any{"status": "pending"}}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("pure_ref_loop_terminates", func(t *testing.T) {
		// Resolved.Validate itself recurses without end on this schema,
		// so the default walk is called directly.
		schema := resolveSchemaJSON(t, `{"$ref": "#/$defs/A", "$defs": {"A": {"$ref": "#/$defs/A"}}}`)
		root := schema.Schema()

		changed, err := fillDefaults(root, root, map[string]any{})
		if err != nil {
			t.Fatalf("fillDefaults: %v", err)
		}
		if changed {
			t.Error("fillDefaults reported a change, want none")
		}
	})

	t.Run("additional_properties", func(t *testing.T) {
		schema := resolveSchemaJSON(t, `{
			"type": "object",
			"properties": {"known": {"type": "object"}},
			"additionalProperties": {"type": "object", "properties": {"tag": {"type": "string", "default": "v1"}}}
		}`)
		node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

		got, err := node.ValidateOutput(map[string]any{
			"known": map[string]any{},
			"x":     map[string]any{},
			"y":     map[string]any{"tag": "custom"},
		})
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		want := map[string]any{
			"known": map[string]any{},
			"x":     map[string]any{"tag": "v1"},
			"y":     map[string]any{"tag": "custom"},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestValidateOutput_NullsStrippedThroughRefAndAdditionalProperties(t *testing.T) {
	t.Run("ref_in_array_items", func(t *testing.T) {
		schema := resolveSchemaJSON(t, `{
			"type": "object",
			"properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
			"$defs": {"Item": {"type": "object", "properties": {"note": {"type": ["string", "null"]}}}}
		}`)
		node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

		got, err := node.ValidateOutput(map[string]any{"items": []any{map[string]any{"note": nil}}})
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		want := map[string]any{"items": []any{map[string]any{}}}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("recursive_ref", func(t *testing.T) {
		schema := resolveSchemaJSON(t, `{
			"$ref": "#/$defs/Node",
			"$defs": {"Node": {
				"type": "object",
				"properties": {
					"note": {"type": ["string", "null"]},
					"child": {"$ref": "#/$defs/Node"}
				}
			}}
		}`)
		node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

		got, err := node.ValidateOutput(map[string]any{
			"note":  nil,
			"child": map[string]any{"note": nil, "child": map[string]any{"note": "kept"}},
		})
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		want := map[string]any{"child": map[string]any{"child": map[string]any{"note": "kept"}}}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("ref_to_root", func(t *testing.T) {
		schema := resolveSchemaJSON(t, `{
			"type": "object",
			"properties": {
				"note": {"type": ["string", "null"]},
				"next": {"$ref": "#"}
			}
		}`)
		node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

		got, err := node.ValidateOutput(map[string]any{"next": map[string]any{"note": nil}})
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		want := map[string]any{"next": map[string]any{}}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("ref_to_definitions", func(t *testing.T) {
		schema := resolveSchemaJSON(t, `{
			"type": "object",
			"properties": {"item": {"$ref": "#/definitions/Entry"}},
			"definitions": {"Entry": {"type": "object", "properties": {"note": {"type": ["string", "null"]}}}}
		}`)
		node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

		got, err := node.ValidateOutput(map[string]any{"item": map[string]any{"note": nil}})
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		want := map[string]any{"item": map[string]any{}}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("additional_properties", func(t *testing.T) {
		schema := resolveSchemaJSON(t, `{
			"type": "object",
			"properties": {"known": {"type": "object"}},
			"additionalProperties": {"type": "object", "properties": {"note": {"type": ["string", "null"]}}}
		}`)
		node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

		got, err := node.ValidateOutput(map[string]any{
			"known": map[string]any{"note": nil},
			"x":     map[string]any{"note": nil},
			"y":     map[string]any{"note": "kept"},
		})
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		want := map[string]any{
			"known": map[string]any{"note": nil},
			"x":     map[string]any{},
			"y":     map[string]any{"note": "kept"},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
	})
}

// A filled default is used as written: defaults its own schema declares
// are not filled into it, as pydantic does not validate a field default
// (validate_default is off by default).
func TestValidateOutput_FilledDefaultIsNotFilledFurther(t *testing.T) {
	schema := resolveSchemaJSON(t, `{
		"type": "object",
		"properties": {"sub": {
			"type": "object",
			"default": {},
			"properties": {"x": {"type": "integer", "default": 1}}
		}}
	}`)
	node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

	got, err := node.ValidateOutput(map[string]any{})
	if err != nil {
		t.Fatalf("ValidateOutput: %v", err)
	}
	if diff := cmp.Diff(map[string]any{"sub": map[string]any{}}, got); diff != "" {
		t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
	}
}

// Every allOf member describes the same value, so each member's defaults
// are filled and each member's optional nulls are dropped.
func TestValidateOutput_AllOfMembers(t *testing.T) {
	t.Run("members_at_root_and_ref_member", func(t *testing.T) {
		schema := resolveSchemaJSON(t, `{
			"type": "object",
			"allOf": [
				{"properties": {"a": {"type": "integer", "default": 1}, "b": {"type": ["integer", "null"]}}},
				{"properties": {"c": {"type": "integer", "default": 2}}}
			],
			"properties": {"item": {"allOf": [{"$ref": "#/$defs/Item"}]}},
			"$defs": {"Item": {"type": "object", "properties": {"tag": {"type": "string", "default": "v1"}, "note": {"type": ["string", "null"]}}}}
		}`)
		node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

		got, err := node.ValidateOutput(map[string]any{"b": nil, "item": map[string]any{"note": nil}})
		if err != nil {
			t.Fatalf("ValidateOutput: %v", err)
		}
		want := map[string]any{"a": float64(1), "c": float64(2), "item": map[string]any{"tag": "v1"}}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("ref_loop_through_allOf_terminates", func(t *testing.T) {
		// Resolved.Validate itself recurses without end on this schema,
		// so the default walk is called directly.
		schema := resolveSchemaJSON(t, `{
			"allOf": [{"$ref": "#/$defs/A"}],
			"$defs": {"A": {
				"allOf": [{"$ref": "#"}],
				"properties": {"a": {"type": "integer", "default": 1}}
			}}
		}`)
		root := schema.Schema()
		v := map[string]any{}

		changed, err := fillDefaults(root, root, v)
		if err != nil {
			t.Fatalf("fillDefaults: %v", err)
		}
		if !changed {
			t.Error("fillDefaults reported no change, want one")
		}
		if diff := cmp.Diff(map[string]any{"a": float64(1)}, v); diff != "" {
			t.Errorf("fillDefaults mismatch (-want +got):\n%s", diff)
		}
	})
}

// prefixItems[i] describes item i; items describes the items after them.
func TestValidateOutput_PrefixItems(t *testing.T) {
	schema := resolveSchemaJSON(t, `{
		"type": "array",
		"prefixItems": [
			{"type": "object", "properties": {"a": {"type": "integer", "default": 1}, "b": {"type": ["integer", "null"]}}},
			{"type": "object", "properties": {"f": {"type": "integer", "default": 3}}}
		],
		"items": {"type": "object", "properties": {"c": {"type": "integer", "default": 2}, "d": {"type": ["integer", "null"]}}}
	}`)
	node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

	got, err := node.ValidateOutput([]any{
		map[string]any{"b": nil},
		map[string]any{},
		map[string]any{"d": nil},
		map[string]any{},
	})
	if err != nil {
		t.Fatalf("ValidateOutput: %v", err)
	}
	want := []any{
		map[string]any{"a": float64(1)},
		map[string]any{"f": float64(3)},
		map[string]any{"c": float64(2)},
		map[string]any{"c": float64(2)},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
	}
}

// A patternProperties schema describes each undeclared key its pattern
// matches; additionalProperties describes only the keys no property or
// pattern matches.
func TestValidateOutput_PatternProperties(t *testing.T) {
	schema := resolveSchemaJSON(t, `{
		"type": "object",
		"properties": {"x0": {"type": "object"}},
		"patternProperties": {
			"^x": {"type": "object", "properties": {"a": {"type": "integer", "default": 1}, "b": {"type": ["integer", "null"]}}},
			"1$": {"type": "object", "properties": {"e": {"type": "integer", "default": 5}}}
		},
		"additionalProperties": {"type": "object", "properties": {"c": {"type": "integer", "default": 2}, "d": {"type": ["integer", "null"]}}}
	}`)
	node := NewBaseNodeWithSchemas("n", "", NodeConfig{}, nil, schema)

	got, err := node.ValidateOutput(map[string]any{
		"x0": map[string]any{"b": nil},
		"x1": map[string]any{"b": nil},
		"x2": map[string]any{"a": 7},
		"y":  map[string]any{"d": nil},
	})
	if err != nil {
		t.Fatalf("ValidateOutput: %v", err)
	}
	want := map[string]any{
		"x0": map[string]any{"b": nil},
		"x1": map[string]any{"a": float64(1), "e": float64(5)},
		"x2": map[string]any{"a": 7},
		"y":  map[string]any{"c": float64(2)},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("ValidateOutput mismatch (-want +got):\n%s", diff)
	}
}

// outputsByNode maps each node name to the Output of the event it
// emitted.
func outputsByNode(events []*session.Event) map[string]any {
	out := map[string]any{}
	for _, ev := range events {
		if ev == nil || ev.Output == nil {
			continue
		}
		out[ev.NodeInfo.Name()] = ev.Output
	}
	return out
}
