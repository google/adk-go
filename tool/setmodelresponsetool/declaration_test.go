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
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/toolinternal"
)

func TestDeclaration_Shapes(t *testing.T) {
	const wantDescription = "Set your final response using the required output schema.\n\n" +
		"Use this tool to provide your final structured answer instead\n" +
		"of outputting text directly."

	record := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
		Required:   []string{"city"},
	}
	recordLower := &genai.Schema{
		Type:       "object",
		Properties: map[string]*genai.Schema{"city": {Type: "string"}},
	}
	recordList := &genai.Schema{
		Type: genai.TypeArray,
		Items: &genai.Schema{
			Type:       genai.TypeObject,
			Title:      "Item",
			Properties: map[string]*genai.Schema{"id": {Type: genai.TypeInteger}},
		},
	}
	recordListLower := &genai.Schema{
		Type: "array",
		Items: &genai.Schema{
			Type:       "object",
			Properties: map[string]*genai.Schema{"id": {Type: "integer"}},
		},
	}
	primitiveList := &genai.Schema{Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}}
	freeForm := &genai.Schema{Type: genai.TypeObject}
	primitive := &genai.Schema{Type: genai.TypeString}
	empty := &genai.Schema{}
	raw := &jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"result": {Type: "string"}},
	}

	tests := []struct {
		name string
		tool toolinternal.FunctionTool
		// samePointer, when set, is the schema ParametersJsonSchema must be
		// identical to; otherwise want is compared by value.
		samePointer *genai.Schema
		want        any
	}{
		{name: "record", tool: newTool(t, record), samePointer: record},
		{name: "record_lowercase_type", tool: newTool(t, recordLower), samePointer: recordLower},
		{
			name: "record_list",
			tool: newTool(t, recordList),
			want: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{"items": {
					Type:  "array",
					Items: &jsonschema.Schema{Ref: "#/$defs/Item"},
				}},
				Required: []string{"items"},
				Defs: map[string]*jsonschema.Schema{"Item": {
					Type:       "object",
					Title:      "Item",
					Properties: map[string]*jsonschema.Schema{"id": {Type: "integer"}},
				}},
			},
		},
		{
			name: "record_list_lowercase_type",
			tool: newTool(t, recordListLower),
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{"items": {
					Type: "array",
					Items: &genai.Schema{
						Type:       "object",
						Properties: map[string]*genai.Schema{"id": {Type: "integer"}},
					},
				}},
				Required: []string{"items"},
			},
		},
		{
			name: "primitive_list",
			tool: newTool(t, primitiveList),
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{"response": {
					Type:  genai.TypeArray,
					Items: &genai.Schema{Type: genai.TypeString},
				}},
				Required: []string{"response"},
			},
		},
		{
			name: "free_form_object",
			tool: newTool(t, freeForm),
			want: &genai.Schema{
				Type:       genai.TypeObject,
				Properties: map[string]*genai.Schema{"response": {Type: genai.TypeObject}},
				Required:   []string{"response"},
			},
		},
		{
			name: "primitive_root",
			tool: newTool(t, primitive),
			want: &genai.Schema{
				Type:       genai.TypeObject,
				Properties: map[string]*genai.Schema{"response": {Type: genai.TypeString}},
				Required:   []string{"response"},
			},
		},
		{
			name: "empty_type_schema",
			tool: newTool(t, empty),
			want: &genai.Schema{
				Type:       genai.TypeObject,
				Properties: map[string]*genai.Schema{"response": {}},
				Required:   []string{"response"},
			},
		},
		{
			name: "raw_json_schema",
			tool: newRawTool(t, raw),
			want: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{"response": {
					Type:       "object",
					Properties: map[string]*jsonschema.Schema{"result": {Type: "string"}},
				}},
				Required: []string{"response"},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.tool.Name(); got != "set_model_response" {
				t.Errorf("Name() = %q, want %q", got, "set_model_response")
			}
			if tc.tool.IsLongRunning() {
				t.Error("IsLongRunning() = true, want false")
			}
			if got := tc.tool.Description(); got != wantDescription {
				t.Errorf("Description() = %q, want %q", got, wantDescription)
			}
			decl := tc.tool.Declaration()
			if decl == nil {
				t.Fatal("Declaration() = nil")
			}
			if decl.Name != "set_model_response" || decl.Description != wantDescription {
				t.Errorf("Declaration() name, description = %q, %q; want %q, %q", decl.Name, decl.Description, "set_model_response", wantDescription)
			}
			if tc.samePointer != nil {
				if got, ok := decl.ParametersJsonSchema.(*genai.Schema); !ok || got != tc.samePointer {
					t.Errorf("ParametersJsonSchema = %#v, want the developer's schema pointer %p", decl.ParametersJsonSchema, tc.samePointer)
				}
				return
			}
			if diff := cmp.Diff(tc.want, decl.ParametersJsonSchema); diff != "" {
				t.Errorf("ParametersJsonSchema mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestDeclaration_RawJSONSchemaWithDefs checks that a raw schema's root
// definitions ($defs or definitions) move to the wrapper object, so its
// references still resolve from the root of the declared parameters, and that
// the caller's schema is left as it was.
func TestDeclaration_RawJSONSchemaWithDefs(t *testing.T) {
	item := func() map[string]*jsonschema.Schema {
		return map[string]*jsonschema.Schema{"Item": {
			Type:       "object",
			Properties: map[string]*jsonschema.Schema{"id": {Type: "integer"}},
		}}
	}
	tests := []struct {
		name string
		// raw returns a new copy of the developer's schema on each call.
		raw  func() *jsonschema.Schema
		want *jsonschema.Schema
	}{
		{
			name: "defs",
			raw: func() *jsonschema.Schema {
				return &jsonschema.Schema{
					Type:  "array",
					Items: &jsonschema.Schema{Ref: "#/$defs/Item"},
					Defs:  item(),
				}
			},
			want: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{"response": {
					Type:  "array",
					Items: &jsonschema.Schema{Ref: "#/$defs/Item"},
				}},
				Required: []string{"response"},
				Defs:     item(),
			},
		},
		{
			name: "definitions",
			raw: func() *jsonschema.Schema {
				return &jsonschema.Schema{
					Type:        "array",
					Items:       &jsonschema.Schema{Ref: "#/definitions/Item"},
					Definitions: item(),
				}
			},
			want: &jsonschema.Schema{
				Type: "object",
				Properties: map[string]*jsonschema.Schema{"response": {
					Type:  "array",
					Items: &jsonschema.Schema{Ref: "#/definitions/Item"},
				}},
				Required:    []string{"response"},
				Definitions: item(),
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw := tc.raw()

			got := newRawTool(t, raw).Declaration().ParametersJsonSchema

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("ParametersJsonSchema mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.raw(), raw); diff != "" {
				t.Errorf("caller's schema was modified (-want +got):\n%s", diff)
			}
			if s, ok := got.(*jsonschema.Schema); ok {
				if _, err := s.Resolve(nil); err != nil {
					t.Errorf("Resolve() on the declared parameters: %v", err)
				}
			}
		})
	}
}

// jsonOf returns v marshaled to JSON and decoded into generic values, so two
// schemas compare by their JSON regardless of key order.
func jsonOf(t *testing.T, v any) any {
	t.Helper()
	out, err := decodeJSON(v)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// decodeJSON is jsonOf for goroutines, which must not call t.Fatal.
func decodeJSON(v any) (any, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var out any
	err = json.Unmarshal(b, &out)
	return out, err
}

// literal decodes the JSON text s.
func literal(t *testing.T, s string) any {
	t.Helper()
	var out any
	if err := json.Unmarshal([]byte(s), &out); err != nil {
		t.Fatalf("bad literal: %v", err)
	}
	return out
}

// userSchema returns a new copy of a record whose titled nested model Address
// is used twice: once with a field description, once as a list item.
func userSchema() *genai.Schema {
	address := func(description string) *genai.Schema {
		return &genai.Schema{
			Type:        genai.TypeObject,
			Title:       "Address",
			Description: description,
			Properties: map[string]*genai.Schema{
				"street": {Type: genai.TypeString, Description: "Street address"},
				"city":   {Type: genai.TypeString},
			},
			Required: []string{"street", "city"},
		}
	}
	return &genai.Schema{
		Type:  genai.TypeObject,
		Title: "User",
		Properties: map[string]*genai.Schema{
			"name":    {Type: genai.TypeString},
			"address": address("User's address"),
			"previous_addresses": {
				Type:        genai.TypeArray,
				Description: "Past addresses",
				Default:     []any{},
				Items:       address(""),
			},
		},
		Required: []string{"name", "address"},
	}
}

const userJSON = `{
	"type": "object",
	"title": "User",
	"properties": {
		"name": {"type": "string"},
		"address": {"$ref": "#/$defs/Address", "description": "User's address"},
		"previous_addresses": {
			"type": "array",
			"description": "Past addresses",
			"default": [],
			"items": {"$ref": "#/$defs/Address"}
		}
	},
	"required": ["name", "address"],
	"$defs": {
		"Address": {
			"type": "object",
			"title": "Address",
			"properties": {
				"street": {"type": "string", "description": "Street address"},
				"city": {"type": "string"}
			},
			"required": ["street", "city"]
		}
	}
}`

// TestDeclaration_TitledNestedRecordIsHoistedOnce fails if a use site loses
// its description, if the second use of Address gets a def of its own, if a
// field description is copied into the def, or if nothing is hoisted.
func TestDeclaration_TitledNestedRecordIsHoistedOnce(t *testing.T) {
	got := newTool(t, userSchema()).Declaration().ParametersJsonSchema

	if diff := cmp.Diff(literal(t, userJSON), jsonOf(t, got)); diff != "" {
		t.Errorf("ParametersJsonSchema JSON mismatch (-want +got):\n%s", diff)
	}
}

// TestDeclaration_SameTitleDifferentSchemas fails if properties are visited in
// map order (the names swap between calls), if the suffix is omitted so one
// def overwrites the other, or if equal and different defs are not told apart.
func TestDeclaration_SameTitleDifferentSchemas(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"a": {Type: genai.TypeObject, Title: "Item", Properties: map[string]*genai.Schema{"x": {Type: genai.TypeString}}},
			"b": {Type: genai.TypeObject, Title: "Item", Properties: map[string]*genai.Schema{"y": {Type: genai.TypeInteger}}},
		},
	}
	want := literal(t, `{
		"type": "object",
		"properties": {
			"a": {"$ref": "#/$defs/Item"},
			"b": {"$ref": "#/$defs/Item_1"}
		},
		"$defs": {
			"Item": {"type": "object", "title": "Item", "properties": {"x": {"type": "string"}}},
			"Item_1": {"type": "object", "title": "Item", "properties": {"y": {"type": "integer"}}}
		}
	}`)
	tl := newTool(t, schema)

	for i := range 50 {
		if diff := cmp.Diff(want, jsonOf(t, tl.Declaration().ParametersJsonSchema)); diff != "" {
			t.Fatalf("call %d: ParametersJsonSchema JSON mismatch (-want +got):\n%s", i, diff)
		}
	}
}

// TestDeclaration_DoesNotModifyTheInputSchema fails if the conversion rewrites
// the developer's schema in place, which also makes the second call differ.
func TestDeclaration_DoesNotModifyTheInputSchema(t *testing.T) {
	passed := userSchema()
	tl := newTool(t, passed)

	first := jsonOf(t, tl.Declaration().ParametersJsonSchema)
	second := jsonOf(t, tl.Declaration().ParametersJsonSchema)

	if diff := cmp.Diff(userSchema(), passed); diff != "" {
		t.Errorf("developer's schema was modified (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(first, second); diff != "" {
		t.Errorf("second Declaration() differs from the first (-first +second):\n%s", diff)
	}
}

// TestDeclaration_FlatRootStaysGenaiSchema fails if the root's own title, or
// an untitled nested object, switches the declaration to the JSON form.
func TestDeclaration_FlatRootStaysGenaiSchema(t *testing.T) {
	schema := &genai.Schema{
		Type:  genai.TypeObject,
		Title: "Weather",
		Properties: map[string]*genai.Schema{
			"city":   {Type: genai.TypeString},
			"detail": {Type: genai.TypeObject, Properties: map[string]*genai.Schema{"k": {Type: genai.TypeString}}},
		},
	}
	want := &genai.Schema{
		Type:  genai.TypeObject,
		Title: "Weather",
		Properties: map[string]*genai.Schema{
			"city":   {Type: genai.TypeString},
			"detail": {Type: genai.TypeObject, Properties: map[string]*genai.Schema{"k": {Type: genai.TypeString}}},
		},
	}

	got, ok := newTool(t, schema).Declaration().ParametersJsonSchema.(*genai.Schema)

	if !ok {
		t.Fatalf("ParametersJsonSchema is not a *genai.Schema")
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("ParametersJsonSchema mismatch (-want +got):\n%s", diff)
	}
}

// TestDeclaration_RecordListJSONShape fails if types stay upper case, if a
// def loses its required list or a property default, or if the items wrapper
// is missing.
func TestDeclaration_RecordListJSONShape(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeArray,
		Items: &genai.Schema{
			Type:  genai.TypeObject,
			Title: "Item",
			Properties: map[string]*genai.Schema{
				"id":   {Type: genai.TypeInteger, Description: "Item ID"},
				"name": {Type: genai.TypeString},
				"qty":  {Type: genai.TypeInteger, Default: 1},
			},
			Required: []string{"id", "name"},
		},
	}
	want := literal(t, `{
		"type": "object",
		"properties": {
			"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}
		},
		"required": ["items"],
		"$defs": {
			"Item": {
				"type": "object",
				"title": "Item",
				"properties": {
					"id": {"type": "integer", "description": "Item ID"},
					"name": {"type": "string"},
					"qty": {"type": "integer", "default": 1}
				},
				"required": ["id", "name"]
			}
		}
	}`)

	got := newTool(t, schema).Declaration().ParametersJsonSchema

	if diff := cmp.Diff(want, jsonOf(t, got)); diff != "" {
		t.Errorf("ParametersJsonSchema JSON mismatch (-want +got):\n%s", diff)
	}
}

// TestDeclaration_DefsFormKeepsKeywords fails if the JSON form drops enum,
// format, pattern, a minimum or maximum, a length or item bound, the example
// or the anyOf arms of a property, at the root or inside a def.
func TestDeclaration_DefsFormKeepsKeywords(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"status": {Type: genai.TypeString, Enum: []string{"open", "closed"}},
			"email":  {Type: genai.TypeString, Format: "email", Pattern: "^.+@.+$", MinLength: genai.Ptr[int64](3), MaxLength: genai.Ptr[int64](64)},
			"qty":    {Type: genai.TypeInteger, Minimum: new(1.0), Maximum: new(10.0), Example: 2},
			"tags":   {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}, MinItems: genai.Ptr[int64](1), MaxItems: genai.Ptr[int64](5)},
			"id":     {AnyOf: []*genai.Schema{{Type: genai.TypeString}, {Type: genai.TypeInteger}}},
			"detail": {
				Type:  genai.TypeObject,
				Title: "Detail",
				Properties: map[string]*genai.Schema{
					"level": {Type: genai.TypeString, Enum: []string{"low", "high"}},
					"date":  {Type: genai.TypeString, Format: "date", Pattern: "^[0-9]{4}", MinLength: genai.Ptr[int64](4), MaxLength: genai.Ptr[int64](10)},
					"score": {Type: genai.TypeNumber, Minimum: new(0.5), Maximum: new(9.5), Example: 1.5},
					"codes": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeInteger}, MinItems: genai.Ptr[int64](2), MaxItems: genai.Ptr[int64](4)},
					"key":   {AnyOf: []*genai.Schema{{Type: genai.TypeInteger}, {Type: genai.TypeBoolean}}},
				},
			},
		},
	}
	want := literal(t, `{
		"type": "object",
		"properties": {
			"status": {"type": "string", "enum": ["open", "closed"]},
			"email": {"type": "string", "format": "email", "pattern": "^.+@.+$", "minLength": 3, "maxLength": 64},
			"qty": {"type": "integer", "minimum": 1, "maximum": 10, "examples": [2]},
			"tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5},
			"id": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
			"detail": {"$ref": "#/$defs/Detail"}
		},
		"$defs": {
			"Detail": {
				"type": "object",
				"title": "Detail",
				"properties": {
					"level": {"type": "string", "enum": ["low", "high"]},
					"date": {"type": "string", "format": "date", "pattern": "^[0-9]{4}", "minLength": 4, "maxLength": 10},
					"score": {"type": "number", "minimum": 0.5, "maximum": 9.5, "examples": [1.5]},
					"codes": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 4},
					"key": {"anyOf": [{"type": "integer"}, {"type": "boolean"}]}
				}
			}
		}
	}`)

	got := newTool(t, schema).Declaration().ParametersJsonSchema

	if diff := cmp.Diff(want, jsonOf(t, got)); diff != "" {
		t.Errorf("ParametersJsonSchema JSON mismatch (-want +got):\n%s", diff)
	}
}

// TestDeclaration_NullableUseSitesUseAnyOf fails if Nullable is dropped, if it
// is put inside the def, or if a nullable use site loses its description.
func TestDeclaration_NullableUseSitesUseAnyOf(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"customer": {
				Type:        genai.TypeObject,
				Title:       "Customer",
				Nullable:    new(true),
				Description: "Who",
				Properties:  map[string]*genai.Schema{"email": {Type: genai.TypeString}},
			},
			"note": {Type: genai.TypeString, Nullable: new(true), Description: "Free text"},
		},
		Required: []string{"customer"},
	}
	want := literal(t, `{
		"type": "object",
		"properties": {
			"customer": {"anyOf": [{"$ref": "#/$defs/Customer"}, {"type": "null"}], "description": "Who"},
			"note": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Free text"}
		},
		"required": ["customer"],
		"$defs": {
			"Customer": {"type": "object", "title": "Customer", "properties": {"email": {"type": "string"}}}
		}
	}`)

	got := newTool(t, schema).Declaration().ParametersJsonSchema

	if diff := cmp.Diff(want, jsonOf(t, got)); diff != "" {
		t.Errorf("ParametersJsonSchema JSON mismatch (-want +got):\n%s", diff)
	}
}

// TestDeclaration_DefsAreFlatAtTheRoot fails if a titled object inside a def
// is left inline or gets a $defs map of its own, where "#/$defs/..." from the
// parameters root would not find it.
func TestDeclaration_DefsAreFlatAtTheRoot(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"customer": {
				Type:  genai.TypeObject,
				Title: "Customer",
				Properties: map[string]*genai.Schema{
					"address": {Type: genai.TypeObject, Title: "Address", Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}}},
				},
			},
		},
	}
	want := literal(t, `{
		"type": "object",
		"properties": {"customer": {"$ref": "#/$defs/Customer"}},
		"$defs": {
			"Customer": {"type": "object", "title": "Customer", "properties": {"address": {"$ref": "#/$defs/Address"}}},
			"Address": {"type": "object", "title": "Address", "properties": {"city": {"type": "string"}}}
		}
	}`)

	got := newTool(t, schema).Declaration().ParametersJsonSchema

	if diff := cmp.Diff(want, jsonOf(t, got)); diff != "" {
		t.Errorf("ParametersJsonSchema JSON mismatch (-want +got):\n%s", diff)
	}
}

// treeNode returns an object schema with the given title whose "children"
// items are the schema itself: the only way a *genai.Schema can describe a
// recursive type.
func treeNode(title string) *genai.Schema {
	node := &genai.Schema{
		Type:  genai.TypeObject,
		Title: title,
		Properties: map[string]*genai.Schema{
			"name":     {Type: genai.TypeString},
			"children": {Type: genai.TypeArray},
		},
	}
	node.Properties["children"].Items = node
	return node
}

// TestDeclaration_TitledRecursiveObjectRefersToItself fails if the def of a
// titled object that contains itself is built without a visited set, which
// overflows the stack, or if its inner use is not a reference to its own def.
func TestDeclaration_TitledRecursiveObjectRefersToItself(t *testing.T) {
	schema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"root": treeNode("Node")},
	}
	want := literal(t, `{
		"type": "object",
		"properties": {"root": {"$ref": "#/$defs/Node"}},
		"$defs": {
			"Node": {
				"type": "object",
				"title": "Node",
				"properties": {
					"name": {"type": "string"},
					"children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}
				}
			}
		}
	}`)

	got := newTool(t, schema).Declaration().ParametersJsonSchema

	if diff := cmp.Diff(want, jsonOf(t, got)); diff != "" {
		t.Errorf("ParametersJsonSchema JSON mismatch (-want +got):\n%s", diff)
	}
}

// TestDeclaration_UntitledCycleKeepsGenaiSchema fails if a cycle with no
// titled object to refer to is walked without a visited set, which overflows
// the stack, or if it is declared as anything but the *genai.Schema itself.
func TestDeclaration_UntitledCycleKeepsGenaiSchema(t *testing.T) {
	schema := treeNode("")

	got := newTool(t, schema).Declaration().ParametersJsonSchema

	if got != schema {
		t.Errorf("ParametersJsonSchema = %T %p, want the schema passed to New (%p)", got, got, schema)
	}
}

// TestDeclaration_RefResolvesForAnyTitle fails if a def's key is put into the
// "$ref" without escaping it as a JSON Pointer token ("/" and "~") and then as
// a URI fragment ("%" and space), so the reference does not resolve or is not
// a valid URI reference.
func TestDeclaration_RefResolvesForAnyTitle(t *testing.T) {
	tests := []struct {
		title   string
		wantRef string
	}{
		{title: "Request/Reply", wantRef: "#/$defs/Request~1Reply"},
		{title: "Price %", wantRef: "#/$defs/Price%20%25"},
		{title: "A~B", wantRef: "#/$defs/A~0B"},
		{title: "Shipping address", wantRef: "#/$defs/Shipping%20address"},
		{title: "Address", wantRef: "#/$defs/Address"},
	}
	for _, tt := range tests {
		t.Run(tt.title, func(t *testing.T) {
			schema := &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"nested": {
						Type:       genai.TypeObject,
						Title:      tt.title,
						Properties: map[string]*genai.Schema{"x": {Type: genai.TypeString}},
						Required:   []string{"x"},
					},
				},
			}

			got, ok := newTool(t, schema).Declaration().ParametersJsonSchema.(*jsonschema.Schema)

			if !ok {
				t.Fatal("ParametersJsonSchema is not a *jsonschema.Schema")
			}
			if ref := got.Properties["nested"].Ref; ref != tt.wantRef {
				t.Errorf("nested $ref = %q, want %q", ref, tt.wantRef)
			}
			if _, ok := got.Defs[tt.title]; !ok {
				t.Errorf("$defs has no key %q", tt.title)
			}
			resolved, err := got.Resolve(nil)
			if err != nil {
				t.Fatalf("Resolve() error = %v", err)
			}
			if err := resolved.Validate(map[string]any{"nested": map[string]any{"x": "s"}}); err != nil {
				t.Errorf("Validate(valid value) error = %v", err)
			}
			if err := resolved.Validate(map[string]any{"nested": map[string]any{"x": 1.0}}); err == nil {
				t.Error("Validate(x is a number) = nil, want an error from the def")
			}
		})
	}
}
