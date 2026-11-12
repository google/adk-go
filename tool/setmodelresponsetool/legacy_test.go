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
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/setmodelresponsetool"
)

// legacy returns AsLegacy(tl) as a tool the flow can run.
func legacy(t *testing.T, tl tool.Tool) toolinternal.FunctionTool {
	t.Helper()
	return functionTool(t, setmodelresponsetool.AsLegacy(tl), nil)
}

// typedParameters returns the declaration's typed parameters, and fails the
// test if the JSON-schema field is also set.
func typedParameters(t *testing.T, decl *genai.FunctionDeclaration) *genai.Schema {
	t.Helper()
	if decl.ParametersJsonSchema != nil {
		t.Errorf("ParametersJsonSchema is a %T, want nil", decl.ParametersJsonSchema)
	}
	return decl.Parameters
}

// userTyped is the typed form of userSchema: every title is cleared, and
// nothing else changes.
func userTyped() *genai.Schema {
	address := func(description string) *genai.Schema {
		return &genai.Schema{
			Type:        genai.TypeObject,
			Description: description,
			Properties: map[string]*genai.Schema{
				"street": {Type: genai.TypeString, Description: "Street address"},
				"city":   {Type: genai.TypeString},
			},
			Required: []string{"street", "city"},
		}
	}
	return &genai.Schema{
		Type: genai.TypeObject,
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

// TestAsLegacy_RecordDeclaration fails if the JSON-schema field stays set, if
// a title is kept, or if Nullable or a nested description is dropped.
func TestAsLegacy_RecordDeclaration(t *testing.T) {
	schema := &genai.Schema{
		Type:  genai.TypeObject,
		Title: "Order",
		Properties: map[string]*genai.Schema{
			"customer": {
				Type:        genai.TypeObject,
				Title:       "Customer",
				Nullable:    new(true),
				Description: "Who",
				Properties:  map[string]*genai.Schema{"email": {Type: genai.TypeString, Description: "Mail"}},
			},
			"note": {Type: genai.TypeString, Nullable: new(true), Description: "Free text"},
		},
		Required: []string{"customer"},
	}
	want := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"customer": {
				Type:        genai.TypeObject,
				Nullable:    new(true),
				Description: "Who",
				Properties:  map[string]*genai.Schema{"email": {Type: genai.TypeString, Description: "Mail"}},
			},
			"note": {Type: genai.TypeString, Nullable: new(true), Description: "Free text"},
		},
		Required: []string{"customer"},
	}
	tl, err := setmodelresponsetool.New(schema)
	if err != nil {
		t.Fatal(err)
	}

	got := typedParameters(t, legacy(t, tl).Declaration())

	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Parameters mismatch (-want +got):\n%s", diff)
	}
}

// TestAsLegacy_ClearsTitlesInAnyOfArms fails if the typed form keeps the title
// of an anyOf arm, or clears it on the developer's schema.
func TestAsLegacy_ClearsTitlesInAnyOfArms(t *testing.T) {
	schema := func() *genai.Schema {
		return &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"contact": {AnyOf: []*genai.Schema{
					{Type: genai.TypeObject, Title: "Phone", Properties: map[string]*genai.Schema{"number": {Type: genai.TypeString}}},
					{Type: genai.TypeString},
				}},
			},
		}
	}
	want := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"contact": {AnyOf: []*genai.Schema{
				{Type: genai.TypeObject, Properties: map[string]*genai.Schema{"number": {Type: genai.TypeString}}},
				{Type: genai.TypeString},
			}},
		},
	}
	passed := schema()
	tl, err := setmodelresponsetool.New(passed)
	if err != nil {
		t.Fatal(err)
	}

	got := typedParameters(t, legacy(t, tl).Declaration())

	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Parameters mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(schema(), passed); diff != "" {
		t.Errorf("developer's schema was modified (-want +got):\n%s", diff)
	}
}

// TestAsLegacy_WrapperParameters fails if a wrapper key is wrong, if Required
// is empty, or if the record-list wrapper is missing.
func TestAsLegacy_WrapperParameters(t *testing.T) {
	tests := []struct {
		name   string
		schema *genai.Schema
		want   *genai.Schema
	}{
		{
			name: "record_list",
			schema: &genai.Schema{
				Type: genai.TypeArray,
				Items: &genai.Schema{
					Type:       genai.TypeObject,
					Title:      "Item",
					Properties: map[string]*genai.Schema{"id": {Type: genai.TypeInteger, Description: "Item ID"}},
				},
			},
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{"items": {
					Type: genai.TypeArray,
					Items: &genai.Schema{
						Type:       genai.TypeObject,
						Properties: map[string]*genai.Schema{"id": {Type: genai.TypeInteger, Description: "Item ID"}},
					},
				}},
				Required: []string{"items"},
			},
		},
		{
			name:   "list_of_strings",
			schema: &genai.Schema{Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}},
			want: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{"response": {
					Type:  genai.TypeArray,
					Items: &genai.Schema{Type: genai.TypeString},
				}},
				Required: []string{"response"},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tl, err := setmodelresponsetool.New(tc.schema)
			if err != nil {
				t.Fatal(err)
			}

			got := typedParameters(t, legacy(t, tl).Declaration())

			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("Parameters mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestAsLegacy_RawJSONSchemaTool fails if a tool built from a raw JSON schema
// is returned unchanged, so additionalProperties reaches a typed request.
func TestAsLegacy_RawJSONSchemaTool(t *testing.T) {
	tl, err := setmodelresponsetool.NewFromJSONSchema(&jsonschema.Schema{
		Type:                 "object",
		AdditionalProperties: &jsonschema.Schema{Type: "integer"},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"response": {Type: genai.TypeObject}},
		Required:   []string{"response"},
	}

	got := typedParameters(t, legacy(t, tl).Declaration())

	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("Parameters mismatch (-want +got):\n%s", diff)
	}
}

// TestAsLegacy_RunDelegatesToTheTool fails if the wrapper does not forward
// Run, Name, Description or IsLongRunning to the tool it wraps.
func TestAsLegacy_RunDelegatesToTheTool(t *testing.T) {
	tl, err := setmodelresponsetool.New(&genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"count": {Type: genai.TypeInteger}},
	})
	if err != nil {
		t.Fatal(err)
	}
	original := functionTool(t, tl, nil)
	wrapped := legacy(t, tl)

	for name, ft := range map[string]toolinternal.FunctionTool{"original": original, "legacy": wrapped} {
		ctx := newToolContext(t)
		got, err := ft.Run(ctx, map[string]any{"count": 10.0})
		if err != nil {
			t.Fatalf("%s: Run() error: %v", name, err)
		}
		if diff := cmp.Diff(map[string]any{"count": 10.0}, got); diff != "" {
			t.Errorf("%s: Run() result mismatch (-want +got):\n%s", name, diff)
		}
		if diff := cmp.Diff(any(map[string]any{"count": 10.0}), ctx.Actions().SetModelResponse); diff != "" {
			t.Errorf("%s: SetModelResponse mismatch (-want +got):\n%s", name, diff)
		}
	}
	if got := wrapped.Name(); got != "set_model_response" {
		t.Errorf("Name() = %q, want %q", got, "set_model_response")
	}
	if got, want := wrapped.Description(), original.Description(); got != want {
		t.Errorf("Description() = %q, want %q", got, want)
	}
	if wrapped.IsLongRunning() {
		t.Error("IsLongRunning() = true, want false")
	}
}

// TestAsLegacy_OtherToolsAreReturnedAsTheyAre fails if AsLegacy wraps a tool
// of another package, or panics on nil.
func TestAsLegacy_OtherToolsAreReturnedAsTheyAre(t *testing.T) {
	other, err := functiontool.New(functiontool.Config{Name: "other"}, func(agent.Context, struct{}) (struct{}, error) {
		return struct{}{}, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	if got := setmodelresponsetool.AsLegacy(nil); got != nil {
		t.Errorf("AsLegacy(nil) = %#v, want nil", got)
	}
	if got := setmodelresponsetool.AsLegacy(other); got != other {
		t.Errorf("AsLegacy(functiontool) = %#v, want the same tool %#v", got, other)
	}
}

// TestAsLegacy_KeepsTheCallersSchema fails if the typed form clears the titles
// of the developer's schema in place, after which the JSON form finds no
// titled object and sends no $defs.
func TestAsLegacy_KeepsTheCallersSchema(t *testing.T) {
	passed := userSchema()
	tl, err := setmodelresponsetool.New(passed)
	if err != nil {
		t.Fatal(err)
	}

	legacy(t, tl).Declaration()
	got := functionTool(t, tl, nil).Declaration().ParametersJsonSchema

	if diff := cmp.Diff(userSchema(), passed); diff != "" {
		t.Errorf("developer's schema was modified (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(literal(t, userJSON), jsonOf(t, got)); diff != "" {
		t.Errorf("JSON form after AsLegacy mismatch (-want +got):\n%s", diff)
	}
}

// TestAsLegacy_DeclarationIsRepeatable fails if the wrapper hands out one
// cached tree, so a change a caller makes to one declaration shows up in the
// next.
func TestAsLegacy_DeclarationIsRepeatable(t *testing.T) {
	tl, err := setmodelresponsetool.New(userSchema())
	if err != nil {
		t.Fatal(err)
	}
	ft := legacy(t, tl)

	first := typedParameters(t, ft.Declaration())
	if first == nil {
		t.Fatal("Parameters = nil")
	}
	first.Properties["name"].Description = "changed by a caller"
	second := typedParameters(t, ft.Declaration())

	if diff := cmp.Diff(userTyped(), second); diff != "" {
		t.Errorf("second Declaration() mismatch (-want +got):\n%s", diff)
	}
}

// TestDeclaration_ConcurrentUse fails under -race if a declaration is built
// into shared state, and fails outright if concurrent calls see each other's
// partial results. The JSON form and the typed form read the same schema.
func TestDeclaration_ConcurrentUse(t *testing.T) {
	tl, err := setmodelresponsetool.New(userSchema())
	if err != nil {
		t.Fatal(err)
	}
	jsonTool := functionTool(t, tl, nil)
	typedTool := legacy(t, tl)
	// Canonical JSON (sorted keys) of each expected form, so the loop compares
	// bytes. A race needs overlapping calls, not many of them, so each
	// goroutine makes 25.
	wantJSON := canonicalJSON(t, literal(t, userJSON))
	wantTyped := canonicalJSON(t, jsonOf(t, userTyped()))

	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			for range 25 {
				if got, err := canonical(jsonTool.Declaration().ParametersJsonSchema); err != nil || got != wantJSON {
					t.Errorf("JSON form = %s, %v; want %s", got, err, wantJSON)
					return
				}
				if got, err := canonical(typedTool.Declaration().Parameters); err != nil || got != wantTyped {
					t.Errorf("typed form = %s, %v; want %s", got, err, wantTyped)
					return
				}
			}
		})
	}
	wg.Wait()
}

func canonicalJSON(t *testing.T, v any) string {
	t.Helper()
	s, err := canonical(v)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// canonical returns the JSON of v with object keys sorted.
func canonical(v any) (string, error) {
	decoded, err := decodeJSON(v)
	if err != nil {
		return "", err
	}
	b, err := json.Marshal(decoded)
	return string(b), err
}

// TestAsLegacy_CyclicSchema fails if the titles are cleared without a visited
// set, which overflows the stack on a schema that contains itself, if the copy
// loses its cycle, or if the developer's schema is modified.
func TestAsLegacy_CyclicSchema(t *testing.T) {
	passed := treeNode("Node")
	tl, err := setmodelresponsetool.New(passed)
	if err != nil {
		t.Fatal(err)
	}

	got := typedParameters(t, legacy(t, tl).Declaration())

	if got == nil || got == passed {
		t.Fatalf("Parameters = %p, want a copy of the schema %p", got, passed)
	}
	if got.Title != "" {
		t.Errorf("Parameters.Title = %q, want empty", got.Title)
	}
	if items := got.Properties["children"].Items; items != got {
		t.Errorf("Parameters children items = %p, want the copy itself (%p)", items, got)
	}
	if passed.Title != "Node" {
		t.Errorf("developer's schema Title = %q, want %q", passed.Title, "Node")
	}
}
