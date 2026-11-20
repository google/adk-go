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
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/toolinternal"
)

func TestRun_ValueAndRawShapes_Valid(t *testing.T) {
	tests := []struct {
		name       string
		tool       toolinternal.FunctionTool
		args       map[string]any
		wantResult map[string]any
		wantStored any
	}{
		{
			name:       "value_list_of_strings",
			tool:       newTool(t, &genai.Schema{Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeString}}),
			args:       map[string]any{"response": []any{"a", "b"}},
			wantResult: map[string]any{"result": []any{"a", "b"}},
			wantStored: []any{"a", "b"},
		},
		{
			name:       "value_primitive_int_unvalidated",
			tool:       newTool(t, &genai.Schema{Type: genai.TypeInteger}),
			args:       map[string]any{"response": "not_validated"},
			wantResult: map[string]any{"result": "not_validated"},
			wantStored: "not_validated",
		},
		{
			name:       "value_freeform_object_map",
			tool:       newTool(t, &genai.Schema{Type: genai.TypeObject}),
			args:       map[string]any{"response": map[string]any{"k": "v"}},
			wantResult: map[string]any{"k": "v"},
			wantStored: map[string]any{"k": "v"},
		},
		{
			name: "raw_map",
			tool: newRawTool(t, &jsonschema.Schema{
				Type:       "object",
				Properties: map[string]*jsonschema.Schema{"result": {Type: "string"}},
			}),
			args:       map[string]any{"response": map[string]any{"result": float64(999)}},
			wantResult: map[string]any{"result": float64(999)},
			wantStored: map[string]any{"result": float64(999)},
		},
		{
			name:       "raw_non_map",
			tool:       newRawTool(t, &jsonschema.Schema{Type: "array"}),
			args:       map[string]any{"response": []any{float64(1)}},
			wantResult: map[string]any{"result": []any{float64(1)}},
			wantStored: []any{float64(1)},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := newToolContext(t)

			got, err := tc.tool.Run(ctx, tc.args)
			if err != nil {
				t.Fatalf("Run() error: %v", err)
			}
			if diff := cmp.Diff(tc.wantResult, got); diff != "" {
				t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantStored, ctx.Actions().SetModelResponse); diff != "" {
				t.Errorf("SetModelResponse mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRun_RecordList_ValidAndEmpty(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeArray,
		Items: &genai.Schema{
			Type:  genai.TypeObject,
			Title: "ItemSchema",
			Properties: map[string]*genai.Schema{
				"id":   {Type: genai.TypeInteger},
				"name": {Type: genai.TypeString, Default: "unnamed"},
			},
			Required: []string{"id"},
		},
	}

	t.Run("empty_list", func(t *testing.T) {
		ctx := newToolContext(t)

		got, err := newTool(t, schema).Run(ctx, map[string]any{"items": []any{}})
		if err != nil {
			t.Fatalf("Run() error: %v", err)
		}
		if diff := cmp.Diff(map[string]any{"result": []any{}}, got); diff != "" {
			t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
		}
		stored, ok := ctx.Actions().SetModelResponse.([]any)
		if !ok || stored == nil || len(stored) != 0 {
			t.Errorf("SetModelResponse = %#v, want a non-nil empty []any", ctx.Actions().SetModelResponse)
		}
	})

	t.Run("non_empty_list", func(t *testing.T) {
		ctx := newToolContext(t)
		args := map[string]any{"items": []any{
			map[string]any{"id": "1", "extra": "drop"},
			map[string]any{"id": float64(2), "name": "two"},
		}}
		wantItems := []any{
			map[string]any{"id": float64(1), "name": "unnamed"},
			map[string]any{"id": float64(2), "name": "two"},
		}

		got, err := newTool(t, schema).Run(ctx, args)
		if err != nil {
			t.Fatalf("Run() error: %v", err)
		}
		if diff := cmp.Diff(map[string]any{"result": wantItems}, got); diff != "" {
			t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
		}
		if diff := cmp.Diff(wantItems, ctx.Actions().SetModelResponse); diff != "" {
			t.Errorf("SetModelResponse mismatch (-want +got):\n%s", diff)
		}
	})
}
