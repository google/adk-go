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

package typeutil

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"
)

func TestGenaiSchemaToJSONMap(t *testing.T) {
	tests := []struct {
		name    string
		schema  *genai.Schema
		want    map[string]any
		wantErr bool
	}{
		{
			name:   "nil schema",
			schema: nil,
			want:   nil,
		},
		{
			name:   "string type",
			schema: &genai.Schema{Type: genai.TypeString},
			want:   map[string]any{"type": "string"},
		},
		{
			name: "nested types are lowercased at every depth",
			schema: &genai.Schema{
				Type: genai.TypeObject,
				Properties: map[string]*genai.Schema{
					"a": {Type: genai.TypeArray, Items: &genai.Schema{Type: genai.TypeInteger}},
				},
			},
			want: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"a": map[string]any{
						"type":  "array",
						"items": map[string]any{"type": "integer"},
					},
				},
			},
		},
		{
			name: "types inside a list of schemas are lowercased",
			schema: &genai.Schema{
				AnyOf: []*genai.Schema{{Type: genai.TypeString}, {Type: genai.TypeInteger}},
			},
			want: map[string]any{
				"anyOf": []any{
					map[string]any{"type": "string"},
					map[string]any{"type": "integer"},
				},
			},
		},
		{
			name:    "schema that does not marshal",
			schema:  &genai.Schema{Example: make(chan int)},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := GenaiSchemaToJSONMap(tc.schema)
			if (err != nil) != tc.wantErr {
				t.Fatalf("GenaiSchemaToJSONMap() error = %v, wantErr %v", err, tc.wantErr)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("GenaiSchemaToJSONMap() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
