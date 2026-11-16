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

package converters

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestCamelizeKeys_Collision(t *testing.T) {
	tests := []struct {
		name string
		in   map[string]any
		want any
	}{
		{
			name: "camelCase key beats its snake_case spelling",
			in:   map[string]any{"tool_type": "A", "toolType": "B"},
			want: map[string]any{"toolType": "B"},
		},
		{
			name: "camelCase key wins when a snake_case key sorts before it",
			in:   map[string]any{"_type": "A", "type": "B"},
			want: map[string]any{"type": "B"},
		},
		{
			name: "first snake_case key in sorted order wins",
			in:   map[string]any{"tool__type": "A", "tool_type": "C"},
			want: map[string]any{"toolType": "A"},
		},
		{
			name: "user-data key follows the same rule",
			in: map[string]any{
				"part_metadata": map[string]any{"a": 1},
				"partMetadata":  map[string]any{"b": 2},
			},
			want: map[string]any{"partMetadata": map[string]any{"b": 2}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// A winner picked by map iteration order is right about half the
			// time, so one call proves little.
			for range 20 {
				if diff := cmp.Diff(tt.want, CamelizeKeys(tt.in)); diff != "" {
					t.Fatalf("CamelizeKeys(%v) mismatch (-want +got):\n%s", tt.in, diff)
				}
			}
		})
	}
}
