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

package telemetry

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestConvertersRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		val  any
		want any
	}{
		{
			name: "nil",
			val:  nil,
			want: nil,
		},
		{
			name: "string",
			val:  "hello",
			want: "hello",
		},
		{
			name: "bool",
			val:  true,
			want: true,
		},
		{
			name: "float64",
			val:  123.456,
			want: 123.456,
		},
		{
			name: "int to int64",
			val:  int(123),
			want: int64(123),
		},
		{
			name: "slice of mixed types",
			val:  []any{1.0, true, "foo"},
			want: []any{1.0, true, "foo"},
		},
		{
			name: "map",
			val: map[string]any{
				"foo": "bar",
				"baz": 123.0,
			},
			want: map[string]any{
				"foo": "bar",
				"baz": 123.0,
			},
		},
		{
			name: "nested structure",
			val: map[string]any{
				"list": []any{
					map[string]any{"a": 1.0},
				},
			},
			want: map[string]any{
				"list": []any{
					map[string]any{"a": 1.0},
				},
			},
		},
		{
			name: "fallback for unsupported type",
			val:  struct{ A int }{A: 1},
			want: "{1}",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Convert to log.Value
			val := toLogValue(tc.val)
			// Convert back to any
			got := FromLogValue(val)

			// Assert that result is the same as the expected want
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("Round trip conversion mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
