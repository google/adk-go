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

type strictLoopState struct {
	CurrentSubAgent string `json:"current_sub_agent"`
	TimesLooped     uint   `json:"times_looped"`
}

func TestFromMapStructureStrict(t *testing.T) {
	tests := []struct {
		name    string
		data    map[string]any
		want    *strictLoopState
		wantErr bool
	}{
		{
			name: "int value",
			data: map[string]any{"current_sub_agent": "a2", "times_looped": 2},
			want: &strictLoopState{CurrentSubAgent: "a2", TimesLooped: 2},
		},
		{
			name: "float64 value from a JSON round trip",
			data: map[string]any{"current_sub_agent": "a2", "times_looped": float64(3)},
			want: &strictLoopState{CurrentSubAgent: "a2", TimesLooped: 3},
		},
		{
			name: "empty map",
			data: map[string]any{},
			want: &strictLoopState{},
		},
		{
			name:    "unknown key",
			data:    map[string]any{"unknown_field": "x"},
			wantErr: true,
		},
		{
			name:    "negative uint",
			data:    map[string]any{"times_looped": -1},
			wantErr: true,
		},
		{
			name:    "fractional uint",
			data:    map[string]any{"times_looped": 1.5},
			wantErr: true,
		},
		{
			name:    "string for uint",
			data:    map[string]any{"times_looped": "1"},
			wantErr: true,
		},
		{
			name:    "number for string",
			data:    map[string]any{"current_sub_agent": 123},
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := FromMapStructureStrict[strictLoopState](tc.data)
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Fatalf("FromMapStructureStrict(%v) error = %v; want error %v", tc.data, err, tc.wantErr)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("FromMapStructureStrict(%v) mismatch (-want +got):\n%s", tc.data, diff)
			}
		})
	}
}
