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
)

// A nil []any is JSON null, and its copy stays nil rather than becoming an
// empty slice, which would encode as [].
func TestDeepCopy_NilStaysNil(t *testing.T) {
	tests := []struct {
		name string
		in   any
		want any
	}{
		{name: "nil slice", in: []any(nil), want: []any(nil)},
		{name: "nil slice in a map", in: map[string]any{"tags": []any(nil)}, want: map[string]any{"tags": []any(nil)}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if diff := cmp.Diff(tc.want, DeepCopy(tc.in)); diff != "" {
				t.Errorf("DeepCopy(%#v) mismatch (-want +got):\n%s", tc.in, diff)
			}
		})
	}
}
