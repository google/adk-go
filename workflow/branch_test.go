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

import "testing"

func TestDeriveSubBranch(t *testing.T) {
	tests := []struct {
		name, parent, segment, want string
	}{
		{"root_and_segment", "", "child@1", "child@1"},
		{"parent_and_segment", "wf", "child@1", "wf.child@1"},
		{"nested_parent", "wf.outer@1", "child@2", "wf.outer@1.child@2"},
		{"empty_segment_noop", "wf", "", "wf"},
		{"empty_both", "", "", ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := deriveSubBranch(tc.parent, tc.segment)
			if got != tc.want {
				t.Errorf("deriveSubBranch(%q, %q) = %q, want %q",
					tc.parent, tc.segment, got, tc.want)
			}
		})
	}
}

func TestCommonBranchPrefix(t *testing.T) {
	tests := []struct {
		name     string
		branches []string
		want     string
	}{
		{"empty_input", nil, ""},
		{"single_branch", []string{"a.b.c"}, "a.b.c"},
		{"single_empty_branch", []string{""}, ""},
		{"any_empty_short_circuits_to_root", []string{"a.b", "", "a.b.c"}, ""},
		{"identical_branches", []string{"a.b", "a.b", "a.b"}, "a.b"},
		{"shared_prefix_one_segment", []string{"a.b", "a.c"}, "a"},
		{"shared_prefix_two_segments", []string{"a.b.c", "a.b.d"}, "a.b"},
		{"no_common_prefix", []string{"a", "b"}, ""},
		{
			// segment-aware comparison: "a" and "ab" share NO segments
			// (not "a" as string prefix).
			name:     "string_prefix_but_no_segment_prefix",
			branches: []string{"a", "ab"},
			want:     "",
		},
		{
			// Mixed depth: deeper branches still share the shallower prefix.
			name:     "mixed_depth",
			branches: []string{"a.b", "a.b.c.d"},
			want:     "a.b",
		},
		{
			// Three branches, deepest common is two segments.
			name:     "three_branches_two_common",
			branches: []string{"a.b.x", "a.b.y", "a.b.z.w"},
			want:     "a.b",
		},
		{
			// First segment differs → root.
			name:     "first_segment_differs",
			branches: []string{"a.b.c", "x.b.c"},
			want:     "",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := commonBranchPrefix(tc.branches)
			if got != tc.want {
				t.Errorf("commonBranchPrefix(%v) = %q, want %q",
					tc.branches, got, tc.want)
			}
		})
	}
}
