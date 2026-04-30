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

package engine

import "testing"

func TestJoinPath(t *testing.T) {
	cases := []struct {
		parent, name string
		run          int
		want         string
	}{
		{"", "wf", 1, "wf@1"},
		{"wf@1", "classify", 1, "wf@1/classify@1"},
		{"wf@1/loop@1", "step", 3, "wf@1/loop@1/step@3"},
	}
	for _, tc := range cases {
		got := JoinPath(tc.parent, tc.name, tc.run)
		if got != tc.want {
			t.Errorf("JoinPath(%q,%q,%d) = %q, want %q", tc.parent, tc.name, tc.run, got, tc.want)
		}
	}
}
