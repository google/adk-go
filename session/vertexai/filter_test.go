// Copyright 2025 Google LLC
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

package vertexai

import "testing"

func TestQuoteFilterLiteral(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "plain value",
			input: "alice",
			want:  `"alice"`,
		},
		{
			// A double quote in the value must not break out of the literal and
			// append an OR predicate that would return every user's sessions.
			name:  "quote injection is neutralized",
			input: `attacker" OR userId!="`,
			want:  `"attacker\" OR userId!=\""`,
		},
		{
			name:  "backslash only",
			input: `\`,
			want:  `"\\"`,
		},
		{
			name:  "empty string",
			input: "",
			want:  `""`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := quoteFilterLiteral(tc.input); got != tc.want {
				t.Errorf("quoteFilterLiteral(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}
