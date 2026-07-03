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

package replayplugin

import "testing"

func TestNormalizeDescription(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "Simple single line",
			input:    "Simple description",
			expected: "Simple description",
		},
		{
			name:     "Single line with leading and trailing spaces",
			input:    "   Simple description   ",
			expected: "Simple description",
		},
		{
			name:     "Multiple lines with indentation",
			input:    "  Line 1  \n  Line 2  ",
			expected: "Line 1\nLine 2",
		},
		{
			name:     "Empty lines at start and end",
			input:    "\n\nLine 1\nLine 2\n\n",
			expected: "Line 1\nLine 2",
		},
		{
			name:     "Whitespace-only lines at start and end",
			input:    "   \n  \nLine 1\nLine 2\n  \n   ",
			expected: "Line 1\nLine 2",
		},
		{
			name:     "Empty lines in the middle are preserved",
			input:    "Line 1\n\nLine 2",
			expected: "Line 1\n\nLine 2",
		},
		{
			name:     "Empty input",
			input:    "",
			expected: "",
		},
		{
			name:     "Whitespace-only input",
			input:    "    \n   \n  ",
			expected: "",
		},
		{
			name:     "Complex formatting with tabs and newlines",
			input:    "\t\n  Line 1 \t \n\tLine 2\n  \t\n",
			expected: "Line 1\nLine 2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := normalizeDescription(tt.input)
			if actual != tt.expected {
				t.Errorf("normalizeDescription(%q) = %q; want %q", tt.input, actual, tt.expected)
			}
		})
	}
}
