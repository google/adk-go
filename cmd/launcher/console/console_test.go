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

package console

import (
	"strings"
	"testing"
)

func TestConsolePipedPromptWithoutTrailingNewline(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		wantSubstr []string
	}{
		{
			name:  "single line without newline",
			input: "hello",
			wantSubstr: []string{
				"Agent -> hello",
				"EOF detected, exiting...",
			},
		},
		{
			name:  "multiple lines ending without newline",
			input: "first\nlast",
			wantSubstr: []string{
				"Agent -> first",
				"Agent -> last",
				"EOF detected, exiting...",
			},
		},
		{
			name:  "empty input",
			input: "",
			wantSubstr: []string{
				"EOF detected, exiting...",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			config := testConfig(t, nil)
			out := runConsole(t, config, tc.input)

			for _, substr := range tc.wantSubstr {
				if !strings.Contains(out, substr) {
					t.Errorf("expected output to contain %q, but got:\n%s", substr, out)
				}
			}
		})
	}
}
