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
		name        string
		input       string
		wantTurns   int
		wantPrompts []string
	}{
		{
			name:        "newline terminated input",
			input:       "hello\n",
			wantTurns:   1,
			wantPrompts: []string{"Agent -> hello"},
		},
		{
			name:        "single line without newline",
			input:       "hello",
			wantTurns:   1,
			wantPrompts: []string{"Agent -> hello"},
		},
		{
			name:        "multiple lines ending without newline",
			input:       "first\nlast",
			wantTurns:   2,
			wantPrompts: []string{"Agent -> first", "Agent -> last"},
		},
		{
			name:        "empty input",
			input:       "",
			wantTurns:   0,
			wantPrompts: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			config := testConfig(t, nil)
			out := runConsole(t, config, tc.input)

			if !strings.Contains(out, "EOF detected, exiting...") {
				t.Errorf("expected output to contain EOF exit message, but got:\n%s", out)
			}

			if gotTurns := strings.Count(out, "Agent -> "); gotTurns != tc.wantTurns {
				t.Errorf("got %d agent turns, want %d. Output:\n%s", gotTurns, tc.wantTurns, out)
			}

			lastIdx := 0
			for _, prompt := range tc.wantPrompts {
				idx := strings.Index(out[lastIdx:], prompt)
				if idx == -1 {
					t.Errorf("expected prompt %q not found in order after index %d. Output:\n%s", prompt, lastIdx, out)
					break
				}
				lastIdx += idx + len(prompt)
			}
		})
	}
}
