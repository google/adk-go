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

import (
	"testing"

	"gopkg.in/yaml.v3"
)

func TestRemoveUnderscores(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name: "strips underscores from standard keys",
			input: `
first_name: "John"
last_name: "Doe"
`,
			expected: `
firstname: "John"
lastname: "Doe"
`,
		},
		{
			name: "ignores fields specified in toIgnore",
			input: `
thought_signature: "SGVsbG8="
http_options: "options"
args: "arguments"
response: "result"
`,
			expected: `
thought_signature: "SGVsbG8="
http_options: "options"
args: "arguments"
response: "result"
`,
		},
		{
			name: "ignores but recurses fields specified in toIgnoreButRecurse",
			input: `
user_message_index: 0
agent_name: "test"
llm_recording:
  nested_field_with_underscore: "nested"
`,
			expected: `
user_message_index: 0
agent_name: "test"
llm_recording:
  nestedfieldwithunderscore: "nested"
`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var node yaml.Node
			err := yaml.Unmarshal([]byte(tc.input), &node)
			if err != nil {
				t.Fatalf("Unmarshal failed: %v", err)
			}

			normalizeYAMLNode(&node)

			output, err := yaml.Marshal(&node)
			if err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}

			gotClean := normalizeYAML(string(output))
			expectedClean := normalizeYAML(tc.expected)
			if gotClean != expectedClean {
				t.Errorf("mismatch.\nGot:\n%s\nExpected:\n%s", gotClean, expectedClean)
			}
		})
	}
}

func TestFixTypeMismatches(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name: "wraps llmresponse mapping into sequence llm_responses",
			input: `
llmresponse:
  content:
    role: "model"
`,
			expected: `
llm_responses:
  - content:
      role: "model"
`,
		},
		{
			name: "wraps llm_response mapping into sequence llm_responses",
			input: `
llm_response:
  content:
    role: "model"
`,
			expected: `
llm_responses:
  - content:
      role: "model"
`,
		},
		{
			name: "normalizes systeminstruction scalar string to structured object",
			input: `
systeminstruction: "You are a helpful assistant."
`,
			expected: `
systeminstruction:
  role: user
  parts:
    - text: You are a helpful assistant.
`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var node yaml.Node
			err := yaml.Unmarshal([]byte(tc.input), &node)
			if err != nil {
				t.Fatalf("Unmarshal failed: %v", err)
			}

			normalizeYAMLNode(&node)

			output, err := yaml.Marshal(&node)
			if err != nil {
				t.Fatalf("Marshal failed: %v", err)
			}

			gotClean := normalizeYAML(string(output))
			expectedClean := normalizeYAML(tc.expected)
			if gotClean != expectedClean {
				t.Errorf("mismatch.\nGot:\n%s\nExpected:\n%s", gotClean, expectedClean)
			}
		})
	}
}

func normalizeYAML(s string) string {
	var node yaml.Node
	if err := yaml.Unmarshal([]byte(s), &node); err != nil {
		return s
	}
	out, _ := yaml.Marshal(&node)
	return string(out)
}
