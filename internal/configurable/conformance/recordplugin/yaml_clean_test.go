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

package recordplugin

import (
	"testing"

	"google.golang.org/genai"
	"gopkg.in/yaml.v3"
)

func TestCleanYAMLNode(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name: "prunes null fields",
			input: `
field1: null
field2: ~
field3: "value"
`,
			expected: `field3: "value"
`,
		},
		{
			name: "prunes default zero values and false booleans outside tool args",
			input: `
candidate_count: 0
max_output_tokens: 0
thought: false
text: "keep me"
`,
			expected: `text: "keep me"
`,
		},
		{
			name: "retains zero values and false booleans inside tool args",
			input: `
tool_call:
  name: "my_tool"
  args:
    param1: 0
    param2: false
`,
			expected: `tool_call:
  name: "my_tool"
  args:
    param1: 0
    param2: false
`,
		},
		{
			name: "retains zero values and false booleans inside tool responses",
			input: `
tool_response:
  response:
    success: false
    count: 0
`,
			expected: `tool_response:
  response:
    success: false
    count: 0
`,
		},
		{
			name: "prunes empty sequences but retains empty mappings",
			input: `
thought_signature: []
google_search: {}
text: "test"
`,
			expected: `google_search: {}
text: "test"
`,
		},
		{
			name: "retains thought_signature base64 scalar as string",
			input: `
text: "test"
thought_signature: "SGVsbG8="
`,
			expected: `text: "test"
thought_signature: "SGVsbG8="
`,
		},
		{
			name: "retains thoughtsignature base64 scalar as string",
			input: `
text: "test"
thoughtsignature: "SGVsbG8="
`,
			expected: `text: "test"
thought_signature: "SGVsbG8="
`,
		},
		{
			name: "converts thought_signature sequence of integers to base64 string",
			input: `
text: "test"
thought_signature:
  - 72
  - 101
  - 108
  - 108
  - 111
`,
			expected: `text: "test"
thought_signature: "SGVsbG8="
`,
		},
		{
			name: "converts thoughtsignature sequence of integers to base64 string",
			input: `
text: "test"
thoughtsignature:
  - 72
  - 101
  - 108
  - 108
  - 111
`,
			expected: `text: "test"
thought_signature: "SGVsbG8="
`,
		},
		{
			name: "deep nested pruning",
			input: `
recordings:
  - user_message_index: 0
    agent_name: "search"
    llm_recording:
      llm_request:
        model: "gemini"
        contents:
          - parts:
              - media_resolution: null
                text: "hello"
                thought_signature: []
`,
			expected: `recordings:
  - user_message_index: 0
    agent_name: "search"
    llm_recording:
      llm_request:
        model: "gemini"
        contents:
          - parts:
              - text: "hello"
`,
		},
		{
			name: "cleans bloated parameters schema and translates keys",
			input: `
functiondeclarations:
  - name: "validate_email"
    description: "checks email"
    parametersjsonschema:
      id: ""
      schema: ""
      ref: ""
      comment: ""
      defs: {}
      definitions: {}
      dependentrequired: {}
      extra: {}
      type: "object"
      required:
        - "email"
      properties:
        email:
          id: ""
          type: "string"
          description: "the email address"
          dependentrequired: {}
    responsejsonschema:
      type: "boolean"
`,
			expected: `
function_declarations:
  - name: "validate_email"
    description: "checks email"
    parameters_json_schema:
      title: "validate_emailParams"
      type: "object"
      required:
        - "email"
      properties:
        email:
          title: "Email"
          type: "string"
          description: "the email address"
`,
		},
		{
			name: "cleans schema parameters mapping additionalproperties and propertyorder, and pruning unsupported openapi properties",
			input: `
functiondeclarations:
  - name: "validate_email"
    parametersjsonschema:
      type: "object"
      required:
        - "email"
      properties:
        email:
          type: "string"
      additionalproperties:
        not: {}
      propertyorder:
        - "email"
      deprecated: false
      readonly: false
      writeonly: false
      uniqueitems: false
`,
			expected: `
function_declarations:
  - name: "validate_email"
    parameters_json_schema:
      title: "validate_emailParams"
      type: "object"
      required:
        - "email"
      properties:
        email:
          title: "Email"
          type: "string"
`,
		},
		{
			name: "prunes empty strings for general fields but retains them inside tool args",
			input: `
llm_recording:
  llm_responses:
    - content:
        parts:
          - text: "done"
      session_resumption_handle: ""
      error_code: ""
      error_message: ""
tool_call:
  name: "my_tool"
  args:
    param_empty_str: ""
`,
			expected: `
llm_recording:
  llm_responses:
    - content:
        parts:
          - text: "done"
tool_call:
  name: "my_tool"
  args:
    param_empty_str: ""
`,
		},
		{
			name: "prunes empty GenerateContentConfig and google search properties",
			input: `
llm_recording:
  llm_request:
    config:
      response_mime_type: ""
      cached_content: ""
      media_resolution: ""
      service_tier: ""
      tools:
        - google_search:
            blocking_confidence: ""
      labels: {}
`,
			expected: `
llm_recording:
  llm_request:
    config:
      tools:
        - google_search: {}
`,
		},
		{
			name: "normalizes raw string system_instruction to structured object",
			input: `
llm_request:
  config:
    system_instruction: "You are a booking assistant."
`,
			expected: `
llm_request:
  config:
    system_instruction: "You are a booking assistant."
`,
		},
		{
			name: "injects part_metadata to contents parts",
			input: `
llm_request:
  contents:
    - role: "user"
      parts:
        - text: "Hello"
`,
			expected: `
llm_request:
  contents:
    - role: "user"
      parts:
        - text: "Hello"
`,
		},
		{
			name: "preserves existing part_metadata if already present",
			input: `
llm_request:
  contents:
    - role: "user"
      parts:
        - text: "Hello"
          part_metadata:
            some_key: "some_value"
`,
			expected: `
llm_request:
  contents:
    - role: "user"
      parts:
        - text: "Hello"
          part_metadata:
            some_key: "some_value"
`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var node yaml.Node
			err := yaml.Unmarshal([]byte(tc.input), &node)
			if err != nil {
				t.Fatalf("failed to unmarshal test input: %v", err)
			}

			cleanYAMLNode(&node, false)

			output, err := yaml.Marshal(&node)
			if err != nil {
				t.Fatalf("failed to marshal cleaned node: %v", err)
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

func TestFunctionDeclarationTitleInjection(t *testing.T) {
	decl := &genai.FunctionDeclaration{
		Name:        "validate_email",
		Description: "checks email",
		Parameters: &genai.Schema{
			Type:     genai.TypeObject,
			Required: []string{"email"},
			Properties: map[string]*genai.Schema{
				"email": {
					Type:        genai.TypeString,
					Description: "the email address",
				},
			},
		},
	}

	var node yaml.Node
	err := node.Encode(decl)
	if err != nil {
		t.Fatalf("failed to encode: %v", err)
	}

	cleanYAMLNode(&node, false)

	output, err := yaml.Marshal(&node)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	t.Logf("Serialized:\n%s", string(output))
}
