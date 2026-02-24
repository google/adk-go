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

package toolconfirmation_test

import (
	"strings"
	"testing"

	"google.golang.org/genai"

	"github.com/sjzsdu/adk-go/tool/toolconfirmation"
)

// --- The Test Suite ---

func TestOriginalCallFrom(t *testing.T) {
	tests := []struct {
		name          string
		input         *genai.FunctionCall
		wantErr       bool
		wantErrSubstr string // Substring we expect to see in the error message
	}{
		{
			name: "Success - Valid Structure",
			input: &genai.FunctionCall{
				ID: "call_123",
				Args: map[string]any{
					"originalFunctionCall": map[string]any{
						"ID":   "call_999",
						"Name": "weather_lookup",
					},
				},
			},
			wantErr: false,
		},
		{
			name:          "Failure - Nil Input",
			input:         nil,
			wantErr:       true,
			wantErrSubstr: "cannot be nil",
		},
		{
			name: "Failure - Missing Key",
			input: &genai.FunctionCall{
				ID: "call_456",
				Args: map[string]any{
					"someOtherKey": "foo",
				},
			},
			wantErr:       true,
			wantErrSubstr: "required argument \"originalFunctionCall\" is missing",
		},
		{
			name: "Failure - Invalid Type (String instead of Map)",
			input: &genai.FunctionCall{
				ID: "call_789",
				Args: map[string]any{
					// LLM hallucination: sending a string instead of an object
					"originalFunctionCall": "{\"Name\": \"bad_json\"}",
				},
			},
			wantErr:       true,
			wantErrSubstr: "got string", // Verifies our %T check works
		},
		{
			name: "Failure - Invalid Type (Nil value)",
			input: &genai.FunctionCall{
				ID: "call_nil",
				Args: map[string]any{
					"originalFunctionCall": nil,
				},
			},
			wantErr:       true,
			wantErrSubstr: "got <nil>", // Verifies %T handles nil gracefully
		},
		{
			name: "Failure - Converter Error (Bad Internal Structure id with wrong type)",
			input: &genai.FunctionCall{
				ID: "call_bad_struct",
				Args: map[string]any{
					"originalFunctionCall": map[string]any{
						"ID": 11,
					},
				},
			},
			wantErr:       true,
			wantErrSubstr: "failed to decode",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := toolconfirmation.OriginalCallFrom(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("OriginalCallFrom() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr && !strings.Contains(err.Error(), tt.wantErrSubstr) {
				t.Errorf("OriginalCallFrom() error = %q, expected substring %q", err.Error(), tt.wantErrSubstr)
			}

			if !tt.wantErr {
				if got == nil {
					t.Error("Expected result, got nil")
				} else {
					// Verify we actually extracted data from the inner map
					// This relies on the specific "Success" test case data
					if got.Name != "weather_lookup" {
						t.Errorf("Expected extracted name 'weather_lookup', got %q", got.Name)
					}
				}
			}
		})
	}
}
