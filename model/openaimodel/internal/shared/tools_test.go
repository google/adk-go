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

package shared

import (
	"strings"
	"testing"

	"google.golang.org/genai"
)

func TestEnsureFunctionToolOnly(t *testing.T) {
	tests := []struct {
		name    string
		tool    *genai.Tool
		wantErr string
	}{
		{
			name:    "nil tool",
			tool:    nil,
			wantErr: "tool 0 is nil",
		},
		{
			name:    "non-function tool",
			tool:    &genai.Tool{GoogleSearch: &genai.GoogleSearch{}},
			wantErr: "non-function tools",
		},
		{
			name:    "no functions",
			tool:    &genai.Tool{},
			wantErr: "does not declare any functions",
		},
		{
			name: "valid",
			tool: &genai.Tool{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "fn1"}}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := EnsureFunctionToolOnly(0, tc.tool)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected no error, got %v", err)
				}
			} else {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected error containing %q, got: %v", tc.wantErr, err)
				}
			}
		})
	}
}
