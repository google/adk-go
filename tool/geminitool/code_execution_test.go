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

package geminitool_test

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/geminitool"
)

func TestCodeExecution_ProcessRequest(t *testing.T) {
	codeExec := &genai.Tool{CodeExecution: &genai.ToolCodeExecution{}}

	testCases := []struct {
		name        string
		req         *model.LLMRequest
		wantTools   []*genai.Tool
		wantErrText string
	}{
		{
			name:      "gemini 2 adds tool",
			req:       &model.LLMRequest{Model: "gemini-2.5-flash"},
			wantTools: []*genai.Tool{codeExec},
		},
		{
			name: "gemini 2 keeps existing tools",
			req: &model.LLMRequest{
				Model: "gemini-2.5-flash",
				Config: &genai.GenerateContentConfig{
					Tools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}},
				},
			},
			wantTools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}, codeExec},
		},
		{
			name:        "gemini 1 is unsupported",
			req:         &model.LLMRequest{Model: "gemini-1.5-flash"},
			wantErrText: "cannot be used in Gemini 1.x",
		},
		{
			name:        "non-gemini model is unsupported",
			req:         &model.LLMRequest{Model: "gpt-4o"},
			wantErrText: "code execution tool is not supported for model gpt-4o",
		},
		{
			name:        "nil request errors",
			req:         nil,
			wantErrText: "llm request is nil",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			var rp toolinternal.RequestProcessor = geminitool.CodeExecution{}

			err := rp.ProcessRequest(nil, tt.req)
			if tt.wantErrText != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErrText) {
					t.Fatalf("ProcessRequest() error = %v, want error containing %q", err, tt.wantErrText)
				}
				return
			}
			if err != nil {
				t.Fatalf("ProcessRequest() unexpected error = %v", err)
			}
			if diff := cmp.Diff(tt.wantTools, tt.req.Config.Tools); diff != "" {
				t.Errorf("ProcessRequest() tools mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
