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

package llminternal_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/model"
)

// The transfer tool prefers the first genai.Tool that already holds
// declarations, falls back to the first bare genai.Tool placeholder (nil or
// empty declaration list and no other field set), and skips non-function tools
// even when their declaration list is an empty slice.
func TestTransferToAgentTool_EmptyDeclarationsSelection(t *testing.T) {
	tests := []struct {
		name  string
		tools []*genai.Tool
		want  [][]string
	}{
		{
			name:  "bare genai.Tool with empty declarations",
			tools: []*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{}}},
			want:  [][]string{{"transfer_to_agent"}},
		},
		{
			name: "two bare genai.Tools fills the first",
			tools: []*genai.Tool{
				{FunctionDeclarations: []*genai.FunctionDeclaration{}},
				{FunctionDeclarations: []*genai.FunctionDeclaration{}},
			},
			want: [][]string{{"transfer_to_agent"}, {}},
		},
		{
			name:  "zero genai.Tool with nil declarations",
			tools: []*genai.Tool{{}},
			want:  [][]string{{"transfer_to_agent"}},
		},
		{
			name: "genai.Tool with GoogleSearch and empty declarations",
			tools: []*genai.Tool{{
				GoogleSearch:         &genai.GoogleSearch{},
				FunctionDeclarations: []*genai.FunctionDeclaration{},
			}},
			want: [][]string{{}, {"transfer_to_agent"}},
		},
		{
			name: "empty genai.Tool before two with declarations",
			tools: []*genai.Tool{
				{FunctionDeclarations: []*genai.FunctionDeclaration{}},
				{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "first"}}},
				{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "second"}}},
			},
			want: [][]string{{}, {"first", "transfer_to_agent"}, {"second"}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := &model.LLMRequest{Config: &genai.GenerateContentConfig{Tools: tc.tools}}

			if err := (&llminternal.TransferToAgentTool{}).ProcessRequest(nil, req); err != nil {
				t.Fatalf("ProcessRequest failed: %v", err)
			}

			got := [][]string{}
			for _, gt := range req.Config.Tools {
				names := []string{}
				for _, d := range gt.FunctionDeclarations {
					names = append(names, d.Name)
				}
				got = append(got, names)
			}
			if diff := cmp.Diff(tc.want, got); diff != "" {
				t.Errorf("declaration names per genai.Tool mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
