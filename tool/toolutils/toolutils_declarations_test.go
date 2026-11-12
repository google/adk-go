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

package toolutils_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/toolutils"
)

// A genai.Tool with an empty declaration list is not a function tool, as in
// adk-python's LlmRequest.append_tools (models/llm_request.py:46), so PackTool
// must not reuse it.
func TestPackTool_SkipsGenaiToolWithEmptyDeclarations(t *testing.T) {
	decls := func(names ...string) []*genai.FunctionDeclaration {
		out := []*genai.FunctionDeclaration{}
		for _, n := range names {
			out = append(out, &genai.FunctionDeclaration{Name: n})
		}
		return out
	}
	tests := []struct {
		name  string
		tools []*genai.Tool
		want  [][]string
	}{
		{
			name:  "only an empty genai.Tool",
			tools: []*genai.Tool{{FunctionDeclarations: decls()}},
			want:  [][]string{{}, {"new"}},
		},
		{
			name: "empty genai.Tool before two with declarations",
			tools: []*genai.Tool{
				{FunctionDeclarations: decls()},
				{FunctionDeclarations: decls("first")},
				{FunctionDeclarations: decls("second")},
			},
			want: [][]string{{}, {"first", "new"}, {"second"}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := &model.LLMRequest{Config: &genai.GenerateContentConfig{Tools: tc.tools}}

			if err := toolutils.PackTool(req, fakeTool{name: "new", decl: &genai.FunctionDeclaration{Name: "new"}}); err != nil {
				t.Fatalf("PackTool() returned error: %v", err)
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
