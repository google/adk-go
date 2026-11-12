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

// With several genai.Tools that hold declarations, the transfer tool adds its
// declaration to the first of them, as adk-python's LlmRequest.append_tools does.
func TestTransferToAgentTool_AppendsToFirstGenaiToolWithDeclarations(t *testing.T) {
	decls := func(names ...string) []*genai.FunctionDeclaration {
		out := []*genai.FunctionDeclaration{}
		for _, n := range names {
			out = append(out, &genai.FunctionDeclaration{Name: n})
		}
		return out
	}
	req := &model.LLMRequest{Config: &genai.GenerateContentConfig{Tools: []*genai.Tool{
		{FunctionDeclarations: decls("first")},
		{FunctionDeclarations: decls("second")},
	}}}

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
	if diff := cmp.Diff([][]string{{"first", "transfer_to_agent"}, {"second"}}, got); diff != "" {
		t.Errorf("declaration names per genai.Tool mismatch (-want +got):\n%s", diff)
	}
}

// A nil entry in the request's genai.Tools is skipped, not dereferenced.
func TestTransferToAgentTool_SkipsNilGenaiTool(t *testing.T) {
	req := &model.LLMRequest{Config: &genai.GenerateContentConfig{Tools: []*genai.Tool{nil}}}

	if err := (&llminternal.TransferToAgentTool{}).ProcessRequest(nil, req); err != nil {
		t.Fatalf("ProcessRequest failed: %v", err)
	}

	if len(req.Config.Tools) != 2 || req.Config.Tools[0] != nil {
		t.Fatalf("req.Config.Tools = %v, want the nil entry followed by one genai.Tool", req.Config.Tools)
	}
	if got := req.Config.Tools[1].FunctionDeclarations; len(got) != 1 || got[0].Name != "transfer_to_agent" {
		t.Errorf("declarations of the new genai.Tool = %v, want transfer_to_agent only", got)
	}
}
