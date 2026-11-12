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

func TestTransferToAgentTool_SkipsGenaiToolWithEmptyDeclarations(t *testing.T) {
	req := &model.LLMRequest{Config: &genai.GenerateContentConfig{
		Tools: []*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{}}},
	}}

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
	if diff := cmp.Diff([][]string{{}, {"transfer_to_agent"}}, got); diff != "" {
		t.Errorf("declaration names per genai.Tool mismatch (-want +got):\n%s", diff)
	}
}
