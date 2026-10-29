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

package exampletool

import (
	"fmt"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func benchmarkExamples() []*Example {
	examples := make([]*Example, 4)
	for i := range examples {
		examples[i] = &Example{
			Input: genai.NewContentFromText(fmt.Sprintf("user request %d", i), genai.RoleUser),
			Output: []*genai.Content{
				{
					Role: genai.RoleModel,
					Parts: []*genai.Part{
						{
							FunctionCall: &genai.FunctionCall{
								Name: fmt.Sprintf("search_tool_%d", i),
								Args: map[string]any{
									"alpha":   "val_a",
									"bravo":   "val_b",
									"charlie": 42,
									"delta":   true,
									"echo":    "val_e",
									"foxtrot": "val_f",
								},
							},
						},
					},
				},
				{
					Role: genai.RoleUser,
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								Name:     fmt.Sprintf("search_tool_%d", i),
								Response: map[string]any{"status": "ok"},
							},
						},
					},
				},
				genai.NewContentFromText(fmt.Sprintf("final answer %d", i), genai.RoleModel),
			},
		}
	}
	return examples
}

// BenchmarkExampleToolProcessRequest measures rendering few-shot examples in
// ProcessRequest and reports distinct instructions across 100 calls.
func BenchmarkExampleToolProcessRequest(b *testing.B) {
	et, err := New(ExampleToolConfig{Examples: benchmarkExamples()})
	if err != nil {
		b.Fatalf("New() failed: %v", err)
	}
	ctx := &mockToolContext{
		Context:     b.Context(),
		userContent: genai.NewContentFromText("hello", genai.RoleUser),
	}

	distinct := make(map[string]struct{}, 100)
	for range 100 {
		req := &model.LLMRequest{Model: "gemini-2.5-flash"}
		if err := et.ProcessRequest(ctx, req); err != nil {
			b.Fatalf("ProcessRequest failed: %v", err)
		}
		distinct[req.Config.SystemInstruction.Parts[0].Text] = struct{}{}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		req := &model.LLMRequest{Model: "gemini-2.5-flash"}
		if err := et.ProcessRequest(ctx, req); err != nil {
			b.Fatalf("ProcessRequest failed: %v", err)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(len(distinct)), "distinct-instructions/100-calls")
}
