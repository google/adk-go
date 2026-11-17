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

package llminternal

import (
	"context"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// scriptedModel answers call i with turns[i] (the last entry once the script
// runs out), one LLMResponse per element, and counts its calls.
func scriptedModel(calls *int, turns ...[]*model.LLMResponse) *mockModelForTest {
	return &mockModelForTest{
		name: "scripted",
		generateContent: func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
			i := min(*calls, len(turns)-1)
			*calls++
			return func(yield func(*model.LLMResponse, error) bool) {
				for _, r := range turns[i] {
					if !yield(r, nil) {
						return
					}
				}
			}
		},
	}
}

func newTool(t *testing.T, name string, run func(agent.Context) (map[string]any, error)) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: name}, func(ctx agent.Context, _ struct{}) (map[string]any, error) {
		return run(ctx)
	})
	if err != nil {
		t.Fatalf("functiontool.New(%q) error = %v", name, err)
	}
	return tl
}

func textResponse(text string) *model.LLMResponse {
	return &model.LLMResponse{Content: genai.NewContentFromText(text, genai.RoleModel)}
}

func callResponse(id, name string) *model.LLMResponse {
	return &model.LLMResponse{Content: &genai.Content{
		Role:  genai.RoleModel,
		Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: id, Name: name, Args: map[string]any{}}}},
	}}
}
