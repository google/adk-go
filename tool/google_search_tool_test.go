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

package tool_test

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/adk"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

func TestGoogleSearchTool_ProcessRequest(t *testing.T) {
	ctx := t.Context()
	gsTool := tool.NewGoogleSearchTool()

	testCases := []struct {
		name          string
		modelName     string
		existingTools []*genai.Tool
		wantTools     []*genai.Tool
		wantErr       string
	}{
		{
			name:      "gemini-1_no_existing_tools",
			modelName: "gemini-1.0-pro",
			wantTools: []*genai.Tool{
				{GoogleSearchRetrieval: &genai.GoogleSearchRetrieval{}},
			},
		},
		{
			name:      "gemini-1_with_existing_tools",
			modelName: "gemini-1.0-pro",
			existingTools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{Name: "test_function"},
					},
				},
			},
			wantErr: "google search tool cannot be used with other tools in Gemini 1.x",
		},
		{
			name:      "gemini-2_no_existing_tools",
			modelName: "gemini-2.0-pro",
			wantTools: []*genai.Tool{
				{GoogleSearch: &genai.GoogleSearch{}},
			},
		},
		{
			name:      "gemini-2_with_existing_tools",
			modelName: "gemini-2.0-pro",
			existingTools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{Name: "test_function"},
					},
				},
			},
			wantTools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{Name: "test_function"},
					},
				},
				{GoogleSearch: &genai.GoogleSearch{}},
			},
		},
		{
			name:      "unsupported_model",
			modelName: "unsupported-model",
			wantErr:   "google search tool is not supported for model",
		},
		{
			name:      "gemini-2_nil_config_init",
			modelName: "gemini-2.0-pro",
			wantTools: []*genai.Tool{
				{GoogleSearch: &genai.GoogleSearch{}},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			modelName := tc.modelName
			m, err := model.NewGeminiModel(ctx, modelName, &genai.ClientConfig{
				APIKey: "fakeApiKey",
			})
			if err != nil {
				t.Fatalf("model.NewGeminiModel(%q) failed: %v", modelName, err)
			}
			req := &adk.LLMRequest{
				Model: m,
			}

			if tc.existingTools != nil {
				req.GenerateConfig = &genai.GenerateContentConfig{
					Tools: tc.existingTools,
				}
			}

			if err := gsTool.ProcessRequest(t.Context(), &adk.ToolContext{}, req); err != nil {
				if tc.wantErr != "" {
					if !strings.Contains(err.Error(), tc.wantErr) {
						t.Fatalf("ProcessRequest error: got %v, want %v", err, tc.wantErr)
					}
					return
				}
			}
			if err != nil {
				t.Fatalf("ProcessRequest failed: %v", err)
			}

			if req.GenerateConfig == nil {
				t.Fatal("GenerateConfig should not be nil")
			}
			gotTools := req.GenerateConfig.Tools

			if diff := cmp.Diff(tc.wantTools, gotTools); diff != "" {
				t.Errorf("ProcessRequest returned unexpected tools (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGoogleSearchTool_Run(t *testing.T) {
	ctx := t.Context()
	gsTool := tool.NewGoogleSearchTool()

	_, err := gsTool.Run(ctx, &adk.ToolContext{}, nil)

	if err == nil {
		t.Fatal("Run expected error, but got nil")
	}
}
