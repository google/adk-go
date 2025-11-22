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

package geminitool_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/tool/geminitool"
)

func TestGeminiTool_ProcessRequest(t *testing.T) {
	testCases := []struct {
		name      string
		inputTool *genai.Tool
		req       *model.LLMRequest
		wantTools []*genai.Tool
		wantErr   bool
	}{
		{
			name: "add to empty request",
			inputTool: &genai.Tool{
				GoogleSearch: &genai.GoogleSearch{},
			},
			req: &model.LLMRequest{},
			wantTools: []*genai.Tool{
				{GoogleSearch: &genai.GoogleSearch{}},
			},
		},
		{
			name: "add to existing tools",
			inputTool: &genai.Tool{
				GoogleSearch: &genai.GoogleSearch{},
			},
			req: &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					Tools: []*genai.Tool{
						{
							GoogleMaps: &genai.GoogleMaps{},
						},
					},
				},
			},
			wantTools: []*genai.Tool{
				{GoogleMaps: &genai.GoogleMaps{}},
				{GoogleSearch: &genai.GoogleSearch{}},
			},
		},
		{
			name:    "error on nil request",
			wantErr: true,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			geminiTool := geminitool.New("test_tool", tt.inputTool)

			requestProcessor, ok := geminiTool.(toolinternal.RequestProcessor)
			if !ok {
				t.Fatal("geminiTool does not implement RequestProcessor")
			}

			err := requestProcessor.ProcessRequest(nil, tt.req)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ProcessRequest() error = %v, wantErr %v", err, tt.wantErr)
			}
			if tt.wantErr {
				return
			}

			if diff := cmp.Diff(tt.wantTools, tt.req.Config.Tools); diff != "" {
				t.Errorf("ProcessRequest returned unexpected tools (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSimpleTools_ProcessRequest(t *testing.T) {
	testCases := []struct {
		name     string
		tool     toolinternal.RequestProcessor
		wantTool *genai.Tool
	}{
		{
			name:     "GoogleSearch",
			tool:     geminitool.GoogleSearch{},
			wantTool: &genai.Tool{GoogleSearch: &genai.GoogleSearch{}},
		},
		{
			name:     "GoogleMaps",
			tool:     geminitool.GoogleMaps{},
			wantTool: &genai.Tool{GoogleMaps: &genai.GoogleMaps{}},
		},
		{
			name:     "EnterpriseWebSearch",
			tool:     geminitool.EnterpriseWebSearch{},
			wantTool: &genai.Tool{EnterpriseWebSearch: &genai.EnterpriseWebSearch{}},
		},
		{
			name:     "URLContext",
			tool:     geminitool.URLContext{},
			wantTool: &genai.Tool{URLContext: &genai.URLContext{}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			runStandardToolTests(t, tc.tool, tc.wantTool)
		})
	}
}

func runStandardToolTests(t *testing.T, tool toolinternal.RequestProcessor, wantTool *genai.Tool) {
	t.Run("add to empty request", func(t *testing.T) {
		req := &model.LLMRequest{}

		err := tool.ProcessRequest(nil, req)
		if err != nil {
			t.Fatalf("ProcessRequest() error = %v, wantErr false", err)
		}

		wantTools := []*genai.Tool{wantTool}
		if diff := cmp.Diff(wantTools, req.Config.Tools); diff != "" {
			t.Errorf("ProcessRequest returned unexpected tools (-want +got):\n%s", diff)
		}
	})

	t.Run("add to existing tools", func(t *testing.T) {
		req := &model.LLMRequest{
			Config: &genai.GenerateContentConfig{
				Tools: []*genai.Tool{
					{GoogleSearch: &genai.GoogleSearch{}},
				},
			},
		}

		err := tool.ProcessRequest(nil, req)
		if err != nil {
			t.Fatalf("ProcessRequest() error = %v, wantErr false", err)
		}

		wantTools := []*genai.Tool{
			{GoogleSearch: &genai.GoogleSearch{}},
			wantTool,
		}
		if diff := cmp.Diff(wantTools, req.Config.Tools); diff != "" {
			t.Errorf("ProcessRequest returned unexpected tools (-want +got):\n%s", diff)
		}
	})

	t.Run("error on nil request", func(t *testing.T) {
		err := tool.ProcessRequest(nil, nil)
		if err == nil {
			t.Fatal("ProcessRequest() error = nil, wantErr true")
		}
	})
}
