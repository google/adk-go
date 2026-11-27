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

package llmagent_test

import (
	"context"
	"iter"
	"net/http"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropicmodel"
	"google.golang.org/adk/v2/model/apigee"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/model/openaimodel"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type capabilityModel struct {
	name string
	caps model.Capabilities
}

func (m *capabilityModel) Name() string                     { return m.name }
func (m *capabilityModel) Capabilities() model.Capabilities { return m.caps }

func (*capabilityModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(func(*model.LLMResponse, error) bool) {}
}

func declaresFunction(cfg *genai.GenerateContentConfig, name string) bool {
	if cfg == nil {
		return false
	}
	for _, t := range cfg.Tools {
		for _, fd := range t.FunctionDeclarations {
			if fd.Name == name {
				return true
			}
		}
	}
	return false
}

func newDummyTool(t *testing.T) tool.Tool {
	t.Helper()
	dummyTool, err := functiontool.New(functiontool.Config{
		Name:        "dummy_tool",
		Description: "does nothing",
	}, func(agent.Context, struct{}) (struct{}, error) {
		return struct{}{}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}
	return dummyTool
}

func captureFirstRequestConfig(
	t *testing.T,
	llm model.LLM,
	schema *genai.Schema,
	tools []tool.Tool,
) *genai.GenerateContentConfig {
	t.Helper()
	var requests []*model.LLMRequest
	a, err := llmagent.New(llmagent.Config{
		Name:         "schema_agent",
		Model:        llm,
		OutputSchema: schema,
		Tools:        tools,
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
			requests = append(requests, req)
			return &model.LLMResponse{
				Content:      genai.NewContentFromText(`{"city": "Paris"}`, genai.RoleModel),
				FinishReason: genai.FinishReasonStop,
			}, nil
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	runner := testutil.NewTestAgentRunner(t, a)
	if _, err := testutil.CollectEvents(runner.Run(t, "session", "hi")); err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("model received no request")
	}
	if requests[0].Config == nil {
		t.Fatal("request Config is nil")
	}
	return requests[0].Config
}

func TestOutputSchemaWithTools_RespectsModelCapabilities(t *testing.T) {
	schema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
		Required:   []string{"city"},
	}
	tests := []struct {
		name             string
		model            model.LLM
		withTool         bool
		wantSchema       *genai.Schema
		wantMIMEType     string
		wantSetModelResp bool
	}{
		{
			name:         "custom_model_declaring_support_with_tools",
			model:        &capabilityModel{name: "custom-model", caps: model.Capabilities{AcceptsOutputSchemaAndTools: true}},
			withTool:     true,
			wantSchema:   schema,
			wantMIMEType: "application/json",
		},
		{
			name:             "custom_model_declaring_no_support_with_tools",
			model:            &capabilityModel{name: "custom-model", caps: model.Capabilities{AcceptsOutputSchemaAndTools: false}},
			withTool:         true,
			wantSetModelResp: true,
		},
		{
			name:         "custom_model_declaring_no_support_without_tools",
			model:        &capabilityModel{name: "custom-model", caps: model.Capabilities{AcceptsOutputSchemaAndTools: false}},
			withTool:     false,
			wantSchema:   schema,
			wantMIMEType: "application/json",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var tools []tool.Tool
			if tc.withTool {
				tools = []tool.Tool{newDummyTool(t)}
			}
			cfg := captureFirstRequestConfig(t, tc.model, schema, tools)
			if diff := cmp.Diff(tc.wantSchema, cfg.ResponseSchema); diff != "" {
				t.Errorf("ResponseSchema mismatch (-want +got):\n%s", diff)
			}
			if got := cfg.ResponseMIMEType; got != tc.wantMIMEType {
				t.Errorf("ResponseMIMEType = %q, want %q", got, tc.wantMIMEType)
			}
			if got := declaresFunction(cfg, "set_model_response"); got != tc.wantSetModelResp {
				t.Errorf("request declares set_model_response = %t, want %t", got, tc.wantSetModelResp)
			}
		})
	}
}

func TestOutputSchemaWithTools_BuiltInModels(t *testing.T) {
	schema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
		Required:   []string{"city"},
	}
	vertexCfg := &genai.ClientConfig{
		Backend:    genai.BackendVertexAI,
		Project:    "test-project",
		Location:   "us-central1",
		HTTPClient: &http.Client{},
	}
	geminiAPICfg := &genai.ClientConfig{
		Backend:    genai.BackendGeminiAPI,
		APIKey:     "test-key",
		HTTPClient: &http.Client{},
	}

	tests := []struct {
		name             string
		newModel         func(t *testing.T) model.LLM
		wantSchema       *genai.Schema
		wantMIMEType     string
		wantSetModelResp bool
	}{
		{
			name: "gemini_2_5_on_vertex",
			newModel: func(t *testing.T) model.LLM {
				m, err := gemini.NewModel(t.Context(), "gemini-2.5-flash", vertexCfg)
				if err != nil {
					t.Fatalf("gemini.NewModel: %v", err)
				}
				return m
			},
			wantSchema:   schema,
			wantMIMEType: "application/json",
		},
		{
			name: "gemini_1_5_on_vertex",
			newModel: func(t *testing.T) model.LLM {
				m, err := gemini.NewModel(t.Context(), "gemini-1.5-pro", vertexCfg)
				if err != nil {
					t.Fatalf("gemini.NewModel: %v", err)
				}
				return m
			},
			wantSetModelResp: true,
		},
		{
			name: "gemini_on_gemini_api",
			newModel: func(t *testing.T) model.LLM {
				m, err := gemini.NewModel(t.Context(), "gemini-2.5-flash", geminiAPICfg)
				if err != nil {
					t.Fatalf("gemini.NewModel: %v", err)
				}
				return m
			},
			wantSetModelResp: true,
		},
		{
			name: "gemma_on_vertex",
			newModel: func(t *testing.T) model.LLM {
				m, err := gemini.NewModel(t.Context(), "gemma-3-27b-it", vertexCfg)
				if err != nil {
					t.Fatalf("gemini.NewModel: %v", err)
				}
				return m
			},
			wantSchema:   schema,
			wantMIMEType: "application/json",
		},
		{
			name: "tuned_endpoint_on_vertex",
			newModel: func(t *testing.T) model.LLM {
				m, err := gemini.NewModel(t.Context(), "projects/p/locations/us-central1/endpoints/12345", vertexCfg)
				if err != nil {
					t.Fatalf("gemini.NewModel: %v", err)
				}
				return m
			},
			wantSchema:   schema,
			wantMIMEType: "application/json",
		},
		{
			name: "apigee_vertex_gemini",
			newModel: func(t *testing.T) model.LLM {
				t.Setenv("GOOGLE_API_KEY", "")
				t.Setenv("GEMINI_API_KEY", "")
				t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "")
				t.Setenv("GOOGLE_CLOUD_PROJECT", "test-project")
				t.Setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
				m, err := apigee.NewModel(t.Context(), "apigee/vertex_ai/gemini-2.5-flash",
					apigee.WithProxyURL("https://proxy.example.com"),
					apigee.WithHTTPClient(&http.Client{}))
				if err != nil {
					t.Fatalf("apigee.NewModel: %v", err)
				}
				return m
			},
			wantSetModelResp: true,
		},
		{
			name: "apigee_vertex_non_gemini",
			newModel: func(t *testing.T) model.LLM {
				t.Setenv("GOOGLE_API_KEY", "")
				t.Setenv("GEMINI_API_KEY", "")
				t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "")
				t.Setenv("GOOGLE_CLOUD_PROJECT", "test-project")
				t.Setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
				m, err := apigee.NewModel(t.Context(), "apigee/vertex_ai/gemma-3-27b-it",
					apigee.WithProxyURL("https://proxy.example.com"),
					apigee.WithHTTPClient(&http.Client{}))
				if err != nil {
					t.Fatalf("apigee.NewModel: %v", err)
				}
				return m
			},
			wantSchema:   schema,
			wantMIMEType: "application/json",
		},
		{
			name: "anthropic",
			newModel: func(t *testing.T) model.LLM {
				m, err := anthropicmodel.NewModel(t.Context(), "claude-3-7-sonnet-latest", anthropicmodel.ClientConfig{
					Backend: anthropicmodel.BackendAnthropic,
					APIKey:  "test-key",
				})
				if err != nil {
					t.Fatalf("anthropicmodel.NewModel: %v", err)
				}
				return m
			},
			wantSchema:   schema,
			wantMIMEType: "application/json",
		},
		{
			name: "openai_responses",
			newModel: func(t *testing.T) model.LLM {
				m, err := openaimodel.NewModel(t.Context(), "gpt-4o", &openaimodel.ClientConfig{
					APIKey: "test-key",
					API:    openaimodel.APIResponses,
				})
				if err != nil {
					t.Fatalf("openaimodel.NewModel: %v", err)
				}
				return m
			},
			wantSchema:   schema,
			wantMIMEType: "application/json",
		},
		{
			name: "openai_chat_completions",
			newModel: func(t *testing.T) model.LLM {
				m, err := openaimodel.NewModel(t.Context(), "gpt-4o", &openaimodel.ClientConfig{
					APIKey: "test-key",
					API:    openaimodel.APIChatCompletions,
				})
				if err != nil {
					t.Fatalf("openaimodel.NewModel: %v", err)
				}
				return m
			},
			wantSchema:   schema,
			wantMIMEType: "application/json",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := captureFirstRequestConfig(t, tc.newModel(t), schema, []tool.Tool{newDummyTool(t)})
			if diff := cmp.Diff(tc.wantSchema, cfg.ResponseSchema); diff != "" {
				t.Errorf("ResponseSchema mismatch (-want +got):\n%s", diff)
			}
			if got := cfg.ResponseMIMEType; got != tc.wantMIMEType {
				t.Errorf("ResponseMIMEType = %q, want %q", got, tc.wantMIMEType)
			}
			if got := declaresFunction(cfg, "set_model_response"); got != tc.wantSetModelResp {
				t.Errorf("request declares set_model_response = %t, want %t", got, tc.wantSetModelResp)
			}
		})
	}
}
