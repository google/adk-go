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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/openaimodel"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// undeclaredModel is a model that does not implement
// model.CapabilitiesProvider. It reports a backend the way the built-in
// Gemini model does, records each request, and answers with JSON text.
type undeclaredModel struct {
	name     string
	backend  genai.Backend
	requests []*model.LLMRequest
}

func (m *undeclaredModel) Name() string                       { return m.name }
func (m *undeclaredModel) GetGoogleLLMVariant() genai.Backend { return m.backend }

func (m *undeclaredModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	m.requests = append(m.requests, req)
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			Content:      genai.NewContentFromText(`{"city": "Paris"}`, genai.RoleModel),
			FinishReason: genai.FinishReasonStop,
		}, nil)
	}
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

// TestOutputSchemaWithTools_UndeclaredModelFallbackByNameAndBackend pins
// adk-python's fallback for a model that declares no capabilities
// (models/base_llm.py:92-124): an output schema goes into the request next to
// tools only for a Gemini model name on Vertex AI; every other model gets the
// set_model_response tool instead.
func TestOutputSchemaWithTools_UndeclaredModelFallbackByNameAndBackend(t *testing.T) {
	schema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
		Required:   []string{"city"},
	}
	tests := []struct {
		name             string
		modelName        string
		backend          genai.Backend
		wantSchema       *genai.Schema
		wantMIMEType     string
		wantSetModelResp bool
	}{
		{
			name:         "gemini_1_5_on_vertex_gets_native_schema",
			modelName:    "gemini-1.5-pro",
			backend:      genai.BackendVertexAI,
			wantSchema:   schema,
			wantMIMEType: "application/json",
		},
		{
			name:             "gemini_on_gemini_api_gets_set_model_response",
			modelName:        "gemini-2.5-flash",
			backend:          genai.BackendGeminiAPI,
			wantSetModelResp: true,
		},
		{
			name:             "non_gemini_on_vertex_gets_set_model_response",
			modelName:        "claude-3-7-sonnet",
			backend:          genai.BackendVertexAI,
			wantSetModelResp: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dummyTool, err := functiontool.New(functiontool.Config{
				Name:        "dummy_tool",
				Description: "does nothing",
			}, func(agent.Context, struct{}) (struct{}, error) {
				return struct{}{}, nil
			})
			if err != nil {
				t.Fatalf("functiontool.New: %v", err)
			}
			m := &undeclaredModel{name: tc.modelName, backend: tc.backend}
			a, err := llmagent.New(llmagent.Config{
				Name:         "schema_agent",
				Model:        m,
				OutputSchema: schema,
				Tools:        []tool.Tool{dummyTool},
			})
			if err != nil {
				t.Fatalf("llmagent.New: %v", err)
			}
			runner := testutil.NewTestAgentRunner(t, a)

			if _, err := testutil.CollectEvents(runner.Run(t, "session", "hi")); err != nil {
				t.Fatalf("run failed: %v", err)
			}

			if len(m.requests) == 0 {
				t.Fatal("model received no request")
			}
			cfg := m.requests[0].Config
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

// backendlessModel is a model that implements neither
// model.CapabilitiesProvider nor a GetGoogleLLMVariant method, as most custom
// models do. It implements only Name; a before-model callback must answer in
// its place.
type backendlessModel struct {
	model.LLM
	name string
}

func (m *backendlessModel) Name() string { return m.name }

// TestOutputSchemaWithTools_UndeclaredModelFallbackIgnoresEnvironment pins
// that the fallback for a model without model.CapabilitiesProvider takes the
// backend from the model's GetGoogleLLMVariant method, never from the
// GOOGLE_GENAI_USE_ENTERPRISE and GOOGLE_GENAI_USE_VERTEXAI environment
// variables. adk-python's fallback reads those variables
// (models/_capabilities.py:30-51, utils/variant_utils.py:43-46), so there a
// model named like Gemini that reports no backend gets the native schema under
// them. A before-model callback records the request the model would receive
// and answers in its place.
func TestOutputSchemaWithTools_UndeclaredModelFallbackIgnoresEnvironment(t *testing.T) {
	t.Setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
	t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
	schema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
		Required:   []string{"city"},
	}
	tests := []struct {
		name             string
		model            model.LLM
		wantSchema       *genai.Schema
		wantMIMEType     string
		wantSetModelResp bool
	}{
		{
			name:             "gemini_reporting_no_backend_gets_set_model_response",
			model:            &backendlessModel{name: "gemini-2.5-flash"},
			wantSetModelResp: true,
		},
		{
			name:             "non_gemini_reporting_no_backend_gets_set_model_response",
			model:            &backendlessModel{name: "my-model"},
			wantSetModelResp: true,
		},
		{
			name:         "gemini_reporting_vertex_gets_native_schema",
			model:        &undeclaredModel{name: "gemini-2.5-flash", backend: genai.BackendVertexAI},
			wantSchema:   schema,
			wantMIMEType: "application/json",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dummyTool, err := functiontool.New(functiontool.Config{
				Name:        "dummy_tool",
				Description: "does nothing",
			}, func(agent.Context, struct{}) (struct{}, error) {
				return struct{}{}, nil
			})
			if err != nil {
				t.Fatalf("functiontool.New: %v", err)
			}
			var requests []*model.LLMRequest
			a, err := llmagent.New(llmagent.Config{
				Name:         "schema_agent",
				Model:        tc.model,
				OutputSchema: schema,
				Tools:        []tool.Tool{dummyTool},
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
			cfg := requests[0].Config
			if diff := cmp.Diff(tc.wantSchema, cfg.ResponseSchema); diff != "" {
				t.Errorf("ResponseSchema mismatch (-want +got):\n%s", diff)
			}
			if cfg.ResponseJsonSchema != nil {
				t.Errorf("ResponseJsonSchema = %v, want nil", cfg.ResponseJsonSchema)
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

// TestOutputSchemaWithTools_OpenAIModelGetsSetModelResponse checks, on both
// OpenAI APIs, that an agent on an openaimodel model with tools and an output
// schema sends no native response schema and gets the set_model_response tool,
// as adk-python's OpenAILlm and OpenAIResponsesLlm agents do. A before-model
// callback records the request the model would receive and answers in its
// place, so no request leaves the process.
func TestOutputSchemaWithTools_OpenAIModelGetsSetModelResponse(t *testing.T) {
	schema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
		Required:   []string{"city"},
	}
	for _, api := range []openaimodel.API{openaimodel.APIResponses, openaimodel.APIChatCompletions} {
		t.Run(string(api), func(t *testing.T) {
			m, err := openaimodel.NewModel(t.Context(), "gpt-4o", &openaimodel.ClientConfig{APIKey: "test", API: api})
			if err != nil {
				t.Fatalf("openaimodel.NewModel: %v", err)
			}
			dummyTool, err := functiontool.New(functiontool.Config{
				Name:        "dummy_tool",
				Description: "does nothing",
			}, func(agent.Context, struct{}) (struct{}, error) {
				return struct{}{}, nil
			})
			if err != nil {
				t.Fatalf("functiontool.New: %v", err)
			}
			var requests []*model.LLMRequest
			a, err := llmagent.New(llmagent.Config{
				Name:         "schema_agent",
				Model:        m,
				OutputSchema: schema,
				Tools:        []tool.Tool{dummyTool},
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
			cfg := requests[0].Config
			if cfg.ResponseSchema != nil {
				t.Errorf("ResponseSchema = %v, want nil", cfg.ResponseSchema)
			}
			if got := cfg.ResponseMIMEType; got != "" {
				t.Errorf("ResponseMIMEType = %q, want empty", got)
			}
			if !declaresFunction(cfg, "set_model_response") {
				t.Error("request does not declare set_model_response")
			}
		})
	}
}
