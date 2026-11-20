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
	"encoding/json"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/anthropicmodel"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/model/openaimodel"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// recordedRequest is what a fake model backend saw: the headers and the JSON
// body of one request.
type recordedRequest struct {
	header http.Header
	body   map[string]any
}

// mixedModelBackends records the requests that reach a fake Vertex AI Gemini
// backend and a fake Anthropic backend.
type mixedModelBackends struct {
	mu     sync.Mutex
	gemini []recordedRequest
	claude []recordedRequest
}

func (b *mixedModelBackends) record(t *testing.T, dst *[]recordedRequest, r *http.Request) {
	t.Helper()
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		t.Errorf("reading request body: %v", err)
		return
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Errorf("json.Unmarshal(request body) error = %v", err)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	*dst = append(*dst, recordedRequest{header: r.Header.Clone(), body: body})
}

// roundTripFunc adapts a function to http.RoundTripper.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// newMixedModelAgents returns a Gemini root agent on Vertex AI whose first
// turn transfers to a Claude sub-agent, both backed by fakes that record into
// b.
func newMixedModelAgents(t *testing.T, b *mixedModelBackends) agent.Agent {
	t.Helper()
	geminiTransport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		b.record(t, &b.gemini, r)
		body := `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"transfer_to_agent","args":{"agent_name":"claude_agent"}}}]},"finishReason":"STOP"}]}`
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       io.NopCloser(strings.NewReader(body)),
		}, nil
	})
	geminiModel, err := gemini.NewModel(t.Context(), "gemini-2.5-flash", &genai.ClientConfig{
		Backend:    genai.BackendVertexAI,
		Project:    "test-project",
		Location:   "us-central1",
		HTTPClient: &http.Client{Transport: geminiTransport},
	})
	if err != nil {
		t.Fatalf("gemini.NewModel() error = %v", err)
	}

	claudeServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b.record(t, &b.claude, r)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"claude answer"}],"usage":{"input_tokens":3,"output_tokens":2}}`)
	}))
	t.Cleanup(claudeServer.Close)
	claudeModel, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
		APIKey:  "test-key",
		BaseURL: claudeServer.URL,
	})
	if err != nil {
		t.Fatalf("anthropicmodel.NewModel() error = %v", err)
	}

	claudeAgent, err := llmagent.New(llmagent.Config{Name: "claude_agent", Description: "Answers on Claude.", Model: claudeModel})
	if err != nil {
		t.Fatalf("llmagent.New(claude_agent) error = %v", err)
	}
	root, err := llmagent.New(llmagent.Config{Name: "gemini_root", Model: geminiModel, SubAgents: []agent.Agent{claudeAgent}})
	if err != nil {
		t.Fatalf("llmagent.New(gemini_root) error = %v", err)
	}
	return root
}

// TestRunConfig_MixedModels_HTTPOptions runs a Gemini root agent that
// transfers to a Claude sub-agent with RunConfig.HTTPOptions that set fields
// the Anthropic connector cannot send. The Gemini request carries the run's
// header, and the Claude request is sent instead of failing, as in
// adk-python, whose AnthropicLlm ignores HTTP options.
func TestRunConfig_MixedModels_HTTPOptions(t *testing.T) {
	b := &mixedModelBackends{}
	runner := testutil.NewTestAgentRunner(t, newMixedModelAgents(t, b))
	cfg := agent.RunConfig{HTTPOptions: &genai.HTTPOptions{
		Headers:      http.Header{"X-Request-Source": {"batch"}},
		RetryOptions: &genai.HTTPRetryOptions{Attempts: genai.Ptr[int32](1)},
		ExtraBody:    map[string]any{"run_extra": "v"},
	}}

	events, err := testutil.CollectEvents(runner.RunContentWithConfig(t, "session", genai.NewContentFromText("hi", genai.RoleUser), cfg))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if len(b.gemini) != 1 {
		t.Fatalf("Gemini backend got %d requests, want 1", len(b.gemini))
	}
	if got := b.gemini[0].header.Get("X-Request-Source"); got != "batch" {
		t.Errorf("Gemini request header X-Request-Source = %q, want %q", got, "batch")
	}
	if got := b.gemini[0].body["run_extra"]; got != "v" {
		t.Errorf("Gemini request body run_extra = %v, want %q", got, "v")
	}
	if len(b.claude) != 1 {
		t.Fatalf("Claude backend got %d requests, want 1", len(b.claude))
	}
	if _, ok := b.claude[0].body["run_extra"]; ok {
		t.Errorf("Claude request body carries run_extra: %v", b.claude[0].body)
	}
	last := events[len(events)-1]
	if last.Author != "claude_agent" || last.Content == nil || len(last.Content.Parts) == 0 || last.Content.Parts[0].Text != "claude answer" {
		t.Errorf("last event = %+v, want the Claude agent's answer", last)
	}
}

// TestRunConfig_MixedModels_Labels runs a Gemini root agent that transfers to
// a Claude sub-agent with RunConfig.Labels. The Gemini request on Vertex AI
// carries the run's labels, and the Claude request is sent instead of
// failing, as in adk-python, whose AnthropicLlm ignores labels.
func TestRunConfig_MixedModels_Labels(t *testing.T) {
	b := &mixedModelBackends{}
	runner := testutil.NewTestAgentRunner(t, newMixedModelAgents(t, b))
	labels := map[string]string{"cost_center": "x"}

	if _, err := testutil.CollectEvents(runner.RunContentWithConfig(t, "session", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{Labels: &labels})); err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if len(b.gemini) != 1 {
		t.Fatalf("Gemini backend got %d requests, want 1", len(b.gemini))
	}
	wantGeminiLabels := map[string]any{"cost_center": "x", "adk_agent_name": "gemini_root"}
	if diff := cmp.Diff(wantGeminiLabels, b.gemini[0].body["labels"]); diff != "" {
		t.Errorf("Gemini request labels mismatch (-want +got):\n%s", diff)
	}
	if len(b.claude) != 1 {
		t.Fatalf("Claude backend got %d requests, want 1", len(b.claude))
	}
}

// TestRunConfig_Labels_RunConfigWinsOnConflict pins that RunConfig.Labels are
// merged over the agent's GenerateContentConfig.Labels, winning on a shared
// key, and that an adk_agent_name label from RunConfig survives the flow's
// default agent-name label.
func TestRunConfig_Labels_RunConfigWinsOnConflict(t *testing.T) {
	mockModel := &testutil.MockModel{
		Responses: []*genai.Content{genai.NewContentFromText("done", genai.RoleModel)},
	}
	a, err := llmagent.New(llmagent.Config{
		Name:  "labels_agent",
		Model: mockModel,
		GenerateContentConfig: &genai.GenerateContentConfig{
			Labels: map[string]string{"shared": "agent-val", "agent-only": "a"},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New: %v", err)
	}
	runner := testutil.NewTestAgentRunner(t, a)
	runLabels := map[string]string{"shared": "run-val", "adk_agent_name": "custom-owner"}
	cfg := agent.RunConfig{Labels: &runLabels}

	if _, err := testutil.CollectEvents(runner.RunContentWithConfig(t, "session", genai.NewContentFromText("hi", genai.RoleUser), cfg)); err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if len(mockModel.Requests) == 0 {
		t.Fatal("model received no request")
	}
	want := map[string]string{"shared": "run-val", "agent-only": "a", "adk_agent_name": "custom-owner"}
	if diff := cmp.Diff(want, mockModel.Requests[0].Config.Labels); diff != "" {
		t.Errorf("request Config.Labels mismatch (-want +got):\n%s", diff)
	}
}

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

func TestRunConfig_InheritedClaudeModel_FiltersHTTPOptionsAndLabels(t *testing.T) {
	b := &mixedModelBackends{}
	var calls int
	claudeServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b.record(t, &b.claude, r)
		calls++
		w.Header().Set("Content-Type", "application/json")
		if calls == 1 {
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"tool_use","content":[{"type":"tool_use","id":"tu_1","name":"transfer_to_agent","input":{"agent_name":"claude_child"}}],"usage":{"input_tokens":3,"output_tokens":2}}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"msg_2","type":"message","role":"assistant","model":"claude-sonnet-4-5-20250929","stop_reason":"end_turn","content":[{"type":"text","text":"inherited claude answer"}],"usage":{"input_tokens":3,"output_tokens":2}}`)
	}))
	t.Cleanup(claudeServer.Close)
	claudeModel, err := anthropicmodel.NewModel(t.Context(), "claude-sonnet-4-5-20250929", anthropicmodel.ClientConfig{
		APIKey:  "test-key",
		BaseURL: claudeServer.URL,
	})
	if err != nil {
		t.Fatalf("anthropicmodel.NewModel() error = %v", err)
	}

	child, err := llmagent.New(llmagent.Config{Name: "claude_child", Description: "Inherits Claude."})
	if err != nil {
		t.Fatalf("llmagent.New(claude_child) error = %v", err)
	}
	root, err := llmagent.New(llmagent.Config{Name: "claude_parent", Model: claudeModel, SubAgents: []agent.Agent{child}})
	if err != nil {
		t.Fatalf("llmagent.New(claude_parent) error = %v", err)
	}

	r := testutil.NewTestAgentRunner(t, root)
	labels := map[string]string{"cost_center": "billing"}
	events, err := testutil.CollectEvents(r.RunContentWithConfig(t, "session", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{
		Labels: &labels,
		HTTPOptions: &genai.HTTPOptions{
			Headers:   http.Header{"X-Run-Header": {"h1"}},
			ExtraBody: map[string]any{"unsupported": true},
		},
	}))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	if len(events) == 0 {
		t.Fatal("run yielded no events")
	}
	if len(b.claude) != 2 {
		t.Fatalf("claude backend received %d requests, want 2", len(b.claude))
	}
	if _, ok := b.claude[1].body["unsupported"]; ok {
		t.Errorf("inherited claude request body carries unsupported ExtraBody: %v", b.claude[1].body)
	}
	last := events[len(events)-1]
	if last.Author != "claude_child" || last.Content == nil || len(last.Content.Parts) == 0 || last.Content.Parts[0].Text != "inherited claude answer" {
		t.Errorf("last event = %+v, want claude_child's answer", last)
	}
}
