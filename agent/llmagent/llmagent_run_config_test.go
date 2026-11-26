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
	"encoding/json"
	"io"
	"maps"
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

// forwardingModel wraps a model and forwards the run-option filters the
// wrapped model implements, as a model that wraps a connector does.
type forwardingModel struct {
	model.LLM
}

func (m forwardingModel) RunHTTPOptions(opts *genai.HTTPOptions) *genai.HTTPOptions {
	if f, ok := m.LLM.(model.RunHTTPOptionsFilter); ok {
		return f.RunHTTPOptions(opts)
	}
	return opts
}

func (m forwardingModel) RunLabels(labels map[string]string) map[string]string {
	if f, ok := m.LLM.(model.RunLabelsFilter); ok {
		return f.RunLabels(labels)
	}
	return labels
}

// TestRunConfig_WrappedModelForwardsFilters runs a Claude agent whose model is
// wrapped in a type that forwards model.RunHTTPOptionsFilter and
// model.RunLabelsFilter, with run labels and RetryOptions that the Anthropic
// connector cannot send. The Claude request is sent instead of failing.
func TestRunConfig_WrappedModelForwardsFilters(t *testing.T) {
	b := &mixedModelBackends{}
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
	a, err := llmagent.New(llmagent.Config{Name: "claude_agent", Model: forwardingModel{claudeModel}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	runner := testutil.NewTestAgentRunner(t, a)
	labels := map[string]string{"cost_center": "x"}
	cfg := agent.RunConfig{
		Labels:      &labels,
		HTTPOptions: &genai.HTTPOptions{RetryOptions: &genai.HTTPRetryOptions{Attempts: genai.Ptr[int32](3)}},
	}

	events, err := testutil.CollectEvents(runner.RunContentWithConfig(t, "session", genai.NewContentFromText("hi", genai.RoleUser), cfg))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if len(b.claude) != 1 {
		t.Fatalf("Claude backend got %d requests, want 1", len(b.claude))
	}
	if len(events) == 0 {
		t.Fatal("run produced no events")
	}
	last := events[len(events)-1]
	if last.Content == nil || len(last.Content.Parts) == 0 || last.Content.Parts[0].Text != "claude answer" {
		t.Errorf("last event = %+v, want the Claude agent's answer", last)
	}
}

// TestRunConfig_Labels_GeminiAPI runs a Gemini agent on the Gemini Developer
// API with labels on both the agent and the run. A before-model callback sees
// both, and the request sent to that API carries no labels.
func TestRunConfig_Labels_GeminiAPI(t *testing.T) {
	var (
		mu     sync.Mutex
		bodies []map[string]any
	)
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, err
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			return nil, err
		}
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": {"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`)),
		}, nil
	})
	geminiModel, err := gemini.NewModel(t.Context(), "gemini-2.5-flash", &genai.ClientConfig{
		Backend:    genai.BackendGeminiAPI,
		APIKey:     "test-key",
		HTTPClient: &http.Client{Transport: transport},
	})
	if err != nil {
		t.Fatalf("gemini.NewModel() error = %v", err)
	}
	var callbackLabels map[string]string
	recordLabels := func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
		callbackLabels = maps.Clone(req.Config.Labels)
		return nil, nil
	}
	a, err := llmagent.New(llmagent.Config{
		Name:                  "gemini_agent",
		Model:                 geminiModel,
		GenerateContentConfig: &genai.GenerateContentConfig{Labels: map[string]string{"team": "core"}},
		BeforeModelCallbacks:  []llmagent.BeforeModelCallback{recordLabels},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	runner := testutil.NewTestAgentRunner(t, a)
	runLabels := map[string]string{"cost_center": "x"}

	if _, err := testutil.CollectEvents(runner.RunContentWithConfig(t, "session", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{Labels: &runLabels})); err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if diff := cmp.Diff(map[string]string{"team": "core", "cost_center": "x"}, callbackLabels); diff != "" {
		t.Errorf("before-model callback labels mismatch (-want +got):\n%s", diff)
	}
	if len(bodies) != 1 {
		t.Fatalf("Gemini backend got %d requests, want 1", len(bodies))
	}
	if got, sent := bodies[0]["labels"]; sent {
		t.Errorf("Gemini API request carries labels %v, want none", got)
	}
}
