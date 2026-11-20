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
	"net/http"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool"
)

// TestBasicRequestProcessor_OutputSchemaPerMode pins how
// basicRequestProcessor populates LLMRequest.Config.ResponseSchema /
// ResponseMIMEType across the three LlmAgent modes:
//
//   - task: the FinishTaskTool's declaration is the authoritative
//     schema for the model's structured output, so basic must NOT
//     write ResponseSchema/ResponseMIMEType — the LLM emits its
//     answer inside the finish_task FC args, not as a structured
//     text response.
//
//   - single_turn / chat: basic populates ResponseSchema +
//     ResponseMIMEType from the agent's OutputSchema, allowing the
//     model to return a JSON-shaped response directly.
func TestBasicRequestProcessor_OutputSchemaPerMode(t *testing.T) {
	t.Parallel()

	schema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"answer": {Type: genai.TypeString}},
	}

	cases := []struct {
		name          string
		mode          Mode
		wantSchemaSet bool // true => ResponseSchema=schema + ResponseMIMEType=json
	}{
		{
			name:          "task mode skips OutputSchema",
			mode:          ModeTask,
			wantSchemaSet: false,
		},
		{
			name:          "single_turn mode sets OutputSchema",
			mode:          ModeSingleTurn,
			wantSchemaSet: true,
		},
		{
			name:          "chat mode sets OutputSchema",
			mode:          ModeChat,
			wantSchemaSet: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			mockAgent, err := agent.New(agent.Config{Name: "agent_" + string(tc.mode)})
			if err != nil {
				t.Fatal(err)
			}
			mock := &mockLLMAgent{
				Agent: mockAgent,
				s: &State{
					Mode:         tc.mode,
					Model:        &mockLLM{name: "m"},
					OutputSchema: schema,
				},
			}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent: mock,
			})
			req := &model.LLMRequest{}
			for ev, err := range basicRequestProcessor(ctx, req, &Flow{}) {
				if ev != nil {
					t.Fatalf("basicRequestProcessor unexpectedly yielded an event: %+v", ev)
				}
				if err != nil {
					t.Fatalf("basicRequestProcessor failed: %v", err)
				}
			}
			if req.Config == nil {
				t.Fatal("req.Config is nil; want non-nil")
			}

			var (
				wantSchema   *genai.Schema
				wantMIMEType string
			)
			if tc.wantSchemaSet {
				wantSchema = schema
				wantMIMEType = "application/json"
			}
			if diff := cmp.Diff(wantSchema, req.Config.ResponseSchema); diff != "" {
				t.Errorf("ResponseSchema mismatch (-want +got):\n%s", diff)
			}
			if got := req.Config.ResponseMIMEType; got != wantMIMEType {
				t.Errorf("ResponseMIMEType = %q, want %q", got, wantMIMEType)
			}
		})
	}
}

// runBasicRequestProcessor runs basicRequestProcessor for an LLM agent with
// state s under rc, which may be nil, and returns the request it built.
func runBasicRequestProcessor(t *testing.T, s *State, rc *agent.RunConfig) *model.LLMRequest {
	t.Helper()
	a, err := agent.New(agent.Config{Name: "test_agent"})
	if err != nil {
		t.Fatal(err)
	}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:     &mockLLMAgent{Agent: a, s: s},
		RunConfig: rc,
	})
	req := &model.LLMRequest{}
	for ev, err := range basicRequestProcessor(ctx, req, &Flow{}) {
		if ev != nil {
			t.Fatalf("basicRequestProcessor unexpectedly yielded an event: %+v", ev)
		}
		if err != nil {
			t.Fatalf("basicRequestProcessor failed: %v", err)
		}
	}
	if req.Config == nil {
		t.Fatal("req.Config is nil; want non-nil")
	}
	return req
}

// TestBasicProcessor_RunConfigHTTPOptions_P1PortedFlaw pins that BaseURL and
// APIVersion from RunConfig.HTTPOptions reach the request only when the agent
// configures no HTTPOptions, as adk-python's _merge_run_config_http_options
// does (flows/llm_flows/basic.py:45-60).
func TestBasicProcessor_RunConfigHTTPOptions_P1PortedFlaw(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		agentCfg *genai.GenerateContentConfig
		want     *genai.HTTPOptions
	}{
		{
			name:     "no_agent_http_options",
			agentCfg: nil,
			want:     &genai.HTTPOptions{BaseURL: "https://run-proxy.example", APIVersion: "v1beta"},
		},
		{
			name:     "agent_has_http_options",
			agentCfg: &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: genai.Ptr(1000 * time.Millisecond)}},
			want:     &genai.HTTPOptions{Timeout: genai.Ptr(1000 * time.Millisecond)},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			rc := &agent.RunConfig{HTTPOptions: &genai.HTTPOptions{BaseURL: "https://run-proxy.example", APIVersion: "v1beta"}}

			req := runBasicRequestProcessor(t, &State{GenerateContentConfig: tc.agentCfg}, rc)

			if diff := cmp.Diff(tc.want, req.Config.HTTPOptions); diff != "" {
				t.Errorf("req.Config.HTTPOptions mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestBasicProcessor_RunConfigHTTPOptions_NoLeak pins that writes into the
// request's Timeout, RetryOptions and ExtraBody, as a BeforeModelCallback can
// make, reach neither the caller's RunConfig nor the agent, and that a later
// run without RunConfig.HTTPOptions sees only the agent's options.
func TestBasicProcessor_RunConfigHTTPOptions_NoLeak(t *testing.T) {
	t.Parallel()

	state := &State{GenerateContentConfig: &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{
		Timeout: genai.Ptr(1000 * time.Millisecond),
		Headers: http.Header{"Agent-Header": {"agent"}},
	}}}
	runCfg := &agent.RunConfig{HTTPOptions: &genai.HTTPOptions{
		Timeout:      genai.Ptr(2000 * time.Millisecond),
		RetryOptions: &genai.HTTPRetryOptions{Attempts: genai.Ptr[int32](3), HTTPStatusCodes: []int32{500}},
		ExtraBody:    map[string]any{"k": "orig"},
	}}

	first := runBasicRequestProcessor(t, state, runCfg)
	wantFirst := &genai.HTTPOptions{
		Timeout:      genai.Ptr(2000 * time.Millisecond),
		Headers:      http.Header{"Agent-Header": {"agent"}},
		RetryOptions: &genai.HTTPRetryOptions{Attempts: genai.Ptr[int32](3), HTTPStatusCodes: []int32{500}},
		ExtraBody:    map[string]any{"k": "orig"},
	}
	if diff := cmp.Diff(wantFirst, first.Config.HTTPOptions); diff != "" {
		t.Fatalf("first request HTTPOptions mismatch (-want +got):\n%s", diff)
	}
	*first.Config.HTTPOptions.Timeout = 5 * time.Second
	first.Config.HTTPOptions.RetryOptions.HTTPStatusCodes[0] = 429
	first.Config.HTTPOptions.ExtraBody["k"] = "changed"

	wantRun := &genai.HTTPOptions{
		Timeout:      genai.Ptr(2000 * time.Millisecond),
		RetryOptions: &genai.HTTPRetryOptions{Attempts: genai.Ptr[int32](3), HTTPStatusCodes: []int32{500}},
		ExtraBody:    map[string]any{"k": "orig"},
	}
	if diff := cmp.Diff(wantRun, runCfg.HTTPOptions); diff != "" {
		t.Errorf("RunConfig.HTTPOptions changed by writes to the request (-want +got):\n%s", diff)
	}
	if got := *state.GenerateContentConfig.HTTPOptions.Timeout; got != 1000*time.Millisecond {
		t.Errorf("agent Timeout = %v after writes to the request, want 1s", got)
	}

	second := runBasicRequestProcessor(t, state, &agent.RunConfig{})
	wantSecond := &genai.HTTPOptions{
		Timeout: genai.Ptr(1000 * time.Millisecond),
		Headers: http.Header{"Agent-Header": {"agent"}},
	}
	if diff := cmp.Diff(wantSecond, second.Config.HTTPOptions); diff != "" {
		t.Errorf("second request HTTPOptions mismatch (-want +got):\n%s", diff)
	}
}

// TestBasicProcessor_RunConfigHTTPOptions_NoLeakWhenAgentHasNone pins that
// when the agent sets no HTTPOptions, writes into the request's Timeout and
// Headers, as a BeforeModelCallback can make, do not reach the caller's
// RunConfig.
func TestBasicProcessor_RunConfigHTTPOptions_NoLeakWhenAgentHasNone(t *testing.T) {
	t.Parallel()

	runCfg := &agent.RunConfig{HTTPOptions: &genai.HTTPOptions{
		Timeout: genai.Ptr(2000 * time.Millisecond),
		Headers: http.Header{"X-Run": {"run"}},
	}}

	got := runBasicRequestProcessor(t, &State{}, runCfg).Config.HTTPOptions
	if got == nil || got.Timeout == nil || got.Headers == nil {
		t.Fatalf("req.Config.HTTPOptions = %+v, want a Timeout and Headers", got)
	}
	*got.Timeout = 5 * time.Second
	got.Headers.Set("X-Callback", "cb")

	want := &genai.HTTPOptions{
		Timeout: genai.Ptr(2000 * time.Millisecond),
		Headers: http.Header{"X-Run": {"run"}},
	}
	if diff := cmp.Diff(want, runCfg.HTTPOptions); diff != "" {
		t.Errorf("RunConfig.HTTPOptions changed by writes to the request (-want +got):\n%s", diff)
	}
}

// TestBasicProcessor_RunConfigHTTPOptions_CanonicalHeaderNames pins that
// header names on the request are canonical and that a RunConfig header
// replaces the agent header with the same canonical name.
func TestBasicProcessor_RunConfigHTTPOptions_CanonicalHeaderNames(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		agentCfg *genai.GenerateContentConfig
		runHTTP  *genai.HTTPOptions
		want     http.Header
	}{
		{
			name:     "no_agent_http_options",
			agentCfg: nil,
			runHTTP:  &genai.HTTPOptions{Headers: http.Header{"x-custom-key": {"run-val"}}},
			want:     http.Header{"X-Custom-Key": {"run-val"}},
		},
		{
			name:     "agent_and_run_header_differ_in_case",
			agentCfg: &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Headers: http.Header{"X-Custom-Key": {"agent-val"}}}},
			runHTTP:  &genai.HTTPOptions{Headers: http.Header{"x-custom-key": {"run-val"}}},
			want:     http.Header{"X-Custom-Key": {"run-val"}},
		},
		{
			name:     "non_canonical_agent_header",
			agentCfg: &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Headers: http.Header{"x-custom": {"val"}}}},
			runHTTP:  &genai.HTTPOptions{Timeout: genai.Ptr(1000 * time.Millisecond)},
			want:     http.Header{"X-Custom": {"val"}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			req := runBasicRequestProcessor(t, &State{GenerateContentConfig: tc.agentCfg}, &agent.RunConfig{HTTPOptions: tc.runHTTP})

			if req.Config.HTTPOptions == nil {
				t.Fatal("req.Config.HTTPOptions is nil; want non-nil")
			}
			if diff := cmp.Diff(tc.want, req.Config.HTTPOptions.Headers); diff != "" {
				t.Errorf("req.Config.HTTPOptions.Headers mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestBasicProcessor_RunConfigHTTPOptions_AgentTimeoutOnlyKeepsRunHeaders
// pins the merge into agent HTTPOptions that set only a Timeout: the run's
// headers are added and the agent's Timeout is kept.
func TestBasicProcessor_RunConfigHTTPOptions_AgentTimeoutOnlyKeepsRunHeaders(t *testing.T) {
	t.Parallel()

	state := &State{GenerateContentConfig: &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{Timeout: genai.Ptr(1000 * time.Millisecond)}}}
	rc := &agent.RunConfig{HTTPOptions: &genai.HTTPOptions{Headers: http.Header{"X-Run": {"run"}}}}

	req := runBasicRequestProcessor(t, state, rc)

	want := &genai.HTTPOptions{
		Timeout: genai.Ptr(1000 * time.Millisecond),
		Headers: http.Header{"X-Run": {"run"}},
	}
	if diff := cmp.Diff(want, req.Config.HTTPOptions); diff != "" {
		t.Errorf("req.Config.HTTPOptions mismatch (-want +got):\n%s", diff)
	}
}

// timeoutOnlyLLM is a model whose connector accepts only the Timeout of a
// run's HTTP options, as a RunHTTPOptionsFilter.
type timeoutOnlyLLM struct {
	model.LLM
}

func (*timeoutOnlyLLM) RunHTTPOptions(opts *genai.HTTPOptions) *genai.HTTPOptions {
	if opts.Timeout == nil {
		return nil
	}
	return &genai.HTTPOptions{Timeout: opts.Timeout}
}

// TestBasicProcessor_RunConfigHTTPOptions_FilteredForModel pins that a model
// implementing RunHTTPOptionsFilter receives only the run HTTP options it
// keeps, while HTTP options set on the agent reach the request whole.
func TestBasicProcessor_RunConfigHTTPOptions_FilteredForModel(t *testing.T) {
	t.Parallel()

	runHTTP := &genai.HTTPOptions{
		Timeout:      genai.Ptr(2000 * time.Millisecond),
		Headers:      http.Header{"X-Run": {"run"}},
		RetryOptions: &genai.HTTPRetryOptions{Attempts: genai.Ptr[int32](3)},
		ExtraBody:    map[string]any{"k": "v"},
	}
	tests := []struct {
		name     string
		agentCfg *genai.GenerateContentConfig
		runHTTP  *genai.HTTPOptions
		want     *genai.HTTPOptions
	}{
		{
			name:    "agent_without_http_options",
			runHTTP: runHTTP,
			want:    &genai.HTTPOptions{Timeout: genai.Ptr(2000 * time.Millisecond)},
		},
		{
			name:     "agent_options_kept_whole",
			agentCfg: &genai.GenerateContentConfig{HTTPOptions: &genai.HTTPOptions{ExtraBody: map[string]any{"agent": "set"}}},
			runHTTP:  runHTTP,
			want:     &genai.HTTPOptions{Timeout: genai.Ptr(2000 * time.Millisecond), ExtraBody: map[string]any{"agent": "set"}},
		},
		{
			name:    "nothing_left",
			runHTTP: &genai.HTTPOptions{RetryOptions: &genai.HTTPRetryOptions{Attempts: genai.Ptr[int32](3)}},
			want:    nil,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			state := &State{Model: &timeoutOnlyLLM{}, GenerateContentConfig: tc.agentCfg}

			req := runBasicRequestProcessor(t, state, &agent.RunConfig{HTTPOptions: tc.runHTTP})

			if diff := cmp.Diff(tc.want, req.Config.HTTPOptions); diff != "" {
				t.Errorf("req.Config.HTTPOptions mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestBasicProcessor_RunConfigLabels_AgentWithoutLabels pins that RunConfig
// labels reach the request when the agent configures no labels at all, in a
// map of the request's own.
func TestBasicProcessor_RunConfigLabels_AgentWithoutLabels(t *testing.T) {
	t.Parallel()

	labels := map[string]string{"team": "billing"}
	req := runBasicRequestProcessor(t, &State{}, &agent.RunConfig{Labels: &labels})

	if diff := cmp.Diff(map[string]string{"team": "billing"}, req.Config.Labels); diff != "" {
		t.Errorf("req.Config.Labels mismatch (-want +got):\n%s", diff)
	}
	req.Config.Labels["written"] = "by callback"
	if diff := cmp.Diff(map[string]string{"team": "billing"}, labels); diff != "" {
		t.Errorf("RunConfig labels changed (-want +got):\n%s", diff)
	}
}

// labelDroppingLLM is a model whose connector sends no labels, as a
// RunLabelsFilter.
type labelDroppingLLM struct {
	model.LLM
}

func (*labelDroppingLLM) RunLabels(map[string]string) map[string]string { return nil }

// TestBasicProcessor_RunConfigLabels_FilteredForModel pins that a model
// implementing RunLabelsFilter receives none of the run's labels, while labels
// set on the agent reach the request, and that neither the caller's map nor
// the agent's map is written.
func TestBasicProcessor_RunConfigLabels_FilteredForModel(t *testing.T) {
	t.Parallel()

	runLabels := map[string]string{"team": "billing"}
	agentLabels := map[string]string{"env": "prod"}
	state := &State{Model: &labelDroppingLLM{}, GenerateContentConfig: &genai.GenerateContentConfig{Labels: agentLabels}}

	req := runBasicRequestProcessor(t, state, &agent.RunConfig{Labels: &runLabels})

	if diff := cmp.Diff(map[string]string{"env": "prod"}, req.Config.Labels); diff != "" {
		t.Errorf("req.Config.Labels mismatch (-want +got):\n%s", diff)
	}
	req.Config.Labels["written"] = "by callback"
	if diff := cmp.Diff(map[string]string{"env": "prod"}, agentLabels); diff != "" {
		t.Errorf("agent labels changed (-want +got):\n%s", diff)
	}
}

// TestBasicProcessor_RunConfigLabels_NotWritten pins that writes into the
// request's labels, as a BeforeModelCallback can make, reach neither the
// caller's RunConfig.Labels nor the agent's labels.
func TestBasicProcessor_RunConfigLabels_NotWritten(t *testing.T) {
	t.Parallel()

	runLabels := map[string]string{"team": "billing"}
	agentLabels := map[string]string{"env": "prod"}
	state := &State{GenerateContentConfig: &genai.GenerateContentConfig{Labels: agentLabels}}

	req := runBasicRequestProcessor(t, state, &agent.RunConfig{Labels: &runLabels})
	req.Config.Labels["written"] = "by callback"

	if diff := cmp.Diff(map[string]string{"team": "billing"}, runLabels); diff != "" {
		t.Errorf("RunConfig labels changed (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]string{"env": "prod"}, agentLabels); diff != "" {
		t.Errorf("agent labels changed (-want +got):\n%s", diff)
	}
}

// capableLLM is a model that declares its capabilities through
// model.CapabilitiesProvider and reports a backend like the built-in Gemini
// model does.
type capableLLM struct {
	model.LLM
	name    string
	backend genai.Backend
	caps    model.Capabilities
}

func (m *capableLLM) Name() string                       { return m.name }
func (m *capableLLM) GetGoogleLLMVariant() genai.Backend { return m.backend }
func (m *capableLLM) Capabilities() model.Capabilities   { return m.caps }

// TestBasicRequestProcessor_DeclaredCapabilityDecides pins that a model's
// declared Capabilities decide whether an agent with tools gets the native
// output schema, over the name-and-backend rule for undeclared models.
func TestBasicRequestProcessor_DeclaredCapabilityDecides(t *testing.T) {
	t.Parallel()

	schema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
		Required:   []string{"city"},
	}
	tests := []struct {
		name         string
		model        *capableLLM
		wantSchema   *genai.Schema
		wantMIMEType string
	}{
		{
			name:         "non_gemini_model_declares_support",
			model:        &capableLLM{name: "mock", backend: genai.BackendUnspecified, caps: model.Capabilities{OutputSchemaAndTools: true}},
			wantSchema:   schema,
			wantMIMEType: "application/json",
		},
		{
			name:  "gemini_on_vertex_declares_no_support",
			model: &capableLLM{name: "gemini-2.5-flash", backend: genai.BackendVertexAI, caps: model.Capabilities{}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			state := &State{
				Mode:         ModeChat,
				Model:        tc.model,
				Tools:        []tool.Tool{&mockTool{name: "dummy_tool"}},
				OutputSchema: schema,
			}

			req := runBasicRequestProcessor(t, state, nil)

			if diff := cmp.Diff(tc.wantSchema, req.Config.ResponseSchema); diff != "" {
				t.Errorf("ResponseSchema mismatch (-want +got):\n%s", diff)
			}
			if got := req.Config.ResponseMIMEType; got != tc.wantMIMEType {
				t.Errorf("ResponseMIMEType = %q, want %q", got, tc.wantMIMEType)
			}
		})
	}
}

// TestBasicRequestProcessor_TaskModeIgnoresDeclaredCapability pins that a
// task-mode agent with tools gets no native output schema even when its model
// declares OutputSchemaAndTools: the finish_task tool carries the schema.
func TestBasicRequestProcessor_TaskModeIgnoresDeclaredCapability(t *testing.T) {
	t.Parallel()

	state := &State{
		Mode:  ModeTask,
		Model: &capableLLM{name: "mock", caps: model.Capabilities{OutputSchemaAndTools: true}},
		Tools: []tool.Tool{&mockTool{name: "dummy_tool"}},
		OutputSchema: &genai.Schema{
			Type:       genai.TypeObject,
			Properties: map[string]*genai.Schema{"city": {Type: genai.TypeString}},
			Required:   []string{"city"},
		},
	}

	req := runBasicRequestProcessor(t, state, nil)

	if req.Config.ResponseSchema != nil {
		t.Errorf("ResponseSchema = %+v, want nil", req.Config.ResponseSchema)
	}
	if got := req.Config.ResponseMIMEType; got != "" {
		t.Errorf("ResponseMIMEType = %q, want empty", got)
	}
}
