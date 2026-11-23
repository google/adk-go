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
	"testing"

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
