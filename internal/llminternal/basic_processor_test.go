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
	"encoding/json"
	"errors"
	"math/big"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
)

func TestBasicRequestProcessor_ConfigIsolation(t *testing.T) {
	newConfig := func() *genai.GenerateContentConfig {
		return &genai.GenerateContentConfig{
			ResponseJsonSchema: map[string]any{
				"properties": map[string]any{"answer": map[string]any{"type": "string"}},
			},
			ResponseSchema: &genai.Schema{
				Default: map[string]any{"answer": "default"},
				Example: []string{"example"},
			},
			Tools: []*genai.Tool{{FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name: "lookup",
				ParametersJsonSchema: map[string]any{
					"properties": map[string]any{"query": map[string]any{"type": "string"}},
				},
			}}}},
		}
	}
	config := newConfig()
	base, err := agent.New(agent.Config{Name: "testAgent"})
	if err != nil {
		t.Fatal(err)
	}
	a := &mockLLMAgent{Agent: base, s: &State{GenerateContentConfig: config}}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a})
	requests := []*model.LLMRequest{{}, {}}
	for _, req := range requests {
		for _, err := range basicRequestProcessor(ctx, req, &Flow{}) {
			if err != nil {
				t.Fatal(err)
			}
		}
		if !reflect.DeepEqual(req.Config, config) {
			t.Fatalf("request config = %#v, want %#v", req.Config, config)
		}
	}
	// BeforeModel callbacks can mutate any of these request fields.
	copied := requests[0].Config
	copied.ResponseJsonSchema.(map[string]any)["additionalProperties"] = false
	copied.ResponseJsonSchema.(map[string]any)["properties"].(map[string]any)["answer"].(map[string]any)["type"] = "number"
	copied.ResponseSchema.Default.(map[string]any)["answer"] = "changed"
	copied.ResponseSchema.Example.([]string)[0] = "changed"
	copied.Tools[0].FunctionDeclarations[0].ParametersJsonSchema.(map[string]any)["properties"].(map[string]any)["query"].(map[string]any)["type"] = "number"
	for name, got := range map[string]*genai.GenerateContentConfig{"agent": config, "other request": requests[1].Config} {
		if diff := cmp.Diff(newConfig(), got); diff != "" {
			t.Errorf("%s config mutated (-want +got):\n%s", name, diff)
		}
	}
}

func TestBasicRequestProcessor_JSONSchemaCompatibility(t *testing.T) {
	type schemaWithCache struct {
		Type  string `json:"type"`
		cache string
	}
	var deep any = map[string]any{"type": "string"}
	for range 130 {
		deep = map[string]any{"type": "array", "items": deep}
	}
	for name, schema := range map[string]any{
		"big integer":   map[string]any{"type": "integer", "minimum": big.NewInt(9007199254740993)},
		"private cache": &schemaWithCache{Type: "string", cache: "not serialized"},
		"deep schema":   deep,
	} {
		t.Run(name, func(t *testing.T) {
			base := utils.Must(agent.New(agent.Config{Name: "testAgent"}))
			a := &mockLLMAgent{Agent: base, s: &State{
				GenerateContentConfig: &genai.GenerateContentConfig{ResponseJsonSchema: schema},
			}}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a})
			req := &model.LLMRequest{}
			for _, err := range basicRequestProcessor(ctx, req, &Flow{}) {
				if err != nil {
					t.Fatal(err)
				}
			}
			want, err := json.Marshal(schema)
			if err != nil {
				t.Fatal(err)
			}
			got, err := json.Marshal(req.Config.ResponseJsonSchema)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != string(want) {
				t.Errorf("schema JSON = %s, want %s", got, want)
			}
		})
	}
}

func TestBasicRequestProcessor_ConfigCloneError(t *testing.T) {
	schema := map[string]any{}
	schema["self"] = schema
	base := utils.Must(agent.New(agent.Config{Name: "testAgent"}))
	a := &mockLLMAgent{Agent: base, s: &State{
		GenerateContentConfig: &genai.GenerateContentConfig{ResponseJsonSchema: schema},
	}}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a})
	req := &model.LLMRequest{}
	errorsSeen := 0
	for ev, err := range basicRequestProcessor(ctx, req, &Flow{}) {
		if ev != nil || !errors.Is(err, errCloneDepth) {
			t.Fatalf("processor returned (%v, %v), want depth error", ev, err)
		}
		errorsSeen++
	}
	if errorsSeen != 1 || req.Config != nil {
		t.Errorf("processor returned %d errors and config %p, want one error and no config", errorsSeen, req.Config)
	}
}

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
