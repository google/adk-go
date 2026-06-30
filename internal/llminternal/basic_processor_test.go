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
