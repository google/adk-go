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
	"google.golang.org/adk/v2/codeexecutor"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
)

// TestCodeExecutionRequestProcessor_BuiltInCreatesConfigAndAppendsOwnEntry
// calls the processor directly, because in a full run basicRequestProcessor
// always sets req.Config first and the nil-config branch is not reached.
func TestCodeExecutionRequestProcessor_BuiltInCreatesConfigAndAppendsOwnEntry(t *testing.T) {
	codeExecution := &genai.Tool{CodeExecution: &genai.ToolCodeExecution{}}
	declarations := &genai.Tool{FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "test_func"}}}
	tests := []struct {
		name   string
		config *genai.GenerateContentConfig
		want   *genai.GenerateContentConfig
	}{
		{
			name:   "nil_config",
			config: nil,
			want:   &genai.GenerateContentConfig{Tools: []*genai.Tool{codeExecution}},
		},
		{
			name:   "nil_tools",
			config: &genai.GenerateContentConfig{Temperature: genai.Ptr[float32](0.5)},
			want:   &genai.GenerateContentConfig{Temperature: genai.Ptr[float32](0.5), Tools: []*genai.Tool{codeExecution}},
		},
		{
			name:   "empty_tools",
			config: &genai.GenerateContentConfig{Tools: []*genai.Tool{}},
			want:   &genai.GenerateContentConfig{Tools: []*genai.Tool{codeExecution}},
		},
		{
			name:   "existing_function_declarations_entry",
			config: &genai.GenerateContentConfig{Tools: []*genai.Tool{declarations}},
			want:   &genai.GenerateContentConfig{Tools: []*genai.Tool{declarations, codeExecution}},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			base, err := agent.New(agent.Config{Name: "test_agent"})
			if err != nil {
				t.Fatal(err)
			}
			a := &mockLLMAgent{Agent: base, s: &State{CodeExecutor: codeexecutor.NewBuiltInExecutor()}}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Agent: a})
			req := &model.LLMRequest{Model: "gemini-2.5-flash", Config: tc.config}

			for ev, err := range codeExecutionRequestProcessor(ctx, req, &Flow{}) {
				t.Fatalf("codeExecutionRequestProcessor yielded (%v, %v), want nothing", ev, err)
			}

			if diff := cmp.Diff(tc.want, req.Config); diff != "" {
				t.Errorf("request Config mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
