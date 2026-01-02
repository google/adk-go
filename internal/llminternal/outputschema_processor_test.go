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

package llminternal

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
)

type mockTool struct {
	name string
}

func (m *mockTool) Name() string        { return m.name }
func (m *mockTool) Description() string { return "mock tool" }
func (m *mockTool) IsLongRunning() bool { return false }
func (m *mockTool) Declaration() *genai.FunctionDeclaration {
	return &genai.FunctionDeclaration{Name: m.name}
}
func (m *mockTool) Run(ctx tool.Context, args any) (map[string]any, error) { return nil, nil }

type mockLLM struct {
	model.LLM
	name string
}

func (m *mockLLM) Name() string { return m.name }

// mockLLMAgent satisfies both agent.Agent (via embedding) and llminternal.Agent (via internal() implementation)
type mockLLMAgent struct {
	agent.Agent
	s *State
}

func (m *mockLLMAgent) internal() *State {
	return m.s
}

func TestOutputSchemaRequestProcessor(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"answer": {Type: genai.TypeString},
		},
		Required: []string{"answer"},
	}

	t.Run("InjectsToolAndInstructions", func(t *testing.T) {
		baseAgent := utils.Must(agent.New(agent.Config{Name: "SchemaAgent"}))
		mockAgent := &mockLLMAgent{
			Agent: baseAgent,
			s: &State{
				Model:        &mockLLM{name: "gemini-1.5-flash"},
				OutputSchema: schema,
				Tools:        []tool.Tool{&mockTool{name: "other_tool"}},
			},
		}

		req := &model.LLMRequest{}
		ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
			Agent: mockAgent,
		})

		if err := outputSchemaRequestProcessor(ctx, req); err != nil {
			t.Fatalf("outputSchemaRequestProcessor() error = %v", err)
		}

		// Verify set_model_response tool is present
		if _, ok := req.Tools["set_model_response"]; !ok {
			t.Error("req.Tools['set_model_response'] missing")
		}

		// Verify instructions
		instructions := utils.TextParts(req.Config.SystemInstruction)
		found := false
		for _, s := range instructions {
			if strings.Contains(s, "set_model_response") && strings.Contains(s, "required structured format") {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Instruction about set_model_response not found. Instructions: %v", instructions)
		}
	})

	t.Run("NoOpWhenNoTools", func(t *testing.T) {
		baseAgent := utils.Must(agent.New(agent.Config{Name: "SchemaAgentNoTools"}))
		mockAgent := &mockLLMAgent{
			Agent: baseAgent,
			s: &State{
				Model:        &mockLLM{name: "gemini-1.5-flash"},
				OutputSchema: schema,
				Tools:        nil, // No tools -> optimization skips processor
			},
		}

		req := &model.LLMRequest{}
		ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
			Agent: mockAgent,
		})

		if err := outputSchemaRequestProcessor(ctx, req); err != nil {
			t.Fatalf("outputSchemaRequestProcessor() error = %v", err)
		}

		if _, ok := req.Tools["set_model_response"]; ok {
			t.Error("set_model_response tool should NOT be added when no other tools are present")
		}
	})

	t.Run("NoOpWhenNoSchema", func(t *testing.T) {
		baseAgent := utils.Must(agent.New(agent.Config{Name: "NoSchemaAgent"}))
		mockAgent := &mockLLMAgent{
			Agent: baseAgent,
			s: &State{
				Model:        &mockLLM{name: "gemini-1.5-flash"},
				OutputSchema: nil,
				Tools:        []tool.Tool{&mockTool{name: "other_tool"}},
			},
		}

		req := &model.LLMRequest{}
		ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
			Agent: mockAgent,
		})

		if err := outputSchemaRequestProcessor(ctx, req); err != nil {
			t.Fatalf("outputSchemaRequestProcessor() error = %v", err)
		}

		if _, ok := req.Tools["set_model_response"]; ok {
			t.Error("set_model_response tool should NOT be added when no OutputSchema")
		}
	})

	t.Run("NoOpWhenNativeSupportAvailable", func(t *testing.T) {
		// Native support = Vertex AI + Gemini 2.0+
		t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "1")

		baseAgent := utils.Must(agent.New(agent.Config{Name: "VertexGemini2Agent"}))
		mockAgent := &mockLLMAgent{
			Agent: baseAgent,
			s: &State{
				Model:        &mockLLM{name: "gemini-2.0-flash"},
				OutputSchema: schema,
				Tools:        []tool.Tool{&mockTool{name: "other_tool"}},
			},
		}

		req := &model.LLMRequest{}
		ctx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
			Agent: mockAgent,
		})

		if err := outputSchemaRequestProcessor(ctx, req); err != nil {
			t.Fatalf("outputSchemaRequestProcessor() error = %v", err)
		}

		if _, ok := req.Tools["set_model_response"]; ok {
			t.Error("set_model_response tool should NOT be added when native support is available")
		}
	})
}

func TestCreateFinalModelResponseEvent(t *testing.T) {
	// Setup context
	a := utils.Must(agent.New(agent.Config{Name: "TestAgent"}))
	invCtx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{
		Agent: a,
	})

	jsonResp := `{"answer": "value"}`
	event := createFinalModelResponseEvent(invCtx, jsonResp)

	if event.Author != "TestAgent" {
		t.Errorf("Author = %q, want TestAgent", event.Author)
	}
	if event.Content == nil || event.Content.Role != "model" {
		t.Errorf("Content Role mismatch or nil")
	}
	if event.Branch != invCtx.Branch() {
		t.Errorf("Branch = %q, want %q", event.Branch, invCtx.Branch())
	}
	if event.InvocationID != invCtx.InvocationID() {
		t.Errorf("InvocationID = %q, want %q", event.InvocationID, invCtx.InvocationID())
	}
	if len(event.Content.Parts) != 1 {
		t.Fatalf("Content Parts length = %d, want 1", len(event.Content.Parts))
	}
	if got := event.Content.Parts[0].Text; got != jsonResp {
		t.Errorf("Content Text = %q, want %q", got, jsonResp)
	}
}

func TestGetStructuredModelResponse(t *testing.T) {
	t.Run("ExtractsResponse", func(t *testing.T) {
		event := &session.Event{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								Name: "set_model_response",
								Response: map[string]any{
									"result": 123.0,
								},
							},
						},
					},
				},
			},
		}

		got, err := retrieveStructuredModelResponse(event)
		if err != nil {
			t.Fatalf("GetStructuredModelResponse error: %v", err)
		}

		// The JSON might be formatted differently, so unmarshal to compare
		var gotMap map[string]any
		if err := json.Unmarshal([]byte(got), &gotMap); err != nil {
			t.Fatalf("Result is not valid JSON: %v", err)
		}

		wantMap := map[string]any{"result": 123.0}
		if diff := cmp.Diff(wantMap, gotMap); diff != "" {
			t.Errorf("Extracted JSON mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("NoResponseWhenNameMismatch", func(t *testing.T) {
		event := &session.Event{
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Parts: []*genai.Part{
						{
							FunctionResponse: &genai.FunctionResponse{
								Name:     "other_tool",
								Response: map[string]any{"x": 1},
							},
						},
					},
				},
			},
		}

		got, err := retrieveStructuredModelResponse(event)
		if err == nil {
			t.Fatal("Expected error for tool name mismatch, got nil")
		}
		if got != "" {
			t.Errorf("Expected empty string, got %q", got)
		}
	})

	t.Run("NilEvent", func(t *testing.T) {
		got, err := retrieveStructuredModelResponse(nil)
		if err == nil {
			t.Fatal("Expected error for nil event, got nil")
		}
		if got != "" {
			t.Error("expected empty string")
		}
	})
}

func TestSetModelResponseTool(t *testing.T) {
	schema := &genai.Schema{
		Type: genai.TypeObject,
		Properties: map[string]*genai.Schema{
			"count": {Type: genai.TypeInteger},
		},
		Required: []string{"count"},
	}

	toolInstance := &setModelResponseTool{schema: schema}

	// Check Description
	if !strings.Contains(toolInstance.Description(), "outputting text directly") {
		t.Errorf("Description should contain explicit instruction")
	}

	t.Run("RunSuccess", func(t *testing.T) {
		invCtx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{})
		toolCtx := toolinternal.NewToolContext(invCtx, "", nil)

		input := map[string]any{"count": 10.0} // JSON numbers often come as float64
		got, err := toolInstance.Run(toolCtx, input)
		if err != nil {
			t.Fatalf("Run failed: %v", err)
		}
		if diff := cmp.Diff(input, got); diff != "" {
			t.Errorf("Output mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("RunValidationFailure_Type", func(t *testing.T) {
		invCtx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{})
		toolCtx := toolinternal.NewToolContext(invCtx, "", nil)

		input := map[string]any{"count": "not a number"}
		_, err := toolInstance.Run(toolCtx, input)
		if err == nil {
			t.Error("Expected validation error for invalid type, got nil")
		}
	})

	t.Run("RunValidationFailure_MissingRequired", func(t *testing.T) {
		invCtx := icontext.NewInvocationContext(context.Background(), icontext.InvocationContextParams{})
		toolCtx := toolinternal.NewToolContext(invCtx, "", nil)

		input := map[string]any{"other": 123}
		_, err := toolInstance.Run(toolCtx, input)
		if err == nil {
			t.Error("Expected validation error for missing required field, got nil")
		}
	})
}
