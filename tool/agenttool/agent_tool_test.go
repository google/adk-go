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

package agenttool_test

import (
	"log"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/agenttool"
)

func TestAgentTool_Declaration(t *testing.T) {
	inputSchema := &genai.Schema{
		Type: "OBJECT",
		Properties: map[string]*genai.Schema{
			"request": {Type: "STRING"},
		},
		Required: []string{"request"},
	}
	agent := createAgent(t, inputSchema, nil)
	agentTool := agenttool.New(agent, nil)
	toolImpl, ok := agentTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	decl := toolImpl.Declaration()

	wantDecl := &genai.FunctionDeclaration{
		Name:        "math_agent",
		Description: "Solves math problems.",
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"request": {Type: "STRING"},
			},
			Required: []string{"request"},
		},
	}
	if diff := cmp.Diff(wantDecl, decl); diff != "" {
		t.Errorf("Declaration() returned diff (-want +got):\n%s", diff)
	}
}

func TestAgentTool_DeclarationWithoutSchema(t *testing.T) {
	agent := createAgent(t, nil, nil)
	agentTool := agenttool.New(agent, nil)
	toolImpl, ok := agentTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	decl := toolImpl.Declaration()

	wantDecl := &genai.FunctionDeclaration{
		Name:        "math_agent",
		Description: "Solves math problems.",
		Parameters: &genai.Schema{
			Type: "OBJECT",
			Properties: map[string]*genai.Schema{
				"request": {Type: "STRING"},
			},
			Required: []string{"request"},
		},
	}
	if diff := cmp.Diff(wantDecl, decl); diff != "" {
		t.Errorf("Declaration() returned diff (-want +got):\n%s", diff)
	}
}

func TestAgentTool_Run_InputValidation(t *testing.T) {
	inputSchema := &genai.Schema{
		Type: "OBJECT",
		Properties: map[string]*genai.Schema{
			"is_magic": {Type: "BOOLEAN"},
			"name":     {Type: "STRING"},
		},
		Required: []string{"is_magic", "name"},
	}
	agent := createAgent(t, inputSchema, nil)
	agentTool := agenttool.New(agent, nil)
	toolCtx := createToolContext(t, agent)

	tests := []struct {
		name string
		args map[string]any
	}{
		{
			name: "extra_field",
			args: map[string]any{"is_magic": true, "name_invalid": "test_name", "name": "test"},
		},
		{
			name: "invalid_type",
			args: map[string]any{"is_magic": "invalid_type", "name": "test_name"},
		},
		{
			name: "missing_required",
			args: map[string]any{"is_magic": true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			toolImpl, ok := agentTool.(toolinternal.FunctionTool)
			if !ok {
				t.Fatal("agentTool does not implement FunctionTool")
			}

			_, err := toolImpl.Run(toolCtx, tt.args)
			if err == nil {
				t.Fatalf("Run(%v) succeeded unexpectedly, wanted error", tt.args)
			}
		})
	}
}

func TestAgentTool_Run_OutputValidation(t *testing.T) {
	outputSchema := &genai.Schema{
		Type: "OBJECT",
		Properties: map[string]*genai.Schema{
			"is_valid": {Type: "BOOLEAN"},
			"message":  {Type: "STRING"},
		},
		Required: []string{"is_valid", "message"},
	}

	testLLM := &testutil.MockModel{
		Responses: []*genai.Content{
			genai.NewContentFromText("{\"is_valid\": \"invalid type\", \"message\": \"success\"}", genai.RoleModel),
		},
	}

	agent := createAgentWithModel(t, nil, outputSchema, testLLM)
	agentTool := agenttool.New(agent, nil)
	toolCtx := createToolContext(t, agent)
	toolImpl, ok := agentTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	_, err := toolImpl.Run(toolCtx, map[string]any{"request": "test"})
	if err == nil {
		t.Fatalf("Run() succeeded unexpectedly, want error")
	}
}

func TestAgentTool_Run_Successful(t *testing.T) {
	inputSchema := &genai.Schema{
		Type: "OBJECT",
		Properties: map[string]*genai.Schema{
			"is_magic": {Type: "BOOLEAN"},
		},
		Required: []string{"is_magic"},
	}
	outputSchema := &genai.Schema{
		Type: "OBJECT",
		Properties: map[string]*genai.Schema{
			"is_valid": {Type: "BOOLEAN"},
			"message":  {Type: "STRING"},
		},
		Required: []string{"is_valid", "message"},
	}
	testLLM := &testutil.MockModel{
		Responses: []*genai.Content{
			genai.NewContentFromText("{\"is_valid\": true, \"message\": \"success\"}", genai.RoleModel),
		},
	}
	agent := createAgentWithModel(t, inputSchema, outputSchema, testLLM)
	agentTool := agenttool.New(agent, nil)
	toolCtx := createToolContext(t, agent)
	toolImpl, ok := agentTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	result, err := toolImpl.Run(toolCtx, map[string]any{"is_magic": true})
	if err != nil {
		t.Fatalf("Run() failed unexpectedly: %v", err)
	}
	want := map[string]any{"is_valid": true, "message": "success"}
	if diff := cmp.Diff(want, result); diff != "" {
		t.Errorf("Run() result diff (-want +got):\n%s", diff)
	}
}

func TestAgentTool_Run_WithoutSchema(t *testing.T) {
	testLLM := &testutil.MockModel{
		Responses: []*genai.Content{
			{
				Parts: []*genai.Part{
					{Text: "First text part is returned"},
					{Text: " This should not be ignored"},
				},
				Role: genai.RoleModel,
			},
		},
		StreamResponsesCount: 1,
	}

	agent := createAgentWithModel(t, nil, nil, testLLM)
	agentTool := agenttool.New(agent, nil)
	toolCtx := createToolContext(t, agent)
	toolImpl, ok := agentTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	result, err := toolImpl.Run(toolCtx, map[string]any{"request": "magic"})
	if err != nil {
		t.Fatalf("Run() failed unexpectedly: %v", err)
	}
	want := map[string]any{"result": "First text part is returned This should not be ignored"}
	if diff := cmp.Diff(want, result); diff != "" {
		t.Errorf("Run() result diff (-want +got):\n%s", diff)
	}
}

func TestAgentTool_Run_EmptyModelResponse(t *testing.T) {
	testLLM := &testutil.MockModel{
		Responses: []*genai.Content{
			{Role: genai.RoleModel}, // Empty content
		},
	}
	agent := createAgentWithModel(t, nil, nil, testLLM)
	agentTool := agenttool.New(agent, nil)
	toolCtx := createToolContext(t, agent)
	toolImpl, ok := agentTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	result, err := toolImpl.Run(toolCtx, map[string]any{"request": "magic"})
	if err != nil {
		t.Fatalf("Run() failed unexpectedly: %v", err)
	}
	want := map[string]any{}
	if diff := cmp.Diff(want, result); diff != "" {
		t.Errorf("Run() result diff (-want +got):\n%s", diff)
	}
}

func TestAgentTool_Run_SkipSummarization(t *testing.T) {
	tests := []struct {
		name              string
		skipSummarization bool
	}{
		{name: "skip_summarization_true", skipSummarization: true},
		{name: "skip_summarization_false", skipSummarization: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testLLM := &testutil.MockModel{
				Responses: []*genai.Content{
					genai.NewContentFromText("test response", genai.RoleModel),
				},
			}
			ag := createAgentWithModel(t, nil, nil, testLLM)
			toolCtx := createToolContext(t, ag)

			agentTool := agenttool.New(ag, &agenttool.Config{SkipSummarization: tt.skipSummarization})
			toolImpl, ok := agentTool.(toolinternal.FunctionTool)
			if !ok {
				t.Fatal("agentTool does not implement FunctionTool")
			}

			_, err := toolImpl.Run(toolCtx, map[string]any{"request": "magic"})
			if err != nil {
				t.Fatalf("Run() failed unexpectedly: %v", err)
			}

			// SkipSummarization must never be propagated to the parent tool context.
			// Doing so causes IsFinalResponse() to return true on the function response
			// event, which terminates the parent agent loop before it can process the
			// sub-agent result.
			if toolCtx.Actions().SkipSummarization {
				t.Errorf("SkipSummarization was propagated to parent tool context (skipSummarization=%v), want false", tt.skipSummarization)
			}
		})
	}
}

func createAgent(t *testing.T, inputSchema, outputSchema *genai.Schema) agent.Agent {
	t.Helper()

	model, err := gemini.NewModel(t.Context(), "gemini-2.5-flash", &genai.ClientConfig{
		APIKey: "FAKE_KEY",
	})
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}
	agent, err := llmagent.New(llmagent.Config{
		Name:         "math_agent",
		Model:        model,
		Description:  "Solves math problems.",
		Instruction:  "You solve math problems.",
		InputSchema:  inputSchema,
		OutputSchema: outputSchema,
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}
	return agent
}

func createAgentWithModel(t *testing.T, inputSchema, outputSchema *genai.Schema, llmModel model.LLM) agent.Agent {
	t.Helper()
	agent, err := llmagent.New(llmagent.Config{
		Name:         "math_agent",
		Model:        llmModel,
		Description:  "Solves math problems.",
		Instruction:  "You solve math problems.",
		InputSchema:  inputSchema,
		OutputSchema: outputSchema,
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}
	return agent
}

func createToolContext(t *testing.T, testAgent agent.Agent) tool.Context {
	t.Helper()

	sessionService := session.InMemoryService()
	createResponse, err := sessionService.Create(t.Context(), &session.CreateRequest{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	})
	if err != nil {
		t.Fatalf("Failed to create session: %v", err)
	}

	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Session: createResponse.Session,
	})

	return toolinternal.NewToolContext(ctx, "", &session.EventActions{}, nil)
}
