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
	"context"
	"errors"
	"fmt"
	"iter"
	"log"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
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

// TestAgentTool_Run_CompositeAgentInputSchema checks that a wrapped workflow
// agent whose first sub-agent has an InputSchema takes the arguments its
// declaration advertises, as adk-python's AgentTool does with
// _get_input_schema.
func TestAgentTool_Run_CompositeAgentInputSchema(t *testing.T) {
	inputSchema := &genai.Schema{
		Type: "OBJECT",
		Properties: map[string]*genai.Schema{
			"is_magic": {Type: "BOOLEAN"},
		},
		Required: []string{"is_magic"},
	}
	tests := []struct {
		name    string
		args    map[string]any
		want    map[string]any
		wantErr bool
	}{
		{
			name: "valid_arguments",
			args: map[string]any{"is_magic": true},
			want: map[string]any{"result": "magic"},
		},
		{
			name:    "invalid_arguments",
			args:    map[string]any{"is_magic": "yes"},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testLLM := &testutil.MockModel{
				Responses: []*genai.Content{genai.NewContentFromText("magic", genai.RoleModel)},
			}
			pipeline, err := sequentialagent.New(sequentialagent.Config{
				AgentConfig: agent.Config{
					Name:        "pipeline",
					Description: "Runs the math agent.",
					SubAgents:   []agent.Agent{createAgentWithModel(t, inputSchema, nil, testLLM)},
				},
			})
			if err != nil {
				t.Fatalf("sequentialagent.New() failed: %v", err)
			}
			toolImpl, ok := agenttool.New(pipeline, nil).(toolinternal.FunctionTool)
			if !ok {
				t.Fatal("agentTool does not implement FunctionTool")
			}
			if diff := cmp.Diff(inputSchema, toolImpl.Declaration().Parameters); diff != "" {
				t.Fatalf("Declaration().Parameters diff (-want +got):\n%s", diff)
			}

			got, err := toolImpl.Run(createToolContext(t, pipeline), tt.args)
			if tt.wantErr {
				// The schema rejects the arguments, rather than the missing
				// "request" the declaration never asked for.
				if err == nil || !strings.Contains(err.Error(), "argument validation failed") {
					t.Fatalf("Run(%v) error = %v, want an argument validation error", tt.args, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Run(%v) failed unexpectedly: %v", tt.args, err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Run() result diff (-want +got):\n%s", diff)
			}
		})
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

// Without an input schema, the agent's request is the "request" argument, or,
// without one, the arguments as adk-python's json.dumps(args,
// ensure_ascii=False, sort_keys=True) writes them.
func TestAgentTool_Run_WithoutSchemaRequestText(t *testing.T) {
	tests := []struct {
		name string
		args map[string]any
		want string
	}{
		{name: "request_argument_wins", args: map[string]any{"request": "find shoes", "brand": "Nike"}, want: "find shoes"},
		{name: "empty", args: map[string]any{}, want: "{}"},
		{
			name: "keys_sorted_non_ascii_kept",
			args: map[string]any{"product": "Laufschuhe für Damen", "brand": "Nike & Co"},
			want: `{"brand": "Nike & Co", "product": "Laufschuhe für Damen"}`,
		},
		{
			name: "nested",
			args: map[string]any{"filter": map[string]any{"size": 38.0, "colors": []any{"rot", "grün"}}},
			want: `{"filter": {"colors": ["rot", "grün"], "size": 38}}`,
		},
		{
			name: "string_with_quote_comma_colon",
			args: map[string]any{"note": `say "hi, then: go`},
			want: `{"note": "say \"hi, then: go"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := &requestTextModel{}
			ag := createAgentWithModel(t, nil, nil, recorder)
			toolImpl, ok := agenttool.New(ag, nil).(toolinternal.FunctionTool)
			if !ok {
				t.Fatal("agentTool does not implement FunctionTool")
			}

			if _, err := toolImpl.Run(createToolContext(t, ag), tt.args); err != nil {
				t.Fatalf("Run(%v) failed unexpectedly: %v", tt.args, err)
			}

			if diff := cmp.Diff([]string{tt.want}, recorder.texts); diff != "" {
				t.Errorf("request texts the agent's model received (-want +got):\n%s", diff)
			}
		})
	}
}

// requestTextModel answers every request with "done" and records the text of
// the request's last content.
type requestTextModel struct {
	model.LLM
	texts []string
}

func (m *requestTextModel) Name() string { return "request-text-mock" }

func (m *requestTextModel) GenerateContent(_ context.Context, req *model.LLMRequest, _ bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		var text string
		if n := len(req.Contents); n > 0 && len(req.Contents[n-1].Parts) > 0 {
			text = req.Contents[n-1].Parts[0].Text
		}
		m.texts = append(m.texts, text)
		yield(&model.LLMResponse{Content: genai.NewContentFromText("done", genai.RoleModel)}, nil)
	}
}

func TestAgentTool_Run_CodeParts(t *testing.T) {
	tests := []struct {
		name  string
		parts []*genai.Part
		want  map[string]any
	}{
		{
			name: "code execution output",
			parts: []*genai.Part{
				{CodeExecutionResult: &genai.CodeExecutionResult{Output: "\n 42 \n\n"}},
			},
			want: map[string]any{"result": "\n 42 "},
		},
		{
			name: "executable code",
			parts: []*genai.Part{
				{ExecutableCode: &genai.ExecutableCode{Code: "print(42)\n"}},
			},
			want: map[string]any{"result": "print(42)\n"},
		},
		{
			name: "mixed content",
			parts: []*genai.Part{
				{Text: "Calculating."},
				{ExecutableCode: &genai.ExecutableCode{Code: "print(42)"}},
				{CodeExecutionResult: &genai.CodeExecutionResult{Output: "42\n"}},
				{Text: "Done."},
			},
			want: map[string]any{"result": "Calculating.\nprint(42)\n42\nDone."},
		},
		{
			name: "empty and thought parts",
			parts: []*genai.Part{
				nil,
				{},
				{CodeExecutionResult: &genai.CodeExecutionResult{}},
				{CodeExecutionResult: &genai.CodeExecutionResult{Output: "\n\n"}},
				{ExecutableCode: &genai.ExecutableCode{}},
				{Thought: true, CodeExecutionResult: &genai.CodeExecutionResult{Output: "hidden"}},
				{Thought: true, ExecutableCode: &genai.ExecutableCode{Code: "hidden"}},
			},
			want: map[string]any{"result": ""},
		},
		{
			name: "text and code precedence",
			parts: []*genai.Part{
				{Text: "text", CodeExecutionResult: &genai.CodeExecutionResult{Output: "unused"}, ExecutableCode: &genai.ExecutableCode{Code: "unused"}},
				{CodeExecutionResult: &genai.CodeExecutionResult{Output: "output"}, ExecutableCode: &genai.ExecutableCode{Code: "unused"}},
				{CodeExecutionResult: &genai.CodeExecutionResult{}, ExecutableCode: &genai.ExecutableCode{Code: "code"}},
			},
			want: map[string]any{"result": "text\noutput\ncode"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			child, err := agent.New(agent.Config{
				Name: "calculator",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						event := session.NewEvent(ctx, ctx.InvocationID())
						event.Author = "calculator"
						event.Content = &genai.Content{Role: genai.RoleModel, Parts: tt.parts}
						yield(event, nil)
					}
				},
			})
			if err != nil {
				t.Fatalf("agent.New failed: %v", err)
			}
			wrapped := agenttool.New(child, nil).(toolinternal.FunctionTool)
			got, err := wrapped.Run(createToolContext(t, child), map[string]any{"request": "calculate"})
			if err != nil {
				t.Fatalf("Run failed: %v", err)
			}
			if !cmp.Equal(got, tt.want) {
				t.Error("Run returned an unexpected tool result")
			}
		})
	}
}

func TestAgentTool_Run_FiltersThoughtParts(t *testing.T) {
	testLLM := &testutil.MockModel{
		Responses: []*genai.Content{
			{
				Parts: []*genai.Part{
					{Text: "Let me think about this...", Thought: true},
					{Text: "", Thought: true}, // edge case: empty text + thought
					{Text: "The answer is 42"},
					{Text: "Still reasoning...", Thought: true},
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

	result, err := toolImpl.Run(toolCtx, map[string]any{"request": "what is the answer?"})
	if err != nil {
		t.Fatalf("Run() failed unexpectedly: %v", err)
	}
	want := map[string]any{"result": "The answer is 42"}
	if diff := cmp.Diff(want, result); diff != "" {
		t.Errorf("Run() result diff (-want +got):\n%s", diff)
	}
}

// A sub-agent turn that finishes with STOP holding only thought parts is an
// error event with code MODEL_RETURNED_NO_CONTENT, and AgentTool reports a
// sub-agent's error event as an error, with or without an output schema.
// adk-python's AgentTool returns the event's error message as the tool result
// instead (tools/agent_tool.py).
func TestAgentTool_Run_AllThoughtPartsStopTurnIsAnError(t *testing.T) {
	outputSchema := &genai.Schema{
		Type:       "OBJECT",
		Properties: map[string]*genai.Schema{"is_valid": {Type: "BOOLEAN"}},
		Required:   []string{"is_valid"},
	}
	for _, tc := range []struct {
		name         string
		outputSchema *genai.Schema
	}{
		{name: "no_output_schema"},
		{name: "output_schema", outputSchema: outputSchema},
	} {
		t.Run(tc.name, func(t *testing.T) {
			testLLM := &testutil.MockModel{
				Responses: []*genai.Content{
					{
						Parts: []*genai.Part{
							{Text: "Just thinking...", Thought: true},
						},
						Role: genai.RoleModel,
					},
				},
				StreamResponsesCount: 1,
			}

			agent := createAgentWithModel(t, nil, tc.outputSchema, testLLM)
			agentTool := agenttool.New(agent, nil)
			toolCtx := createToolContext(t, agent)
			toolImpl, ok := agentTool.(toolinternal.FunctionTool)
			if !ok {
				t.Fatal("agentTool does not implement FunctionTool")
			}

			result, err := toolImpl.Run(toolCtx, map[string]any{"request": "think only"})
			if err == nil || !strings.Contains(err.Error(), `code: "MODEL_RETURNED_NO_CONTENT"`) {
				t.Fatalf("Run() = %v, %v; want an error with code MODEL_RETURNED_NO_CONTENT", result, err)
			}
		})
	}
}

func TestAgentTool_Run_FiltersThoughtParts_WithOutputSchema(t *testing.T) {
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
			{
				Parts: []*genai.Part{
					{Text: "Let me validate the input carefully...", Thought: true},
					{Text: "{\"is_valid\": true, \"message\": \"success\"}"},
				},
				Role: genai.RoleModel,
			},
		},
		StreamResponsesCount: 1,
	}

	agent := createAgentWithModel(t, nil, outputSchema, testLLM)
	agentTool := agenttool.New(agent, nil)
	toolCtx := createToolContext(t, agent)
	toolImpl, ok := agentTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	result, err := toolImpl.Run(toolCtx, map[string]any{"request": "validate"})
	if err != nil {
		t.Fatalf("Run() failed unexpectedly: %v", err)
	}
	want := map[string]any{"is_valid": true, "message": "success"}
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
	want := map[string]any{"result": ""}
	if diff := cmp.Diff(want, result); diff != "" {
		t.Errorf("Run() result diff (-want +got):\n%s", diff)
	}
}

func TestAgentTool_Run_NoContentReturnsEmptyResult(t *testing.T) {
	child, err := agent.New(agent.Config{
		Name: "silent",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := session.NewEvent(ctx, ctx.InvocationID())
				event.Author = "silent"
				yield(event, nil)
			}
		},
	})
	if err != nil {
		t.Fatalf("agent.New failed: %v", err)
	}
	wrapped := agenttool.New(child, nil).(toolinternal.FunctionTool)

	result, err := wrapped.Run(createToolContext(t, child), map[string]any{"request": "anything"})
	if err != nil {
		t.Fatalf("Run() failed unexpectedly: %v", err)
	}
	want := map[string]any{"result": ""}
	if diff := cmp.Diff(want, result); diff != "" {
		t.Errorf("Run() result diff (-want +got):\n%s", diff)
	}
}

func TestAgentTool_Run_SkipSummarization(t *testing.T) {
	inputSchema := &genai.Schema{
		Type:       "OBJECT",
		Properties: map[string]*genai.Schema{"city": {Type: "STRING"}},
		Required:   []string{"city"},
	}
	tests := []struct {
		name              string
		skipSummarization bool
		inputSchema       *genai.Schema
		args              map[string]any
	}{
		{name: "skip_summarization_true", skipSummarization: true, args: map[string]any{"request": "magic"}},
		{name: "skip_summarization_false", skipSummarization: false, args: map[string]any{"request": "magic"}},
		{name: "input_schema_skip_summarization_true", skipSummarization: true, inputSchema: inputSchema, args: map[string]any{"city": "Paris"}},
		{name: "input_schema_skip_summarization_false", skipSummarization: false, inputSchema: inputSchema, args: map[string]any{"city": "Paris"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testLLM := &testutil.MockModel{
				Responses: []*genai.Content{
					genai.NewContentFromText("test response", genai.RoleModel),
				},
			}
			ag := createAgentWithModel(t, tt.inputSchema, nil, testLLM)
			toolCtx := createToolContext(t, ag)

			agentTool := agenttool.New(ag, &agenttool.Config{SkipSummarization: tt.skipSummarization})
			toolImpl, ok := agentTool.(toolinternal.FunctionTool)
			if !ok {
				t.Fatal("agentTool does not implement FunctionTool")
			}

			_, err := toolImpl.Run(toolCtx, tt.args)
			if err != nil {
				t.Fatalf("Run() failed unexpectedly: %v", err)
			}

			if got := toolCtx.Actions().SkipSummarization; got != tt.skipSummarization {
				t.Errorf("SkipSummarization propagated to parent tool context = %v, want %v", got, tt.skipSummarization)
			}
		})
	}
}

type transferArgs struct {
	Amount int `json:"amount"`
}

func TestAgentTool_Run_PendingConfirmationReturnsError(t *testing.T) {
	ran := false
	transfer, err := functiontool.New(functiontool.Config{
		Name:                "transfer",
		Description:         "Transfers money.",
		RequireConfirmation: true,
	}, func(agent.Context, transferArgs) (map[string]any, error) {
		ran = true
		return map[string]any{"status": "done"}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() failed: %v", err)
	}
	testLLM := &testutil.MockModel{
		Responses: []*genai.Content{
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
					ID:   "call-1",
					Name: "transfer",
					Args: map[string]any{"amount": 100},
				}}},
			},
		},
	}
	child, err := llmagent.New(llmagent.Config{
		Name:        "payments_agent",
		Description: "Moves money.",
		Model:       testLLM,
		Tools:       []tool.Tool{transfer},
	})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}
	toolImpl, ok := agenttool.New(child, nil).(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	toolCtx := createToolContext(t, child)
	// Something other than the tool's Config, e.g. a BeforeToolCallback, set
	// the flag on this call.
	toolCtx.Actions().SkipSummarization = true

	result, err := toolImpl.Run(toolCtx, map[string]any{"request": "send 100"})

	if toolCtx.Actions().SkipSummarization {
		t.Error("SkipSummarization still set; the parent model would never see the error")
	}
	if ran {
		t.Fatal("transfer ran without confirmation")
	}
	if err == nil {
		t.Fatalf("Run() = %v, nil; want an error for the unresolved confirmation", result)
	}
	if !strings.Contains(err.Error(), "unresolved long-running tool calls") {
		t.Errorf("Run() error = %q, want it to report the unresolved call", err)
	}
}

func TestAgentTool_Run_AnsweredLongRunningCallSucceeds(t *testing.T) {
	start, err := functiontool.New(functiontool.Config{
		Name:          "start_job",
		Description:   "Starts a job.",
		IsLongRunning: true,
	}, func(agent.Context, transferArgs) (map[string]any, error) {
		return map[string]any{"status": "started"}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() failed: %v", err)
	}
	testLLM := &testutil.MockModel{
		Responses: []*genai.Content{
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
					ID:   "call-1",
					Name: "start_job",
					Args: map[string]any{"amount": 1},
				}}},
			},
			genai.NewContentFromText("job started", genai.RoleModel),
		},
	}
	child, err := llmagent.New(llmagent.Config{
		Name:        "jobs_agent",
		Description: "Starts jobs.",
		Model:       testLLM,
		Tools:       []tool.Tool{start},
	})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}
	toolImpl, ok := agenttool.New(child, nil).(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	result, err := toolImpl.Run(createToolContext(t, child), map[string]any{"request": "start"})
	if err != nil {
		t.Fatalf("Run() failed unexpectedly: %v", err)
	}
	if got := result["result"]; got != "job started" {
		t.Errorf("Run() result = %v, want job started", result)
	}
}

// With SkipSummarization, the pending-call error must still go back to the
// parent model instead of ending the parent's turn on an empty response.
func TestAgentTool_Run_SkipSummarizationPendingErrorReachesParent(t *testing.T) {
	transfer, err := functiontool.New(functiontool.Config{
		Name:                "transfer",
		Description:         "Transfers money.",
		RequireConfirmation: true,
	}, func(agent.Context, transferArgs) (map[string]any, error) {
		return map[string]any{"status": "done"}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() failed: %v", err)
	}
	child, err := llmagent.New(llmagent.Config{
		Name:        "payments_agent",
		Description: "Moves money.",
		Model: &testutil.MockModel{
			Responses: []*genai.Content{
				{
					Role: genai.RoleModel,
					Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
						ID:   "call-1",
						Name: "transfer",
						Args: map[string]any{"amount": 1},
					}}},
				},
			},
		},
		Tools: []tool.Tool{transfer},
	})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}
	parentLLM := &testutil.MockModel{
		Responses: []*genai.Content{
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
					ID:   "p-1",
					Name: "payments_agent",
					Args: map[string]any{"request": "send"},
				}}},
			},
			genai.NewContentFromText("handled the error", genai.RoleModel),
		},
	}
	parent, err := llmagent.New(llmagent.Config{
		Name:  "root",
		Model: parentLLM,
		Tools: []tool.Tool{agenttool.New(child, &agenttool.Config{SkipSummarization: true})},
	})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}
	svc := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: parent, SessionService: svc})
	if err != nil {
		t.Fatalf("runner.New() failed: %v", err)
	}
	s, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u"})
	if err != nil {
		t.Fatalf("Create() failed: %v", err)
	}

	for _, err := range r.Run(t.Context(), "u", s.Session.ID(), genai.NewContentFromText("pay", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() failed: %v", err)
		}
	}

	if got := len(parentLLM.Requests); got != 2 {
		t.Fatalf("parent model requests = %d, want 2 (the second carrying the agent tool's error)", got)
	}
	if !requestHasFunctionResponseError(parentLLM.Requests[1], "unresolved long-running tool calls") {
		t.Error("second parent model request does not carry the agent tool's pending-call error")
	}
}

func requestHasFunctionResponseError(req *model.LLMRequest, want string) bool {
	for _, c := range req.Contents {
		if c == nil {
			continue
		}
		for _, p := range c.Parts {
			if p == nil || p.FunctionResponse == nil {
				continue
			}
			if msg, ok := p.FunctionResponse.Response["error"].(string); ok && strings.Contains(msg, want) {
				return true
			}
		}
	}
	return false
}

// A long-running call whose arguments are streamed without an ID gets a
// different generated ID on each partial chunk than on the final event. Only
// the final event's ID is answered, so partial chunks must not count.
func TestAgentTool_Run_StreamedArgsAnsweredLongRunningCallSucceeds(t *testing.T) {
	start, err := functiontool.New(functiontool.Config{
		Name:          "start_job",
		Description:   "Starts a job.",
		IsLongRunning: true,
	}, func(agent.Context, transferArgs) (map[string]any, error) {
		return map[string]any{"status": "started"}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() failed: %v", err)
	}
	testLLM := &testutil.MockModel{
		StreamResponsesCount: 2,
		Responses: []*genai.Content{
			{
				Role:  genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "start_job", WillContinue: new(true)}}},
			},
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
					PartialArgs:  []*genai.PartialArg{{JsonPath: "$.amount", NumberValue: new(1.0)}},
					WillContinue: new(false),
				}}},
			},
			genai.NewContentFromText("job started", genai.RoleModel),
		},
	}
	child, err := llmagent.New(llmagent.Config{
		Name:        "jobs_agent",
		Description: "Starts jobs.",
		Model:       testLLM,
		Tools:       []tool.Tool{start},
	})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}
	toolImpl, ok := agenttool.New(child, nil).(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	result, err := toolImpl.Run(createToolContext(t, child), map[string]any{"request": "start"})
	if err != nil {
		t.Fatalf("Run() failed unexpectedly: %v", err)
	}
	if got := result["result"]; got != "job started" {
		t.Errorf("Run() result = %v, want job started", result)
	}
}

// Skipping partial chunks must not drop the final event's ID when a streamed
// long-running call is never answered.
func TestAgentTool_Run_StreamedArgsUnansweredLongRunningCallReturnsError(t *testing.T) {
	start, err := functiontool.New(functiontool.Config{
		Name:          "start_job",
		Description:   "Starts a job.",
		IsLongRunning: true,
	}, func(agent.Context, transferArgs) (map[string]any, error) {
		return nil, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() failed: %v", err)
	}
	testLLM := &testutil.MockModel{
		StreamResponsesCount: 2,
		Responses: []*genai.Content{
			{
				Role:  genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "start_job", WillContinue: new(true)}}},
			},
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
					PartialArgs:  []*genai.PartialArg{{JsonPath: "$.amount", NumberValue: new(1.0)}},
					WillContinue: new(false),
				}}},
			},
		},
	}
	child, err := llmagent.New(llmagent.Config{
		Name:        "jobs_agent",
		Description: "Starts jobs.",
		Model:       testLLM,
		Tools:       []tool.Tool{start},
	})
	if err != nil {
		t.Fatalf("llmagent.New() failed: %v", err)
	}
	toolImpl, ok := agenttool.New(child, nil).(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	result, err := toolImpl.Run(createToolContext(t, child), map[string]any{"request": "start"})

	if err == nil {
		t.Fatalf("Run() = %v, nil; want an error for the unanswered call", result)
	}
	if !strings.Contains(err.Error(), "unresolved long-running tool calls") {
		t.Errorf("Run() error = %q, want it to report the unresolved call", err)
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

func createToolContext(t *testing.T, testAgent agent.Agent) agent.Context {
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

	return agent.NewToolContext(ctx, "", &session.EventActions{}, nil)
}

// With SkipSummarization, invalid arguments must not end the parent turn: the
// error goes back to the parent model so it can retry with corrected args.
func TestAgentTool_Run_SkipSummarizationNotSetOnInvalidArgs(t *testing.T) {
	inputSchema := &genai.Schema{
		Type: "OBJECT",
		Properties: map[string]*genai.Schema{
			"city": {Type: "STRING"},
			"days": {Type: "INTEGER"},
		},
		Required: []string{"city", "days"},
	}
	tests := []struct {
		name          string
		inputSchema   *genai.Schema
		args          any
		wantErrPrefix string
	}{
		{
			name:          "input_schema_missing_required",
			inputSchema:   inputSchema,
			args:          map[string]any{"city": "Paris"},
			wantErrPrefix: "argument validation failed for agent math_agent: ",
		},
		{
			name:          "no_schema_args_not_an_object",
			args:          "Paris",
			wantErrPrefix: "agentTool expects map[string]any arguments, got string",
		},
		{
			name:          "no_schema_args_not_json",
			args:          map[string]any{"callback": func() {}},
			wantErrPrefix: "error serializing tool arguments for agent math_agent: ",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ag := createAgent(t, tt.inputSchema, nil)
			toolCtx := createToolContext(t, ag)
			toolImpl, ok := agenttool.New(ag, &agenttool.Config{SkipSummarization: true}).(toolinternal.FunctionTool)
			if !ok {
				t.Fatal("agentTool does not implement FunctionTool")
			}

			_, err := toolImpl.Run(toolCtx, tt.args)
			if err == nil {
				t.Fatalf("Run(%v) succeeded unexpectedly, wanted error", tt.args)
			}
			if !strings.HasPrefix(err.Error(), tt.wantErrPrefix) {
				t.Errorf("Run(%v) error = %q, want prefix %q", tt.args, err, tt.wantErrPrefix)
			}

			if toolCtx.Actions().SkipSummarization {
				t.Error("SkipSummarization = true after invalid arguments, want false so the parent model can retry")
			}
		})
	}
}

var errToolsetClosed = errors.New("toolset is closed")

// closeCountingToolset counts its Close calls. Like a closed mcptoolset, it
// fails Tools once it is closed.
type closeCountingToolset struct {
	closes atomic.Int32
}

func (c *closeCountingToolset) Name() string { return "close_counting_toolset" }

func (c *closeCountingToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	if c.closes.Load() > 0 {
		return nil, errToolsetClosed
	}
	return nil, nil
}

func (c *closeCountingToolset) Close() error {
	c.closes.Add(1)
	return nil
}

// An agent tool runs the same child agent, and so the same toolsets, on every
// call. A toolset such as mcptoolset cannot reopen once closed, so Run leaves
// the child's toolsets open; the runner that owns the agent tool closes them.
func TestAgentTool_RunTwiceLeavesChildToolsetOpen(t *testing.T) {
	ts := &closeCountingToolset{}
	testLLM := &testutil.MockModel{
		Responses: []*genai.Content{
			genai.NewContentFromText("first answer", genai.RoleModel),
			genai.NewContentFromText("second answer", genai.RoleModel),
		},
	}
	child, err := llmagent.New(llmagent.Config{
		Name:     "child_agent",
		Model:    testLLM,
		Toolsets: []tool.Toolset{ts},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	toolImpl, ok := agenttool.New(child, nil).(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	first, err := toolImpl.Run(createToolContext(t, child), map[string]any{"request": "one"})
	if err != nil {
		t.Fatalf("first Run() error = %v", err)
	}
	if got := ts.closes.Load(); got != 0 {
		t.Errorf("child toolset closes after first Run() = %d, want 0", got)
	}
	second, err := toolImpl.Run(createToolContext(t, child), map[string]any{"request": "two"})
	if err != nil {
		t.Fatalf("second Run() error = %v", err)
	}
	if got := ts.closes.Load(); got != 0 {
		t.Errorf("child toolset closes after second Run() = %d, want 0", got)
	}

	if diff := cmp.Diff(map[string]any{"result": "first answer"}, first); diff != "" {
		t.Errorf("first Run() result diff (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]any{"result": "second answer"}, second); diff != "" {
		t.Errorf("second Run() result diff (-want +got):\n%s", diff)
	}
}

// TestAgentTool_ForwardsAbortSignalToNestedRunner checks that the agent tool
// gives its nested run the caller's abort signal and returns the abort message
// as its result when the nested run is sealed, as adk-python's AgentTool does
// (tools/agent_tool.py, run_async). The tool contexts carry an abort state but
// no watcher, so they are never canceled: only the forwarded signal can stop
// the nested run.
func TestAgentTool_ForwardsAbortSignalToNestedRunner(t *testing.T) {
	tests := []struct {
		name string
		// signal returns the caller's abort channel and abort, which the
		// child calls after its first event.
		signal func() (sig <-chan struct{}, abort func())
		// child is what the wrapped agent yields before it waits.
		child    func(ctx agent.InvocationContext) *session.Event
		wantRuns int32
	}{
		{
			name: "fired_before_run",
			signal: func() (<-chan struct{}, func()) {
				ch := make(chan struct{})
				close(ch)
				return ch, func() {}
			},
			child: func(ctx agent.InvocationContext) *session.Event {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Content = genai.NewContentFromText("not aborted", genai.RoleModel)
				return ev
			},
			wantRuns: 0,
		},
		{
			name: "fired_during_nested_run",
			signal: func() (<-chan struct{}, func()) {
				ch := make(chan struct{})
				return ch, func() { close(ch) }
			},
			child: func(ctx agent.InvocationContext) *session.Event {
				ev := session.NewEvent(ctx, ctx.InvocationID())
				ev.Content = &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "approve"}},
				}}
				ev.LongRunningToolIDs = []string{"call_1"}
				return ev
			},
			wantRuns: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sig, abort := tt.signal()
			var runs atomic.Int32
			child, err := agent.New(agent.Config{Name: "child", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					runs.Add(1)
					ev := tt.child(ctx)
					ev.Author = "child"
					if !yield(ev, nil) {
						return
					}
					abort()
					select {
					case <-ctx.Done():
					case <-time.After(30 * time.Second):
					}
				}
			}})
			if err != nil {
				t.Fatalf("agent.New() error = %v", err)
			}
			toolImpl, ok := agenttool.New(child, nil).(toolinternal.FunctionTool)
			if !ok {
				t.Fatal("agentTool does not implement FunctionTool")
			}
			created, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{AppName: "testApp", UserID: "testUser"})
			if err != nil {
				t.Fatalf("Create() error = %v", err)
			}
			ctx := icontext.WithAbortState(t.Context(), icontext.NewAbortState(t.Context(), sig))
			ictx := icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{Session: created.Session})
			toolCtx := agent.NewToolContext(ictx, "", &session.EventActions{}, nil)

			result, err := toolImpl.Run(toolCtx, map[string]any{"request": "go"})
			if err != nil {
				t.Fatalf("Run() error = %v, want nil", err)
			}
			if diff := cmp.Diff(map[string]any{"result": "Invocation was aborted by client."}, result); diff != "" {
				t.Errorf("Run() result diff (-want +got):\n%s", diff)
			}
			if got := runs.Load(); got != tt.wantRuns {
				t.Errorf("child ran %d times, want %d", got, tt.wantRuns)
			}
		})
	}
}

// TestAgentTool_SentAbortValueStopsCallingAndNestedRun checks that one value
// sent on the calling run's abort signal while the nested run of an agent tool
// is running stops both runs: the nested agent's context is canceled, the
// nested run is sealed, so the tool's result is the abort message, and the
// calling run calls its model no more and is sealed. adk-python's nested run
// waits on the caller's asyncio.Event, which every reader sees once it is set
// (tools/agent_tool.py, run_async).
func TestAgentTool_SentAbortValueStopsCallingAndNestedRun(t *testing.T) {
	started := make(chan struct{})
	var childCanceled atomic.Bool
	child, err := agent.New(agent.Config{Name: "child", Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
		return func(func(*session.Event, error) bool) {
			close(started)
			select {
			case <-ctx.Done():
				childCanceled.Store(true)
			case <-time.After(5 * time.Second):
			}
		}
	}})
	if err != nil {
		t.Fatalf("agent.New() error = %v", err)
	}
	outerModel := &testutil.MockModel{Responses: []*genai.Content{
		{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "call_1", Name: "child", Args: map[string]any{"request": "go"}}}}},
		genai.NewContentFromText("outer reply", genai.RoleModel),
	}}
	// The calling run is sealed before the tool's response is yielded, so the
	// tool's result is read here.
	var toolResult map[string]any
	var toolErr error
	root, err := llmagent.New(llmagent.Config{
		Name:  "root",
		Model: outerModel,
		Tools: []tool.Tool{agenttool.New(child, nil)},
		AfterToolCallbacks: []llmagent.AfterToolCallback{
			func(_ agent.Context, _ tool.Tool, _, result map[string]any, err error) (map[string]any, error) {
				toolResult, toolErr = result, err
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	sessions := session.InMemoryService()
	r, err := runner.New(runner.Config{AppName: "app", Agent: root, SessionService: sessions, AutoCreateSession: true})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	sig := make(chan struct{}, 1)
	go func() {
		select {
		case <-started:
			sig <- struct{}{}
		case <-t.Context().Done():
		}
	}()

	var got []string
	for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}, runner.WithAbortSignal(sig)) {
		if err != nil {
			t.Fatalf("Run() error = %v, want none", err)
		}
		got = append(got, describeEvent(ev))
	}

	if !childCanceled.Load() {
		t.Error("the nested agent's context was not canceled within 5s of the signal")
	}
	if toolErr != nil {
		t.Errorf("agent tool error = %v, want nil", toolErr)
	}
	if diff := cmp.Diff(map[string]any{"result": "Invocation was aborted by client."}, toolResult); diff != "" {
		t.Errorf("agent tool result mismatch (-want +got):\n%s", diff)
	}
	if got := len(outerModel.Requests); got != 1 {
		t.Errorf("calling run's model was called %d times, want 1", got)
	}
	want := []string{
		"root call call_1 child",
		`root INVOCATION_ABORTED "Invocation was aborted by client." response call_1 child map[error:Invocation was aborted by client.]`,
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("calling run's events mismatch (-want +got):\n%s", diff)
	}
}

// describeEvent names ev by author, error code and message, function calls,
// function responses and text.
func describeEvent(ev *session.Event) string {
	s := ev.Author
	if ev.ErrorCode != "" || ev.ErrorMessage != "" {
		s += fmt.Sprintf(" %s %q", ev.ErrorCode, ev.ErrorMessage)
	}
	if ev.Content == nil {
		return s + " no-content"
	}
	for _, p := range ev.Content.Parts {
		switch {
		case p.FunctionCall != nil:
			s += fmt.Sprintf(" call %s %s", p.FunctionCall.ID, p.FunctionCall.Name)
		case p.FunctionResponse != nil:
			s += fmt.Sprintf(" response %s %s %v", p.FunctionResponse.ID, p.FunctionResponse.Name, p.FunctionResponse.Response)
		case p.Text != "":
			s += fmt.Sprintf(" text %q", p.Text)
		}
	}
	return s
}
