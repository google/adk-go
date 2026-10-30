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
	"iter"
	"log"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
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
			want: map[string]any{},
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

func TestAgentTool_Run_AllThoughtPartsReturnsEmpty(t *testing.T) {
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

	agent := createAgentWithModel(t, nil, nil, testLLM)
	agentTool := agenttool.New(agent, nil)
	toolCtx := createToolContext(t, agent)
	toolImpl, ok := agentTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentTool does not implement FunctionTool")
	}

	result, err := toolImpl.Run(toolCtx, map[string]any{"request": "think only"})
	if err != nil {
		t.Fatalf("Run() failed unexpectedly: %v", err)
	}
	want := map[string]any{}
	if diff := cmp.Diff(want, result); diff != "" {
		t.Errorf("Run() result diff (-want +got):\n%s", diff)
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

	result, err := toolImpl.Run(createToolContext(t, child), map[string]any{"request": "send 100"})

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
		t.Errorf("parent model requests = %d, want 2 (the second carrying the agent tool's error)", got)
	}
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
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "start_job", WillContinue: genai.Ptr(true)}}},
			},
			{
				Role: genai.RoleModel,
				Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
					PartialArgs:  []*genai.PartialArg{{JsonPath: "$.amount", NumberValue: genai.Ptr(1.0)}},
					WillContinue: genai.Ptr(false),
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
