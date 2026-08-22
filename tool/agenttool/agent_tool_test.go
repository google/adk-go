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
	"fmt"
	"iter"
	"log"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/model/gemini"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
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
	testLLM := &testutil.MockModel{
		Responses: []*genai.Content{
			genai.NewContentFromText("test response", genai.RoleModel),
		},
	}
	agent := createAgentWithModel(t, nil, nil, testLLM)
	toolCtx := createToolContext(t, agent)

	// Test with skipSummarization = true
	agentToolSkip := agenttool.New(agent, &agenttool.Config{SkipSummarization: true})
	actions := toolCtx.Actions()
	toolImpl, ok := agentToolSkip.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentToolSkip does not implement FunctionTool")
	}
	_, err := toolImpl.Run(toolCtx, map[string]any{"request": "magic"})
	if err != nil {
		t.Fatalf("Run() with skipSummarization=true failed unexpectedly: %v", err)
	}
	if !actions.SkipSummarization {
		t.Errorf("SkipSummarization flag not set when AgentTool was created with skipSummarization=true")
	}

	// Test with skipSummarization = false
	agentToolNoSkip := agenttool.New(agent, &agenttool.Config{SkipSummarization: false})
	toolImpl, ok = agentToolNoSkip.(toolinternal.FunctionTool)
	if !ok {
		t.Fatal("agentToolNoSkip does not implement FunctionTool")
	}
	actions.SkipSummarization = false // Reset
	// Reset mock for the second call
	testLLM.Responses = []*genai.Content{
		genai.NewContentFromText("test response", genai.RoleModel),
	}
	testLLM.Requests = nil
	_, err = toolImpl.Run(toolCtx, map[string]any{"request": "magic"})
	if err != nil {
		t.Fatalf("Run() with skipSummarization=false failed unexpectedly: %v", err)
	}
	if actions.SkipSummarization {
		t.Errorf("SkipSummarization flag was set when AgentTool was created with skipSummarization=false")
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

// concurrentModel replays scripted turns in order, repeating the last one once
// the script runs out. Parallel function-call dispatch calls one model from
// several goroutines, so the cursor is mutex-guarded (testutil.MockModel is
// not safe for that).
type concurrentModel struct {
	mu    sync.Mutex
	turns []*genai.Content
	next  int
}

func (m *concurrentModel) Name() string { return "concurrent-mock" }

func (m *concurrentModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	turn := m.turns[min(m.next, len(m.turns)-1)]
	m.next++
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: turn}, nil)
	}
}

// TestAgentTool_ParallelDispatch_SharedStateNotWritten is the regression test
// for the agenttool path of issue #1137: one model turn emits several calls to
// the same agent tool, the flow dispatches them on separate goroutines, and
// every dispatch builds a fresh Runner over the one shared sub-agent. Any
// run-time write to that agent's state races between the dispatches, and a
// torn read of State.Mode segfaults instead of merely tripping the detector.
//
// The Mode assertion fails deterministically without -race, so it pins the
// invariant — shared agent state is never written at run time — rather than
// the absence of a race report.
func TestAgentTool_ParallelDispatch_SharedStateNotWritten(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name string
		mode llmagent.Mode
		// calls is the number of parallel calls to the agent tool in one
		// coordinator turn.
		calls int
	}{
		{name: "unset mode, two calls", mode: llmagent.ModeUnset, calls: 2},
		{name: "unset mode, eight calls", mode: llmagent.ModeUnset, calls: 8},
		// An explicitly-configured mode was never written to, so this row
		// passes with or without the fix; it is here to exercise the fan-out
		// under -race.
		{name: "explicit chat mode, eight calls", mode: llmagent.ModeChat, calls: 8},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			const workerName = "worker"
			worker, err := llmagent.New(llmagent.Config{
				Name:        workerName,
				Description: "Resolves one descriptor.",
				Model: &concurrentModel{turns: []*genai.Content{
					genai.NewContentFromText("resolved", genai.RoleModel),
				}},
				Mode: tc.mode,
			})
			if err != nil {
				t.Fatalf("llmagent.New(worker) failed: %v", err)
			}

			parts := make([]*genai.Part, tc.calls)
			for i := range parts {
				parts[i] = &genai.Part{FunctionCall: &genai.FunctionCall{
					ID:   fmt.Sprintf("call-%d", i),
					Name: workerName,
					Args: map[string]any{"request": fmt.Sprintf("descriptor %d", i)},
				}}
			}
			coordinator, err := llmagent.New(llmagent.Config{
				Name: "coordinator",
				Model: &concurrentModel{turns: []*genai.Content{
					{Role: genai.RoleModel, Parts: parts},
					genai.NewContentFromText("all resolved", genai.RoleModel),
				}},
				Tools: []tool.Tool{agenttool.New(worker, nil)},
			})
			if err != nil {
				t.Fatalf("llmagent.New(coordinator) failed: %v", err)
			}

			sessionService := session.InMemoryService()
			created, err := sessionService.Create(t.Context(), &session.CreateRequest{
				AppName: "race-repro", UserID: "user",
			})
			if err != nil {
				t.Fatalf("session Create() failed: %v", err)
			}
			r, err := runner.New(runner.Config{
				AppName: "race-repro", Agent: coordinator, SessionService: sessionService,
			})
			if err != nil {
				t.Fatalf("runner.New() failed: %v", err)
			}

			gotResponses := map[string]bool{}
			for event, err := range r.Run(t.Context(), created.Session.UserID(), created.Session.ID(),
				genai.NewContentFromText("resolve these", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("Run() failed unexpectedly: %v", err)
				}
				if event == nil || event.Content == nil {
					continue
				}
				for _, part := range event.Content.Parts {
					if part != nil && part.FunctionResponse != nil {
						gotResponses[part.FunctionResponse.ID] = true
					}
				}
			}

			// Every parallel call was dispatched: the fan-out really happened.
			for i := range tc.calls {
				if id := fmt.Sprintf("call-%d", i); !gotResponses[id] {
					t.Errorf("no FunctionResponse for %s; got responses for %v", id, gotResponses)
				}
			}

			// The shared sub-agent's mode is still what it was configured
			// with: no dispatch wrote it (issue #1137).
			if got, want := llminternal.Reveal(worker.(llminternal.Agent)).Mode, llminternal.Mode(tc.mode); got != want {
				t.Errorf("worker Mode after run = %q, want unchanged %q: shared agent state was written at run time (issue #1137)", got, want)
			}
		})
	}
}
