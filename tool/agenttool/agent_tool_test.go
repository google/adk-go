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
	"errors"
	"iter"
	"log"
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

func TestAgentTool_Run_StateDelta(t *testing.T) {
	for _, tc := range []struct {
		name      string
		outputKey string
		state     map[string]any
		want      map[string]any
	}{
		{
			name:  "callback without content",
			state: map[string]any{"result": "updated", "app:shared": true, "user:preference": true, "temp:progress": 1},
			want:  map[string]any{"result": "updated", "app:shared": true, "user:preference": true, "temp:progress": 1},
		},
		{
			name:      "output overwrites earlier callback update",
			outputKey: "result",
			state:     map[string]any{"result": "pending", "callback_ran": true},
			want:      map[string]any{"result": "done", "callback_ran": true},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			child, err := llmagent.New(llmagent.Config{
				Name:      "child",
				OutputKey: tc.outputKey,
				Model: &testutil.MockModel{Responses: []*genai.Content{
					genai.NewContentFromText("done", "model"),
				}},
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(ctx agent.Context) (*genai.Content, error) {
					for key, value := range tc.state {
						if err := ctx.State().Set(key, value); err != nil {
							return nil, err
						}
					}
					return nil, nil
				}},
			})
			if err != nil {
				t.Fatal(err)
			}
			ctx := createToolContext(t, child)
			if err := ctx.State().Set("untouched", true); err != nil {
				t.Fatal(err)
			}
			if err := ctx.State().Set("result", "old"); err != nil {
				t.Fatal(err)
			}
			wrapped := agenttool.New(child, nil).(toolinternal.FunctionTool)
			result, err := wrapped.Run(ctx, map[string]any{"request": "run"})
			if err != nil {
				t.Fatal(err)
			}
			if result["result"] != "done" {
				t.Fatal("child did not return its result")
			}
			for key, want := range tc.want {
				got, err := ctx.ReadonlyState().Get(key)
				if err != nil || !cmp.Equal(got, want) {
					t.Error("child state update missing from parent session")
				}
				if !cmp.Equal(ctx.Actions().StateDelta[key], want) {
					t.Error("child state update missing from parent event")
				}
			}
			if got, err := ctx.State().Get("untouched"); err != nil || got != true {
				t.Error("unrelated parent state changed")
			}
		})
	}
}

func TestAgentTool_Run_PersistsStateDelta(t *testing.T) {
	child, err := llmagent.New(llmagent.Config{
		Name:      "child",
		OutputKey: "child_result",
		Model: &testutil.MockModel{Responses: []*genai.Content{
			genai.NewContentFromText("done", "model"),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	parent, err := llmagent.New(llmagent.Config{
		Name:  "parent",
		Tools: []tool.Tool{agenttool.New(child, nil)},
		Model: &testutil.MockModel{Responses: []*genai.Content{
			{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
				ID: "call_child", Name: "child", Args: map[string]any{"request": "run"},
			}}}},
			genai.NewContentFromText("finished", "model"),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	r := testutil.NewTestAgentRunner(t, parent)
	for _, err := range r.Run(t, "parent_session", "run child") {
		if err != nil {
			t.Fatal(err)
		}
	}
	stored, err := r.SessionService().Get(t.Context(), &session.GetRequest{
		AppName: "test_app", UserID: "test_user", SessionID: "parent_session",
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, err := stored.Session.State().Get("child_result"); err != nil || got != "done" {
		t.Error("child output was not persisted in parent session")
	}
}

func TestAgentTool_Run_StateOnlyEvents(t *testing.T) {
	for _, tc := range []struct {
		name    string
		partial bool
	}{
		{name: "complete"},
		{name: "partial", partial: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			child, err := agent.New(agent.Config{
				Name: "child",
				Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
					return func(yield func(*session.Event, error) bool) {
						event := session.NewEvent(ctx, ctx.InvocationID())
						event.Author = "child"
						event.Partial = tc.partial
						event.Actions.StateDelta = map[string]any{"result": "done"}
						yield(event, nil)
					}
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			ctx := createToolContext(t, child)
			wrapped := agenttool.New(child, nil).(toolinternal.FunctionTool)
			result, err := wrapped.Run(ctx, map[string]any{"request": "run"})
			if err != nil {
				t.Fatal(err)
			}
			if len(result) != 0 {
				t.Error("state-only event produced a tool result")
			}
			got, err := ctx.State().Get("result")
			if tc.partial {
				if !errors.Is(err, session.ErrStateKeyNotExist) || len(ctx.Actions().StateDelta) != 0 {
					t.Error("partial state delta reached the parent")
				}
			} else if err != nil || got != "done" || ctx.Actions().StateDelta["result"] != "done" {
				t.Error("state-only event did not update the parent")
			}
		})
	}
}

type failingState struct {
	session.State
	err error
}

func (s failingState) Set(string, any) error { return s.err }

type contextWithState struct {
	agent.Context
	state session.State
}

func (c contextWithState) State() session.State { return c.state }

func TestAgentTool_Run_StateDeltaError(t *testing.T) {
	continued := false
	child, err := agent.New(agent.Config{
		Name: "child",
		Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
			return func(yield func(*session.Event, error) bool) {
				event := session.NewEvent(ctx, ctx.InvocationID())
				event.Author = "child"
				event.Actions.StateDelta = map[string]any{"result": "done"}
				if !yield(event, nil) {
					return
				}
				continued = true
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	wantErr := errors.New("state write failed")
	ctx := createToolContext(t, child)
	wrappedCtx := contextWithState{Context: ctx, state: failingState{State: ctx.State(), err: wantErr}}
	wrapped := agenttool.New(child, nil).(toolinternal.FunctionTool)
	result, err := wrapped.Run(wrappedCtx, map[string]any{"request": "run"})
	if !errors.Is(err, wantErr) {
		t.Error("Run did not preserve the state write error")
	}
	if result != nil {
		t.Error("Run returned a result after a failed state write")
	}
	if continued {
		t.Error("child continued after a failed state write")
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
