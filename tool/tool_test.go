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

package tool_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/memory"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/geminitool"
	"google.golang.org/adk/v2/tool/loadartifactstool"
	"google.golang.org/adk/v2/tool/toolconfirmation"
	"google.golang.org/adk/v2/tool/toolutils"
)

func TestTypes(t *testing.T) {
	const (
		functionTool = "FunctionTool"
		requestProc  = "RequestProcessor"
	)

	type intInput struct {
		Value int `json:"value"`
	}
	type intOutput struct {
		Value int `json:"value"`
	}

	tests := []struct {
		name          string
		constructor   func() (tool.Tool, error)
		expectedTypes []string
	}{
		{
			name: "FunctionTool",
			constructor: func() (tool.Tool, error) {
				return functiontool.New(functiontool.Config{}, func(_ agent.Context, input intInput) (intOutput, error) {
					return intOutput(input), nil
				})
			},
			expectedTypes: []string{requestProc, functionTool},
		},
		{
			name:          "geminitool",
			constructor:   func() (tool.Tool, error) { return geminitool.New("", "", nil), nil },
			expectedTypes: []string{requestProc},
		},
		{
			name:          "geminitool.GoogleSearch{}",
			constructor:   func() (tool.Tool, error) { return geminitool.GoogleSearch{}, nil },
			expectedTypes: []string{requestProc},
		},
		{
			name:          "LoadArtifactsTool",
			constructor:   func() (tool.Tool, error) { return loadartifactstool.New(), nil },
			expectedTypes: []string{requestProc, functionTool},
		},
		{
			name:          "AgentTool",
			constructor:   func() (tool.Tool, error) { return agenttool.New(nil, nil), nil },
			expectedTypes: []string{requestProc, functionTool},
		},
		{
			name:          "LoadArtifactsTool",
			constructor:   func() (tool.Tool, error) { return loadartifactstool.New(), nil },
			expectedTypes: []string{requestProc, functionTool},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool, err := tt.constructor()
			if err != nil {
				t.Fatalf("Failed to create tool %s: %v", tt.name, err)
			}

			for _, s := range tt.expectedTypes {
				switch s {
				case functionTool:
					if _, ok := tool.(toolinternal.FunctionTool); !ok {
						t.Errorf("Expected %s to implement toolinternal.FunctionTool", tt.name)
					}
				case requestProc:
					if _, ok := tool.(toolinternal.RequestProcessor); !ok {
						t.Errorf("Expected %s to implement toolinternal.RequestProcessor", tt.name)
					}
				default:
					t.Fatalf("Unknown expected type: %s", s)
				}
			}
		})
	}
}

type testContext struct {
	agent.ContextMock
	context.Context
	toolConfirmationResult    *toolconfirmation.ToolConfirmation
	requestConfirmationCalled bool
	eventActions              *session.EventActions
}

// Deadline implements [agent.InvocationContext].
func (c *testContext) Deadline() (deadline time.Time, ok bool) {
	return c.Context.Deadline()
}

// Done implements [agent.InvocationContext].
func (c *testContext) Done() <-chan struct{} {
	return c.Context.Done()
}

// Err implements [agent.InvocationContext].
func (c *testContext) Err() error {
	return c.Context.Err()
}

// Value implements [agent.InvocationContext].
func (c *testContext) Value(key any) any {
	return c.Context.Value(key)
}

func (c *testContext) ToolConfirmation() *toolconfirmation.ToolConfirmation {
	return c.toolConfirmationResult
}

func (c *testContext) RequestConfirmation(string, any) error {
	c.requestConfirmationCalled = true
	return nil
}

func (c *testContext) Actions() *session.EventActions {
	if c.eventActions == nil {
		c.eventActions = &session.EventActions{}
	}
	return c.eventActions
}
func (c *testContext) FunctionCallID() string { return "test-function-call-id" }
func (c *testContext) SearchMemory(context.Context, string) (*memory.SearchResponse, error) {
	return nil, nil
}
func (c *testContext) AgentName() string                                       { return "test-agent" }
func (c *testContext) InvocationID() string                                    { return "test-invocation-id" }
func (c *testContext) AppName() string                                         { return "test-app" }
func (c *testContext) Branch() string                                          { return "test-branch" }
func (c *testContext) SessionID() string                                       { return "test-session-id" }
func (c *testContext) UserID() string                                          { return "test-user-id" }
func (m *testContext) WithContext(ctx context.Context) agent.InvocationContext { return m }

var _ agent.InvocationContext = (*testContext)(nil)

type testToolset struct {
	tools []tool.Tool
}

func (tts *testToolset) Name() string { return "testToolset" }
func (tts *testToolset) Tools(agent.ReadonlyContext) ([]tool.Tool, error) {
	return tts.tools, nil
}

func TestWithConfirmation(t *testing.T) {
	toolRan := false
	noOpTool, err := functiontool.New(functiontool.Config{Name: "noOpTool"}, func(ctx agent.Context, input struct{}) (struct{}, error) {
		toolRan = true
		return struct{}{}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() failed: %v", err)
	}
	ts := &testToolset{tools: []tool.Tool{noOpTool}}

	tests := []struct {
		name                      string
		requireConfirmation       bool
		provider                  tool.ConfirmationProvider
		toolConfirmation          *toolconfirmation.ToolConfirmation
		wantConfirmationRequested bool
		wantSkipSummarization     bool
		wantErr                   bool
		wantErrMsg                string
		wantToolRan               bool
	}{
		{
			name:                      "confirmation required, no confirmation in context",
			requireConfirmation:       true,
			wantConfirmationRequested: true,
			wantSkipSummarization:     true,
			wantErr:                   true,
			wantErrMsg:                "requires confirmation",
			wantToolRan:               false,
		},
		{
			name:                "confirmation required, confirmed in context",
			requireConfirmation: true,
			toolConfirmation:    &toolconfirmation.ToolConfirmation{Confirmed: true},
			wantToolRan:         true,
		},
		{
			name:                "confirmation required, rejected in context",
			requireConfirmation: true,
			toolConfirmation:    &toolconfirmation.ToolConfirmation{Confirmed: false},
			wantErr:             true,
			wantErrMsg:          "call is rejected",
			wantToolRan:         false,
		},
		{
			name:                      "confirmation not required",
			requireConfirmation:       false,
			wantConfirmationRequested: false,
			wantToolRan:               true,
		},
		{
			name: "confirmation provider requires confirmation",
			provider: func(toolName string, toolInput any) bool {
				return true
			},
			wantConfirmationRequested: true,
			wantSkipSummarization:     true,
			wantErr:                   true,
			wantErrMsg:                "requires confirmation",
			wantToolRan:               false,
		},
		{
			name: "confirmation provider does not require confirmation",
			provider: func(toolName string, toolInput any) bool {
				return false
			},
			wantConfirmationRequested: false,
			wantToolRan:               true,
		},
		{
			name:                "requireConfirmation=true but provider returns false",
			requireConfirmation: true,
			provider: func(toolName string, toolInput any) bool {
				return false
			},
			wantConfirmationRequested: false,
			wantToolRan:               true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			toolRan = false
			ctx := &testContext{Context: context.Background(), toolConfirmationResult: tt.toolConfirmation}

			cts := tool.WithConfirmation(ts, tt.requireConfirmation, tt.provider)
			tools, err := cts.Tools(nil)
			if err != nil {
				t.Fatalf("cts.Tools() failed: %v", err)
			}
			if len(tools) != 1 {
				t.Fatalf("cts.Tools() returned %d tools, want 1", len(tools))
			}
			confirmedTool, ok := tools[0].(toolinternal.FunctionTool)
			if !ok {
				t.Fatalf("tools[0] is not a FunctionTool")
			}

			_, err = confirmedTool.Run(ctx, map[string]any{})
			if (err != nil) != tt.wantErr {
				t.Errorf("Run() error = %v, wantErr %v", err, tt.wantErr)
			}
			if err != nil && tt.wantErrMsg != "" && !strings.Contains(err.Error(), tt.wantErrMsg) {
				t.Errorf("Run() error msg = %q, want it to contain %q", err.Error(), tt.wantErrMsg)
			}

			if ctx.Actions().SkipSummarization != tt.wantSkipSummarization {
				t.Errorf("Run() skipSummarization = %v, want %v", ctx.Actions().SkipSummarization, tt.wantSkipSummarization)
			}
			if ctx.requestConfirmationCalled != tt.wantConfirmationRequested {
				t.Errorf("Run() requestConfirmationCalled = %v, want %v", ctx.requestConfirmationCalled, tt.wantConfirmationRequested)
			}
			if toolRan != tt.wantToolRan {
				t.Errorf("toolRan = %v, want %v", toolRan, tt.wantToolRan)
			}
		})
	}
}

type mockCustomRequestTool struct {
	tool.Tool
	decl                 *genai.FunctionDeclaration
	processRequestCalled *bool
}

func (m *mockCustomRequestTool) Declaration() *genai.FunctionDeclaration { return m.decl }
func (m *mockCustomRequestTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	return map[string]any{}, nil
}

func (m *mockCustomRequestTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	*m.processRequestCalled = true
	utils.AppendInstructions(req, "custom tool system instruction")
	return nil
}

func TestWithConfirmation_ProcessRequest(t *testing.T) {
	processRequestCalled := false
	baseTool, err := functiontool.New(functiontool.Config{Name: "customReqTool"}, func(ctx agent.Context, input struct{}) (struct{}, error) {
		return struct{}{}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() failed: %v", err)
	}

	customTool := &mockCustomRequestTool{
		Tool:                 baseTool,
		decl:                 &genai.FunctionDeclaration{Name: "customReqTool"},
		processRequestCalled: &processRequestCalled,
	}

	ts := &testToolset{tools: []tool.Tool{customTool}}
	cts := tool.WithConfirmation(ts, true, nil)

	tools, err := cts.Tools(nil)
	if err != nil {
		t.Fatalf("cts.Tools() failed: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("cts.Tools() returned %d tools, want 1", len(tools))
	}

	processor, ok := tools[0].(toolinternal.RequestProcessor)
	if !ok {
		t.Fatalf("wrapped tool does not implement RequestProcessor")
	}

	req := &model.LLMRequest{}
	if err := processor.ProcessRequest(nil, req); err != nil {
		t.Fatalf("ProcessRequest() failed: %v", err)
	}

	if !processRequestCalled {
		t.Errorf("expected wrapped tool's ProcessRequest to be called, but it was not")
	}
	if req.Config == nil || req.Config.SystemInstruction == nil || len(req.Config.SystemInstruction.Parts) == 0 {
		t.Fatalf("expected SystemInstruction to be set on req.Config, got nil/empty")
	}
	if got := req.Config.SystemInstruction.Parts[0].Text; got != "custom tool system instruction" {
		t.Errorf("SystemInstruction text = %q, want %q", got, "custom tool system instruction")
	}
}

type mockCustomRequestPackingTool struct {
	tool.Tool
	decl                 *genai.FunctionDeclaration
	processRequestCalled *bool
}

func (m *mockCustomRequestPackingTool) Declaration() *genai.FunctionDeclaration { return m.decl }
func (m *mockCustomRequestPackingTool) Run(ctx agent.Context, args any) (map[string]any, error) {
	return map[string]any{}, nil
}

func (m *mockCustomRequestPackingTool) ProcessRequest(ctx agent.Context, req *model.LLMRequest) error {
	*m.processRequestCalled = true
	return toolutils.PackTool(req, m)
}

func TestWithConfirmation_ProcessRequest_Packing(t *testing.T) {
	processRequestCalled := false
	baseTool, err := functiontool.New(functiontool.Config{Name: "customPackingTool"}, func(ctx agent.Context, input struct{}) (struct{}, error) {
		return struct{}{}, nil
	})
	if err != nil {
		t.Fatalf("functiontool.New() failed: %v", err)
	}

	customTool := &mockCustomRequestPackingTool{
		Tool:                 baseTool,
		decl:                 &genai.FunctionDeclaration{Name: "customPackingTool"},
		processRequestCalled: &processRequestCalled,
	}

	ts := &testToolset{tools: []tool.Tool{customTool}}
	cts := tool.WithConfirmation(ts, true, nil)

	tools, err := cts.Tools(nil)
	if err != nil {
		t.Fatalf("cts.Tools() failed: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("cts.Tools() returned %d tools, want 1", len(tools))
	}

	processor, ok := tools[0].(toolinternal.RequestProcessor)
	if !ok {
		t.Fatalf("wrapped tool does not implement RequestProcessor")
	}

	req := &model.LLMRequest{}
	if err := processor.ProcessRequest(nil, req); err != nil {
		t.Fatalf("ProcessRequest() failed: %v", err)
	}

	if !processRequestCalled {
		t.Errorf("expected wrapped tool's ProcessRequest to be called, but it was not")
	}

	if req.Tools["customPackingTool"] != tools[0] {
		t.Errorf("expected req.Tools[%q] to be the confirmation wrapper %T, got %T", "customPackingTool", tools[0], req.Tools["customPackingTool"])
	}

	packedTool, ok := req.Tools["customPackingTool"].(toolinternal.FunctionTool)
	if !ok {
		t.Fatalf("expected packed tool to implement toolinternal.FunctionTool, got %T", req.Tools["customPackingTool"])
	}

	ctx := &testContext{Context: context.Background()}
	_, err = packedTool.Run(ctx, map[string]any{})
	if !errors.Is(err, tool.ErrConfirmationRequired) {
		t.Errorf("expected Run() to return ErrConfirmationRequired, got %v", err)
	}
	if !ctx.requestConfirmationCalled {
		t.Errorf("expected RequestConfirmation to be called on ctx, but it was not")
	}
}
