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

package functioncallmodifier_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/internal/workflowinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/functioncallmodifier"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolutils"
)

type SimpleArgs struct {
	Num int
}

func okFunc(_ agent.Context, _ SimpleArgs) (string, error) {
	return "ok", nil
}

func TestBeforeModelCallback(t *testing.T) {
	invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	ctx := icontext.NewCallbackContextWithDelta(invCtx, nil, nil)

	transferTool := &llminternal.TransferToAgentTool{}
	transferToolDecl := transferTool.Declaration()

	agentToolDefault := createAgentTool(t, "agent_default", "desc", nil)
	agentToolDefaultDecl := agentToolDefault.(toolinternal.FunctionTool).Declaration()
	functionTool, err := functiontool.New(functiontool.Config{
		Name: "other_tool",
	}, okFunc)
	if err != nil {
		t.Fatalf("functiontool.New failed: %v", err)
	}

	testCases := []struct {
		name       string
		req        *model.LLMRequest
		wantParams map[string]bool
		checkTools []string
	}{
		{
			name: "no relevant tools",
			req: &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					Tools: []*genai.Tool{{
						FunctionDeclarations: []*genai.FunctionDeclaration{{Name: "other_tool"}},
					}},
				},
				Tools: map[string]any{"other_tool": functionTool},
			},
		},
		{
			name: "agent tool default schema",
			req: &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					Tools: []*genai.Tool{{
						FunctionDeclarations: []*genai.FunctionDeclaration{agentToolDefaultDecl},
					}},
				},
				Tools: map[string]any{"agent_default": agentToolDefault},
			},
			wantParams: map[string]bool{"request": true, "skill_id": true, "rationale": true},
			checkTools: []string{"agent_default"},
		},
		{
			name: "transfer to agent tool",
			req: &model.LLMRequest{
				Config: &genai.GenerateContentConfig{
					Tools: []*genai.Tool{{
						FunctionDeclarations: []*genai.FunctionDeclaration{transferToolDecl},
					}},
				},
				Tools: map[string]any{"transfer_to_agent": transferTool},
			},
			wantParams: map[string]bool{"agent_name": true, "skill_id": true, "rationale": true},
			checkTools: []string{"transfer_to_agent"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			plugin, err := functioncallmodifier.NewPlugin(functioncallmodifier.FunctionCallModifierConfig{
				Predicate: func(toolName string) bool {
					return toolName == "transfer_to_agent" || toolName == "agent_default"
				},
				Args: map[string]*genai.Schema{
					"skill_id": {
						Description: "The specific skill to be utilized by the agent.",
						Type:        "STRING",
					},
					"rationale": {
						Description: "The reasoning behind selecting this agent and skill.",
						Type:        "STRING",
					},
				},
				OverrideDescription: func(originalDescription string) string {
					return fmt.Sprintf("This tool can now optionally accept skill_id and rationale parameters to guide skill-based orchestration. %s", originalDescription)
				},
			})
			if err != nil {
				t.Fatalf("New plugin failed: %v", err)
			}
			// Clone req to avoid modification across test cases
			currentReq := cloneLLMRequest(t, tc.req)

			beforeModelCallback := plugin.BeforeModelCallback()
			if _, err := beforeModelCallback(ctx, currentReq); err != nil {
				t.Fatalf("BeforeModelCallback failed: %v", err)
			}

			for _, toolName := range tc.checkTools {
				decl := findDeclaration(currentReq, toolName)
				if decl == nil {
					t.Errorf("Tool %s: Declaration not found in request", toolName)
					continue
				}

				if len(decl.Parameters.Properties) != len(tc.wantParams) {
					t.Errorf("Tool %s: Expected %d parameters, got %d", toolName, len(tc.wantParams), len(decl.Parameters.Properties))
				}
				params := make(map[string]bool)
				for k := range decl.Parameters.Properties {
					params[k] = true
				}
				if diff := cmp.Diff(tc.wantParams, params); diff != "" {
					t.Errorf("Tool %s: Parameter mismatch (-want +got):\n%s", toolName, diff)
				}

				if !strings.Contains(decl.Description, "skill-based orchestration") {
					t.Errorf("Tool %s: Description not updated", toolName)
				}
			}
		})
	}
}

func TestAfterModelCallback(t *testing.T) {
	testCases := []struct {
		name                     string
		content                  *genai.Content
		originalDecls            map[string]*genai.FunctionDeclaration
		wantArgs                 map[string]any
		wantSkillStateKey        string
		wantSkillStateValue      string
		wantRationaleStateKey    string
		wantRationaleStateValue  string
		shouldHaveSkillState     bool
		shouldHaveRationaleState bool
	}{
		{
			name:                     "no function calls",
			content:                  &genai.Content{Parts: []*genai.Part{{Text: "hello"}}},
			originalDecls:            map[string]*genai.FunctionDeclaration{},
			shouldHaveSkillState:     false,
			shouldHaveRationaleState: false,
		},
		{
			name:                     "unmodified function call",
			content:                  &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "other_tool", Args: map[string]any{"foo": "bar"}}}}},
			originalDecls:            map[string]*genai.FunctionDeclaration{},
			wantArgs:                 map[string]any{"foo": "bar"},
			shouldHaveSkillState:     false,
			shouldHaveRationaleState: false,
		},
		{
			name:                     "agent tool with skill and rationale",
			content:                  &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "fcId", Name: "test_agent", Args: map[string]any{"request": "do", "skill_id": "s1", "rationale": "r1"}}}}},
			originalDecls:            map[string]*genai.FunctionDeclaration{"test_agent": {Name: "test_agent"}},
			wantArgs:                 map[string]any{"request": "do"},
			wantSkillStateKey:        "fcId/skill_id",
			wantSkillStateValue:      "s1",
			wantRationaleStateKey:    "fcId/rationale",
			wantRationaleStateValue:  "r1",
			shouldHaveSkillState:     true,
			shouldHaveRationaleState: true,
		},
		{
			name:                     "agent tool with only skill_id",
			content:                  &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "fcId", Name: "test_agent", Args: map[string]any{"request": "do", "skill_id": "s1"}}}}},
			originalDecls:            map[string]*genai.FunctionDeclaration{"test_agent": {Name: "test_agent"}},
			wantArgs:                 map[string]any{"request": "do"},
			wantSkillStateKey:        "fcId/skill_id",
			wantSkillStateValue:      "s1",
			shouldHaveSkillState:     true,
			shouldHaveRationaleState: false,
		},
		{
			name:                     "agent tool without skill/rationale",
			content:                  &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "fcId", Name: "test_agent", Args: map[string]any{"request": "do"}}}}},
			originalDecls:            map[string]*genai.FunctionDeclaration{"test_agent": {Name: "test_agent"}},
			wantArgs:                 map[string]any{"request": "do"},
			shouldHaveSkillState:     false,
			shouldHaveRationaleState: false,
		},
		{
			name:                     "transfer tool with skill and rationale",
			content:                  &genai.Content{Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "fcId", Name: "transfer_to_agent", Args: map[string]any{"agent_name": "a1", "skill_id": "s2", "rationale": "r2"}}}}},
			originalDecls:            map[string]*genai.FunctionDeclaration{"transfer_to_agent": {Name: "transfer_to_agent"}},
			wantArgs:                 map[string]any{"agent_name": "a1"},
			wantSkillStateKey:        "fcId/skill_id",
			wantSkillStateValue:      "s2",
			wantRationaleStateKey:    "fcId/rationale",
			wantRationaleStateValue:  "r2",
			shouldHaveSkillState:     true,
			shouldHaveRationaleState: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			plugin, err := functioncallmodifier.NewPlugin(functioncallmodifier.FunctionCallModifierConfig{
				Predicate: func(toolName string) bool {
					return toolName == "transfer_to_agent" || toolName == "test_agent"
				},
				Args: map[string]*genai.Schema{
					"skill_id": {
						Description: "The specific skill to be utilized by the agent.",
						Type:        "STRING",
					},
					"rationale": {
						Description: "The reasoning behind selecting this agent and skill.",
						Type:        "STRING",
					},
				},
				OverrideDescription: func(originalDescription string) string {
					return fmt.Sprintf("This tool can now optionally accept skill_id and rationale parameters to guide skill-based orchestration. %s", originalDescription)
				},
			})
			if err != nil {
				t.Fatalf("New plugin failed: %v", err)
			}
			service := session.InMemoryService()
			sesResp, err := service.Create(t.Context(), &session.CreateRequest{AppName: "test", UserID: "user", SessionID: "ses1"})
			if err != nil {
				t.Fatalf("Failed to create session: %v", err)
			}
			invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Session: sesResp.Session,
			})
			ctx := icontext.NewCallbackContextWithDelta(invCtx, nil, nil)

			afterModelCallback := plugin.AfterModelCallback()
			if _, err := afterModelCallback(ctx, &model.LLMResponse{Content: tc.content}, nil); err != nil {
				t.Fatalf("AfterModelCallback failed: %v", err)
			}

			// Check if args are pruned
			if len(tc.content.Parts) > 0 {
				part := tc.content.Parts[0]
				if part.FunctionCall != nil && tc.wantArgs != nil {
					if diff := cmp.Diff(tc.wantArgs, part.FunctionCall.Args); diff != "" {
						t.Errorf("Args mismatch (-want +got):\n%s", diff)
					}
				}
			}

			// Check session state
			skillStateValue, err := sesResp.Session.State().Get(tc.wantSkillStateKey)
			if tc.shouldHaveSkillState {
				if err != nil {
					t.Errorf("State().Get(%q) unexpected error: %v", tc.wantSkillStateKey, err)
				} else {
					gotState, ok := skillStateValue.(string)
					if !ok {
						t.Errorf("State value for key %s is not string: %T", tc.wantSkillStateKey, skillStateValue)
					} else if diff := cmp.Diff(tc.wantSkillStateValue, gotState); diff != "" {
						t.Errorf("State value mismatch for key %s (-want +got):\n%s", tc.wantSkillStateKey, diff)
					}
				}
			} else { // Should NOT have state
				if tc.wantSkillStateKey != "" {
					_, err := sesResp.Session.State().Get(tc.wantSkillStateKey)
					if err == nil { // Key WAS found, which is unexpected
						t.Errorf("Unexpected state key %s found", tc.wantSkillStateKey)
					} else if !errors.Is(err, session.ErrStateKeyNotExist) {
						// Unexpected error other than not existing
						t.Errorf("State().Get(%q) unexpected error when expecting key not to exist: %v", tc.wantSkillStateKey, err)
					}
				}
			}

			rationaleStateValue, err := sesResp.Session.State().Get(tc.wantRationaleStateKey)
			if tc.shouldHaveRationaleState {
				if err != nil {
					t.Errorf("State().Get(%q) unexpected error: %v", tc.wantRationaleStateKey, err)
				} else {
					gotState, ok := rationaleStateValue.(string)
					if !ok {
						t.Errorf("State value for key %s is not string: %T", tc.wantRationaleStateKey, rationaleStateValue)
					} else if diff := cmp.Diff(tc.wantRationaleStateValue, gotState); diff != "" {
						t.Errorf("State value mismatch for key %s (-want +got):\n%s", tc.wantRationaleStateKey, diff)
					}
				}
			} else { // Should NOT have state
				if tc.wantRationaleStateKey != "" {
					_, err := sesResp.Session.State().Get(tc.wantRationaleStateKey)
					if err == nil { // Key WAS found, which is unexpected
						t.Errorf("Unexpected state key %s found", tc.wantRationaleStateKey)
					} else if !errors.Is(err, session.ErrStateKeyNotExist) {
						// Unexpected error other than not existing
						t.Errorf("State().Get(%q) unexpected error when expecting key not to exist: %v", tc.wantRationaleStateKey, err)
					}
				}
			}
		})
	}
}

// contentlessModel answers every request with a response that carries no
// content, as a model does when it stops on MAX_TOKENS or SAFETY.
type contentlessModel struct{}

func (contentlessModel) Name() string { return "contentless" }

func (contentlessModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{
			FinishReason: genai.FinishReasonMaxTokens,
			ErrorCode:    string(genai.FinishReasonMaxTokens),
		}, nil)
	}
}

// TestAfterModelCallback_ContentlessResponseKeepsChain pins that the plugin
// leaves a response with nothing to rewrite to the rest of the after-model
// chain, so an agent's own AfterModelCallbacks still see it.
func TestAfterModelCallback_ContentlessResponseKeepsChain(t *testing.T) {
	p, err := functioncallmodifier.NewPlugin(functioncallmodifier.FunctionCallModifierConfig{
		Predicate: func(string) bool { return true },
		Args:      map[string]*genai.Schema{"skill_id": {Type: genai.TypeString}},
	})
	if err != nil {
		t.Fatalf("NewPlugin() error = %v", err)
	}

	const recovered = "recovered from an empty response"
	agentCallbackCalls := 0
	a, err := llmagent.New(llmagent.Config{
		Name:  "agent",
		Model: contentlessModel{},
		AfterModelCallbacks: []llmagent.AfterModelCallback{
			func(_ agent.Context, resp *model.LLMResponse, _ error) (*model.LLMResponse, error) {
				agentCallbackCalls++
				if resp != nil && resp.Content == nil {
					return &model.LLMResponse{Content: genai.NewContentFromText(recovered, genai.RoleModel)}, nil
				}
				return nil, nil
			},
		},
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	sessionService := session.InMemoryService()
	if _, err := sessionService.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s"}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:        "app",
		Agent:          a,
		SessionService: sessionService,
		PluginConfig:   runner.PluginConfig{Plugins: []*plugin.Plugin{p}},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}

	var texts []string
	for ev, err := range r.Run(t.Context(), "user", "s", genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}) {
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if part.Text != "" {
				texts = append(texts, part.Text)
			}
		}
	}

	if agentCallbackCalls != 1 {
		t.Errorf("agent AfterModelCallback ran %d times, want 1", agentCallbackCalls)
	}
	if want := []string{recovered}; !slices.Equal(texts, want) {
		t.Errorf("event texts = %q, want %q", texts, want)
	}
}

// TestBeforeModelCallback_LeavesToolDeclarationUnchanged pins that the plugin
// edits only the request. A task agent tool hands out the same declaration on
// every call, and an agent tool reuses its agent's InputSchema, so editing
// them in place would wrap the description once more on every request and add
// the arguments to the agent's own schema. It also pins that the request's
// copy keeps every other field of the tool's declaration.
func TestBeforeModelCallback_LeavesToolDeclarationUnchanged(t *testing.T) {
	const prefix = "Accepts skill_id. "
	p, err := functioncallmodifier.NewPlugin(functioncallmodifier.FunctionCallModifierConfig{
		Predicate: func(string) bool { return true },
		Args:      map[string]*genai.Schema{"skill_id": {Type: genai.TypeString}},
		OverrideDescription: func(original string) string {
			return prefix + original
		},
	})
	if err != nil {
		t.Fatalf("NewPlugin() error = %v", err)
	}

	inputSchema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: map[string]*genai.Schema{"query": {Type: genai.TypeString}},
	}
	schemaAgent, err := llmagent.New(llmagent.Config{
		Name:        "schema_agent",
		Description: "answers queries",
		InputSchema: inputSchema,
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	taskAgent, err := llmagent.New(llmagent.Config{
		Name:        "task_agent",
		Description: "runs a task",
		Mode:        llmagent.ModeTask,
	})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	taskTool, err := workflowinternal.NewTaskAgentTool(taskAgent)
	if err != nil {
		t.Fatalf("NewTaskAgentTool() error = %v", err)
	}

	for _, tc := range []struct {
		name string
		tool tool.Tool
	}{
		{name: "task agent tool", tool: taskTool},
		{name: "agent tool with input schema", tool: agenttool.New(schemaAgent, nil)},
		{
			name: "cached declaration without parameters",
			tool: &cachedDeclarationTool{decl: &genai.FunctionDeclaration{Name: "no_params", Description: "takes nothing"}},
		},
		{
			name: "cached declaration with response schema and behavior",
			tool: &cachedDeclarationTool{decl: &genai.FunctionDeclaration{
				Name:        "full",
				Description: "has a response schema and a behavior",
				Behavior:    genai.BehaviorNonBlocking,
				Parameters: &genai.Schema{
					Type:        genai.TypeObject,
					Description: "a query",
					Required:    []string{"query"},
					Properties:  map[string]*genai.Schema{"query": {Type: genai.TypeString}},
				},
				ResponseJsonSchema: map[string]any{"type": "string"},
			}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fnTool := tc.tool.(toolinternal.FunctionTool)
			original := fnTool.Declaration()
			originalDescription := original.Description
			originalProperties := propertyNames(original)
			// The request's copy may differ from the tool's declaration only in its
			// description and the added skill_id, plus the object schema that holds
			// skill_id when the declaration has no parameters.
			wantDecl := *original
			wantParams := genai.Schema{Type: genai.TypeObject}
			if original.Parameters != nil {
				wantParams = *original.Parameters
			}
			wantParams.Properties = maps.Clone(wantParams.Properties)
			if wantParams.Properties == nil {
				wantParams.Properties = map[string]*genai.Schema{}
			}
			wantDecl.Parameters = &wantParams
			ignoreModified := cmp.Options{
				cmpopts.IgnoreFields(genai.FunctionDeclaration{}, "Description"),
				cmpopts.IgnoreMapEntries(func(name string, _ *genai.Schema) bool { return name == "skill_id" }),
			}

			invCtx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
			ctx := icontext.NewCallbackContextWithDelta(invCtx, nil, nil)
			for request := range 2 {
				req := &model.LLMRequest{}
				if err := toolutils.PackTool(req, fnTool); err != nil {
					t.Fatalf("PackTool() error = %v", err)
				}
				if _, err := p.BeforeModelCallback()(ctx, req); err != nil {
					t.Fatalf("BeforeModelCallback() error = %v", err)
				}
				decl := findDeclaration(req, tc.tool.Name())
				if want := prefix + originalDescription; decl.Description != want {
					t.Errorf("request %d: description = %q, want %q", request, decl.Description, want)
				}
				if !slices.Contains(propertyNames(decl), "skill_id") {
					t.Errorf("request %d: declaration lacks skill_id", request)
				}
				if diff := cmp.Diff(&wantDecl, decl, ignoreModified); diff != "" {
					t.Errorf("request %d: declaration differs beyond the description and skill_id (-tool +request):\n%s", request, diff)
				}
			}

			after := fnTool.Declaration()
			if after.Description != originalDescription {
				t.Errorf("tool description = %q, want %q", after.Description, originalDescription)
			}
			if got := propertyNames(after); !slices.Equal(got, originalProperties) {
				t.Errorf("tool parameters = %q, want %q", got, originalProperties)
			}
		})
	}
	if _, ok := inputSchema.Properties["skill_id"]; ok {
		t.Error("agent InputSchema gained skill_id")
	}
}

// cachedDeclarationTool hands out one declaration on every call, as the task
// and single_turn tools do.
type cachedDeclarationTool struct {
	decl *genai.FunctionDeclaration
}

func (c *cachedDeclarationTool) Name() string                                   { return c.decl.Name }
func (c *cachedDeclarationTool) Description() string                            { return c.decl.Description }
func (c *cachedDeclarationTool) IsLongRunning() bool                            { return false }
func (c *cachedDeclarationTool) Declaration() *genai.FunctionDeclaration        { return c.decl }
func (c *cachedDeclarationTool) Run(agent.Context, any) (map[string]any, error) { return nil, nil }

func propertyNames(decl *genai.FunctionDeclaration) []string {
	if decl.Parameters == nil {
		return nil
	}
	return slices.Sorted(maps.Keys(decl.Parameters.Properties))
}

// Mock agent for testing agenttool
type mockAgent struct {
	agent.Agent
	name        string
	description string
	inputSchema *genai.Schema
}

func (m *mockAgent) Name() string               { return m.name }
func (m *mockAgent) Description() string        { return m.description }
func (m *mockAgent) InputSchema() *genai.Schema { return m.inputSchema }
func (m *mockAgent) SubAgents() []agent.Agent   { return nil }

func createAgentTool(t *testing.T, name, desc string, schema *genai.Schema) tool.Tool {
	t.Helper()
	tA := agenttool.New(&mockAgent{name: name, description: desc, inputSchema: schema}, nil)

	return tA
}

func findDeclaration(req *model.LLMRequest, toolName string) *genai.FunctionDeclaration {
	if req.Config == nil {
		return nil
	}
	for _, tool := range req.Config.Tools {
		for _, decl := range tool.FunctionDeclarations {
			if decl.Name == toolName {
				return decl
			}
		}
	}
	return nil
}

func cloneLLMRequest(t *testing.T, req *model.LLMRequest) *model.LLMRequest {
	t.Helper()
	newReq := &model.LLMRequest{
		Tools: make(map[string]any),
	}
	if req.Config != nil {
		newReq.Config = &genai.GenerateContentConfig{}
		for _, tool := range req.Config.Tools {
			newTool := &genai.Tool{}
			for _, decl := range tool.FunctionDeclarations {
				newDecl := *decl // Shallow copy of declaration
				if decl.Parameters != nil {
					newParams := *decl.Parameters // Shallow copy of Schema
					newParams.Properties = make(map[string]*genai.Schema)
					for k, v := range decl.Parameters.Properties {
						prop := *v
						newParams.Properties[k] = &prop
					}
					newDecl.Parameters = &newParams
				}
				newTool.FunctionDeclarations = append(newTool.FunctionDeclarations, &newDecl)
			}
			newReq.Config.Tools = append(newReq.Config.Tools, newTool)
		}
	}
	// Shallow copy of tool instances
	maps.Copy(newReq.Tools, req.Tools)
	return newReq
}
