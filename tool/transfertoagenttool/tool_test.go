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

package transfertoagenttool_test

import (
	"encoding/json"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/exp/feature"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal/googlellm"
	"google.golang.org/adk/v2/internal/toolinternal"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/transfertoagenttool"
)

// descWithoutReason and descWithReason are adk-python's cleaned descriptions
// of the tool without and with transfer_reason
// (tools/transfer_to_agent_tool.py).
const (
	descWithoutReason = "Transfer the query to another agent.\n\nUse this tool to hand off control to another agent that is more suitable to\nanswer the user's query according to the agent's description.\n\nArgs:\n  agent_name: the agent name to transfer to."
	descWithReason    = descWithoutReason + "\n  transfer_reason: the reason for transferring to the target agent."
)

// requestContext returns the context a tool's ProcessRequest gets from the
// flow, with the legacy declaration switch set to legacy and the model's
// backend set to backend.
func requestContext(t *testing.T, legacy bool, backend genai.Backend) agent.Context {
	t.Helper()
	ctx := feature.WithLegacyFunctionDeclarationSchema(t.Context(), legacy)
	ctx = googlellm.ContextWithBackend(ctx, backend)
	return agent.NewToolContext(icontext.NewInvocationContext(ctx, icontext.InvocationContextParams{}), "", &session.EventActions{}, nil)
}

// declare registers tl on a new request, as the flow does, and returns the
// declaration the request carries.
func declare(t *testing.T, tl tool.Tool, ctx agent.Context) *genai.FunctionDeclaration {
	t.Helper()
	rp, ok := tl.(toolinternal.RequestProcessor)
	if !ok {
		t.Fatalf("%T does not implement ProcessRequest", tl)
	}
	req := &model.LLMRequest{}
	if err := rp.ProcessRequest(ctx, req); err != nil {
		t.Fatalf("ProcessRequest() error = %v", err)
	}
	if req.Config == nil || len(req.Config.Tools) != 1 || len(req.Config.Tools[0].FunctionDeclarations) != 1 {
		t.Fatalf("ProcessRequest() registered %+v, want one function declaration", req.Config)
	}
	return req.Config.Tools[0].FunctionDeclarations[0]
}

// asJSON returns v marshaled and unmarshaled into a map, so that two JSON
// schemas compare without regard to key order.
func asJSON(t *testing.T, v any) map[string]any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("json.Marshal(%v) error = %v", v, err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("json.Unmarshal(%s) error = %v", b, err)
	}
	return m
}

// runTool calls tl's Run with args on ctx.
func runTool(t *testing.T, tl tool.Tool, ctx agent.Context, args any) (map[string]any, error) {
	t.Helper()
	ft, ok := tl.(toolinternal.FunctionTool)
	if !ok {
		t.Fatalf("%T does not implement Run", tl)
	}
	return ft.Run(ctx, args)
}

// typedEnum returns the agent_name enum of a typed-form declaration.
func typedEnum(t *testing.T, decl *genai.FunctionDeclaration) []string {
	t.Helper()
	if decl.Parameters == nil || decl.Parameters.Properties["agent_name"] == nil {
		t.Fatalf("Parameters = %v, want the typed form with agent_name", decl.Parameters)
	}
	return decl.Parameters.Properties["agent_name"].Enum
}

func TestNew_DeclarationForms(t *testing.T) {
	names := []string{"zeta", "alpha"}
	t.Run("JSON", func(t *testing.T) {
		tests := []struct {
			name   string
			reason bool
			want   map[string]any
		}{
			{
				name: "without reason",
				want: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"agent_name": map[string]any{"type": "string", "enum": []any{"zeta", "alpha"}},
					},
					"required": []any{"agent_name"},
				},
			},
			{
				name:   "with reason",
				reason: true,
				want: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"agent_name":      map[string]any{"type": "string", "enum": []any{"zeta", "alpha"}},
						"transfer_reason": map[string]any{"type": "string", "default": ""},
					},
					"required": []any{"agent_name"},
				},
			},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: names, IncludeTransferReason: tc.reason})
				decl := declare(t, tl, requestContext(t, false, genai.BackendGeminiAPI))
				if decl.Parameters != nil {
					t.Errorf("Parameters = %+v, want nil in the JSON form", decl.Parameters)
				}
				if diff := cmp.Diff(tc.want, asJSON(t, decl.ParametersJsonSchema)); diff != "" {
					t.Errorf("ParametersJsonSchema mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})
	t.Run("typed", func(t *testing.T) {
		tests := []struct {
			name   string
			reason bool
			want   *genai.Schema
		}{
			{
				name: "without reason",
				want: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"agent_name": {Type: genai.TypeString, Enum: []string{"zeta", "alpha"}},
					},
					Required: []string{"agent_name"},
				},
			},
			{
				name:   "with reason",
				reason: true,
				want: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"agent_name":      {Type: genai.TypeString, Enum: []string{"zeta", "alpha"}},
						"transfer_reason": {Type: genai.TypeString, Default: ""},
					},
					Required: []string{"agent_name"},
				},
			},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: names, IncludeTransferReason: tc.reason})
				decl := declare(t, tl, requestContext(t, true, genai.BackendGeminiAPI))
				if decl.ParametersJsonSchema != nil {
					t.Errorf("ParametersJsonSchema = %+v, want nil in the typed form", decl.ParametersJsonSchema)
				}
				if diff := cmp.Diff(tc.want, decl.Parameters); diff != "" {
					t.Errorf("Parameters mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})
}

func TestNew_DescriptionMatchesDeclaration(t *testing.T) {
	tests := []struct {
		name   string
		legacy bool
		reason bool
		want   string
	}{
		{name: "JSON without reason", want: descWithoutReason},
		{name: "JSON with reason", reason: true, want: descWithReason},
		{name: "typed without reason", legacy: true, want: descWithoutReason},
		// adk-python sends the raw, indented docstring as the declaration's
		// description here; the tool sends the cleaned text in both forms.
		{name: "typed with reason", legacy: true, reason: true, want: descWithReason},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{"a"}, IncludeTransferReason: tc.reason})
			if got := tl.Description(); got != tc.want {
				t.Errorf("Description() = %q, want %q", got, tc.want)
			}
			decl := declare(t, tl, requestContext(t, tc.legacy, genai.BackendGeminiAPI))
			if decl.Description != tc.want {
				t.Errorf("declaration description = %q, want %q", decl.Description, tc.want)
			}
		})
	}
}

func TestNew_VertexAIAddsNullResponseSchema(t *testing.T) {
	tests := []struct {
		name             string
		legacy           bool
		backend          genai.Backend
		wantResponse     *genai.Schema
		wantResponseJSON map[string]any
	}{
		{name: "Vertex AI JSON", backend: genai.BackendVertexAI, wantResponseJSON: map[string]any{"type": "null"}},
		{name: "Vertex AI typed", legacy: true, backend: genai.BackendVertexAI, wantResponse: &genai.Schema{Type: genai.TypeNULL}},
		{name: "Gemini API JSON", backend: genai.BackendGeminiAPI},
		{name: "Gemini API typed", legacy: true, backend: genai.BackendGeminiAPI},
		{name: "no backend", backend: genai.BackendUnspecified},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{"a"}})
			decl := declare(t, tl, requestContext(t, tc.legacy, tc.backend))
			if diff := cmp.Diff(tc.wantResponse, decl.Response); diff != "" {
				t.Errorf("Response mismatch (-want +got):\n%s", diff)
			}
			var gotResponseJSON map[string]any
			if decl.ResponseJsonSchema != nil {
				gotResponseJSON = asJSON(t, decl.ResponseJsonSchema)
			}
			if diff := cmp.Diff(tc.wantResponseJSON, gotResponseJSON); diff != "" {
				t.Errorf("ResponseJsonSchema mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestNew_FormFollowsEachRequest(t *testing.T) {
	tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{"a"}})
	for i, legacy := range []bool{false, true, false} {
		decl := declare(t, tl, requestContext(t, legacy, genai.BackendGeminiAPI))
		gotTyped := decl.Parameters != nil && decl.ParametersJsonSchema == nil
		gotJSON := decl.Parameters == nil && decl.ParametersJsonSchema != nil
		if legacy && !gotTyped || !legacy && !gotJSON {
			t.Errorf("request %d with switch %v: Parameters = %v, ParametersJsonSchema = %v, want only the %s form", i, legacy, decl.Parameters, decl.ParametersJsonSchema, map[bool]string{true: "typed", false: "JSON"}[legacy])
		}
	}
}

func TestNew_ConcurrentRequestsKeepTheirOwnForm(t *testing.T) {
	tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{"a"}})
	var wg sync.WaitGroup
	for g := range 16 {
		legacy := g%2 == 0
		ctx := requestContext(t, legacy, genai.BackendGeminiAPI)
		wg.Go(func() {
			for range 50 {
				req := &model.LLMRequest{}
				if err := tl.(toolinternal.RequestProcessor).ProcessRequest(ctx, req); err != nil {
					t.Errorf("ProcessRequest() error = %v", err)
					return
				}
				decl := req.Config.Tools[0].FunctionDeclarations[0]
				if typed := decl.Parameters != nil && decl.ParametersJsonSchema == nil; typed != legacy {
					t.Errorf("goroutine %d with switch %v got the typed form = %v", g, legacy, typed)
					return
				}
			}
		})
	}
	wg.Wait()
}

func TestNew_ProcessRequestNilContextDeclaresJSONForm(t *testing.T) {
	tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{"a"}})
	decl := declare(t, tl, nil)
	if decl.Parameters != nil || decl.ParametersJsonSchema == nil {
		t.Errorf("Parameters = %v, ParametersJsonSchema = %v, want only the JSON form", decl.Parameters, decl.ParametersJsonSchema)
	}
	if decl.Response != nil || decl.ResponseJsonSchema != nil {
		t.Errorf("Response = %v, ResponseJsonSchema = %v, want no response schema", decl.Response, decl.ResponseJsonSchema)
	}
}

func TestNew_CopiesAgentNames(t *testing.T) {
	names := []string{"a", "b"}
	tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: names})
	names[0] = "x"

	decl := declare(t, tl, requestContext(t, true, genai.BackendGeminiAPI))
	if diff := cmp.Diff([]string{"a", "b"}, typedEnum(t, decl)); diff != "" {
		t.Errorf("agent_name enum mismatch (-want +got):\n%s", diff)
	}
	ctx := requestContext(t, false, genai.BackendGeminiAPI)
	if _, err := runTool(t, tl, ctx, map[string]any{"agent_name": "x"}); err == nil {
		t.Errorf("Run(agent_name: x) succeeded, want an error: x was never in the tool's list")
	}
}

func TestNew_DeclarationEnumIsACopy(t *testing.T) {
	tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{"a", "b"}})
	first := declare(t, tl, requestContext(t, true, genai.BackendGeminiAPI))
	typedEnum(t, first)[0] = "x"

	second := declare(t, tl, requestContext(t, true, genai.BackendGeminiAPI))
	if diff := cmp.Diff([]string{"a", "b"}, typedEnum(t, second)); diff != "" {
		t.Errorf("second declaration's agent_name enum mismatch (-want +got):\n%s", diff)
	}
	ctx := requestContext(t, false, genai.BackendGeminiAPI)
	if _, err := runTool(t, tl, ctx, map[string]any{"agent_name": "a"}); err != nil {
		t.Errorf("Run(agent_name: a) error = %v, want success", err)
	}
}

func TestNew_NoNamesDeclaresNoEnum(t *testing.T) {
	configs := map[string]transfertoagenttool.Config{
		"nil":   {},
		"empty": {AgentNames: []string{}},
	}
	for name, cfg := range configs {
		t.Run(name, func(t *testing.T) {
			tl := transfertoagenttool.New(cfg)

			jsonDecl := declare(t, tl, requestContext(t, false, genai.BackendGeminiAPI))
			wantJSON := map[string]any{
				"type":       "object",
				"properties": map[string]any{"agent_name": map[string]any{"type": "string"}},
				"required":   []any{"agent_name"},
			}
			if diff := cmp.Diff(wantJSON, asJSON(t, jsonDecl.ParametersJsonSchema)); diff != "" {
				t.Errorf("ParametersJsonSchema mismatch (-want +got):\n%s", diff)
			}

			typedDecl := declare(t, tl, requestContext(t, true, genai.BackendGeminiAPI))
			if got := typedEnum(t, typedDecl); got != nil {
				t.Errorf("typed agent_name enum = %#v, want nil", got)
			}
		})
	}

	t.Run("Run refuses a name outside the tree", func(t *testing.T) {
		tl := transfertoagenttool.New(transfertoagenttool.Config{AgentNames: []string{}, IncludeTransferReason: true})
		root := newTree(t)
		ctx := treeContext(t, root, root)
		_, err := runTool(t, tl, ctx, map[string]any{"agent_name": "ghost", "transfer_reason": "r"})
		if err == nil || err.Error() != `agent "ghost" is not a valid transfer target; available agents: ` {
			t.Errorf("Run(agent_name: ghost) error = %v, want the not-a-valid-target error with no agents listed", err)
		}
		if got := ctx.Actions(); got.TransferToAgent != "" || got.TransferReason != "" {
			t.Errorf("Run(agent_name: ghost) set TransferToAgent = %q, TransferReason = %q, want both unset", got.TransferToAgent, got.TransferReason)
		}
	})
}
