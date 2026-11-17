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

package authtool_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/exp/authtool"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// toolBaseHandler is a BaseHandler that records its calls and returns result.
type toolBaseHandler struct {
	toolRecorder
	result map[string]any
}

func (h *toolBaseHandler) Call(ctx agent.Context, args any, cred *toolauth.Credential, authErr error) (map[string]any, error) {
	h.record(ctx, args, cred, authErr)
	return h.result, nil
}

// toolSchema is the input schema of the base tool tests.
func toolSchema() *jsonschema.Schema {
	return &jsonschema.Schema{
		Type:       "object",
		Properties: map[string]*jsonschema.Schema{"a": {Type: "string"}},
		Required:   []string{"a"},
	}
}

// Test 12. Fails if the handler of a custom tool gets no credential, gets one
// when none is configured, gets other arguments than the model sent, or the
// declaration loses the given schema.
func TestNewBaseTool_HandlerReceivesCredential(t *testing.T) {
	tests := []struct {
		name     string
		auth     *toolauth.Config
		wantCred *toolauth.Credential
	}{
		{name: "api key", auth: apiKeyConfig(), wantCred: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "test-api-key-1"}},
		{name: "no auth config", auth: nil, wantCred: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &toolBaseHandler{result: map[string]any{"k": "v"}}
			tl, err := authtool.NewBaseTool(authtool.Config{
				Config:     functiontool.Config{Name: "custom", Description: "d", InputSchema: toolSchema()},
				AuthConfig: tt.auth,
			}, h)
			if err != nil {
				t.Fatalf("NewBaseTool() error = %v", err)
			}

			got, err := toolRun(t, tl, newToolContext(t, nil), map[string]any{"a": "b"})
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if diff := cmp.Diff(map[string]any{"k": "v"}, got); diff != "" {
				t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
			}
			calls := h.all()
			if len(calls) != 1 {
				t.Fatalf("handler ran %d times, want 1", len(calls))
			}
			if diff := cmp.Diff(map[string]any{"a": "b"}, calls[0].args); diff != "" {
				t.Errorf("handler args mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantCred, calls[0].cred); diff != "" || calls[0].authErr != nil {
				t.Errorf("handler credential mismatch (-want +got):\n%s\nerror = %v, want nil", diff, calls[0].authErr)
			}
			decl := toolDeclaration(t, tl)
			if diff := cmp.Diff(toolSchema(), decl.ParametersJsonSchema); diff != "" || decl.Name != "custom" || decl.Description != "d" {
				t.Errorf("declaration = %s %q, parameters mismatch (-want +got):\n%s", decl.Name, decl.Description, diff)
			}
		})
	}
}

// Test 13. Fails if the custom tool runs the handler without a credential,
// skips the request, or answers with something other than the
// auth-required map.
func TestNewBaseTool_AuthRequiredResponse(t *testing.T) {
	tests := []struct {
		name     string
		response any
		want     map[string]any
	}{
		{name: "default", response: nil, want: map[string]any{"result": pending}},
		{name: "string", response: "p", want: map[string]any{"result": "p"}},
		{name: "map", response: map[string]any{"status": "wait"}, want: map[string]any{"status": "wait"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := &toolBaseHandler{result: map[string]any{"k": "v"}}
			tl, err := authtool.NewBaseTool(authtool.Config{
				Config:                  functiontool.Config{Name: "custom", InputSchema: toolSchema()},
				AuthConfig:              toolNoRawCredential(),
				ResponseForAuthRequired: tt.response,
			}, h)
			if err != nil {
				t.Fatalf("NewBaseTool() error = %v", err)
			}
			ctx := newToolContext(t, nil)

			got, err := toolRun(t, tl, ctx, map[string]any{"a": "b"})
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
			}
			if n := len(h.all()); n != 0 {
				t.Errorf("handler ran %d times, want 0", n)
			}
			if _, ok := ctx.actions.RequestedAuthConfigs["call-1"]; len(ctx.actions.RequestedAuthConfigs) != 1 || !ok {
				t.Errorf("RequestedAuthConfigs = %v, want one entry for call-1", ctx.actions.RequestedAuthConfigs)
			}
		})
	}
}

// Test 14. Fails if NewBaseTool builds a tool over a schema nobody can
// validate against, or over a confirmation setting it would ignore.
func TestNewBaseTool_InvalidConfig(t *testing.T) {
	tests := []struct {
		name string
		cfg  functiontool.Config
	}{
		{name: "bad pattern", cfg: functiontool.Config{InputSchema: &jsonschema.Schema{
			Type: "object", Properties: map[string]*jsonschema.Schema{"a": {Type: "string", Pattern: "["}},
		}}},
		{name: "dangling reference", cfg: functiontool.Config{InputSchema: &jsonschema.Schema{Ref: "#/$defs/missing"}}},
		{name: "bad output schema", cfg: functiontool.Config{InputSchema: toolSchema(), OutputSchema: &jsonschema.Schema{Ref: "#/$defs/missing"}}},
		{name: "require confirmation", cfg: functiontool.Config{InputSchema: toolSchema(), RequireConfirmation: true}},
		{name: "confirmation provider", cfg: functiontool.Config{InputSchema: toolSchema(), RequireConfirmationProvider: func(any) bool { return true }}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.cfg.Name = "custom"
			tl, err := authtool.NewBaseTool(authtool.Config{Config: tt.cfg, AuthConfig: apiKeyConfig()}, &toolBaseHandler{})
			if tl != nil || err == nil {
				t.Errorf("NewBaseTool() = %v, %v; want nil and an error", tl, err)
			}
		})
	}
}

// nilResultHandler returns a nil map.
type nilResultHandler struct{}

func (nilResultHandler) Call(agent.Context, any, *toolauth.Credential, error) (map[string]any, error) {
	return nil, nil
}

// Fails if a custom tool's declaration drops the output schema, if it does not
// declare itself to the model in a request, or if RunRaw returns a nil map
// as a non-nil interface (the flow tells "no result" by comparing with nil).
func TestNewBaseTool_DeclaresItselfAndKeepsNilResult(t *testing.T) {
	out := &jsonschema.Schema{Type: "object", Properties: map[string]*jsonschema.Schema{"k": {Type: "string"}}}
	tl, err := authtool.NewBaseTool(authtool.Config{
		Config: functiontool.Config{Name: "custom", Description: "d", InputSchema: toolSchema(), OutputSchema: out},
	}, nilResultHandler{})
	if err != nil {
		t.Fatalf("NewBaseTool() error = %v", err)
	}

	decl := toolDeclaration(t, tl)
	if diff := cmp.Diff(out, decl.ResponseJsonSchema); diff != "" {
		t.Errorf("declaration response schema mismatch (-want +got):\n%s", diff)
	}

	req := &model.LLMRequest{}
	p, ok := tl.(interface {
		ProcessRequest(agent.Context, *model.LLMRequest) error
	})
	if !ok {
		t.Fatalf("tool %T has no ProcessRequest", tl)
	}
	if err := p.ProcessRequest(newToolContext(t, nil), req); err != nil {
		t.Fatalf("ProcessRequest() error = %v", err)
	}
	if req.Tools["custom"] == nil || req.Config == nil || len(req.Config.Tools) != 1 ||
		len(req.Config.Tools[0].FunctionDeclarations) != 1 || req.Config.Tools[0].FunctionDeclarations[0].Name != "custom" {
		t.Errorf("request after ProcessRequest = %+v, want the tool in Tools and declared as custom", req)
	}

	raw, ok := tl.(interface {
		RunRaw(agent.Context, any) (any, error)
	})
	if !ok {
		t.Fatalf("tool %T has no RunRaw", tl)
	}
	got, err := raw.RunRaw(newToolContext(t, nil), map[string]any{"a": "b"})
	if err != nil || got != nil {
		t.Errorf("RunRaw() = %#v, %v, want an untyped nil and no error", got, err)
	}
}
