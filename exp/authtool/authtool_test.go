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
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/credentialservice"
	"google.golang.org/adk/v2/exp/authtool"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// toolRunner is the part of a tool the LLM flow calls.
type toolRunner interface {
	Declaration() *genai.FunctionDeclaration
	Run(ctx agent.Context, args any) (map[string]any, error)
}

// toolRun calls tl as the flow does.
func toolRun(t *testing.T, tl tool.Tool, ctx agent.Context, args map[string]any) (map[string]any, error) {
	t.Helper()
	r, ok := tl.(toolRunner)
	if !ok {
		t.Fatalf("tool %T has no Declaration and Run methods", tl)
	}
	return r.Run(ctx, args)
}

// toolDeclaration returns tl's declaration.
func toolDeclaration(t *testing.T, tl tool.Tool) *genai.FunctionDeclaration {
	t.Helper()
	r, ok := tl.(toolRunner)
	if !ok {
		t.Fatalf("tool %T has no Declaration and Run methods", tl)
	}
	return r.Declaration()
}

// toolProperties returns the sorted property names and the required names of
// decl's parameters.
func toolProperties(t *testing.T, decl *genai.FunctionDeclaration) (props, required []string) {
	t.Helper()
	s, ok := decl.ParametersJsonSchema.(*jsonschema.Schema)
	if !ok {
		t.Fatalf("ParametersJsonSchema is %T, want *jsonschema.Schema", decl.ParametersJsonSchema)
	}
	for name := range s.Properties {
		props = append(props, name)
	}
	slices.Sort(props)
	return props, s.Required
}

// toolCall is what one call of a handler received.
type toolCall struct {
	user    string
	args    any
	cred    *toolauth.Credential
	authErr error
}

// toolRecorder records the calls of a handler; it is safe for concurrent use.
type toolRecorder struct {
	mu    sync.Mutex
	calls []toolCall
}

func (r *toolRecorder) record(ctx agent.Context, args any, cred *toolauth.Credential, authErr error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, toolCall{user: ctx.UserID(), args: args, cred: cred, authErr: authErr})
}

func (r *toolRecorder) all() []toolCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.calls)
}

type twoArgs struct {
	Arg1 string `json:"arg1"`
	Arg2 int    `json:"arg2"`
}

type oneArg struct {
	Arg1 string `json:"arg1"`
}

type noArgs struct{}

// toolNoRawCredential is an API-key config with no key, so the manager has
// no credential and the tool asks the user for one.
func toolNoRawCredential() *toolauth.Config {
	cfg := apiKeyConfig()
	cfg.RawAuthCredential = nil
	return cfg
}

const pending = "Pending User Authorization."

// Test 1. Fails if a manager is built for a missing or zero scheme, the
// handler gets a credential, or the declaration changes.
func TestNew_WithoutAuthSchemeIsPlainTool(t *testing.T) {
	tests := []struct {
		name string
		auth *toolauth.Config
	}{
		{name: "nil auth config", auth: nil},
		{name: "zero scheme", auth: &toolauth.Config{RawAuthCredential: apiKeyConfig().RawAuthCredential}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rec toolRecorder
			tl, err := authtool.New(authtool.Config{
				Config:     functiontool.Config{Name: "f", Description: "d"},
				AuthConfig: tt.auth,
			}, func(ctx agent.Context, a twoArgs, cred *toolauth.Credential, authErr error) (string, error) {
				rec.record(ctx, a, cred, authErr)
				return fmt.Sprintf("sync_%s_%d", a.Arg1, a.Arg2), nil
			})
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			ctx := newToolContext(t, nil)

			got, err := toolRun(t, tl, ctx, map[string]any{"arg1": "x", "arg2": 2})
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if diff := cmp.Diff(map[string]any{"result": "sync_x_2"}, got); diff != "" {
				t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
			}
			calls := rec.all()
			if len(calls) != 1 || calls[0].cred != nil || calls[0].authErr != nil {
				t.Errorf("handler calls = %+v, want one with a nil credential and a nil error", calls)
			}
			if n := len(ctx.actions.RequestedAuthConfigs); n != 0 {
				t.Errorf("RequestedAuthConfigs has %d entries, want 0", n)
			}
			decl := toolDeclaration(t, tl)
			props, required := toolProperties(t, decl)
			if decl.Name != "f" || decl.Description != "d" ||
				!slices.Equal(props, []string{"arg1", "arg2"}) || !slices.Equal(required, []string{"arg1", "arg2"}) {
				t.Errorf("declaration = %s %q, properties %v, required %v; want f \"d\", [arg1 arg2], [arg1 arg2]",
					decl.Name, decl.Description, props, required)
			}
		})
	}
}

// Test 2. Fails if New drops functiontool's check of the argument type, with
// or without an auth config.
func TestNew_InvalidArgsTypeIsRejected(t *testing.T) {
	tests := []struct {
		name string
		auth *toolauth.Config
	}{
		{name: "no auth config", auth: nil},
		{name: "api key", auth: apiKeyConfig()},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "f"}, AuthConfig: tt.auth},
				func(agent.Context, string, *toolauth.Credential, error) (string, error) { return "", nil })
			if tl != nil || !errors.Is(err, functiontool.ErrInvalidArgument) {
				t.Errorf("New() = %v, %v; want nil and an error wrapping functiontool.ErrInvalidArgument", tl, err)
			}
		})
	}
}

// Fails if New accepts a config the tool cannot serve: an auth-required
// response that is neither a string nor a map, or a scheme without a type
// that is not the zero scheme.
func TestNew_RejectsConfigItCannotServe(t *testing.T) {
	tests := []struct {
		name string
		cfg  authtool.Config
	}{
		{name: "response of another type", cfg: authtool.Config{AuthConfig: apiKeyConfig(), ResponseForAuthRequired: 42}},
		{name: "scheme without type", cfg: authtool.Config{AuthConfig: &toolauth.Config{AuthScheme: toolauth.Scheme{Description: "d"}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.cfg.Name = "f"
			tl, err := authtool.New(tt.cfg, func(agent.Context, noArgs, *toolauth.Credential, error) (string, error) { return "", nil })
			if tl != nil || err == nil {
				t.Errorf("New() = %v, %v; want nil and an error", tl, err)
			}
		})
	}
}

// Test 3. Fails if the tool declares a credential parameter, calls the
// handler without the credential, or passes an error on success.
func TestNew_HandlerReceivesCredentialAndDeclarationHidesIt(t *testing.T) {
	var rec toolRecorder
	tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "f"}, AuthConfig: apiKeyConfig()},
		func(ctx agent.Context, a oneArg, cred *toolauth.Credential, authErr error) (map[string]any, error) {
			rec.record(ctx, a, cred, authErr)
			return map[string]any{"k": "ok"}, nil
		})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	got, err := toolRun(t, tl, newToolContext(t, nil), map[string]any{"arg1": "v"})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if diff := cmp.Diff(map[string]any{"k": "ok"}, got); diff != "" {
		t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
	}
	calls := rec.all()
	if len(calls) != 1 {
		t.Fatalf("handler ran %d times, want 1", len(calls))
	}
	c := calls[0]
	if c.args != (oneArg{Arg1: "v"}) || c.authErr != nil || c.cred == nil ||
		c.cred.AuthType != toolauth.CredentialTypeAPIKey || c.cred.APIKey != "test-api-key-1" {
		t.Errorf("handler got args %+v, credential %+v, error %v; want {v}, the API key test-api-key-1 and nil", c.args, c.cred, c.authErr)
	}
	if props, _ := toolProperties(t, toolDeclaration(t, tl)); !slices.Equal(props, []string{"arg1"}) {
		t.Errorf("declaration properties = %v, want [arg1]", props)
	}
}

// Test 4. adk-python gets the credential before FunctionTool checks the
// mandatory arguments (tools/authenticated_function_tool.py:85-91). Fails if
// the argument check runs first.
func TestNew_CredentialProcuredBeforeArgumentCheck(t *testing.T) {
	var rec toolRecorder
	tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "f"}, AuthConfig: toolNoRawCredential()},
		func(ctx agent.Context, a twoArgs, cred *toolauth.Credential, authErr error) (string, error) {
			rec.record(ctx, a, cred, authErr)
			return "ran", nil
		})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx := newToolContext(t, nil)

	got, err := toolRun(t, tl, ctx, map[string]any{"arg1": "x"})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if diff := cmp.Diff(map[string]any{"result": pending}, got); diff != "" {
		t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
	}
	if _, ok := ctx.actions.RequestedAuthConfigs["call-1"]; !ok {
		t.Errorf("RequestedAuthConfigs has no entry for call-1")
	}
	if n := len(rec.all()); n != 0 {
		t.Errorf("handler ran %d times, want 0", n)
	}
}

// Test 5. Fails if the function runs without a credential, the request is
// missing or recorded under another call, or the default text changes.
func TestNew_NoCredentialRequestsAuth(t *testing.T) {
	var rec toolRecorder
	tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "f"}, AuthConfig: codeConfig("https://idp.example/token")},
		func(ctx agent.Context, a noArgs, cred *toolauth.Credential, authErr error) (string, error) {
			rec.record(ctx, a, cred, authErr)
			return "ran", nil
		})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx := newToolContext(t, nil)

	got, err := toolRun(t, tl, ctx, map[string]any{})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if diff := cmp.Diff(map[string]any{"result": pending}, got); diff != "" {
		t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
	}
	if n := len(rec.all()); n != 0 {
		t.Errorf("handler ran %d times, want 0", n)
	}
	reqs := ctx.actions.RequestedAuthConfigs
	req, ok := reqs["call-1"]
	if len(reqs) != 1 || !ok || req.AuthScheme.Type != toolauth.SchemeTypeOAuth2 {
		t.Errorf("RequestedAuthConfigs = %d entries, call-1 present = %v with scheme %q; want one oauth2 entry for call-1",
			len(reqs), ok, req.AuthScheme.Type)
	}
}

// Test 6. adk-python answers response_for_auth_required or the default, so an
// empty string or map gives the default (tools/authenticated_function_tool.py:91).
// Fails if the default is chosen by a nil check only, or a string is not
// wrapped.
func TestNew_ResponseForAuthRequired(t *testing.T) {
	tests := []struct {
		name     string
		response any
		want     map[string]any
	}{
		{name: "nil", response: nil, want: map[string]any{"result": pending}},
		{name: "string", response: "custom pending", want: map[string]any{"result": "custom pending"}},
		{name: "map", response: map[string]any{"status": "wait"}, want: map[string]any{"status": "wait"}},
		{name: "empty string", response: "", want: map[string]any{"result": pending}},
		{name: "empty map", response: map[string]any{}, want: map[string]any{"result": pending}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tl, err := authtool.New(authtool.Config{
				Config:                  functiontool.Config{Name: "f"},
				AuthConfig:              toolNoRawCredential(),
				ResponseForAuthRequired: tt.response,
			}, func(agent.Context, noArgs, *toolauth.Credential, error) (string, error) { return "ran", nil })
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}

			got, err := toolRun(t, tl, newToolContext(t, nil), map[string]any{})
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("nested map is detached from caller and across concurrent calls", func(t *testing.T) {
		nested := map[string]any{"code": "AUTH_REQUIRED"}
		tl, err := authtool.New(authtool.Config{
			Config:                  functiontool.Config{Name: "f"},
			AuthConfig:              toolNoRawCredential(),
			ResponseForAuthRequired: map[string]any{"error": nested},
		}, func(agent.Context, noArgs, *toolauth.Credential, error) (string, error) { return "ran", nil })
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		nested["code"] = "MUTATED_BY_CALLER"

		var wg sync.WaitGroup
		for i := range 8 {
			wg.Go(func() {
				got, err := toolRun(t, tl, newToolContext(t, nil), map[string]any{})
				if err != nil {
					t.Errorf("Run() error = %v", err)
					return
				}
				inner, _ := got["error"].(map[string]any)
				if inner["code"] != "AUTH_REQUIRED" {
					t.Errorf("Run() error.code = %v, want AUTH_REQUIRED", inner["code"])
				}
				inner["code"] = fmt.Sprintf("mutated-%d", i)
			})
		}
		wg.Wait()
	})
}

// Test 7. Fails if the auth-required answer has to pass through TResults.
func TestNew_AuthRequiredWithTypedResults(t *testing.T) {
	cfg := authtool.Config{Config: functiontool.Config{Name: "f"}, AuthConfig: toolNoRawCredential()}
	structTool, err := authtool.New(cfg, func(agent.Context, noArgs, *toolauth.Credential, error) (struct{ N int }, error) {
		return struct{ N int }{N: 1}, nil
	})
	if err != nil {
		t.Fatalf("New() with struct results error = %v", err)
	}
	stringTool, err := authtool.New(cfg, func(agent.Context, noArgs, *toolauth.Credential, error) (string, error) {
		return "ran", nil
	})
	if err != nil {
		t.Fatalf("New() with string results error = %v", err)
	}
	for name, tl := range map[string]tool.Tool{"struct": structTool, "string": stringTool} {
		got, err := toolRun(t, tl, newToolContext(t, nil), map[string]any{})
		if err != nil {
			t.Fatalf("%s tool: Run() error = %v", name, err)
		}
		if diff := cmp.Diff(map[string]any{"result": pending}, got); diff != "" {
			t.Errorf("%s tool: Run() result mismatch (-want +got):\n%s", name, diff)
		}
	}
}

// Test 8. Fails if the tool swallows the error of RequestCredential and
// answers that it is waiting, when no request was recorded (decision A8).
func TestNew_AuthRequestNeedsFunctionCall(t *testing.T) {
	var rec toolRecorder
	tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "f"}, AuthConfig: toolNoRawCredential()},
		func(ctx agent.Context, a noArgs, cred *toolauth.Credential, authErr error) (string, error) {
			rec.record(ctx, a, cred, authErr)
			return "ran", nil
		})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	ctx := newToolContext(t, nil)
	ctx.callID = ""

	got, err := toolRun(t, tl, ctx, map[string]any{})
	if err == nil || got != nil {
		t.Errorf("Run() = %v, %v; want nil and an error", got, err)
	}
	if n := len(rec.all()); n != 0 {
		t.Errorf("handler ran %d times, want 0", n)
	}
}

var errTool = errors.New("tool failed")

// Test 9. Fails if an error of the manager or of the handler is replaced by a
// result, or wrapped without %w.
func TestNew_ErrorsPropagate(t *testing.T) {
	t.Run("manager error", func(t *testing.T) {
		var rec toolRecorder
		cfg := codeConfig("https://idp.example/token")
		cfg.RawAuthCredential = nil
		tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "f"}, AuthConfig: cfg},
			func(ctx agent.Context, a noArgs, cred *toolauth.Credential, authErr error) (string, error) {
				rec.record(ctx, a, cred, authErr)
				return "ran", nil
			})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}

		got, err := toolRun(t, tl, newToolContext(t, nil), map[string]any{})
		if got != nil || err == nil || err.Error() != "raw_auth_credential is required for auth_scheme type oauth2" {
			t.Errorf("Run() = %v, %v; want nil and the manager's error", got, err)
		}
		if n := len(rec.all()); n != 0 {
			t.Errorf("handler ran %d times, want 0", n)
		}
	})
	t.Run("handler error", func(t *testing.T) {
		tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "f"}, AuthConfig: apiKeyConfig()},
			func(agent.Context, noArgs, *toolauth.Credential, error) (string, error) { return "", errTool })
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}

		if _, err := toolRun(t, tl, newToolContext(t, nil), map[string]any{}); !errors.Is(err, errTool) {
			t.Errorf("Run() error = %v, want errTool", err)
		}
	})
}

// Test 10. A failed exchange leaves the credential unexchanged, as in
// adk-python, and the handler gets the error beside it. Fails if the tool
// aborts, drops the error, or the error carries the client secret.
func TestNew_SoftExchangeErrorReachesHandler(t *testing.T) {
	srv := newTokenServer(t)
	srv.set(func(s *tokenServer) { s.status, s.body = http.StatusBadRequest, invalidClient })
	var rec toolRecorder
	tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "f"}, AuthConfig: ccConfig(srv.URL)},
		func(ctx agent.Context, a noArgs, cred *toolauth.Credential, authErr error) (string, error) {
			rec.record(ctx, a, cred, authErr)
			return "ran", nil
		})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	got, err := toolRun(t, tl, newToolContext(t, nil), map[string]any{})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if diff := cmp.Diff(map[string]any{"result": "ran"}, got); diff != "" {
		t.Errorf("Run() result mismatch (-want +got):\n%s", diff)
	}
	calls := rec.all()
	if len(calls) != 1 {
		t.Fatalf("handler ran %d times, want 1", len(calls))
	}
	c := calls[0]
	if c.cred == nil || c.cred.OAuth2 == nil || c.cred.OAuth2.ClientID != "client-1" || c.cred.OAuth2.AccessToken != "" {
		t.Errorf("handler got credential %+v, want the unexchanged client-1 credential", c.cred)
	}
	if c.authErr == nil || !strings.Contains(c.authErr.Error(), "invalid_client") || strings.Contains(c.authErr.Error(), "secret-1") {
		t.Errorf("handler got error %v, want one naming invalid_client without the client secret", c.authErr)
	}
}

// Test 11. Fails if tools sharing one config do not share the process-wide
// lock, so each redeems the client credentials (decision A2).
func TestNew_ToolsSharingAConfigExchangeOnce(t *testing.T) {
	srv := newTokenServer(t)
	srv.set(func(s *tokenServer) { s.delay = 100 * time.Millisecond })
	cfg := ccConfig(srv.URL)
	svc := newFakeService()
	var rec toolRecorder
	newCCTool := func() tool.Tool {
		tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "f"}, AuthConfig: cfg},
			func(ctx agent.Context, a noArgs, cred *toolauth.Credential, authErr error) (string, error) {
				rec.record(ctx, a, cred, authErr)
				return "ran", nil
			})
		if err != nil {
			t.Fatalf("New() error = %v", err)
		}
		return tl
	}
	tools := []tool.Tool{newCCTool(), newCCTool()}
	ctxs := make([]*fakeToolContext, 8)
	for i := range ctxs {
		ctxs[i] = newToolContext(t, svc)
	}

	var wg sync.WaitGroup
	errs := make([]error, len(ctxs))
	for i, ctx := range ctxs {
		wg.Go(func() { _, errs[i] = toolRun(t, tools[i%2], ctx, map[string]any{}) })
	}
	wg.Wait()

	if err := errors.Join(errs...); err != nil {
		t.Fatalf("Run() errors: %v", err)
	}
	if n := len(srv.seen()); n != 1 {
		t.Errorf("token requests = %d, want 1", n)
	}
	calls := rec.all()
	if len(calls) != 8 {
		t.Fatalf("handlers ran %d times, want 8", len(calls))
	}
	for _, c := range calls {
		if got := accessToken(c.cred); got != "cc-access" {
			t.Errorf("a handler got access token %q, want cc-access", got)
		}
	}
}

// Test 42. The flow calls the tool its call table holds, which ProcessRequest
// fills. Fails if the wrapper leaves the inner function tool there, so the
// flow runs the function without asking for the credential (decision A1).
func TestNew_FlowCallsTheWrapper(t *testing.T) {
	var rec toolRecorder
	tl, err := authtool.New(authtool.Config{
		Config:     functiontool.Config{Name: "get_data", Description: "Returns the data."},
		AuthConfig: toolNoRawCredential(),
	}, func(ctx agent.Context, a noArgs, cred *toolauth.Credential, authErr error) (string, error) {
		rec.record(ctx, a, cred, authErr)
		return "data", nil
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	llm := &testutil.MockModel{Responses: []*genai.Content{
		genai.NewContentFromFunctionCall("get_data", map[string]any{}, genai.RoleModel),
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: llm, Tools: []tool.Tool{tl}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	r := testutil.NewTestAgentRunner(t, a)

	events, err := testutil.CollectEvents(r.Run(t, "s1", "get the data"))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	var toolCallID string
	var requests []*genai.FunctionCall
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			switch {
			case p.FunctionCall == nil:
			case p.FunctionCall.Name == "get_data":
				toolCallID = p.FunctionCall.ID
			case p.FunctionCall.Name == toolauth.RequestCredentialFunctionCallName:
				requests = append(requests, p.FunctionCall)
			}
		}
	}
	if len(requests) != 1 {
		t.Fatalf("run sent %d %s calls, want 1", len(requests), toolauth.RequestCredentialFunctionCallName)
	}
	if got := requests[0].Args["functionCallId"]; toolCallID == "" || got != toolCallID {
		t.Errorf("credential request names call %v, want the get_data call %q", got, toolCallID)
	}
	if n := len(rec.all()); n != 0 {
		t.Errorf("handler ran %d times, want 0", n)
	}
}

// toolStoreToken stores an OAuth2 credential holding token for user under
// cfg's key in svc.
func toolStoreToken(t *testing.T, svc credentialservice.Service, cfg *toolauth.Config, user, token string) {
	t.Helper()
	c := *cfg
	c.ExchangedAuthCredential = &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{AccessToken: token}}
	if err := svc.SaveCredential(t.Context(), "app-1", user, &c); err != nil {
		t.Fatalf("SaveCredential() error = %v", err)
	}
}

// Test 43. One tool, two users calling at once, each with a credential of
// their own in the credential service. Fails, under -race or by a wrong
// token, if the call's credential is kept in a field of the tool or the
// manager.
func TestNew_ConcurrentCallsGetTheirOwnCredential(t *testing.T) {
	cfg := codeConfig("https://idp.example/token")
	svc := credentialservice.InMemoryService()
	want := map[string]string{"user-1": "user-1-access", "user-2": "user-2-access"}
	for user, token := range want {
		toolStoreToken(t, svc, cfg, user, token)
	}
	var rec toolRecorder
	tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "f"}, AuthConfig: cfg},
		func(ctx agent.Context, a noArgs, cred *toolauth.Credential, authErr error) (string, error) {
			rec.record(ctx, a, cred, authErr)
			return "ran", nil
		})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	const perUser = 10
	var ctxs []*fakeToolContext
	for i := range 2 * perUser {
		ctx := newToolContext(t, svc)
		ctx.userID = fmt.Sprintf("user-%d", i%2+1)
		ctx.callID = fmt.Sprintf("call-%d", i)
		ctxs = append(ctxs, ctx)
	}

	start := make(chan struct{})
	var wg sync.WaitGroup
	errs := make([]error, len(ctxs))
	for i, ctx := range ctxs {
		wg.Go(func() {
			<-start
			_, errs[i] = toolRun(t, tl, ctx, map[string]any{})
		})
	}
	close(start)
	wg.Wait()

	if err := errors.Join(errs...); err != nil {
		t.Fatalf("Run() errors: %v", err)
	}
	calls := rec.all()
	if len(calls) != 2*perUser {
		t.Fatalf("handlers ran %d times, want %d", len(calls), 2*perUser)
	}
	for _, c := range calls {
		if got := accessToken(c.cred); got != want[c.user] {
			t.Errorf("a call of %s got access token %q, want %q", c.user, got, want[c.user])
		}
	}
}

// The function gets the tool context ADK built, with the credential added, so
// the helpers that need one of ADK's own contexts work in it as in a plain
// function tool. Fails if the call's context hides them, as an embedding
// wrapper does: agent.RenderUIWidget then returns agent.ErrUnsupportedContext.
func TestNew_HandlerKeepsTheToolContext(t *testing.T) {
	tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "show"}, AuthConfig: apiKeyConfig()},
		func(ctx agent.Context, _ noArgs, cred *toolauth.Credential, _ error) (string, error) {
			if cred == nil {
				return "", errors.New("no credential")
			}
			if err := agent.RenderUIWidget(ctx, session.UIWidget{ID: "w1", Provider: "p"}); err != nil {
				return "", err
			}
			return "shown", nil
		})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	llm := &testutil.MockModel{Responses: []*genai.Content{
		genai.NewContentFromFunctionCall("show", map[string]any{}, genai.RoleModel),
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: llm, Tools: []tool.Tool{tl}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	events, err := testutil.CollectEvents(testutil.NewTestAgentRunner(t, a).Run(t, "s1", "show it"))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	var resp *genai.FunctionResponse
	var widgets []session.UIWidget
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil && p.FunctionResponse.Name == "show" {
				resp, widgets = p.FunctionResponse, ev.Actions.RenderUIWidgets
			}
		}
	}
	if resp == nil {
		t.Fatalf("run has no function response of show")
	}
	if diff := cmp.Diff(map[string]any{"result": "shown"}, resp.Response); diff != "" {
		t.Errorf("function response mismatch (-want +got):\n%s", diff)
	}
	if len(widgets) != 1 || widgets[0].ID != "w1" {
		t.Errorf("response event widgets = %+v, want w1", widgets)
	}
}

// The function records an artifact delta on its context, as in a plain
// function tool, and the delta reaches the function response event. Fails if
// the context the function receives hides the tool context's recorder, so
// that agent.MergeArtifactDelta returns errors.ErrUnsupported.
func TestNew_HandlerMergesArtifactDelta(t *testing.T) {
	tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "report"}, AuthConfig: apiKeyConfig()},
		func(ctx agent.Context, _ noArgs, cred *toolauth.Credential, _ error) (string, error) {
			if cred == nil {
				return "", errors.New("no credential")
			}
			if err := agent.MergeArtifactDelta(ctx, map[string]int64{"report.txt": 1}); err != nil {
				return "", err
			}
			return "saved", nil
		})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	llm := &testutil.MockModel{Responses: []*genai.Content{
		genai.NewContentFromFunctionCall("report", map[string]any{}, genai.RoleModel),
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: llm, Tools: []tool.Tool{tl}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}

	events, err := testutil.CollectEvents(testutil.NewTestAgentRunner(t, a).Run(t, "s1", "save the report"))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	var resp *genai.FunctionResponse
	var delta map[string]int64
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil && p.FunctionResponse.Name == "report" {
				resp, delta = p.FunctionResponse, ev.Actions.ArtifactDelta
			}
		}
	}
	if resp == nil {
		t.Fatalf("run has no function response of report")
	}
	if diff := cmp.Diff(map[string]any{"result": "saved"}, resp.Response); diff != "" {
		t.Errorf("function response mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string]int64{"report.txt": 1}, delta); diff != "" {
		t.Errorf("response event artifact delta mismatch (-want +got):\n%s", diff)
	}
}

// The flow reads these settings from the tool it calls, which is the
// wrapper. Fails if the wrapper or the custom tool drops one of them.
func TestNew_KeepsFunctionToolSettings(t *testing.T) {
	cfg := functiontool.Config{
		Name:               "f",
		InputSchema:        toolSchema(),
		IsLongRunning:      true,
		ResponseScheduling: genai.FunctionResponseSchedulingWhenIdle,
		Sequential:         true,
	}
	plain, err := authtool.New(authtool.Config{Config: cfg, AuthConfig: apiKeyConfig()},
		func(agent.Context, map[string]any, *toolauth.Credential, error) (string, error) { return "", nil })
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	custom, err := authtool.NewBaseTool(authtool.Config{Config: cfg, AuthConfig: apiKeyConfig()}, &toolBaseHandler{})
	if err != nil {
		t.Fatalf("NewBaseTool() error = %v", err)
	}
	for name, tl := range map[string]tool.Tool{"New": plain, "NewBaseTool": custom} {
		sched, _ := tl.(tool.ResponseScheduler)
		seq, _ := tl.(tool.SequentialRunner)
		if !tl.IsLongRunning() || sched == nil || sched.ResponseScheduling() != genai.FunctionResponseSchedulingWhenIdle ||
			seq == nil || !seq.RunsSequentially() {
			t.Errorf("%s tool: long-running %v, scheduling %v, sequential %v; want true, WHEN_IDLE, true",
				name, tl.IsLongRunning(), sched, seq)
		}
	}
}

// A tool without an auth config that runs inside another tool's function, with
// that function's context, gets no credential. Fails if the inner tool's
// function can read the outer tool's credential from the context.
func TestNew_CredentialStaysWithItsTool(t *testing.T) {
	var innerRec toolRecorder
	plain, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "plain"}},
		func(ctx agent.Context, a noArgs, cred *toolauth.Credential, authErr error) (string, error) {
			innerRec.record(ctx, a, cred, authErr)
			return "plain", nil
		})
	if err != nil {
		t.Fatalf("New(plain) error = %v", err)
	}
	outer, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "outer"}, AuthConfig: apiKeyConfig()},
		func(ctx agent.Context, _ noArgs, _ *toolauth.Credential, _ error) (map[string]any, error) {
			return plain.(toolRunner).Run(ctx, map[string]any{})
		})
	if err != nil {
		t.Fatalf("New(outer) error = %v", err)
	}

	if _, err := toolRun(t, outer, newToolContext(t, nil), map[string]any{}); err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if calls := innerRec.all(); len(calls) != 1 || calls[0].cred != nil {
		t.Errorf("inner tool calls = %+v, want one with a nil credential", calls)
	}
}

// Fails if the model is not told that the tool waits for the user: the
// function response of a call that needed a credential must carry the
// auth-required result.
func TestNew_FlowAnswersAuthRequiredToTheModel(t *testing.T) {
	tl, err := authtool.New(authtool.Config{
		Config:     functiontool.Config{Name: "get_data", Description: "Returns the data."},
		AuthConfig: toolNoRawCredential(),
	}, func(agent.Context, noArgs, *toolauth.Credential, error) (string, error) { return "data", nil })
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	llm := &testutil.MockModel{Responses: []*genai.Content{
		genai.NewContentFromFunctionCall("get_data", map[string]any{}, genai.RoleModel),
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: llm, Tools: []tool.Tool{tl}})
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	events, err := testutil.CollectEvents(testutil.NewTestAgentRunner(t, a).Run(t, "s1", "get the data"))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}
	var got []map[string]any
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil && p.FunctionResponse.Name == "get_data" {
				got = append(got, p.FunctionResponse.Response)
			}
		}
	}
	want := []map[string]any{{"result": "Pending User Authorization."}}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("get_data function responses mismatch (-want +got):\n%s", diff)
	}
}

// Fails if the wrapper hands out the one auth-required map it holds, so that
// a callback that edits a response changes the next call's answer.
func TestNew_AuthRequiredResultIsNotShared(t *testing.T) {
	tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "f", Description: "d"}, AuthConfig: toolNoRawCredential()},
		func(agent.Context, noArgs, *toolauth.Credential, error) (string, error) { return "x", nil })
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	first, err := toolRun(t, tl, newToolContext(t, nil), map[string]any{})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	first["result"] = "edited"
	second, err := toolRun(t, tl, newToolContext(t, nil), map[string]any{})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if diff := cmp.Diff(map[string]any{"result": "Pending User Authorization."}, second); diff != "" {
		t.Errorf("second Run() result mismatch (-want +got):\n%s", diff)
	}
}

// Fails if the wrapper's Description is not the function tool's.
func TestNew_DescriptionIsTheConfigs(t *testing.T) {
	tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "f", Description: "Returns the data."}},
		func(agent.Context, noArgs, *toolauth.Credential, error) (string, error) { return "x", nil })
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if got := tl.Description(); got != "Returns the data." {
		t.Errorf("Description() = %q, want %q", got, "Returns the data.")
	}
}

// Fails if the wrapper hides the function tool's refusal to declare the same
// name twice in one request.
func TestNew_DuplicateToolNameIsAnError(t *testing.T) {
	tl, err := authtool.New(authtool.Config{Config: functiontool.Config{Name: "f", Description: "d"}},
		func(agent.Context, noArgs, *toolauth.Credential, error) (string, error) { return "x", nil })
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	p, ok := tl.(interface {
		ProcessRequest(agent.Context, *model.LLMRequest) error
	})
	if !ok {
		t.Fatalf("tool %T has no ProcessRequest", tl)
	}
	req := &model.LLMRequest{}
	if err := p.ProcessRequest(newToolContext(t, nil), req); err != nil {
		t.Fatalf("first ProcessRequest() error = %v", err)
	}
	if err := p.ProcessRequest(newToolContext(t, nil), req); err == nil {
		t.Error("second ProcessRequest() error = nil, want an error for the duplicate tool name")
	}
}
