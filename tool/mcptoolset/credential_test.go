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

package mcptoolset

import (
	"context"
	"errors"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/auth"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/authconsent"
)

// First invocation with a provider that needs consent: the tool requests
// credential consent, marks the pending sentinel, and does not call the server.
func TestMcpToolRun_RequestsConsent(t *testing.T) {
	client := &fakeCredClient{}
	tl := &mcpTool{name: "weather", mcpClient: client, auth: consentProvider()}
	actions := &session.EventActions{}

	_, err := tl.Run(newToolCtx(t, actions), map[string]any{})
	if !errors.Is(err, tool.ErrCredentialRequired) {
		t.Fatalf("Run() error = %v, want tool.ErrCredentialRequired", err)
	}
	got, ok := actions.RequestedCredentials["fn1"]
	if !ok {
		t.Fatalf("RequestedCredentials not set; got %+v", actions.RequestedCredentials)
	}
	if got.AuthURI != "https://consent.example" || got.Key != "k1" {
		t.Errorf("RequestedCredentials[fn1] = %+v, want AuthURI/Key from ConsentRequiredError", got)
	}
	if client.called != 0 {
		t.Errorf("CallTool called %d times, want 0 (paused for consent)", client.called)
	}
}

// After consent (AuthResponse present) and the provider now resolving, the tool
// proceeds to call the server.
func TestMcpToolRun_ProceedsAfterConsent(t *testing.T) {
	client := &fakeCredClient{}
	tl := &mcpTool{name: "weather", mcpClient: client, auth: okProvider()}

	ctx := newToolCtx(t, &session.EventActions{}).
		WithDelta(&agent.CommonContextDelta{CredentialResponse: &authconsent.Response{Token: "tok"}})

	res, err := tl.Run(ctx, map[string]any{})
	if err != nil {
		t.Fatalf("Run() error = %v, want nil", err)
	}
	if res["output"] != "ok" {
		t.Errorf("Run() output = %v, want %q", res["output"], "ok")
	}
	if client.called != 1 {
		t.Errorf("CallTool called %d times, want 1", client.called)
	}
}

// Resumed after consent but the provider still needs consent: fail instead of
// re-requesting (which would loop).
func TestMcpToolRun_ConsentGuardNoLoop(t *testing.T) {
	client := &fakeCredClient{}
	tl := &mcpTool{name: "weather", mcpClient: client, auth: consentProvider()}
	actions := &session.EventActions{}

	ctx := agent.NewToolContext(
		icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{}),
		"fn1", actions, nil,
	).WithDelta(&agent.CommonContextDelta{CredentialResponse: &authconsent.Response{}})

	_, err := tl.Run(ctx, map[string]any{})
	if err == nil || errors.Is(err, tool.ErrCredentialRequired) {
		t.Fatalf("Run() error = %v, want a non-pending failure", err)
	}
	if len(actions.RequestedCredentials) != 0 {
		t.Errorf("RequestedCredentials = %+v, want empty (no re-request on resume)", actions.RequestedCredentials)
	}
	if client.called != 0 {
		t.Errorf("CallTool called %d times, want 0", client.called)
	}
}

// With no auth provider the pre-flight probe is skipped entirely.
func TestMcpToolRun_NoAuthProvider(t *testing.T) {
	client := &fakeCredClient{}
	tl := &mcpTool{name: "weather", mcpClient: client}

	res, err := tl.Run(newToolCtx(t, &session.EventActions{}), map[string]any{})
	if err != nil {
		t.Fatalf("Run() error = %v, want nil", err)
	}
	if res["output"] != "ok" || client.called != 1 {
		t.Errorf("Run() output=%v called=%d, want output=ok called=1", res["output"], client.called)
	}
}

// A non-consent probe error fails fast: the tool surfaces the real resolution
// error and does not call the server (the RoundTripper would fail the same way,
// but the MCP SDK would mangle the error chain).
func TestMcpToolRun_NonConsentErrorFailsFast(t *testing.T) {
	client := &fakeCredClient{}
	tl := &mcpTool{name: "weather", mcpClient: client, auth: errProvider()}
	actions := &session.EventActions{}

	_, err := tl.Run(newToolCtx(t, actions), map[string]any{})
	if err == nil {
		t.Fatal("Run() error = nil, want the resolution error surfaced")
	}
	if errors.Is(err, tool.ErrCredentialRequired) {
		t.Errorf("Run() error = %v, want a non-consent failure (not ErrCredentialRequired)", err)
	}
	if client.called != 0 {
		t.Errorf("CallTool called %d times, want 0 (fail fast before the server call)", client.called)
	}
	if len(actions.RequestedCredentials) != 0 {
		t.Errorf("RequestedCredentials = %+v, want empty (no consent for a non-consent error)", actions.RequestedCredentials)
	}
}

type fakeCredClient struct {
	called int
}

func (f *fakeCredClient) CallTool(context.Context, *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	f.called++
	return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "ok"}}}, nil
}

func (f *fakeCredClient) ListTools(context.Context) ([]*mcp.Tool, error) { return nil, nil }

// consentProvider always reports interactive consent is required.
func consentProvider() auth.CredentialProvider {
	return auth.ProviderFunc(func(context.Context) (*auth.Credential, error) {
		return nil, &auth.ConsentRequiredError{AuthURI: "https://consent.example", Nonce: "n1", Key: "k1"}
	})
}

// okProvider always resolves a credential.
func okProvider() auth.CredentialProvider {
	return auth.ProviderFunc(func(context.Context) (*auth.Credential, error) {
		return &auth.Credential{}, nil
	})
}

// errProvider always fails with a plain (non-consent) error.
func errProvider() auth.CredentialProvider {
	return auth.ProviderFunc(func(context.Context) (*auth.Credential, error) {
		return nil, errors.New("boom")
	})
}

// newToolCtx builds a tool context for function call id "fn1".
func newToolCtx(t *testing.T, actions *session.EventActions) agent.Context {
	t.Helper()
	inv := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	return agent.NewToolContext(inv, "fn1", actions, nil)
}
