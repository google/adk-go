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

package toolinternal

import (
	"testing"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/auth"
	contextinternal "google.golang.org/adk/internal/context"
	"google.golang.org/adk/session"
)

func TestToolContext(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})
	toolCtx := NewToolContext(inv, "fn1", &session.EventActions{})

	if _, ok := toolCtx.(agent.ReadonlyContext); !ok {
		t.Errorf("ToolContext(%+T) is unexpectedly not a ReadonlyContext", toolCtx)
	}
	if _, ok := toolCtx.(agent.CallbackContext); !ok {
		t.Errorf("ToolContext(%+T) is unexpectedly not a CallbackContext", toolCtx)
	}
	if got, ok := toolCtx.(agent.InvocationContext); ok {
		t.Errorf("ToolContext(%+T) is unexpectedly an InvocationContext", got)
	}
}

func TestToolContext_RequestCredential(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})
	actions := &session.EventActions{
		RequestedAuthConfigs: make(map[string]*auth.AuthConfig),
	}
	toolCtx := NewToolContext(inv, "fn-123", actions)

	authConfig := &auth.AuthConfig{
		AuthScheme: &auth.OAuth2Scheme{
			Flows: &auth.OAuthFlows{
				AuthorizationCode: &auth.OAuthFlowAuthorizationCode{
					AuthorizationURL: "https://example.com/auth",
					TokenURL:         "https://example.com/token",
				},
			},
		},
		RawAuthCredential: &auth.AuthCredential{
			AuthType: auth.AuthCredentialTypeOAuth2,
			OAuth2: &auth.OAuth2Auth{
				ClientID:     "client-id",
				ClientSecret: "client-secret",
				RedirectURI:  "https://localhost/callback",
			},
		},
		CredentialKey: "test-key",
	}

	// Request credential
	tc := toolCtx.(*toolContext)
	tc.RequestCredential(authConfig)

	// Verify it was added to RequestedAuthConfigs
	if len(actions.RequestedAuthConfigs) != 1 {
		t.Errorf("RequestedAuthConfigs has %d entries, want 1", len(actions.RequestedAuthConfigs))
	}

	stored, ok := actions.RequestedAuthConfigs["fn-123"]
	if !ok {
		t.Error("RequestedAuthConfigs should have entry for 'fn-123'")
	}
	if stored == nil {
		t.Error("Stored auth config should not be nil")
	}
	// AuthHandler.GenerateAuthRequest adds auth_uri
	if stored.ExchangedAuthCredential == nil || stored.ExchangedAuthCredential.OAuth2 == nil {
		t.Error("ExchangedAuthCredential should be set")
	}
}

func TestToolContext_RequestCredential_Nil(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})
	actions := &session.EventActions{
		RequestedAuthConfigs: make(map[string]*auth.AuthConfig),
	}
	toolCtx := NewToolContext(inv, "fn-123", actions)

	tc := toolCtx.(*toolContext)
	tc.RequestCredential(nil)

	// Should not panic and should not add anything
	if len(actions.RequestedAuthConfigs) != 0 {
		t.Errorf("RequestedAuthConfigs has %d entries, want 0", len(actions.RequestedAuthConfigs))
	}
}

// Note: TestToolContext_GetAuthResponse tests are skipped as they require
// a full invocation context with session state. The GetAuthResponse function
// works correctly when session state is available.

func TestToolContext_CredentialService(t *testing.T) {
	inv := contextinternal.NewInvocationContext(t.Context(), contextinternal.InvocationContextParams{})
	toolCtx := NewToolContext(inv, "fn-123", &session.EventActions{})

	tc := toolCtx.(*toolContext)
	svc := tc.CredentialService()

	// ToolContext doesn't have a credential service by default
	if svc != nil {
		t.Errorf("CredentialService() = %v, want nil", svc)
	}
}
