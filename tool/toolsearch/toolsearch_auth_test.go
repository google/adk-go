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

package toolsearch_test

import (
	"encoding/json"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
	"google.golang.org/adk/v2/tool/toolsearch"
)

// keyedToolset declares cfg as its auth config and records the API key it
// finds under cfg's key each time it lists its tools.
type keyedToolset struct {
	cfg   *toolauth.Config
	tools []tool.Tool

	mu   sync.Mutex
	seen []string
}

func (*keyedToolset) Name() string { return "keyed" }

func (s *keyedToolset) AuthConfig() *toolauth.Config { return s.cfg }

func (s *keyedToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	key := "<nil>"
	if c := agent.ResolvedCredential(ctx, s.cfg.CredentialKey); c != nil {
		key = c.APIKey
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.seen = append(s.seen, key)
	return s.tools, nil
}

func (s *keyedToolset) listings() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.seen...)
}

// credentialRequest is what a client reads from one adk_request_credential
// call.
type credentialRequest struct {
	FunctionCallID, CredentialKey string
}

func credentialRequests(t *testing.T, events []*session.Event) []credentialRequest {
	t.Helper()
	var got []credentialRequest
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionCall == nil || p.FunctionCall.Name != toolauth.RequestCredentialFunctionCallName {
				continue
			}
			b, err := json.Marshal(p.FunctionCall.Args)
			if err != nil {
				t.Fatalf("json.Marshal(args): %v", err)
			}
			var args struct {
				FunctionCallID string `json:"functionCallId"`
				AuthConfig     struct {
					CredentialKey string `json:"credentialKey"`
				} `json:"authConfig"`
			}
			if err := json.Unmarshal(b, &args); err != nil {
				t.Fatalf("json.Unmarshal(args): %v", err)
			}
			got = append(got, credentialRequest{args.FunctionCallID, args.AuthConfig.CredentialKey})
		}
	}
	return got
}

// TestGatingToolset_AuthConfigThroughRunner runs a toolsearch toolset whose
// base declares an auth config through a runner: ADK asks for the base's
// credential under the gating toolset's name, and the base lists its tools
// with the credential ADK resolved (tools/base_toolset.py:248-264,
// flows/llm_flows/_toolset_auth.py:73-123).
func TestGatingToolset_AuthConfigThroughRunner(t *testing.T) {
	lookup, err := functiontool.New(functiontool.Config{Name: "lookup", Description: "Looks something up."},
		func(agent.Context, struct{}) (map[string]any, error) { return nil, nil })
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	tests := []struct {
		name         string
		cfg          *toolauth.Config
		wantRequests []credentialRequest
		wantListings []string
		wantModel    int
	}{
		{
			name: "no credential yet",
			cfg: &toolauth.Config{
				AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: &toolauth.OAuthFlows{
					AuthorizationCode: &toolauth.OAuthFlow{
						AuthorizationURL: "https://idp.example/auth",
						TokenURL:         "https://idp.example/token",
					},
				}},
				RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
					ClientID: "cid", ClientSecret: "s3cret",
				}},
				CredentialKey: "wrapped-key",
			},
			wantRequests: []credentialRequest{{"_adk_toolset_auth_SearchGatedToolset", "wrapped-key"}},
			wantListings: nil,
			wantModel:    0,
		},
		{
			name: "ready api key",
			cfg: &toolauth.Config{
				AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
				RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "k-1"},
				CredentialKey:     "wrapped-key",
			},
			wantRequests: nil,
			wantListings: []string{"k-1"},
			wantModel:    1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := &keyedToolset{cfg: tt.cfg, tools: []tool.Tool{lookup}}
			gated, err := toolsearch.New(base, toolsearch.Config{CoreToolNames: []string{"lookup"}})
			if err != nil {
				t.Fatalf("toolsearch.New() error = %v", err)
			}
			llm := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("ok", genai.RoleModel)}}
			a, err := llmagent.New(llmagent.Config{Name: "assistant", Model: llm, Toolsets: []tool.Toolset{gated}})
			if err != nil {
				t.Fatalf("llmagent.New() error = %v", err)
			}
			r, err := runner.New(runner.Config{AppName: "app", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true})
			if err != nil {
				t.Fatalf("runner.New() error = %v", err)
			}

			events, err := testutil.CollectEvents(r.Run(t.Context(), "user", "session",
				genai.NewContentFromText("hi", genai.RoleUser), agent.RunConfig{}))
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}

			if diff := cmp.Diff(tt.wantRequests, credentialRequests(t, events)); diff != "" {
				t.Errorf("credential requests mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.wantListings, base.listings()); diff != "" {
				t.Errorf("API keys the base listed its tools with mismatch (-want +got):\n%s", diff)
			}
			if got := len(llm.Requests); got != tt.wantModel {
				t.Errorf("model calls = %d, want %d", got, tt.wantModel)
			}
		})
	}
}
