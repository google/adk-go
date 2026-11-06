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

package llminternal_test

import (
	"context"
	"encoding/json"
	"iter"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
)

const authFlowAgentName = "auth_agent"

// authCallsModel answers its first request with a call to fetch_data for each
// ID in callIDs, and every later one with text.
type authCallsModel struct {
	callIDs []string
	calls   atomic.Int32
}

func (m *authCallsModel) Name() string { return "auth-calls-model" }

func (m *authCallsModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.calls.Add(1) > 1 {
			yield(&model.LLMResponse{Content: genai.NewContentFromText("done", genai.RoleModel)}, nil)
			return
		}
		var parts []*genai.Part
		for _, id := range m.callIDs {
			parts = append(parts, &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: "fetch_data", Args: map[string]any{"query": id}}})
		}
		yield(&model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}}, nil)
	}
}

// authMixedCallsModel answers its first request with a call to plain_tool and
// a call to fetch_data, and every later one with text.
type authMixedCallsModel struct {
	calls *atomic.Int32
}

func (m *authMixedCallsModel) Name() string { return "auth-mixed-calls-model" }

func (m *authMixedCallsModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.calls.Add(1) > 1 {
			yield(&model.LLMResponse{Content: genai.NewContentFromText("done", genai.RoleModel)}, nil)
			return
		}
		yield(&model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "plain-call", Name: "plain_tool", Args: map[string]any{"query": "p"}}},
			{FunctionCall: &genai.FunctionCall{ID: "auth-call", Name: "fetch_data", Args: map[string]any{"query": "a"}}},
		}}}, nil)
	}
}

type fetchArgs struct {
	Query string `json:"query"`
}

// newFetchTool returns a tool that asks for cfg's credential until it finds
// it in state under "temp:" followed by cfg's key.
func newFetchTool(t *testing.T, cfg toolauth.Config) tool.Tool {
	t.Helper()
	key, err := cfg.Key()
	if err != nil {
		t.Fatal(err)
	}
	ft, err := functiontool.New(functiontool.Config{Name: "fetch_data", Description: "fetches data"},
		func(ctx agent.Context, args fetchArgs) (map[string]any, error) {
			if v, err := ctx.State().Get(session.KeyPrefixTemp + key); err == nil {
				cred, _ := v.(*toolauth.Credential)
				return map[string]any{"query": args.Query, "authResponseUri": cred.OAuth2.AuthResponseURI}, nil
			}
			actions := ctx.Actions()
			if actions.RequestedAuthConfigs == nil {
				actions.RequestedAuthConfigs = make(map[string]toolauth.Config)
			}
			actions.RequestedAuthConfigs[ctx.FunctionCallID()] = cfg
			return map[string]any{"status": "pending credential"}, nil
		})
	if err != nil {
		t.Fatal(err)
	}
	return ft
}

func oauth2FlowConfig() toolauth.Config {
	return toolauth.Config{
		AuthScheme: toolauth.Scheme{
			Type: toolauth.SchemeTypeOAuth2,
			Flows: &toolauth.OAuthFlows{AuthorizationCode: &toolauth.OAuthFlow{
				AuthorizationURL: "https://example.com/auth",
				TokenURL:         "https://example.com/token",
				Scopes:           map[string]string{"read": "Read access"},
			}},
		},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
			ClientID:     "client-id",
			ClientSecret: "client-secret",
			RedirectURI:  "https://example.com/callback",
		}},
	}
}

type authFlowRun struct {
	runner *runner.Runner
	model  *authCallsModel
}

func newAuthFlowRun(t *testing.T, cfg toolauth.Config, callIDs ...string) *authFlowRun {
	t.Helper()
	r := &authFlowRun{model: &authCallsModel{callIDs: callIDs}}
	a, err := llmagent.New(llmagent.Config{
		Name:  authFlowAgentName,
		Model: r.model,
		Tools: []tool.Tool{newFetchTool(t, cfg)},
	})
	if err != nil {
		t.Fatal(err)
	}
	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s"}); err != nil {
		t.Fatal(err)
	}
	r.runner, err = runner.New(runner.Config{Agent: a, SessionService: svc, AppName: "app"})
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// run sends msg and returns the events of the run, and its error.
func (r *authFlowRun) run(t *testing.T, msg *genai.Content) ([]*session.Event, error) {
	t.Helper()
	var events []*session.Event
	for ev, err := range r.runner.Run(t.Context(), "user", "s", msg, agent.RunConfig{}) {
		if err != nil {
			return events, err
		}
		events = append(events, ev)
	}
	return events, nil
}

// credentialRequests returns the adk_request_credential calls in ev.
func credentialRequests(ev *session.Event) []*genai.FunctionCall {
	var calls []*genai.FunctionCall
	if ev == nil || ev.Content == nil {
		return nil
	}
	for _, p := range ev.Content.Parts {
		if p.FunctionCall != nil && p.FunctionCall.Name == toolauth.RequestCredentialFunctionCallName {
			calls = append(calls, p.FunctionCall)
		}
	}
	return calls
}

func decodeToolArguments(t *testing.T, fc *genai.FunctionCall) toolauth.ToolArguments {
	t.Helper()
	b, err := json.Marshal(fc.Args)
	if err != nil {
		t.Fatal(err)
	}
	var args toolauth.ToolArguments
	if err := json.Unmarshal(b, &args); err != nil {
		t.Fatalf("adk_request_credential args do not decode: %v", err)
	}
	return args
}

func TestAuthRequestEmission(t *testing.T) {
	userMsg := genai.NewContentFromText("fetch my data", genai.RoleUser)

	t.Run("tool request becomes an adk_request_credential call", func(t *testing.T) {
		cfg := oauth2FlowConfig()
		wantKey, err := cfg.Key()
		if err != nil {
			t.Fatal(err)
		}
		r := newAuthFlowRun(t, cfg, "call-1")
		events, err := r.run(t, userMsg)
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		if len(events) != 3 {
			t.Fatalf("Run() yielded %d events, want the model call, the function response and the auth request", len(events))
		}
		if n := r.model.calls.Load(); n != 1 {
			t.Errorf("model was called %d times, want 1", n)
		}

		respEv, authEv := events[1], events[2]
		req, ok := respEv.Actions.RequestedAuthConfigs["call-1"]
		if !ok || req.RawAuthCredential == nil || req.ExchangedAuthCredential == nil {
			t.Fatalf("function response event has no generated request for call-1")
		}
		if req.RawAuthCredential.OAuth2.ClientSecret != "" || req.ExchangedAuthCredential.OAuth2.ClientSecret != "" {
			t.Errorf("stored request carries the client secret")
		}
		if req.CredentialKey != wantKey {
			t.Errorf("stored request CredentialKey = %q, want %q", req.CredentialKey, wantKey)
		}
		if o := req.ExchangedAuthCredential.OAuth2; o.AuthURI == "" || o.State == "" {
			t.Errorf("stored request has no auth URI or state")
		}

		if authEv.Author != authFlowAgentName || authEv.Branch != respEv.Branch {
			t.Errorf("auth event author, branch = %q, %q; want %q, %q", authEv.Author, authEv.Branch, authFlowAgentName, respEv.Branch)
		}
		if authEv.Content.Role != respEv.Content.Role {
			t.Errorf("auth event role = %q, want the function response's %q", authEv.Content.Role, respEv.Content.Role)
		}
		calls := credentialRequests(authEv)
		if len(calls) != 1 || len(authEv.Content.Parts) != 1 {
			t.Fatalf("auth event has %d parts with %d credential requests, want 1 and 1", len(authEv.Content.Parts), len(calls))
		}
		if diff := cmp.Diff([]string{calls[0].ID}, authEv.LongRunningToolIDs); diff != "" {
			t.Errorf("LongRunningToolIDs mismatch (-want +got):\n%s", diff)
		}
		args := decodeToolArguments(t, calls[0])
		if args.FunctionCallID != "call-1" {
			t.Errorf("request FunctionCallID = %q, want call-1", args.FunctionCallID)
		}
		if args.AuthConfig.RawAuthCredential.OAuth2.ClientSecret != "" {
			t.Errorf("adk_request_credential args carry the client secret")
		}
		if args.AuthConfig.ExchangedAuthCredential.OAuth2.State != req.ExchangedAuthCredential.OAuth2.State {
			t.Errorf("adk_request_credential args and the stored request have different states")
		}
	})

	t.Run("one request per credential key, for the first call", func(t *testing.T) {
		r := newAuthFlowRun(t, oauth2FlowConfig(), "call-b", "call-a")
		events, err := r.run(t, userMsg)
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		last := events[len(events)-1]
		calls := credentialRequests(last)
		if len(calls) != 1 {
			t.Fatalf("auth event has %d credential requests, want 1", len(calls))
		}
		if got := decodeToolArguments(t, calls[0]).FunctionCallID; got != "call-b" {
			t.Errorf("request FunctionCallID = %q, want call-b, the model's first call", got)
		}
		if n := len(events[1].Actions.RequestedAuthConfigs); n != 2 {
			t.Errorf("function response event has %d requests, want 2", n)
		}
		for _, id := range []string{"call-b", "call-a"} {
			req, ok := events[1].Actions.RequestedAuthConfigs[id]
			if !ok || req.RawAuthCredential == nil || req.ExchangedAuthCredential == nil {
				t.Fatalf("function response event has no generated request for %s", id)
			}
			if req.RawAuthCredential.OAuth2.ClientSecret != "" || req.ExchangedAuthCredential.OAuth2.ClientSecret != "" {
				t.Errorf("stored request for %s carries the client secret", id)
			}
			if req.CredentialKey == "" || req.ExchangedAuthCredential.OAuth2.AuthURI == "" || req.ExchangedAuthCredential.OAuth2.State == "" {
				t.Errorf("stored request for %s lacks CredentialKey, AuthURI or State", id)
			}
		}
	})

	t.Run("turn mixing a tool that requests a credential with one that does not", func(t *testing.T) {
		cfg := oauth2FlowConfig()
		plainTool, err := functiontool.New(functiontool.Config{Name: "plain_tool", Description: "needs no auth"},
			func(_ agent.Context, args fetchArgs) (map[string]any, error) {
				return map[string]any{"plain": args.Query}, nil
			})
		if err != nil {
			t.Fatal(err)
		}
		var calls atomic.Int32
		m := &authMixedCallsModel{calls: &calls}
		a, err := llmagent.New(llmagent.Config{
			Name:  authFlowAgentName,
			Model: m,
			Tools: []tool.Tool{plainTool, newFetchTool(t, cfg)},
		})
		if err != nil {
			t.Fatal(err)
		}
		svc := session.InMemoryService()
		if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user", SessionID: "s"}); err != nil {
			t.Fatal(err)
		}
		runnr, err := runner.New(runner.Config{Agent: a, SessionService: svc, AppName: "app"})
		if err != nil {
			t.Fatal(err)
		}
		var events []*session.Event
		for ev, err := range runnr.Run(t.Context(), "user", "s", userMsg, agent.RunConfig{}) {
			if err != nil {
				t.Fatalf("Run() error = %v", err)
			}
			events = append(events, ev)
		}
		if len(events) != 3 {
			t.Fatalf("Run() yielded %d events, want 3", len(events))
		}
		reqs := credentialRequests(events[2])
		if len(reqs) != 1 || decodeToolArguments(t, reqs[0]).FunctionCallID != "auth-call" {
			t.Errorf("auth event requests = %v, want only auth-call", reqs)
		}
		if _, ok := events[1].Actions.RequestedAuthConfigs["plain-call"]; ok {
			t.Errorf("plain-call has an entry in RequestedAuthConfigs")
		}
	})

	t.Run("API key scheme passes through without the key", func(t *testing.T) {
		cfg := toolauth.Config{
			AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Key"},
			RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: "configured-key"},
		}
		r := newAuthFlowRun(t, cfg, "call-1")
		events, err := r.run(t, userMsg)
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
		calls := credentialRequests(events[len(events)-1])
		if len(calls) != 1 {
			t.Fatalf("auth event has %d credential requests, want 1", len(calls))
		}
		args := decodeToolArguments(t, calls[0])
		if args.AuthConfig.AuthScheme.Type != toolauth.SchemeTypeAPIKey || args.AuthConfig.RawAuthCredential.APIKey != "" {
			t.Errorf("request does not carry the API key scheme without the key")
		}
		if events[1].Actions.RequestedAuthConfigs["call-1"].RawAuthCredential.APIKey != "" {
			t.Errorf("stored request carries the API key")
		}
	})

	t.Run("config without a client ID fails the run", func(t *testing.T) {
		cfg := oauth2FlowConfig()
		cfg.RawAuthCredential.OAuth2.ClientID = ""
		r := newAuthFlowRun(t, cfg, "call-1")
		events, err := r.run(t, userMsg)
		if err == nil {
			t.Fatalf("Run() error = nil, want one")
		}
		for _, ev := range events {
			if len(ev.Actions.RequestedAuthConfigs) != 0 {
				t.Errorf("Run() yielded an event with a credential request before failing")
			}
		}
	})
}

func TestAuthRoundTrip(t *testing.T) {
	r := newAuthFlowRun(t, oauth2FlowConfig(), "call-1")
	events, err := r.run(t, genai.NewContentFromText("fetch my data", genai.RoleUser))
	if err != nil {
		t.Fatalf("first Run() error = %v", err)
	}
	calls := credentialRequests(events[len(events)-1])
	if len(calls) != 1 {
		t.Fatalf("first run ended with %d credential requests, want 1", len(calls))
	}
	issued := decodeToolArguments(t, calls[0]).AuthConfig

	// The client completes the OAuth2 flow and sends the request back with
	// the URI the provider redirected to.
	const responseURI = "https://example.com/callback?code=code-1"
	o := *issued.ExchangedAuthCredential.OAuth2
	o.AuthResponseURI = responseURI + "&state=" + o.State
	ex := *issued.ExchangedAuthCredential
	ex.OAuth2 = &o
	issued.ExchangedAuthCredential = &ex
	b, err := json.Marshal(issued)
	if err != nil {
		t.Fatal(err)
	}
	var response map[string]any
	if err := json.Unmarshal(b, &response); err != nil {
		t.Fatal(err)
	}
	answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID:       calls[0].ID,
		Name:     toolauth.RequestCredentialFunctionCallName,
		Response: response,
	}}}}

	events, err = r.run(t, answer)
	if err != nil {
		t.Fatalf("second Run() error = %v", err)
	}
	var resumed *genai.FunctionResponse
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionResponse != nil && p.FunctionResponse.ID == "call-1" {
				resumed = p.FunctionResponse
			}
		}
	}
	if resumed == nil {
		t.Fatalf("second run did not run call-1 again")
	}
	want := map[string]any{"query": "call-1", "authResponseUri": o.AuthResponseURI}
	if diff := cmp.Diff(want, resumed.Response); diff != "" {
		t.Errorf("resumed tool response mismatch (-want +got):\n%s", diff)
	}
	if n := r.model.calls.Load(); n != 2 {
		t.Errorf("model was called %d times in total, want 2", n)
	}

	t.Run("rejects a redirect URI whose state parameter does not match", func(t *testing.T) {
		r := newAuthFlowRun(t, oauth2FlowConfig(), "call-1")
		events, err := r.run(t, genai.NewContentFromText("fetch my data", genai.RoleUser))
		if err != nil {
			t.Fatalf("first Run() error = %v", err)
		}
		calls := credentialRequests(events[len(events)-1])
		issued := decodeToolArguments(t, calls[0]).AuthConfig
		o := *issued.ExchangedAuthCredential.OAuth2
		o.AuthResponseURI = "https://example.com/callback?code=forged-code&state=forged-state"
		ex := *issued.ExchangedAuthCredential
		ex.OAuth2 = &o
		issued.ExchangedAuthCredential = &ex
		b, err := json.Marshal(issued)
		if err != nil {
			t.Fatal(err)
		}
		var response map[string]any
		if err := json.Unmarshal(b, &response); err != nil {
			t.Fatal(err)
		}
		answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
			ID:       calls[0].ID,
			Name:     toolauth.RequestCredentialFunctionCallName,
			Response: response,
		}}}}
		events, err = r.run(t, answer)
		if err != nil {
			t.Fatalf("second Run() error = %v", err)
		}
		for _, ev := range events {
			if ev.Content == nil {
				continue
			}
			for _, p := range ev.Content.Parts {
				if p.FunctionResponse != nil && p.FunctionResponse.ID == "call-1" {
					if p.FunctionResponse.Response["authResponseUri"] != nil {
						t.Errorf("resumed tool received credential from forged redirect state")
					}
				}
			}
		}
	})
}
