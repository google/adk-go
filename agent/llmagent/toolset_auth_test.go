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

package llmagent_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/gorilla/websocket"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/credentialservice"
	"google.golang.org/adk/v2/exp/authflow"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
)

// tsaToolset is a toolset that counts its listings and its request hook
// calls, and lists what tools returns for the context it is listed with.
type tsaToolset struct {
	name  string
	tools func(ctx agent.ReadonlyContext) []tool.Tool

	mu       sync.Mutex
	listed   int
	requests int
}

func (s *tsaToolset) Name() string { return s.name }

func (s *tsaToolset) Tools(ctx agent.ReadonlyContext) ([]tool.Tool, error) {
	s.mu.Lock()
	s.listed++
	s.mu.Unlock()
	if s.tools == nil {
		return nil, nil
	}
	return s.tools(ctx), nil
}

func (s *tsaToolset) ProcessRequest(agent.Context, *model.LLMRequest) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests++
	return nil
}

func (s *tsaToolset) counts() (listed, requests int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.listed, s.requests
}

// tsaAuthToolset is a tsaToolset that implements tool.AuthenticatedToolset:
// its n-th AuthConfig call, counting from 1, returns config(n).
type tsaAuthToolset struct {
	*tsaToolset
	config func(call int) *toolauth.Config

	cfgMu sync.Mutex
	calls int
}

func (s *tsaAuthToolset) AuthConfig() *toolauth.Config {
	s.cfgMu.Lock()
	defer s.cfgMu.Unlock()
	s.calls++
	return s.config(s.calls)
}

var _ tool.AuthenticatedToolset = (*tsaAuthToolset)(nil)

// tsaRecorder records values under names, from any goroutine.
type tsaRecorder struct {
	mu   sync.Mutex
	seen map[string][]string
}

func (r *tsaRecorder) add(name, value string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.seen == nil {
		r.seen = map[string][]string{}
	}
	r.seen[name] = append(r.seen[name], value)
}

func (r *tsaRecorder) get(name string) []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.seen[name]...)
}

// tsaAPIKey returns the API key of the credential resolved under key in
// ctx's run, or "<nil>".
func tsaAPIKey(ctx agent.ReadonlyContext, key string) string {
	c := agent.ResolvedCredential(ctx, key)
	if c == nil {
		return "<nil>"
	}
	return c.APIKey
}

// tsaOAuth2 returns the OAuth2 field picked by field from the credential
// resolved under key in ctx's run, or "<nil>".
func tsaOAuth2(ctx agent.ReadonlyContext, key string, field func(*toolauth.OAuth2Auth) string) string {
	c := agent.ResolvedCredential(ctx, key)
	if c == nil || c.OAuth2 == nil {
		return "<nil>"
	}
	return field(c.OAuth2)
}

func tsaAccessToken(o *toolauth.OAuth2Auth) string { return o.AccessToken }

func tsaClientID(o *toolauth.OAuth2Auth) string { return o.ClientID }

// tsaTool returns a function tool named name that calls run, when it is not
// nil, and answers {"status": "ran"}.
func tsaTool(t *testing.T, name string, run func(ctx agent.Context) error) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: name, Description: "Test tool " + name + "."},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			if run != nil {
				if err := run(ctx); err != nil {
					return nil, err
				}
			}
			return map[string]any{"status": "ran"}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New(%q): %v", name, err)
	}
	return tl
}

func tsaAgent(t *testing.T, cfg llmagent.Config) agent.Agent {
	t.Helper()
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New(%q): %v", cfg.Name, err)
	}
	return a
}

// tsaAPIKeyConfig is a ready API-key config: its raw credential is used as it
// is.
func tsaAPIKeyConfig(key, apiKey string) *toolauth.Config {
	return &toolauth.Config{
		AuthScheme:        toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeAPIKey, APIKey: apiKey},
		CredentialKey:     key,
	}
}

// tsaOAuthConfig is an OAuth2 authorization-code config for client clientID
// of a provider whose token endpoint is tokenURL.
func tsaOAuthConfig(key, clientID, tokenURL string) *toolauth.Config {
	return &toolauth.Config{
		AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: &toolauth.OAuthFlows{
			AuthorizationCode: &toolauth.OAuthFlow{
				AuthorizationURL: "https://idp.example/auth",
				TokenURL:         tokenURL,
				Scopes:           map[string]string{"calendar": "Read calendar"},
			},
		}},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
			ClientID: clientID, ClientSecret: "s3cret", TokenEndpointAuthMethod: "client_secret_post",
		}},
		CredentialKey: key,
	}
}

// tsaRunner runs its agent in session "s" of user "u" of app "app".
type tsaRunner struct {
	t *testing.T
	r *runner.Runner
}

func newTsaRunner(t *testing.T, root agent.Agent, creds credentialservice.Service) *tsaRunner {
	t.Helper()
	svc := session.InMemoryService()
	if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
		t.Fatalf("session.Create: %v", err)
	}
	r, err := runner.New(runner.Config{AppName: "app", Agent: root, SessionService: svc, CredentialService: creds})
	if err != nil {
		t.Fatalf("runner.New: %v", err)
	}
	return &tsaRunner{t: t, r: r}
}

func (r *tsaRunner) run(content *genai.Content) ([]*session.Event, error) {
	return testutil.CollectEvents(r.r.Run(r.t.Context(), "u", "s", content, agent.RunConfig{}))
}

func (r *tsaRunner) say(text string) ([]*session.Event, error) {
	return r.run(genai.NewContentFromText(text, genai.RoleUser))
}

// tsaCall is what a client reads from one adk_request_credential call.
type tsaCall struct {
	FunctionCallID, CredentialKey, ClientID string
}

func tsaRequestArgs(t *testing.T, call *genai.FunctionCall) toolauth.ToolArguments {
	t.Helper()
	b, err := json.Marshal(call.Args)
	if err != nil {
		t.Fatalf("json.Marshal(args): %v", err)
	}
	var args toolauth.ToolArguments
	if err := json.Unmarshal(b, &args); err != nil {
		t.Fatalf("json.Unmarshal(args): %v", err)
	}
	return args
}

// tsaRequests returns the adk_request_credential calls of events.
func tsaRequests(events []*session.Event) []*genai.FunctionCall {
	var calls []*genai.FunctionCall
	for _, ev := range events {
		if ev.Content == nil {
			continue
		}
		for _, p := range ev.Content.Parts {
			if p.FunctionCall != nil && p.FunctionCall.Name == toolauth.RequestCredentialFunctionCallName {
				calls = append(calls, p.FunctionCall)
			}
		}
	}
	return calls
}

// tsaCalls reads each adk_request_credential call of events as JSON, the way
// a client does, without the schema checks of toolauth's decoders: a config
// that names a flow without its URLs still reaches the client.
func tsaCalls(t *testing.T, events []*session.Event) []tsaCall {
	t.Helper()
	var got []tsaCall
	for _, call := range tsaRequests(events) {
		b, err := json.Marshal(call.Args)
		if err != nil {
			t.Fatalf("json.Marshal(args): %v", err)
		}
		var args struct {
			FunctionCallID string `json:"functionCallId"`
			AuthConfig     struct {
				CredentialKey     string `json:"credentialKey"`
				RawAuthCredential struct {
					OAuth2 struct {
						ClientID string `json:"clientId"`
					} `json:"oauth2"`
				} `json:"rawAuthCredential"`
			} `json:"authConfig"`
		}
		if err := json.Unmarshal(b, &args); err != nil {
			t.Fatalf("json.Unmarshal(args): %v", err)
		}
		got = append(got, tsaCall{args.FunctionCallID, args.AuthConfig.CredentialKey, args.AuthConfig.RawAuthCredential.OAuth2.ClientID})
	}
	return got
}

// tsaAnswer is what a client sends back once the user has signed in: the
// request's config with the redirect it received, carrying code.
func tsaAnswer(t *testing.T, call *genai.FunctionCall, code string) *genai.Content {
	t.Helper()
	cfg := tsaRequestArgs(t, call).AuthConfig
	o := cfg.ExchangedAuthCredential.OAuth2
	o.RedirectURI = "https://app.example/cb"
	o.AuthResponseURI = o.RedirectURI + "?" + url.Values{"state": {o.State}, "code": {code}}.Encode()
	b, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("json.Marshal(answer): %v", err)
	}
	var resp map[string]any
	if err := json.Unmarshal(b, &resp); err != nil {
		t.Fatalf("json.Unmarshal(answer): %v", err)
	}
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: call.ID, Name: toolauth.RequestCredentialFunctionCallName, Response: resp,
	}}}}
}

// tsaDeclared returns the names of the functions req declares, in order.
func tsaDeclared(req *model.LLMRequest) []string {
	var names []string
	if req.Config == nil {
		return nil
	}
	for _, gt := range req.Config.Tools {
		for _, d := range gt.FunctionDeclarations {
			names = append(names, d.Name)
		}
	}
	return names
}

// tsaTokenEndpoint is a fake OAuth2 token endpoint that answers every request
// with status, handing out "at-1" on 200, and records the forms it received.
type tsaTokenEndpoint struct {
	*httptest.Server
	mu    sync.Mutex
	forms []url.Values
}

func newTsaTokenEndpoint(t *testing.T, status int) *tsaTokenEndpoint {
	t.Helper()
	e := &tsaTokenEndpoint{}
	e.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			http.Error(w, "bad form", http.StatusBadRequest)
			return
		}
		e.mu.Lock()
		e.forms = append(e.forms, r.PostForm)
		e.mu.Unlock()
		if status != http.StatusOK {
			http.Error(w, "unavailable", status)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"access_token": "at-1", "token_type": "Bearer", "expires_in": 3600}`)
	}))
	t.Cleanup(e.Close)
	return e
}

func (e *tsaTokenEndpoint) seen() []url.Values {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]url.Values(nil), e.forms...)
}

// tsaFailingService is a credential service whose every call fails with err.
type tsaFailingService struct{ err error }

func (s tsaFailingService) LoadCredential(context.Context, string, string, *toolauth.Config) (*toolauth.Credential, error) {
	return nil, s.err
}

func (s tsaFailingService) SaveCredential(context.Context, string, string, *toolauth.Config) error {
	return s.err
}

func TestToolsetAuth_NoAuthConfig(t *testing.T) {
	// _toolset_auth.py:65-71: an entry that is not a toolset, and a toolset
	// whose get_auth_config returns None, are skipped; the model is called.
	var rec tsaRecorder
	plainTool, nilAuthTool := tsaTool(t, "plain_tool", nil), tsaTool(t, "nil_auth_tool", nil)
	plain := &tsaToolset{name: "plain", tools: func(ctx agent.ReadonlyContext) []tool.Tool {
		rec.add("plain", tsaAPIKey(ctx, "any"))
		return []tool.Tool{plainTool}
	}}
	nilAuth := &tsaAuthToolset{
		tsaToolset: &tsaToolset{name: "nil_auth", tools: func(ctx agent.ReadonlyContext) []tool.Tool {
			rec.add("nil_auth", tsaAPIKey(ctx, "any"))
			return []tool.Tool{nilAuthTool}
		}},
		config: func(int) *toolauth.Config { return nil },
	}
	llm := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("ok", genai.RoleModel)}}
	root := tsaAgent(t, llmagent.Config{
		Name: "root", Model: llm,
		Tools:    []tool.Tool{tsaTool(t, "own_tool", nil)},
		Toolsets: []tool.Toolset{plain, nilAuth},
	})

	events, err := newTsaRunner(t, root, nil).say("hi")
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if len(events) != 1 || events[0].Author != "root" || events[0].Content.Parts[0].Text != "ok" ||
		len(events[0].LongRunningToolIDs) != 0 {
		t.Errorf("events = %v, want one event of root with text \"ok\" and no long-running call", events)
	}
	if calls := tsaRequests(events); len(calls) != 0 {
		t.Errorf("credential requests = %d, want 0", len(calls))
	}
	if len(llm.Requests) != 1 {
		t.Fatalf("model calls = %d, want 1", len(llm.Requests))
	}
	if diff := cmp.Diff([]string{"own_tool", "plain_tool", "nil_auth_tool"}, tsaDeclared(llm.Requests[0])); diff != "" {
		t.Errorf("declared tools mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(map[string][]string{"plain": {"<nil>"}, "nil_auth": {"<nil>"}}, rec.seen); diff != "" {
		t.Errorf("credentials seen while listing mismatch (-want +got):\n%s", diff)
	}
}

func TestToolsetAuth_CredentialAvailable_PerStepAndIsolation(t *testing.T) {
	// _toolset_auth.py:73-95 resolves before every model call and stores the
	// credential by key in the invocation; readonly_context.py:87-89 reads it
	// from any context; agent_tool.py:265-273 runs the sub-agent in a new
	// invocation; test_toolset_auth.py:189-193: the config is not written.
	var rec tsaRecorder
	var cfgMu sync.Mutex
	var configs []*toolauth.Config

	op := tsaTool(t, "op", func(ctx agent.Context) error {
		rec.add("op", tsaAPIKey(ctx, "tsa-key"))
		return nil
	})
	probe := tsaTool(t, "probe", func(ctx agent.Context) error {
		rec.add("nested", tsaAPIKey(ctx, "tsa-key"))
		return nil
	})
	nested := tsaAgent(t, llmagent.Config{
		Name: "nested", Description: "Runs probe.",
		Model: &testutil.MockModel{Responses: []*genai.Content{
			genai.NewContentFromFunctionCall("probe", map[string]any{}, genai.RoleModel),
			genai.NewContentFromText("nested done", genai.RoleModel),
		}},
		Tools: []tool.Tool{probe},
	})
	keyed := &tsaAuthToolset{
		tsaToolset: &tsaToolset{name: "keyed", tools: func(ctx agent.ReadonlyContext) []tool.Tool {
			c := agent.ResolvedCredential(ctx, "tsa-key")
			if c == nil {
				rec.add("tools", "<nil>")
				return []tool.Tool{op}
			}
			rec.add("tools", c.APIKey)
			c.APIKey = "mutated"
			return []tool.Tool{op}
		}},
		config: func(call int) *toolauth.Config {
			cfg := tsaAPIKeyConfig("tsa-key", fmt.Sprintf("k-%d", call))
			cfgMu.Lock()
			configs = append(configs, cfg)
			cfgMu.Unlock()
			return cfg
		},
	}
	helper := tsaAgent(t, llmagent.Config{
		Name: "helper", Description: "Helps.",
		Model: &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("helper done", genai.RoleModel)}},
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(ctx agent.Context) (*genai.Content, error) {
			rec.add("helper", tsaAPIKey(ctx, "tsa-key"))
			return nil, nil
		}},
	})
	rootLLM := &testutil.MockModel{Responses: []*genai.Content{
		{Role: genai.RoleModel, Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{Name: "op", Args: map[string]any{}}},
			{FunctionCall: &genai.FunctionCall{Name: "nested", Args: map[string]any{"request": "go"}}},
		}},
		genai.NewContentFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": "helper"}, genai.RoleModel),
	}}
	root := tsaAgent(t, llmagent.Config{
		Name: "root", Model: rootLLM,
		Tools:     []tool.Tool{agenttool.New(nested, nil)},
		Toolsets:  []tool.Toolset{keyed},
		SubAgents: []agent.Agent{helper},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
			rec.add("model", tsaAPIKey(ctx, "tsa-key"))
			return nil, nil
		}},
	})

	events, err := newTsaRunner(t, root, nil).say("hi")
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if calls := tsaRequests(events); len(calls) != 0 {
		t.Errorf("credential requests = %d, want 0", len(calls))
	}
	if got := events[len(events)-1]; got.Author != "helper" || got.Content.Parts[0].Text != "helper done" {
		t.Errorf("last event = %v, want helper's \"helper done\"", got)
	}
	want := map[string][]string{
		"tools":  {"k-1", "k-2"},
		"op":     {"k-1"},
		"model":  {"k-1", "k-2"},
		"helper": {"k-2"},
		"nested": {"<nil>"},
	}
	if diff := cmp.Diff(want, rec.seen); diff != "" {
		t.Errorf("credentials seen mismatch (-want +got):\n%s", diff)
	}
	if got := agent.ResolvedCredential(nil, "tsa-key"); got != nil {
		t.Errorf("ResolvedCredential(nil) = %+v, want nil", got)
	}
	if len(configs) != 2 {
		t.Fatalf("AuthConfig calls = %d, want 2", len(configs))
	}
	if configs[0].ExchangedAuthCredential != nil || configs[1].ExchangedAuthCredential != nil {
		t.Errorf("ExchangedAuthCredential after the run = %+v, %+v, want nil, nil",
			configs[0].ExchangedAuthCredential, configs[1].ExchangedAuthCredential)
	}
}

func TestToolsetAuth_RequestCredentialEventAndDeduplication(t *testing.T) {
	// _toolset_auth.py:96-123: one event authored by the agent, keyed by
	// prefix + toolset name (a dict: first place, last config), the invocation
	// ended; _functions.py:143-172: one call per credential key, all
	// long-running; auth_handler.py:242-243: no secrets, an authorization URI.
	type spec struct{ name, key, clientID string }
	for _, tc := range []struct {
		name      string
		toolsets  []spec
		withReady bool
		want      []tsaCall
		wantReady string
	}{
		{
			name:      "one toolset whose name has a space",
			toolsets:  []spec{{"calendar tools", "cal-key", "cid-cal"}},
			want:      []tsaCall{{"_adk_toolset_auth_calendar tools", "cal-key", "cid-cal"}},
			wantReady: "<nil>",
		},
		{
			name:      "two names, one credential key",
			toolsets:  []spec{{"a", "k", "cid-1"}, {"b", "k", "cid-2"}},
			want:      []tsaCall{{"_adk_toolset_auth_a", "k", "cid-1"}},
			wantReady: "<nil>",
		},
		{
			name:      "one name twice, and a ready toolset",
			toolsets:  []spec{{"same", "k1", "cid-1"}, {"other", "k3", "cid-3"}, {"same", "k2", "cid-2"}},
			withReady: true,
			want:      []tsaCall{{"_adk_toolset_auth_same", "k2", "cid-2"}, {"_adk_toolset_auth_other", "k3", "cid-3"}},
			wantReady: "rk-1",
		},
		{
			name:      "two names, two keys",
			toolsets:  []spec{{"a", "k1", "cid-1"}, {"b", "k2", "cid-2"}},
			want:      []tsaCall{{"_adk_toolset_auth_a", "k1", "cid-1"}, {"_adk_toolset_auth_b", "k2", "cid-2"}},
			wantReady: "<nil>",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var sets []*tsaToolset
			var toolsets []tool.Toolset
			for _, s := range tc.toolsets {
				ts := &tsaAuthToolset{
					tsaToolset: &tsaToolset{name: s.name},
					config: func(int) *toolauth.Config {
						return tsaOAuthConfig(s.key, s.clientID, "https://idp.example/token")
					},
				}
				sets = append(sets, ts.tsaToolset)
				toolsets = append(toolsets, ts)
			}
			if tc.withReady {
				ready := &tsaAuthToolset{
					tsaToolset: &tsaToolset{name: "ready"},
					config:     func(int) *toolauth.Config { return tsaAPIKeyConfig("ready-key", "rk-1") },
				}
				sets = append(sets, ready.tsaToolset)
				toolsets = append(toolsets, ready)
			}
			var invocationID, readyKey string
			llm := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("unexpected", genai.RoleModel)}}
			root := tsaAgent(t, llmagent.Config{
				Name: "assistant", Model: llm, Toolsets: toolsets,
				BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(ctx agent.Context) (*genai.Content, error) {
					invocationID = ctx.InvocationID()
					return nil, nil
				}},
				AfterAgentCallbacks: []agent.AfterAgentCallback{func(ctx agent.Context) (*genai.Content, error) {
					readyKey = tsaAPIKey(ctx, "ready-key")
					return nil, nil
				}},
			})

			events, err := newTsaRunner(t, root, nil).say("hi")
			if err != nil {
				t.Fatalf("run failed: %v", err)
			}

			if len(events) != 1 {
				t.Fatalf("events = %d, want 1: %v", len(events), events)
			}
			ev := events[0]
			if ev.Author != "assistant" || ev.InvocationID != invocationID || ev.Content.Role != "model" {
				t.Errorf("event author, invocation, role = %q, %q, %q; want \"assistant\", %q, \"model\"",
					ev.Author, ev.InvocationID, ev.Content.Role, invocationID)
			}
			var ids []string
			for _, p := range ev.Content.Parts {
				if p.FunctionCall == nil || p.FunctionCall.Name != toolauth.RequestCredentialFunctionCallName {
					t.Fatalf("event part %+v, want an adk_request_credential call", p)
				}
				ids = append(ids, p.FunctionCall.ID)
			}
			if diff := cmp.Diff(ids, ev.LongRunningToolIDs); diff != "" {
				t.Errorf("LongRunningToolIDs mismatch, want every call id (-calls +long-running):\n%s", diff)
			}
			if diff := cmp.Diff(tc.want, tsaCalls(t, events)); diff != "" {
				t.Errorf("credential requests mismatch (-want +got):\n%s", diff)
			}
			for _, call := range tsaRequests(events) {
				cfg := tsaRequestArgs(t, call).AuthConfig
				ex := cfg.ExchangedAuthCredential.OAuth2
				if !strings.HasPrefix(ex.AuthURI, "https://idp.example/auth?") || ex.State == "" {
					t.Errorf("request %s: authUri set = %t, state set = %t; want a link to https://idp.example/auth with a state",
						call.ID, strings.HasPrefix(ex.AuthURI, "https://idp.example/auth?"), ex.State != "")
				}
				if cfg.RawAuthCredential.OAuth2.ClientSecret != "" || ex.ClientSecret != "" {
					t.Errorf("request %s carries a client secret, want none", call.ID)
				}
			}
			if readyKey != tc.wantReady {
				t.Errorf("after-agent callback read %q under ready-key, want %q", readyKey, tc.wantReady)
			}
			if len(llm.Requests) != 0 {
				t.Errorf("model calls = %d, want 0", len(llm.Requests))
			}
			for i, s := range sets {
				if listed, requests := s.counts(); listed != 0 || requests != 0 {
					t.Errorf("toolset %d (%s): listed %d times, request hook ran %d times; want 0, 0", i, s.name, listed, requests)
				}
			}
		})
	}
}

func TestToolsetAuth_SignInResumeAndCredentialLifetime(t *testing.T) {
	// auth_preprocessor.py:80-94 stores the answer and reruns nothing for a
	// toolset id; credential_manager.py:284-287, :313-320 exchanges the code
	// and saves it to a credential service; auth_handler.py:113-124 reads the
	// answer from temp: state, which a new invocation does not carry.
	for _, tc := range []struct {
		name  string
		creds credentialservice.Service
		// After the third run, which sends a new message.
		wantThirdRequests int
		wantModelCalls    int
		wantListed        []string
	}{
		{
			name:              "without credential service",
			wantThirdRequests: 1,
			wantModelCalls:    1,
			wantListed:        []string{"at-1"},
		},
		{
			name:              "with credential service",
			creds:             credentialservice.InMemoryService(),
			wantThirdRequests: 0,
			wantModelCalls:    2,
			wantListed:        []string{"at-1", "at-1"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			endpoint := newTsaTokenEndpoint(t, http.StatusOK)
			var rec tsaRecorder
			listEvents := tsaTool(t, "list_events", nil)
			calendar := &tsaAuthToolset{
				tsaToolset: &tsaToolset{name: "calendar tools", tools: func(ctx agent.ReadonlyContext) []tool.Tool {
					rec.add("listed", tsaOAuth2(ctx, "cal-key", tsaAccessToken))
					return []tool.Tool{listEvents}
				}},
				config: func(int) *toolauth.Config { return tsaOAuthConfig("cal-key", "app", endpoint.URL) },
			}
			llm := &testutil.MockModel{Responses: []*genai.Content{
				genai.NewContentFromText("done", genai.RoleModel),
				genai.NewContentFromText("done again", genai.RoleModel),
			}}
			r := newTsaRunner(t, tsaAgent(t, llmagent.Config{Name: "assistant", Model: llm, Toolsets: []tool.Toolset{calendar}}), tc.creds)

			first, err := r.say("hi")
			if err != nil {
				t.Fatalf("first run failed: %v", err)
			}
			requests := tsaRequests(first)
			if diff := cmp.Diff([]tsaCall{{"_adk_toolset_auth_calendar tools", "cal-key", "app"}}, tsaCalls(t, first)); diff != "" {
				t.Fatalf("first run's credential requests mismatch (-want +got):\n%s", diff)
			}
			if listed, _ := calendar.counts(); len(llm.Requests) != 0 || listed != 0 {
				t.Errorf("first run: model calls = %d, listings = %d; want 0, 0", len(llm.Requests), listed)
			}

			second, err := r.run(tsaAnswer(t, requests[0], "c0de"))
			if err != nil {
				t.Fatalf("second run failed: %v", err)
			}
			wantForm := url.Values{
				"grant_type":    {"authorization_code"},
				"code":          {"c0de"},
				"redirect_uri":  {"https://app.example/cb"},
				"client_id":     {"app"},
				"client_secret": {"s3cret"},
			}
			if diff := cmp.Diff([]url.Values{wantForm}, endpoint.seen()); diff != "" {
				t.Errorf("token requests mismatch (-want +got):\n%s", diff)
			}
			if len(second) != 1 || second[0].Content.Parts[0].Text != "done" {
				t.Errorf("second run's events = %v, want only the model's \"done\"", second)
			}
			if diff := cmp.Diff([]string{"at-1"}, rec.get("listed")); diff != "" {
				t.Errorf("second run: tokens seen while listing mismatch (-want +got):\n%s", diff)
			}
			if len(llm.Requests) != 1 {
				t.Fatalf("second run: model calls = %d, want 1", len(llm.Requests))
			}
			if diff := cmp.Diff([]string{"list_events"}, tsaDeclared(llm.Requests[0])); diff != "" {
				t.Errorf("second run: declared tools mismatch (-want +got):\n%s", diff)
			}

			third, err := r.say("again")
			if err != nil {
				t.Fatalf("third run failed: %v", err)
			}
			if got := len(tsaRequests(third)); got != tc.wantThirdRequests {
				t.Errorf("third run: credential requests = %d, want %d", got, tc.wantThirdRequests)
			}
			if len(llm.Requests) != tc.wantModelCalls {
				t.Errorf("model calls after the third run = %d, want %d", len(llm.Requests), tc.wantModelCalls)
			}
			if diff := cmp.Diff(tc.wantListed, rec.get("listed")); diff != "" {
				t.Errorf("tokens seen while listing after the third run mismatch (-want +got):\n%s", diff)
			}
			if got := len(endpoint.seen()); got != 1 {
				t.Errorf("token requests after the third run = %d, want 1", got)
			}
		})
	}
}

func TestToolsetAuth_Errors(t *testing.T) {
	// _toolset_auth.py:76-88 turns only a ValueError of the manager into a
	// request; auth_handler.py:263-267 raises when it cannot build one;
	// credential_manager.py:305-310 returns a credential whose exchange
	// failed. A1 and A2 (decision.md) fail the run for a config without a
	// scheme type and for the Go-only refusals.
	errBoom := errors.New("boom")
	failing := newTsaTokenEndpoint(t, http.StatusInternalServerError)
	clientCredentials := &toolauth.Config{
		AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: &toolauth.OAuthFlows{
			ClientCredentials: &toolauth.OAuthFlow{TokenURL: failing.URL},
		}},
		RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
			ClientID: "cid-cc", ClientSecret: "s3cret",
		}},
		CredentialKey: "cc-key",
	}
	for _, tc := range []struct {
		name           string
		cfg            *toolauth.Config
		creds          credentialservice.Service
		key            string // the key the toolset reads its credential under
		wantErr        string
		wantBoom       bool
		wantRequests   []tsaCall
		wantModelCalls int
		wantListed     []string
	}{
		{
			name:         "custom scheme asks",
			cfg:          &toolauth.Config{AuthScheme: toolauth.Scheme{Type: "myScheme"}, CredentialKey: "ck"},
			key:          "ck",
			wantRequests: []tsaCall{{"_adk_toolset_auth_ts", "ck", ""}},
		},
		{
			name: "oauth2 without raw credential fails building the request",
			cfg: &toolauth.Config{AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: &toolauth.OAuthFlows{
				AuthorizationCode: &toolauth.OAuthFlow{AuthorizationURL: "https://idp.example/auth", TokenURL: "https://idp.example/token"},
			}}},
			wantErr: "authhandler: oauth2 auth scheme requires a raw auth credential",
		},
		{
			name: "oauth2 raw credential without oauth2 field fails building the request",
			cfg: &toolauth.Config{
				AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: &toolauth.OAuthFlows{
					AuthorizationCode: &toolauth.OAuthFlow{AuthorizationURL: "https://idp.example/auth", TokenURL: "https://idp.example/token"},
				}},
				RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2},
			},
			wantErr: "authhandler: oauth2 auth scheme requires oauth2 in the raw auth credential",
		},
		{
			name: "missing oauth urls ask",
			cfg: &toolauth.Config{
				AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: &toolauth.OAuthFlows{Implicit: &toolauth.OAuthFlow{}}},
				RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
					ClientID: "cid", AuthURI: "https://idp.example/login",
				}},
				CredentialKey: "missing-url-key",
			},
			key:          "missing-url-key",
			wantRequests: []tsaCall{{"_adk_toolset_auth_ts", "missing-url-key", "cid"}},
		},
		{
			name:     "credential service failure fails the run",
			cfg:      tsaOAuthConfig("svc-key", "app", "https://idp.example/token"),
			creds:    tsaFailingService{err: errBoom},
			wantErr:  `toolset "ts": authtool: load credential: boom`,
			wantBoom: true,
		},
		{
			name:           "failed exchange beside a credential uses the credential",
			cfg:            clientCredentials,
			key:            "cc-key",
			wantModelCalls: 1,
			wantListed:     []string{"cid-cc"},
		},
		{
			name:    "config without scheme type fails the run",
			cfg:     &toolauth.Config{CredentialKey: "zk"},
			wantErr: `toolset "ts": authtool: NewCredentialManager needs an auth scheme with a type`,
		},
		{
			name: "service account fails the run",
			cfg: &toolauth.Config{
				AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "bearer"},
				RawAuthCredential: &toolauth.Credential{
					AuthType:       toolauth.CredentialTypeServiceAccount,
					ServiceAccount: &toolauth.ServiceAccount{UseDefaultCredential: true},
				},
			},
			wantErr: `toolset "ts": service account is not supported`,
		},
		{
			name: "oauth2 auto-discovery fails the run",
			cfg: &toolauth.Config{
				AuthScheme: toolauth.Scheme{
					Type: toolauth.SchemeTypeOAuth2, IssuerURL: "https://idp.example",
					Flows: &toolauth.OAuthFlows{Implicit: &toolauth.OAuthFlow{}},
				},
				RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{ClientID: "cid"}},
			},
			wantErr: `toolset "ts": OAuth2 auto-discovery is not supported`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var rec tsaRecorder
			ts := &tsaAuthToolset{
				tsaToolset: &tsaToolset{name: "ts", tools: func(ctx agent.ReadonlyContext) []tool.Tool {
					rec.add("listed", tsaOAuth2(ctx, tc.key, tsaClientID))
					return nil
				}},
				config: func(int) *toolauth.Config { return tc.cfg },
			}
			llm := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("ok", genai.RoleModel)}}
			r := newTsaRunner(t, tsaAgent(t, llmagent.Config{Name: "assistant", Model: llm, Toolsets: []tool.Toolset{ts}}), tc.creds)

			events, err := r.say("hi")

			gotErr := ""
			if err != nil {
				gotErr = err.Error()
			}
			if gotErr != tc.wantErr {
				t.Errorf("run error = %q, want %q", gotErr, tc.wantErr)
			}
			if got := errors.Is(err, errBoom); got != tc.wantBoom {
				t.Errorf("errors.Is(run error, the service's error) = %t, want %t", got, tc.wantBoom)
			}
			if diff := cmp.Diff(tc.wantRequests, tsaCalls(t, events)); diff != "" {
				t.Errorf("credential requests mismatch (-want +got):\n%s", diff)
			}
			if len(llm.Requests) != tc.wantModelCalls {
				t.Errorf("model calls = %d, want %d", len(llm.Requests), tc.wantModelCalls)
			}
			if diff := cmp.Diff(tc.wantListed, rec.get("listed")); diff != "" {
				t.Errorf("client IDs seen while listing mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestToolsetAuth_Errors_MissingOAuthURLClientAndResume(t *testing.T) {
	// _toolset_auth.py:80-88 and credential_manager.py:434-438: when a
	// toolset's OAuth2 flow lacks its required URL and has no issuer_url,
	// validate raises ValueError ("OAuth scheme info is missing, and
	// auto-discovery has failed to fill them in."), which toolset_auth turns
	// into an adk_request_credential event when raw_auth_credential supplies
	// auth_uri (auth_handler.py:277-285, F1 / AUTH-TSA-FLAW-1).
	//
	// In adk-python 2.12.0, the client decodes AuthToolArguments and sends
	// back an exchangedAuthCredential; auth_preprocessor stores it under
	// "temp:" + key (_auth_resume.py:173-237), and then toolset_auth calls
	// CredentialManager.get_auth_credential, which runs _validate_credential
	// (credential_manager.py:263) before reading session state (:284), raises
	// the same ValueError again, and emits another adk_request_credential
	// event with 0 model calls.
	//
	// In adk-go, ClientCredentials with an empty TokenURL (AUTH-ARTE-GO-1,
	// tool/toolauth/auth_schemes.go:107, 140-147) matches adk-python on both
	// client decode and the resume run. Implicit with an empty
	// AuthorizationURL hits toolauth.OAuthFlows.UnmarshalJSON's refusal of an
	// empty authorizationUrl (AUTH-ARTE-GAP-1, auth_schemes.go:105-106), both
	// when a client decodes toolauth.ToolArguments and when authPreprocessor's
	// issuedCredentialRequests decodes the session's request event on resume.
	for _, tc := range []struct {
		name            string
		flows           *toolauth.OAuthFlows
		key             string
		wantDecodeErr   string
		wantResumeErr   string
		wantResumeCalls []tsaCall
	}{
		{
			name:          "implicit without authorizationUrl refuses client decode and resume",
			flows:         &toolauth.OAuthFlows{Implicit: &toolauth.OAuthFlow{}},
			key:           "missing-auth-url-key",
			wantDecodeErr: "toolauth: OAuthFlows: implicit flow needs authorizationUrl",
			wantResumeErr: "toolauth: OAuthFlows: implicit flow needs authorizationUrl",
		},
		{
			name:            "clientCredentials with empty tokenUrl decodes, stores answer, and asks again on resume",
			flows:           &toolauth.OAuthFlows{ClientCredentials: &toolauth.OAuthFlow{}},
			key:             "missing-token-url-key",
			wantResumeCalls: []tsaCall{{"_adk_toolset_auth_ts", "missing-token-url-key", "cid"}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &toolauth.Config{
				AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeOAuth2, Flows: tc.flows},
				RawAuthCredential: &toolauth.Credential{AuthType: toolauth.CredentialTypeOAuth2, OAuth2: &toolauth.OAuth2Auth{
					ClientID: "cid", AuthURI: "https://idp.example/login",
				}},
				CredentialKey: tc.key,
			}
			ts := &tsaAuthToolset{
				tsaToolset: &tsaToolset{name: "ts"},
				config:     func(int) *toolauth.Config { return cfg },
			}
			llm := &testutil.MockModel{Responses: []*genai.Content{genai.NewContentFromText("ok", genai.RoleModel)}}
			r := newTsaRunner(t, tsaAgent(t, llmagent.Config{Name: "assistant", Model: llm, Toolsets: []tool.Toolset{ts}}), nil)

			first, err := r.say("hi")
			if err != nil {
				t.Fatalf("first run failed: %v", err)
			}
			if diff := cmp.Diff([]tsaCall{{"_adk_toolset_auth_ts", tc.key, "cid"}}, tsaCalls(t, first)); diff != "" {
				t.Fatalf("first run's credential requests mismatch (-want +got):\n%s", diff)
			}

			call := tsaRequests(first)[0]
			b, err := json.Marshal(call.Args)
			if err != nil {
				t.Fatalf("json.Marshal(args): %v", err)
			}
			var decoded toolauth.ToolArguments
			decodeErr := json.Unmarshal(b, &decoded)
			gotDecodeErr := ""
			if decodeErr != nil {
				gotDecodeErr = decodeErr.Error()
			}
			if gotDecodeErr != tc.wantDecodeErr {
				t.Errorf("client json.Unmarshal(toolauth.ToolArguments) error = %q, want %q", gotDecodeErr, tc.wantDecodeErr)
			}

			resp := maps.Clone(call.Args["authConfig"].(map[string]any))
			resp["exchangedAuthCredential"] = map[string]any{
				"authType": "oauth2",
				"oauth2":   map[string]any{"accessToken": "tok-1"},
			}
			answer := &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
				ID: call.ID, Name: toolauth.RequestCredentialFunctionCallName, Response: resp,
			}}}}

			second, resumeErr := r.run(answer)
			if tc.wantResumeErr != "" {
				if resumeErr == nil || !strings.HasSuffix(resumeErr.Error(), tc.wantResumeErr) {
					t.Errorf("resume run error = %v, want suffix %q", resumeErr, tc.wantResumeErr)
				}
			} else if resumeErr != nil {
				t.Errorf("resume run error = %v, want nil", resumeErr)
			}
			if diff := cmp.Diff(tc.wantResumeCalls, tsaCalls(t, second)); diff != "" {
				t.Errorf("resume run credential requests mismatch (-want +got):\n%s", diff)
			}
			if len(llm.Requests) != 0 {
				t.Errorf("model calls = %d, want 0", len(llm.Requests))
			}
		})
	}
}

// tsaLiveServer is a fake Gemini Live API server that counts the connections
// it accepted. On each it completes the setup, sends one model turn that
// completes, and reads until the client closes it.
type tsaLiveServer struct {
	client *genai.Client

	mu          sync.Mutex
	connections int
}

func newTsaLiveServer(t *testing.T) *tsaLiveServer {
	t.Helper()
	s := &tsaLiveServer{}
	var upgrader websocket.Upgrader
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = conn.Close() }()
		s.mu.Lock()
		s.connections++
		s.mu.Unlock()
		if _, _, err := conn.ReadMessage(); err != nil {
			return
		}
		for _, m := range []string{`{"setupComplete":{}}`, liveText("connected", true)} {
			if err := conn.WriteMessage(websocket.TextMessage, []byte(m)); err != nil {
				return
			}
		}
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				return
			}
		}
	}))
	t.Cleanup(server.Close)
	client, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		Backend:     genai.BackendGeminiAPI,
		APIKey:      "test-api-key",
		HTTPOptions: genai.HTTPOptions{BaseURL: strings.Replace(server.URL, "http", "ws", 1)},
	})
	if err != nil {
		t.Fatalf("genai.NewClient: %v", err)
	}
	s.client = client
	return s
}

func (s *tsaLiveServer) accepted() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.connections
}

// runTsaLive runs root through runner.RunLive until the stream ends or yields
// an event that completes a turn, and returns the events and the stream's
// error.
func runTsaLive(t *testing.T, root agent.Agent) ([]*session.Event, error) {
	t.Helper()
	r, _ := newOutputKeySession(t, root)
	sess, stream, err := r.RunLive(t.Context(), "u", "s", agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("runner.RunLive: %v", err)
	}
	var events []*session.Event
	var streamErr error
	for ev, err := range stream {
		if err != nil {
			streamErr = err
			break
		}
		events = append(events, ev)
		if ev.TurnComplete {
			break
		}
	}
	if err := sess.Close(); err != nil {
		t.Errorf("LiveSession.Close: %v", err)
	}
	return events, streamErr
}

func TestToolsetAuth_RunLive_StopsBeforeConnect(t *testing.T) {
	// live/_live_llm_flow.py:515-521: the live flow does not connect when a
	// request processor ended the invocation.
	server := newTsaLiveServer(t)
	liveTS := &tsaAuthToolset{
		tsaToolset: &tsaToolset{name: "live_ts"},
		config: func(int) *toolauth.Config {
			return tsaOAuthConfig("live-key", "cid-live", "https://idp.example/token")
		},
	}
	root := tsaAgent(t, llmagent.Config{Name: "assistant", Model: &liveModel{client: server.client}, Toolsets: []tool.Toolset{liveTS}})

	events, err := runTsaLive(t, root)
	if err != nil {
		t.Errorf("stream error = %v, want none", err)
	}
	if diff := cmp.Diff([]tsaCall{{"_adk_toolset_auth_live_ts", "live-key", "cid-live"}}, tsaCalls(t, events)); diff != "" || len(events) != 1 {
		t.Errorf("stream yielded %d events, want only the credential request (-want +got):\n%s", len(events), diff)
	}
	if listed, _ := liveTS.counts(); server.accepted() != 0 || listed != 0 {
		t.Errorf("live connections = %d, listings = %d; want 0, 0", server.accepted(), listed)
	}
}

func TestToolsetAuth_RunLive_ListsWithResolvedCredential(t *testing.T) {
	// live/_live_llm_flow.py:515-521 runs the same request processors before it
	// connects, toolset_auth among them (_toolset_auth.py:73-95), so a live
	// run's toolset lists with its resolved credential
	// (readonly_context.py:87-89).
	server := newTsaLiveServer(t)
	var rec tsaRecorder
	liveTS := &tsaAuthToolset{
		tsaToolset: &tsaToolset{name: "live_ts", tools: func(ctx agent.ReadonlyContext) []tool.Tool {
			rec.add("listed", tsaAPIKey(ctx, "live-key"))
			return nil
		}},
		config: func(int) *toolauth.Config { return tsaAPIKeyConfig("live-key", "k-1") },
	}
	root := tsaAgent(t, llmagent.Config{Name: "assistant", Model: &liveModel{client: server.client}, Toolsets: []tool.Toolset{liveTS}})

	events, err := runTsaLive(t, root)
	if err != nil {
		t.Errorf("stream error = %v, want none", err)
	}
	if calls := tsaRequests(events); len(calls) != 0 {
		t.Errorf("credential requests = %d, want 0", len(calls))
	}
	if diff := cmp.Diff([]string{"k-1"}, rec.get("listed")); diff != "" {
		t.Errorf("API keys seen while listing mismatch (-want +got):\n%s", diff)
	}
	if got := server.accepted(); got != 1 {
		t.Errorf("live connections = %d, want 1", got)
	}
}

// tsaTranscript describes each event the way a user reads it: the author,
// then each part as a call, a response with its body, or text.
func tsaTranscript(t *testing.T, events []*session.Event) []string {
	t.Helper()
	var lines []string
	for _, ev := range events {
		var parts []string
		if ev.Content != nil {
			for _, p := range ev.Content.Parts {
				switch {
				case p.FunctionCall != nil:
					parts = append(parts, "call "+p.FunctionCall.Name)
				case p.FunctionResponse != nil:
					b, err := json.Marshal(p.FunctionResponse.Response)
					if err != nil {
						t.Fatalf("json.Marshal(response): %v", err)
					}
					parts = append(parts, "response "+p.FunctionResponse.Name+" "+string(b))
				case p.Text != "":
					parts = append(parts, "text "+p.Text)
				}
			}
		}
		lines = append(lines, ev.Author+": "+strings.Join(parts, ", "))
	}
	return lines
}

// tsaToolCredential is the API-key credential tsaNeedsEUC asks the user for.
var tsaToolCredential = &toolauth.Config{
	AuthScheme:    toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Tool"},
	CredentialKey: "tool-key",
}

// tsaNeedsEUC returns the tool needs_euc, which asks for tsaToolCredential
// and answers {"status": "pending"} until the user's answer has given it, then
// answers {"status": "ran"}. It records each run in rec under "tool runs".
func tsaNeedsEUC(t *testing.T, rec *tsaRecorder) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: "needs_euc", Description: "Needs an end-user credential."},
		func(ctx agent.Context, _ struct{}) (map[string]any, error) {
			cred, err := authflow.GetAuthResponse(ctx, tsaToolCredential)
			if err != nil {
				return nil, err
			}
			if cred == nil {
				rec.add("tool runs", "needs_euc:asks")
				return map[string]any{"status": "pending"}, authflow.RequestCredential(ctx, tsaToolCredential)
			}
			rec.add("tool runs", "needs_euc:ran")
			return map[string]any{"status": "ran"}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New: %v", err)
	}
	return tl
}

// tsaAPIKeyAnswer answers the one adk_request_credential call of first with
// the API key "user-key".
func tsaAPIKeyAnswer(t *testing.T, first []*session.Event) *genai.Content {
	t.Helper()
	return tsaExchangedAnswer(t, first, map[string]any{"authType": "apiKey", "apiKey": "user-key"})
}

// tsaBearerAnswer answers the one adk_request_credential call of first with
// the HTTP bearer token "tok-1".
func tsaBearerAnswer(t *testing.T, first []*session.Event) *genai.Content {
	t.Helper()
	return tsaExchangedAnswer(t, first, map[string]any{"authType": "http", "http": map[string]any{
		"scheme": "bearer", "credentials": map[string]any{"token": "tok-1"},
	}})
}

// tsaExchangedAnswer answers the one adk_request_credential call of first
// with the request's config carrying exchanged as its exchanged credential.
func tsaExchangedAnswer(t *testing.T, first []*session.Event, exchanged map[string]any) *genai.Content {
	t.Helper()
	requests := tsaRequests(first)
	if len(requests) != 1 {
		t.Fatalf("first run's credential requests = %d, want 1", len(requests))
	}
	resp := maps.Clone(requests[0].Args["authConfig"].(map[string]any))
	resp["exchangedAuthCredential"] = exchanged
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
		ID: requests[0].ID, Name: toolauth.RequestCredentialFunctionCallName, Response: resp,
	}}}}
}

// tsaRunAll returns the events of one run of r in session "s" of user "u",
// events without content included (a resumable runner ends a resumed run with
// one, as adk-python's does), and the error that ended it.
func tsaRunAll(t *testing.T, r *runner.Runner, content *genai.Content) ([]*session.Event, error) {
	t.Helper()
	var events []*session.Event
	for ev, err := range r.Run(t.Context(), "u", "s", content, agent.RunConfig{}) {
		if err != nil {
			return events, err
		}
		events = append(events, ev)
	}
	return events, nil
}

func TestToolsetAuth_RerunListsToolsetsWithResolvedCredentials(t *testing.T) {
	// needs-human AU12: a call that resumes after the user's answer runs with
	// the tools the agent's toolsets list once toolset authentication has
	// resolved this invocation's credentials, as a Python user sees it in one
	// process, where the invocation's listing cache
	// (tools/base_toolset.py:137-151) holds the listing made with them.
	// Python, scenarios S1 to S3 in team-a/au12-scenarios.md: the call runs,
	// the model call that follows receives the tool, and get_auth_config runs
	// once per step (_toolset_auth.py:59-71).
	for _, tc := range []struct {
		name      string
		resumable bool
		// config is the toolset's config; nil means the ready API key "k-1".
		config  *toolauth.Config
		newTool func(t *testing.T, rec *tsaRecorder) tool.Tool
		// signIn, when set, answers the toolset's own credential request
		// that ends the first run; the run it starts pauses the call.
		signIn func(t *testing.T, first []*session.Event) *genai.Content
		// answer returns the message that resumes the call first's events paused.
		answer func(t *testing.T, first []*session.Event) *genai.Content
		want   []string
		// wantRequests are the functionCallIds of the second run's
		// credential requests.
		wantRequests    []string
		wantModelTools  [][]string
		wantConfigCalls int
	}{
		{
			name:            "S1 credential answer",
			newTool:         tsaNeedsEUC,
			answer:          tsaAPIKeyAnswer,
			want:            []string{`assistant: response needs_euc {"status":"ran"}`, "assistant: text done"},
			wantModelTools:  [][]string{{"needs_euc"}, {"needs_euc"}},
			wantConfigCalls: 2,
		},
		{
			name:      "S2 credential answer resumes the invocation",
			resumable: true,
			newTool:   tsaNeedsEUC,
			answer:    tsaAPIKeyAnswer,
			// The resumed run ends with an event without content, as in Python.
			want:            []string{`assistant: response needs_euc {"status":"ran"}`, "assistant: text done", "assistant: "},
			wantModelTools:  [][]string{{"needs_euc"}, {"needs_euc"}},
			wantConfigCalls: 2,
		},
		{
			// Opus judge's probe 5b: the toolset's credential is a user answer
			// and there is no credential service, so it is gone on the step
			// that answers the tool's request. Go has no listing cache
			// (FC-BT-GAP-1, NEEDS-HUMAN entry 107), so the rerun lists the
			// toolset without its credential and does not find the tool;
			// adk-python in one process reruns it from the cache, and both
			// then ask for the toolset's credential again (AUTH-TSA-BUG-2).
			name:    "toolset credential gone on the resume step",
			config:  &toolauth.Config{AuthScheme: toolauth.Scheme{Type: toolauth.SchemeTypeHTTP, Scheme: "bearer"}, CredentialKey: "tsa-key"},
			newTool: tsaNeedsEUC,
			signIn:  tsaBearerAnswer,
			answer:  tsaAPIKeyAnswer,
			want: []string{
				"assistant: response needs_euc {\"error\":\"Invoking `needs_euc()` failed as no tool with that name is available. The tools you can call are: none. You could retry, but it is IMPORTANT that you only call a tool from that list.\"}",
				"assistant: call adk_request_credential",
			},
			wantRequests:    []string{"_adk_toolset_auth_keyed"},
			wantModelTools:  [][]string{{"needs_euc"}},
			wantConfigCalls: 3,
		},
		{
			name: "S3 confirmation",
			newTool: func(t *testing.T, rec *tsaRecorder) tool.Tool {
				tl, err := functiontool.New(functiontool.Config{Name: "gated_op", Description: "Needs confirmation.", RequireConfirmation: true},
					func(agent.Context, struct{}) (map[string]any, error) {
						rec.add("tool runs", "gated_op:ran")
						return map[string]any{"status": "ran"}, nil
					})
				if err != nil {
					t.Fatalf("functiontool.New: %v", err)
				}
				return tl
			},
			answer: func(t *testing.T, first []*session.Event) *genai.Content {
				var id string
				for _, ev := range first {
					for _, p := range ev.Content.Parts {
						if p.FunctionCall != nil && p.FunctionCall.Name == "adk_request_confirmation" {
							id = p.FunctionCall.ID
						}
					}
				}
				if id == "" {
					t.Fatalf("first run asked for no confirmation: %v", tsaTranscript(t, first))
				}
				return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{
					ID: id, Name: "adk_request_confirmation", Response: map[string]any{"confirmed": true},
				}}}}
			},
			want:            []string{`assistant: response gated_op {"status":"ran"}`, "assistant: text done"},
			wantModelTools:  [][]string{{"gated_op"}, {"gated_op"}},
			wantConfigCalls: 2,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var rec tsaRecorder
			gated := tc.newTool(t, &rec)
			configCalls := 0
			keyed := &tsaAuthToolset{
				tsaToolset: &tsaToolset{name: "keyed", tools: func(ctx agent.ReadonlyContext) []tool.Tool {
					if agent.ResolvedCredential(ctx, "tsa-key") == nil {
						rec.add("listed", "none")
						return nil
					}
					rec.add("listed", "k-1")
					return []tool.Tool{gated}
				}},
				config: func(call int) *toolauth.Config {
					configCalls = call
					if tc.config != nil {
						return tc.config
					}
					return tsaAPIKeyConfig("tsa-key", "k-1")
				},
			}
			llm := &testutil.MockModel{Responses: []*genai.Content{
				genai.NewContentFromFunctionCall(gated.Name(), map[string]any{}, genai.RoleModel),
				genai.NewContentFromText("done", genai.RoleModel),
			}}
			root := tsaAgent(t, llmagent.Config{Name: "assistant", Model: llm, Toolsets: []tool.Toolset{keyed}})
			svc := session.InMemoryService()
			if _, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "u", SessionID: "s"}); err != nil {
				t.Fatalf("session.Create: %v", err)
			}
			r, err := runner.New(runner.Config{AppName: "app", Agent: root, SessionService: svc, Resumable: tc.resumable})
			if err != nil {
				t.Fatalf("runner.New: %v", err)
			}

			first, err := tsaRunAll(t, r, genai.NewContentFromText("hi", genai.RoleUser))
			if err != nil {
				t.Fatalf("first run failed: %v", err)
			}
			if tc.signIn != nil {
				t.Logf("toolset request run events: %q, listings: %q", tsaTranscript(t, first), rec.get("listed"))
				if first, err = tsaRunAll(t, r, tc.signIn(t, first)); err != nil {
					t.Fatalf("sign-in run failed: %v", err)
				}
			}
			t.Logf("run 1 events: %q, listings: %q, tool runs: %q", tsaTranscript(t, first), rec.get("listed"), rec.get("tool runs"))
			firstListings, firstRuns := len(rec.get("listed")), len(rec.get("tool runs"))

			second, err := tsaRunAll(t, r, tc.answer(t, first))
			var modelTools [][]string
			for _, req := range llm.Requests {
				modelTools = append(modelTools, tsaDeclared(req))
			}
			t.Logf("run 2 events: %q, error: %v, listings: %q, tool runs: %q, model calls (tools received): %q",
				tsaTranscript(t, second), err, rec.get("listed")[firstListings:], rec.get("tool runs")[firstRuns:], modelTools)
			if err != nil {
				t.Fatalf("second run failed: %v", err)
			}

			if diff := cmp.Diff(tc.want, tsaTranscript(t, second)); diff != "" {
				t.Errorf("second run's events mismatch (-want +got):\n%s", diff)
			}
			var requestIDs []string
			for _, call := range tsaRequests(second) {
				requestIDs = append(requestIDs, tsaRequestArgs(t, call).FunctionCallID)
			}
			if diff := cmp.Diff(tc.wantRequests, requestIDs); diff != "" {
				t.Errorf("second run's credential requests (functionCallId) mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantModelTools, modelTools); diff != "" {
				t.Errorf("tools each model call received mismatch (-want +got):\n%s", diff)
			}
			if configCalls != tc.wantConfigCalls {
				t.Errorf("AuthConfig calls = %d, want %d (one per step)", configCalls, tc.wantConfigCalls)
			}
		})
	}
}

func TestToolsetAuth_RerunRunsBeforeToolsetCredentialError(t *testing.T) {
	// Scenario E1 in team-a/au12-scenarios.md: when the toolset's credential
	// cannot be resolved on the step the user's answer reruns a call, the
	// rerun's response comes first and the error ends the run after it, as
	// adk-python's auth_preprocessor reruns before toolset_auth raises
	// (base_llm_flow.py, request_processors before tool_request_processors).
	var rec tsaRecorder
	needsEUC := tsaNeedsEUC(t, &rec)
	keyed := &tsaAuthToolset{
		tsaToolset: &tsaToolset{name: "keyed", tools: func(agent.ReadonlyContext) []tool.Tool {
			return []tool.Tool{needsEUC}
		}},
		// The first step resolves a ready API key; the resume step asks the
		// failing credential service.
		config: func(call int) *toolauth.Config {
			if call == 1 {
				return tsaAPIKeyConfig("tsa-key", "k-1")
			}
			return tsaOAuthConfig("tsa-key", "app", "https://idp.example/token")
		},
	}
	llm := &testutil.MockModel{Responses: []*genai.Content{
		genai.NewContentFromFunctionCall("needs_euc", map[string]any{}, genai.RoleModel),
		genai.NewContentFromText("done", genai.RoleModel),
	}}
	errDown := errors.New("credential service down")
	r := newTsaRunner(t, tsaAgent(t, llmagent.Config{Name: "assistant", Model: llm, Toolsets: []tool.Toolset{keyed}}), tsaFailingService{err: errDown})

	first, err := r.say("hi")
	if err != nil {
		t.Fatalf("first run failed: %v", err)
	}
	second, err := tsaRunAll(t, r.r, tsaAPIKeyAnswer(t, first))

	if diff := cmp.Diff([]string{`assistant: response needs_euc {"status":"ran"}`}, tsaTranscript(t, second)); diff != "" {
		t.Errorf("second run's events mismatch (-want +got):\n%s", diff)
	}
	if !errors.Is(err, errDown) {
		t.Errorf("second run's error = %v, want the credential service's error", err)
	}
	if len(llm.Requests) != 1 {
		t.Errorf("model calls = %d, want 1 (the first run's)", len(llm.Requests))
	}
}
