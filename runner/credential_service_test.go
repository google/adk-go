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

package runner

import (
	"context"
	"iter"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/credentialservice"
	icredentialservice "google.golang.org/adk/v2/internal/credentialservice"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
	"google.golang.org/adk/v2/tool/toolauth"
	"google.golang.org/adk/v2/workflow"
)

// csConfig is the auth config every credential test saves and loads.
func csConfig() *toolauth.Config {
	return &toolauth.Config{
		AuthScheme:    toolauth.Scheme{Type: toolauth.SchemeTypeAPIKey, In: "header", Name: "X-Api-Key"},
		CredentialKey: "key-a",
		ExchangedAuthCredential: &toolauth.Credential{
			AuthType: toolauth.CredentialTypeOAuth2,
			OAuth2:   &toolauth.OAuth2Auth{ClientID: "client-a"},
		},
	}
}

// csServiceWithCredential returns an in-memory credential service holding
// csConfig's credential for cmApp and cmUser.
func csServiceWithCredential(t *testing.T) credentialservice.Service {
	t.Helper()
	svc := credentialservice.InMemoryService()
	if err := svc.SaveCredential(t.Context(), cmApp, cmUser, csConfig()); err != nil {
		t.Fatalf("SaveCredential() error = %v", err)
	}
	return svc
}

// csProbe records what agent.LoadCredential returns in each place a test
// calls it from: the loaded client ID, "none", or "error: <message>".
type csProbe struct {
	mu  sync.Mutex
	got map[string]string
}

func (p *csProbe) load(place string, ctx agent.Context) {
	cred, err := agent.LoadCredential(ctx, csConfig())
	var res string
	switch {
	case err != nil:
		res = "error: " + err.Error()
	case cred == nil || cred.OAuth2 == nil:
		res = "none"
	default:
		res = cred.OAuth2.ClientID
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.got == nil {
		p.got = map[string]string{}
	}
	p.got[place] = res
}

func (p *csProbe) results() map[string]string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.got
}

// csModel answers the n-th request with responses[n], and with the last
// response once the script is used up.
type csModel struct {
	mu        sync.Mutex
	responses []*genai.Content
	call      int
}

func (m *csModel) Name() string { return "cs" }

func (m *csModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	m.mu.Lock()
	i := min(m.call, len(m.responses)-1)
	m.call++
	m.mu.Unlock()
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(&model.LLMResponse{Content: m.responses[i]}, nil)
	}
}

// csProbeTool returns a tool named name that records what it loads under
// place.
func csProbeTool(t *testing.T, p *csProbe, name, place string) tool.Tool {
	t.Helper()
	tl, err := functiontool.New(functiontool.Config{Name: name, Description: "Loads a credential."},
		func(ctx agent.Context, _ map[string]any) (map[string]any, error) {
			p.load(place, ctx)
			return map[string]any{}, nil
		})
	if err != nil {
		t.Fatalf("functiontool.New() error = %v", err)
	}
	return tl
}

// csRecordingAgent returns a custom root agent that records what it loads
// from a callback context built over its invocation context.
func csRecordingAgent(t *testing.T, p *csProbe) agent.Agent {
	t.Helper()
	return cmCustomAgent(t, func(ic agent.InvocationContext) []*session.Event {
		p.load("root", agent.NewCallbackContext(ic, nil))
		return nil
	})
}

func csRunner(t *testing.T, cfg Config) *Runner {
	t.Helper()
	cfg.AppName, cfg.SessionService, cfg.AutoCreateSession = cmApp, session.InMemoryService(), true
	r, err := New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return r
}

func csCheck(t *testing.T, got, want map[string]string) {
	t.Helper()
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("agent.LoadCredential results by place mismatch (-want +got):\n%s", diff)
	}
}

func TestRunner_CredentialServiceReachesContexts(t *testing.T) {
	p := &csProbe{}
	sub, err := llmagent.New(llmagent.Config{
		Name: "sub",
		Model: &csModel{responses: []*genai.Content{
			genai.NewContentFromFunctionCall("sub_probe", map[string]any{}, genai.RoleModel),
			genai.NewContentFromText("done", genai.RoleModel),
		}},
		Tools: []tool.Tool{csProbeTool(t, p, "sub_probe", "sub-agent tool")},
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(ctx agent.Context) (*genai.Content, error) {
			p.load("sub-agent before-agent callback", ctx)
			return nil, nil
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(sub) error = %v", err)
	}
	root, err := llmagent.New(llmagent.Config{
		Name: "root",
		Model: &csModel{responses: []*genai.Content{
			genai.NewContentFromFunctionCall("root_probe", map[string]any{}, genai.RoleModel),
			genai.NewContentFromFunctionCall("transfer_to_agent", map[string]any{"agent_name": "sub"}, genai.RoleModel),
		}},
		Tools:     []tool.Tool{csProbeTool(t, p, "root_probe", "root tool")},
		SubAgents: []agent.Agent{sub},
		BeforeAgentCallbacks: []agent.BeforeAgentCallback{func(ctx agent.Context) (*genai.Content, error) {
			p.load("root before-agent callback", ctx)
			return nil, nil
		}},
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
			p.load("root before-model callback", ctx)
			return nil, nil
		}},
		AfterToolCallbacks: []llmagent.AfterToolCallback{func(ctx agent.Context, _ tool.Tool, _, _ map[string]any, _ error) (map[string]any, error) {
			p.load("root after-tool callback", ctx)
			return nil, nil
		}},
	})
	if err != nil {
		t.Fatalf("llmagent.New(root) error = %v", err)
	}
	r := csRunner(t, Config{Agent: root, CredentialService: csServiceWithCredential(t)})

	cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), agent.RunConfig{}))

	csCheck(t, p.results(), map[string]string{
		"root before-agent callback":      "client-a",
		"root before-model callback":      "client-a",
		"root tool":                       "client-a",
		"root after-tool callback":        "client-a",
		"sub-agent before-agent callback": "client-a",
		"sub-agent tool":                  "client-a",
	})
}

func TestRunner_CredentialServiceOnEveryRootKind(t *testing.T) {
	tests := []struct {
		name string
		cfg  func(p *csProbe) Config
	}{
		{
			name: "custom agent root",
			cfg:  func(p *csProbe) Config { return Config{Agent: csRecordingAgent(t, p)} },
		},
		{
			name: "workflow node root",
			cfg: func(p *csProbe) Config {
				return Config{Node: workflow.NewFunctionNode("root", func(ctx agent.Context, _ any) (string, error) {
					p.load("root", ctx)
					return "ok", nil
				}, workflow.NodeConfig{})}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &csProbe{}
			cfg := tt.cfg(p)
			cfg.CredentialService = csServiceWithCredential(t)
			r := csRunner(t, cfg)

			cmCollect(t, r.Run(t.Context(), cmUser, "s", cmUserText(), agent.RunConfig{}))

			csCheck(t, p.results(), map[string]string{"root": "client-a"})
		})
	}
}

func TestRunner_RunLive_CredentialService(t *testing.T) {
	p := &csProbe{}
	a := cmLiveAgent(t, func(ic agent.InvocationContext) []*session.Event {
		p.load("root", agent.NewCallbackContext(ic, nil))
		return nil
	})
	r := csRunner(t, Config{Agent: a, CredentialService: csServiceWithCredential(t)})

	_, seq, err := r.RunLive(t.Context(), cmUser, "s", agent.LiveRunConfig{})
	if err != nil {
		t.Fatalf("RunLive() error = %v", err)
	}
	cmCollect(t, seq)

	csCheck(t, p.results(), map[string]string{"root": "client-a"})
}

// A runner's own configuration decides whether its invocations have a
// credential service: one built without a service has none, even when the
// context it runs on carries one.
func TestRunner_NoCredentialServiceByDefault(t *testing.T) {
	tests := []struct {
		name   string
		runner func(t *testing.T, a agent.Agent) *Runner
		ctx    func(t *testing.T) context.Context
	}{
		{
			name:   "New without CredentialService",
			runner: func(t *testing.T, a agent.Agent) *Runner { return csRunner(t, Config{Agent: a}) },
			ctx:    func(t *testing.T) context.Context { return t.Context() },
		},
		{
			name: "NewInMemory",
			runner: func(t *testing.T, a agent.Agent) *Runner {
				r, err := NewInMemory(cmApp, a)
				if err != nil {
					t.Fatalf("NewInMemory() error = %v", err)
				}
				return r
			},
			ctx: func(t *testing.T) context.Context { return t.Context() },
		},
		{
			name:   "New without CredentialService, run context carries a service",
			runner: func(t *testing.T, a agent.Agent) *Runner { return csRunner(t, Config{Agent: a}) },
			ctx: func(t *testing.T) context.Context {
				return icredentialservice.WithService(t.Context(), csServiceWithCredential(t))
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &csProbe{}
			r := tt.runner(t, csRecordingAgent(t, p))

			cmCollect(t, r.Run(tt.ctx(t), cmUser, "s", cmUserText(), agent.RunConfig{}))

			csCheck(t, p.results(), map[string]string{"root": "error: credential service is not initialized"})
		})
	}
}
