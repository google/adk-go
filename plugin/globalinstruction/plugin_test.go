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

package globalinstruction_test

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/artifact"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/plugin/globalinstruction"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/util/instructionutil"
)

const (
	appName   = "app"
	userID    = "test_user"
	sessionID = "s1"
)

var errBoom = errors.New("boom")

func newPlugin(t *testing.T, cfg globalinstruction.Config) *plugin.Plugin {
	t.Helper()
	p, err := globalinstruction.New(cfg)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return p
}

// callbackCtx builds the context a plugin's before-model callback receives
// during a run (base_flow.go builds it with
// agent.NewCallbackContextWithArtifactTracking), over a session of user uid
// holding state.
func callbackCtx(t *testing.T, uid string, state map[string]any) agent.Context {
	t.Helper()
	resp, err := session.InMemoryService().Create(t.Context(), &session.CreateRequest{
		AppName: appName,
		UserID:  uid,
		State:   state,
	})
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{Session: resp.Session})
	return agent.NewCallbackContextWithArtifactTracking(ic, nil)
}

func userContent(text string) *genai.Content {
	return &genai.Content{Role: genai.RoleUser, Parts: []*genai.Part{{Text: text}}}
}

func partTexts(c *genai.Content) []string {
	if c == nil {
		return nil
	}
	var out []string
	for _, p := range c.Parts {
		out = append(out, p.Text)
	}
	return out
}

// systemInstruction returns req's system instruction, or nil when req has no
// config.
func systemInstruction(req *model.LLMRequest) *genai.Content {
	if req.Config == nil {
		return nil
	}
	return req.Config.SystemInstruction
}

// firstPartText returns the text of the first part of req's system
// instruction, or "" when there is none.
func firstPartText(req *model.LLMRequest) string {
	texts := partTexts(systemInstruction(req))
	if len(texts) == 0 {
		return ""
	}
	return texts[0]
}

func textReply(text string) *genai.Content {
	return genai.NewContentFromText(text, genai.RoleModel)
}

func newAgent(t *testing.T, cfg llmagent.Config) agent.Agent {
	t.Helper()
	a, err := llmagent.New(cfg)
	if err != nil {
		t.Fatalf("llmagent.New() error = %v", err)
	}
	return a
}

// run sends one user message to root through a runner with plugins, over a
// session holding state, and returns the first error the run yields.
// artifacts may be nil.
func run(t *testing.T, root agent.Agent, state map[string]any, artifacts artifact.Service, plugins ...*plugin.Plugin) error {
	t.Helper()
	sessions := session.InMemoryService()
	if _, err := sessions.Create(t.Context(), &session.CreateRequest{AppName: appName, UserID: userID, SessionID: sessionID, State: state}); err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	r, err := runner.New(runner.Config{
		AppName:         appName,
		Agent:           root,
		SessionService:  sessions,
		ArtifactService: artifacts,
		PluginConfig:    runner.PluginConfig{Plugins: plugins},
	})
	if err != nil {
		t.Fatalf("runner.New() error = %v", err)
	}
	for _, err := range r.Run(t.Context(), userID, sessionID, userContent("hi"), agent.RunConfig{}) {
		if err != nil {
			return err
		}
	}
	return nil
}

func TestNewDefaultName(t *testing.T) {
	tests := []struct {
		name string
		cfg  globalinstruction.Config
		want string
	}{
		{name: "empty", cfg: globalinstruction.Config{}, want: "global_instruction"},
		{name: "custom", cfg: globalinstruction.Config{Name: "org_rules"}, want: "org_rules"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := newPlugin(t, tt.cfg).Name(); got != tt.want {
				t.Errorf("Name() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestNewRejectsInstructionAndProvider(t *testing.T) {
	provider := func(agent.ReadonlyContext) (string, error) { return "p", nil }

	p, err := globalinstruction.New(globalinstruction.Config{Instruction: "a", InstructionProvider: provider})
	if !errors.Is(err, globalinstruction.ErrMultipleProviders) {
		t.Errorf("New(both) error = %v, want %v", err, globalinstruction.ErrMultipleProviders)
	}
	if p != nil {
		t.Errorf("New(both) plugin = %v, want nil", p)
	}

	for _, cfg := range []globalinstruction.Config{{Instruction: "a"}, {InstructionProvider: provider}} {
		if _, err := globalinstruction.New(cfg); err != nil {
			t.Errorf("New(%+v) error = %v, want nil", cfg, err)
		}
	}
}

func TestInstructionInjectsStateFromRunnerCallbackContext(t *testing.T) {
	llm := &testutil.MockModel{Responses: []*genai.Content{textReply("ok")}}
	root := newAgent(t, llmagent.Config{Name: "root", Model: llm, Instruction: "Be brief."})
	p := newPlugin(t, globalinstruction.Config{Instruction: "Hello {user_name}."})

	if err := run(t, root, map[string]any{"user_name": "alice"}, nil, p); err != nil {
		t.Fatalf("run error = %v", err)
	}

	if len(llm.Requests) != 1 {
		t.Fatalf("model got %d requests, want 1", len(llm.Requests))
	}
	want := []string{"Hello alice.", "Be brief.\n\nYou are an agent. Your internal name is \"root\"."}
	if diff := cmp.Diff(want, partTexts(systemInstruction(llm.Requests[0]))); diff != "" {
		t.Errorf("system instruction parts mismatch (-want +got):\n%s", diff)
	}
}

func TestProviderCanInjectStateFromRunnerCallbackContext(t *testing.T) {
	llm := &testutil.MockModel{Responses: []*genai.Content{textReply("ok")}}
	root := newAgent(t, llmagent.Config{Name: "root", Model: llm})
	p := newPlugin(t, globalinstruction.Config{
		InstructionProvider: func(ctx agent.ReadonlyContext) (string, error) {
			return instructionutil.InjectSessionState(ctx, "Provider says hi to {user_name}")
		},
	})

	if err := run(t, root, map[string]any{"user_name": "alice"}, nil, p); err != nil {
		t.Fatalf("run error = %v", err)
	}

	if got := firstPartText(llm.Requests[0]); got != "Provider says hi to alice" {
		t.Errorf("first system instruction part = %q, want %q", got, "Provider says hi to alice")
	}
}

func TestProviderResultIsNotTemplated(t *testing.T) {
	p := newPlugin(t, globalinstruction.Config{
		InstructionProvider: func(agent.ReadonlyContext) (string, error) {
			return "Keep {user_name} as written.", nil
		},
	})
	req := &model.LLMRequest{}

	if _, err := p.BeforeModelCallback()(callbackCtx(t, userID, map[string]any{"user_name": "alice"}), req); err != nil {
		t.Fatalf("BeforeModelCallback() error = %v", err)
	}

	if diff := cmp.Diff([]string{"Keep {user_name} as written."}, partTexts(systemInstruction(req))); diff != "" {
		t.Errorf("system instruction parts mismatch (-want +got):\n%s", diff)
	}
}

func TestInstructionLoadsArtifact(t *testing.T) {
	artifacts := artifact.InMemoryService()
	if _, err := artifacts.Save(t.Context(), &artifact.SaveRequest{
		AppName: appName, UserID: userID, SessionID: sessionID, FileName: "notes.txt",
		Part: genai.NewPartFromText("Be brief."),
	}); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	llm := &testutil.MockModel{Responses: []*genai.Content{textReply("ok")}}
	root := newAgent(t, llmagent.Config{Name: "root", Model: llm})
	p := newPlugin(t, globalinstruction.Config{Instruction: "Rules: {artifact.notes.txt}"})

	if err := run(t, root, nil, artifacts, p); err != nil {
		t.Fatalf("run error = %v", err)
	}

	if got := firstPartText(llm.Requests[0]); got != "Rules: Be brief." {
		t.Errorf("first system instruction part = %q, want %q", got, "Rules: Be brief.")
	}
}

func TestResolutionErrorLeavesRequestUntouched(t *testing.T) {
	tests := []struct {
		name    string
		cfg     globalinstruction.Config
		wantErr error
	}{
		{
			name:    "missing state key",
			cfg:     globalinstruction.Config{Instruction: "Hi {missing}"},
			wantErr: session.ErrStateKeyNotExist,
		},
		{
			name: "provider error",
			cfg: globalinstruction.Config{InstructionProvider: func(agent.ReadonlyContext) (string, error) {
				return "ignored", errBoom
			}},
			wantErr: errBoom,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newPlugin(t, tt.cfg)
			req := &model.LLMRequest{Config: &genai.GenerateContentConfig{SystemInstruction: userContent("Keep me.")}}

			resp, err := p.BeforeModelCallback()(callbackCtx(t, userID, nil), req)

			if !errors.Is(err, tt.wantErr) {
				t.Errorf("BeforeModelCallback() error = %v, want %v", err, tt.wantErr)
			}
			if resp != nil {
				t.Errorf("BeforeModelCallback() response = %v, want nil", resp)
			}
			if diff := cmp.Diff(userContent("Keep me."), req.Config.SystemInstruction); diff != "" {
				t.Errorf("system instruction changed (-want +got):\n%s", diff)
			}
		})
	}
}

func TestEmptyResolutionLeavesRequestUntouched(t *testing.T) {
	tests := []struct {
		name string
		cfg  globalinstruction.Config
	}{
		{name: "zero config", cfg: globalinstruction.Config{}},
		{name: "template resolves to empty", cfg: globalinstruction.Config{Instruction: "{opt?}"}},
		{name: "provider returns empty", cfg: globalinstruction.Config{InstructionProvider: func(agent.ReadonlyContext) (string, error) {
			return "", nil
		}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newPlugin(t, tt.cfg)
			req := &model.LLMRequest{Config: &genai.GenerateContentConfig{SystemInstruction: userContent("Keep me.")}}

			resp, err := p.BeforeModelCallback()(callbackCtx(t, userID, nil), req)
			if resp != nil || err != nil {
				t.Fatalf("BeforeModelCallback() = (%v, %v), want (nil, nil)", resp, err)
			}

			if diff := cmp.Diff(userContent("Keep me."), req.Config.SystemInstruction); diff != "" {
				t.Errorf("system instruction changed (-want +got):\n%s", diff)
			}
		})
	}

	t.Run("nil config stays nil", func(t *testing.T) {
		p := newPlugin(t, globalinstruction.Config{})
		req := &model.LLMRequest{}

		if _, err := p.BeforeModelCallback()(callbackCtx(t, userID, nil), req); err != nil {
			t.Fatalf("BeforeModelCallback() error = %v", err)
		}

		if req.Config != nil {
			t.Errorf("req.Config = %+v, want nil", req.Config)
		}
	})
}

func TestFillsMissingOrEmptySystemInstruction(t *testing.T) {
	tests := []struct {
		name string
		req  *model.LLMRequest
		want *genai.Content
	}{
		{
			name: "nil config",
			req:  &model.LLMRequest{},
			want: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "Global instruction."}}},
		},
		{
			name: "nil system instruction",
			req:  &model.LLMRequest{Config: &genai.GenerateContentConfig{}},
			want: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "Global instruction."}}},
		},
		{
			name: "content without parts keeps its role",
			req:  &model.LLMRequest{Config: &genai.GenerateContentConfig{SystemInstruction: &genai.Content{Role: "model"}}},
			want: &genai.Content{Role: "model", Parts: []*genai.Part{{Text: "Global instruction."}}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newPlugin(t, globalinstruction.Config{Instruction: "Global instruction."})

			resp, err := p.BeforeModelCallback()(callbackCtx(t, userID, nil), tt.req)
			if resp != nil || err != nil {
				t.Fatalf("BeforeModelCallback() = (%v, %v), want (nil, nil)", resp, err)
			}

			if tt.req.Config == nil {
				t.Fatal("req.Config = nil, want a config holding the instruction")
			}
			if diff := cmp.Diff(tt.want, tt.req.Config.SystemInstruction); diff != "" {
				t.Errorf("system instruction mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestPrependsToContentKeepsOrderAndRole(t *testing.T) {
	p := newPlugin(t, globalinstruction.Config{Instruction: "Global instruction."})
	req := &model.LLMRequest{Config: &genai.GenerateContentConfig{SystemInstruction: &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{{Text: "First."}, {Text: "Second."}},
	}}}

	if _, err := p.BeforeModelCallback()(callbackCtx(t, userID, nil), req); err != nil {
		t.Fatalf("BeforeModelCallback() error = %v", err)
	}

	want := &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "Global instruction."}, {Text: "First."}, {Text: "Second."}}}
	if diff := cmp.Diff(want, systemInstruction(req)); diff != "" {
		t.Errorf("system instruction mismatch (-want +got):\n%s", diff)
	}
}

func TestDoesNotModifyTheOriginalContent(t *testing.T) {
	p := newPlugin(t, globalinstruction.Config{Instruction: "Global instruction."})
	shared := userContent("Existing.")
	ctx := callbackCtx(t, userID, nil)

	for i := range 2 {
		req := &model.LLMRequest{Config: &genai.GenerateContentConfig{SystemInstruction: shared}}
		if _, err := p.BeforeModelCallback()(ctx, req); err != nil {
			t.Fatalf("request %d: BeforeModelCallback() error = %v", i, err)
		}
		if diff := cmp.Diff([]string{"Global instruction.", "Existing."}, partTexts(systemInstruction(req))); diff != "" {
			t.Errorf("request %d: system instruction parts mismatch (-want +got):\n%s", i, diff)
		}
	}
	if diff := cmp.Diff([]string{"Existing."}, partTexts(shared)); diff != "" {
		t.Errorf("shared content changed (-want +got):\n%s", diff)
	}
}

func TestInstructionIsResolvedForEachModelCall(t *testing.T) {
	p := newPlugin(t, globalinstruction.Config{Instruction: "Step {n}."})
	first, second := &model.LLMRequest{}, &model.LLMRequest{}

	if _, err := p.BeforeModelCallback()(callbackCtx(t, userID, map[string]any{"n": 1}), first); err != nil {
		t.Fatalf("first BeforeModelCallback() error = %v", err)
	}
	if _, err := p.BeforeModelCallback()(callbackCtx(t, userID, map[string]any{"n": 2}), second); err != nil {
		t.Fatalf("second BeforeModelCallback() error = %v", err)
	}

	got := []string{firstPartText(first), firstPartText(second)}
	if diff := cmp.Diff([]string{"Step 1.", "Step 2."}, got); diff != "" {
		t.Errorf("first system instruction part per call mismatch (-want +got):\n%s", diff)
	}
}

func TestAppliesToEveryModelCallOfEveryAgent(t *testing.T) {
	llm := &testutil.MockModel{Responses: []*genai.Content{
		{Role: genai.RoleModel, Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
			Name: "transfer_to_agent", Args: map[string]any{"agent_name": "child"},
		}}}},
		textReply("ok"),
	}}
	child := newAgent(t, llmagent.Config{Name: "child", Model: llm})
	root := newAgent(t, llmagent.Config{Name: "root", Model: llm, SubAgents: []agent.Agent{child}})
	var calls atomic.Int32
	p := newPlugin(t, globalinstruction.Config{
		InstructionProvider: func(ctx agent.ReadonlyContext) (string, error) {
			calls.Add(1)
			return "Hello from " + ctx.AgentName(), nil
		},
	})

	if err := run(t, root, nil, nil, p); err != nil {
		t.Fatalf("run error = %v", err)
	}

	if got := calls.Load(); got != 2 {
		t.Errorf("provider calls = %d, want 2", got)
	}
	var got []string
	for _, req := range llm.Requests {
		got = append(got, firstPartText(req))
	}
	if diff := cmp.Diff([]string{"Hello from root", "Hello from child"}, got); diff != "" {
		t.Errorf("first system instruction part per request mismatch (-want +got):\n%s", diff)
	}
}

func TestAgentBeforeModelCallbackSeesGlobalInstruction(t *testing.T) {
	var seen string
	llm := &testutil.MockModel{Responses: []*genai.Content{textReply("ok")}}
	root := newAgent(t, llmagent.Config{
		Name:  "root",
		Model: llm,
		BeforeModelCallbacks: []llmagent.BeforeModelCallback{
			func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
				seen = firstPartText(req)
				return nil, nil
			},
		},
	})
	p := newPlugin(t, globalinstruction.Config{Instruction: "Global instruction."})

	if err := run(t, root, nil, nil, p); err != nil {
		t.Fatalf("run error = %v", err)
	}

	if seen != "Global instruction." {
		t.Errorf("agent callback saw first part %q, want %q", seen, "Global instruction.")
	}
}

func TestLeadsLlmAgentGlobalInstructionField(t *testing.T) {
	llm := &testutil.MockModel{Responses: []*genai.Content{textReply("ok")}}
	root := newAgent(t, llmagent.Config{Name: "root", Model: llm, GlobalInstruction: "Field.", Instruction: "Agent."})
	p := newPlugin(t, globalinstruction.Config{Instruction: "Plugin."})

	if err := run(t, root, nil, nil, p); err != nil {
		t.Fatalf("run error = %v", err)
	}

	want := []string{"Plugin.", "Field.\n\nAgent.\n\nYou are an agent. Your internal name is \"root\"."}
	if diff := cmp.Diff(want, partTexts(systemInstruction(llm.Requests[0]))); diff != "" {
		t.Errorf("system instruction parts mismatch (-want +got):\n%s", diff)
	}
}

func TestConcurrentCallbacksDoNotShareState(t *testing.T) {
	const n = 16
	p := newPlugin(t, globalinstruction.Config{
		InstructionProvider: func(ctx agent.ReadonlyContext) (string, error) {
			return "user=" + ctx.UserID(), nil
		},
	})
	ctxs := make([]agent.Context, n)
	reqs := make([]*model.LLMRequest, n)
	for i := range n {
		ctxs[i] = callbackCtx(t, fmt.Sprintf("u%d", i), nil)
		reqs[i] = &model.LLMRequest{}
	}

	var wg sync.WaitGroup
	errs := make([]error, n)
	for i := range n {
		wg.Go(func() {
			_, errs[i] = p.BeforeModelCallback()(ctxs[i], reqs[i])
		})
	}
	wg.Wait()

	for i := range n {
		if errs[i] != nil {
			t.Errorf("call %d: error = %v", i, errs[i])
			continue
		}
		want := []string{fmt.Sprintf("user=u%d", i)}
		if diff := cmp.Diff(want, partTexts(systemInstruction(reqs[i]))); diff != "" {
			t.Errorf("call %d: system instruction parts mismatch (-want +got):\n%s", i, diff)
		}
	}
}
