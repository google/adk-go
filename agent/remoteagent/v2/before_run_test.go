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

package remoteagent

import (
	"context"
	"iter"
	"slices"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/server/adka2a/v2"
	"google.golang.org/adk/v2/session"
)

func TestBeforeRunReplyPreservesRemoteSession(t *testing.T) {
	convertResponse := func(ctx agent.InvocationContext, _ *a2a.SendMessageRequest, ev a2a.Event, err error) (*session.Event, error) {
		if err != nil {
			return nil, err
		}
		return adka2a.ToSessionEvent(ctx, ev)
	}
	for _, tc := range []struct {
		name         string
		contextID    string
		converter    A2AEventConverter
		replace      bool
		functionCall bool
	}{
		{name: "stateful", contextID: "remote-context"},
		{name: "stateless"},
		{name: "converter without metadata", converter: convertResponse},
		{name: "after callback without metadata", replace: true},
		{name: "cached function call", contextID: "remote-context", functionCall: true},
	} {
		for _, mode := range []agent.StreamingMode{agent.StreamingModeNone, agent.StreamingModeSSE} {
			t.Run(tc.name+"/"+string(mode), func(t *testing.T) {
				client := &beforeRunClient{contextID: tc.contextID}
				cfg := A2AConfig{
					Name: "remote", AgentCard: &a2a.AgentCard{}, Converter: tc.converter,
					ClientProvider: func(context.Context, *a2a.AgentCard) (A2AClient, error) { return client, nil },
				}
				if tc.replace {
					cfg.AfterRequestCallbacks = []AfterA2ARequestCallback{
						func(_ agent.Context, _ *a2a.SendMessageRequest, ev *session.Event, err error) (*session.Event, error) {
							out := *ev
							out.CustomMetadata = nil
							return &out, err
						},
					}
				}
				remote, err := NewA2A(cfg)
				if err != nil {
					t.Fatal(err)
				}
				turn := 0
				cache, err := plugin.New(plugin.Config{
					Name: "cache",
					OnEventCallback: func(_ agent.InvocationContext, ev *session.Event) (*session.Event, error) {
						if turn != 2 {
							return nil, nil
						}
						out := *ev
						out.CustomMetadata = map[string]any{"cache": true}
						return &out, nil
					},
					BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
						turn++
						if turn == 2 {
							if tc.functionCall {
								return genai.NewContentFromParts([]*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "cached-call", Name: "lookup"}}}, genai.RoleModel), nil
							}
							return genai.NewContentFromText("cached reply", genai.RoleModel), nil
						}
						return nil, nil
					},
				})
				if err != nil {
					t.Fatal(err)
				}
				r, err := runner.New(runner.Config{
					AppName: "test", Agent: remote, SessionService: session.InMemoryService(), AutoCreateSession: true,
					PluginConfig: runner.PluginConfig{Plugins: []*plugin.Plugin{cache}},
				})
				if err != nil {
					t.Fatal(err)
				}
				for i, question := range []string{"first question", "cached question", "next question"} {
					content := genai.NewContentFromText(question, genai.RoleUser)
					if tc.functionCall && i == 2 {
						content = genai.NewContentFromParts([]*genai.Part{{FunctionResponse: &genai.FunctionResponse{
							ID: "cached-call", Name: "lookup", Response: map[string]any{"ok": true},
						}}}, genai.RoleUser)
					}
					for ev, err := range r.Run(t.Context(), "u", "s", content, agent.RunConfig{StreamingMode: mode}) {
						if err != nil {
							t.Fatal(err)
						}
						if ev.ErrorCode != "" || ev.ErrorMessage != "" {
							t.Fatal("run emitted an error event")
						}
					}
				}
				if client.calls != 2 {
					t.Fatalf("remote calls = %d, want 2", client.calls)
				}
				msg := client.lastMessage
				if msg.ContextID != tc.contextID {
					t.Error("remote context was not preserved")
				}
				if msg.TaskID != "" {
					t.Error("local reply incorrectly resumed a remote task")
				}
				want := []string{"cached question", "For context:", "[remote] said: cached reply", "next question"}
				if tc.functionCall {
					want = []string{"cached question", "For context:", "[remote] called tool lookup with parameters: map[]", `Tool lookup returned: {"ok":true}`}
				}
				if len(msg.Parts) != len(want) {
					t.Fatalf("forwarded parts = %d, want %d", len(msg.Parts), len(want))
				}
				for i, text := range want {
					if msg.Parts[i].Text() != text {
						t.Errorf("forwarded part %d differs", i)
					}
				}
			})
		}
	}
}

func TestCallbackReplyKeepsHistoryBoundary(t *testing.T) {
	for _, kind := range []string{"before agent", "plugin before agent", "before request"} {
		for _, mode := range []agent.StreamingMode{agent.StreamingModeNone, agent.StreamingModeSSE} {
			t.Run(kind+"/"+string(mode), func(t *testing.T) {
				client := &beforeRunClient{contextID: "remote-context"}
				cfg := A2AConfig{
					Name: "remote", AgentCard: &a2a.AgentCard{},
					ClientProvider: func(context.Context, *a2a.AgentCard) (A2AClient, error) { return client, nil },
				}
				denied := 0
				refusal := genai.NewContentFromText("Request withheld.", genai.RoleModel)
				guard := func(ctx agent.Context) (*genai.Content, error) {
					if slices.ContainsFunc(ctx.UserContent().Parts, func(p *genai.Part) bool { return p.Text == "blocked input" }) {
						denied++
						return refusal, nil
					}
					return nil, nil
				}
				var plugins []*plugin.Plugin
				switch kind {
				case "before agent":
					cfg.BeforeAgentCallbacks = []agent.BeforeAgentCallback{guard}
				case "plugin before agent":
					p, err := plugin.New(plugin.Config{Name: "guard", BeforeAgentCallback: guard})
					if err != nil {
						t.Fatal(err)
					}
					plugins = []*plugin.Plugin{p}
				case "before request":
					cfg.BeforeRequestCallbacks = []BeforeA2ARequestCallback{
						func(ctx agent.Context, req *a2a.SendMessageRequest) (*session.Event, error) {
							if slices.ContainsFunc(req.Message.Parts, func(p *a2a.Part) bool { return p.Text() == "blocked input" }) {
								denied++
								ev := session.NewEvent(ctx, "refusal")
								ev.Author = "remote"
								ev.Content = refusal
								return ev, nil
							}
							return nil, nil
						},
					}
				}
				remote, err := NewA2A(cfg)
				if err != nil {
					t.Fatal(err)
				}
				r, err := runner.New(runner.Config{
					AppName: "test", Agent: remote, SessionService: session.InMemoryService(), AutoCreateSession: true,
					PluginConfig: runner.PluginConfig{Plugins: plugins},
				})
				if err != nil {
					t.Fatal(err)
				}
				wantCalls := 0
				for i, question := range []string{"first question", "blocked input", "next question", "last question"} {
					for ev, err := range r.Run(t.Context(), "u", "s", genai.NewContentFromText(question, genai.RoleUser), agent.RunConfig{StreamingMode: mode}) {
						if err != nil {
							t.Fatal(err)
						}
						if ev.ErrorCode != "" || ev.ErrorMessage != "" {
							t.Fatal("run emitted an error event")
						}
					}
					if i != 1 {
						wantCalls++
					}
					if client.calls != wantCalls {
						t.Fatalf("turn %d: remote calls = %d, want %d", i, client.calls, wantCalls)
					}
					if i == 1 {
						continue
					}
					msg := client.lastMessage
					if len(msg.Parts) != 1 || msg.Parts[0].Text() != question {
						t.Errorf("turn %d: callback boundary replayed earlier content", i)
					}
					wantContext := ""
					if i == 3 {
						wantContext = client.contextID
					}
					if msg.ContextID != wantContext {
						t.Errorf("turn %d: context ID did not follow the latest history boundary", i)
					}
				}
				if denied != 1 {
					t.Errorf("denied turns = %d, want 1", denied)
				}
			})
		}
	}
}

type beforeRunClient struct {
	A2AClient
	contextID   string
	calls       int
	lastMessage *a2a.Message
}

func (c *beforeRunClient) SendMessage(_ context.Context, req *a2a.SendMessageRequest) (a2a.SendMessageResult, error) {
	c.calls++
	c.lastMessage = req.Message
	msg := a2a.NewMessage(a2a.MessageRoleAgent, a2a.NewTextPart("remote reply"))
	msg.ContextID = c.contextID
	return msg, nil
}

func (c *beforeRunClient) SendStreamingMessage(ctx context.Context, req *a2a.SendMessageRequest) iter.Seq2[a2a.Event, error] {
	return func(yield func(a2a.Event, error) bool) {
		yield(c.SendMessage(ctx, req))
	}
}

func (c *beforeRunClient) Destroy() error { return nil }
