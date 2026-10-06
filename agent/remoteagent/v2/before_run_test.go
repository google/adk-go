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
	"errors"
	"iter"
	"maps"
	"testing"

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/google/go-cmp/cmp"
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
		{
			name: "converter with context metadata", contextID: "remote-context",
			converter: convertResponse,
		},
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

func TestRemoteReplyOrigin(t *testing.T) {
	responseKey := adka2a.ToADKMetaKey("response")
	for _, tc := range []struct {
		name     string
		metadata map[string]any
		want     map[string]any
		failed   bool
	}{
		{name: "no metadata", want: map[string]any{responseKey: true}},
		{
			name: "existing metadata", metadata: map[string]any{"custom": true},
			want: map[string]any{"custom": true, responseKey: true},
		},
		{
			name: "existing response", metadata: map[string]any{responseKey: map[string]any{"kind": "message"}},
			want: map[string]any{responseKey: map[string]any{"kind": "message"}},
		},
		{name: "local error recovery", failed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := &beforeRunClient{}
			if tc.failed {
				client.err = errors.New("transport failed")
			}
			converted := session.NewEvent(t.Context(), "invocation")
			converted.Author = "remote"
			converted.Content = genai.NewContentFromText("reply", genai.RoleModel)
			converted.CustomMetadata = maps.Clone(tc.metadata)
			remote, err := NewA2A(A2AConfig{
				Name: "remote", AgentCard: &a2a.AgentCard{},
				ClientProvider: func(context.Context, *a2a.AgentCard) (A2AClient, error) { return client, nil },
				Converter: func(agent.InvocationContext, *a2a.SendMessageRequest, a2a.Event, error) (*session.Event, error) {
					return converted, nil
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			ctx := newInvocationContextWithStreamingMode(t, []*session.Event{newEventFromParts("user", genai.NewPartFromText("question"))}, agent.StreamingModeNone)
			count := 0
			for ev, err := range remote.Run(ctx) {
				if err != nil {
					t.Fatal(err)
				}
				count++
				if !cmp.Equal(ev.CustomMetadata, tc.want) {
					t.Error("remote response metadata mismatch")
				}
			}
			if count != 1 {
				t.Errorf("events = %d, want 1", count)
			}
			if !cmp.Equal(converted.CustomMetadata, tc.metadata) {
				t.Error("converter-owned event metadata was mutated")
			}
		})
	}
}

type beforeRunClient struct {
	A2AClient
	contextID   string
	calls       int
	lastMessage *a2a.Message
	err         error
}

func (c *beforeRunClient) SendMessage(_ context.Context, req *a2a.SendMessageRequest) (a2a.SendMessageResult, error) {
	c.calls++
	c.lastMessage = req.Message
	if c.err != nil {
		return nil, c.err
	}
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
