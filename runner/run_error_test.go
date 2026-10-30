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
	"errors"
	"iter"
	"slices"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/agent/workflowagents/sequentialagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/session/compaction"
)

type runErrorModel struct{ err error }

func (m runErrorModel) Name() string { return "error_model" }

func (m runErrorModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		yield(nil, m.err)
	}
}

func TestRunner_OnRunErrorCallback_Run(t *testing.T) {
	runErr := errors.New("run failed")
	for _, tc := range []struct {
		name    string
		kind    string
		failure string
	}{
		{name: "legacy before_run", kind: "legacy", failure: "before_run"},
		{name: "node before_run", kind: "node", failure: "before_run"},
		{name: "legacy on_user_message", kind: "legacy", failure: "on_user_message"},
		{name: "node on_user_message", kind: "node", failure: "on_user_message"},
		{name: "legacy agent", kind: "legacy", failure: "agent"},
		{name: "node model", kind: "node", failure: "model"},
		{name: "legacy on_event", kind: "legacy", failure: "on_event"},
		{name: "legacy append", kind: "legacy", failure: "append"},
		{name: "node append", kind: "node", failure: "append"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var a agent.Agent
			var err error
			if tc.kind == "node" {
				var m model.LLM = noopModel{}
				if tc.failure == "model" {
					m = runErrorModel{err: runErr}
				}
				a, err = llmagent.New(llmagent.Config{Name: "assistant", Model: m})
			} else {
				a, err = agent.New(agent.Config{
					Name: "assistant",
					Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							switch tc.failure {
							case "agent":
								yield(nil, runErr)
							case "on_event":
								yield(session.NewEvent(ctx, ctx.InvocationID()), nil)
							}
						}
					},
				})
			}
			if err != nil {
				t.Fatal(err)
			}

			var calls []string
			var callbackErrors []error
			plugins := make([]*plugin.Plugin, 0, 2)
			for _, name := range []string{"first", "second"} {
				p, err := plugin.New(plugin.Config{
					Name: name,
					OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
						if tc.failure == "on_user_message" {
							return nil, runErr
						}
						return nil, nil
					},
					BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
						if tc.failure == "before_run" {
							return nil, runErr
						}
						if tc.failure == "append" {
							return genai.NewContentFromText("reply", genai.RoleModel), nil
						}
						return nil, nil
					},
					OnEventCallback: func(agent.InvocationContext, *session.Event) (*session.Event, error) {
						if tc.failure == "on_event" {
							return nil, runErr
						}
						return nil, nil
					},
					OnRunErrorCallback: func(ctx agent.InvocationContext, gotErr error) {
						if ctx.Session().ID() != "session" || ctx.Agent().Name() != "assistant" {
							t.Errorf("callback context has session %q and agent %q", ctx.Session().ID(), ctx.Agent().Name())
						}
						if !errors.Is(gotErr, runErr) {
							t.Errorf("callback error = %v, want wrapped run error", gotErr)
						}
						calls = append(calls, name)
						callbackErrors = append(callbackErrors, gotErr)
					},
				})
				if err != nil {
					t.Fatal(err)
				}
				plugins = append(plugins, p)
			}
			noHook, err := plugin.New(plugin.Config{Name: "no_hook"})
			if err != nil {
				t.Fatal(err)
			}
			plugins = []*plugin.Plugin{plugins[0], noHook, plugins[1]}
			svc := session.InMemoryService()
			if tc.failure == "append" {
				svc = &beforeRunSessionService{Service: svc, err: runErr}
			}
			r, err := New(Config{
				AppName: "test", Agent: a, SessionService: svc, AutoCreateSession: true,
				PluginConfig: PluginConfig{Plugins: plugins},
			})
			if err != nil {
				t.Fatal(err)
			}

			var msg *genai.Content
			if tc.failure == "on_user_message" {
				msg = genai.NewContentFromText("question", genai.RoleUser)
			}
			var gotErr error
			for _, err := range r.Run(t.Context(), "user", "session", msg, agent.RunConfig{}) {
				if err != nil {
					gotErr = err
				}
			}
			if !errors.Is(gotErr, runErr) {
				t.Errorf("Run error = %v, want wrapped run error", gotErr)
			}
			if !slices.Equal(calls, []string{"first", "second"}) {
				t.Errorf("callback order = %v, want both plugins once", calls)
			}
			for i, callbackErr := range callbackErrors {
				if callbackErr != gotErr {
					t.Errorf("callback %d got %v, want exact yielded error %v", i, callbackErr, gotErr)
				}
			}
		})
	}
}

func TestRunner_OnRunErrorCallback_CompactionAfterEarlyExit(t *testing.T) {
	for _, kind := range []string{"legacy", "node"} {
		t.Run(kind, func(t *testing.T) {
			leaf, err := llmagent.New(llmagent.Config{Name: "assistant", Model: &scriptedModel{replyFmt: "answer %d"}})
			if err != nil {
				t.Fatal(err)
			}
			root := leaf
			if kind == "legacy" {
				root, err = sequentialagent.New(sequentialagent.Config{AgentConfig: agent.Config{Name: "sequence", SubAgents: []agent.Agent{leaf}}})
				if err != nil {
					t.Fatal(err)
				}
			}
			var callbackErrors []error
			p, err := plugin.New(plugin.Config{
				Name: "observer",
				OnRunErrorCallback: func(_ agent.InvocationContext, err error) {
					callbackErrors = append(callbackErrors, err)
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			r, err := New(Config{
				AppName: "test", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true,
				PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}},
				Compaction:   &compaction.Config{CompactionInterval: 1, Summarizer: failingSummarizer{}},
			})
			if err != nil {
				t.Fatal(err)
			}
			sawReply := false
			for ev, err := range r.Run(t.Context(), "user", "session", genai.NewContentFromText("question", genai.RoleUser), agent.RunConfig{}) {
				if err != nil {
					t.Fatalf("run failed before the caller stopped: %v", err)
				}
				if ev != nil && ev.Author == "assistant" && ev.LLMResponse.Content != nil {
					sawReply = true
					break
				}
			}
			if !sawReply {
				t.Fatal("run yielded no answer before early exit")
			}
			if len(callbackErrors) != 1 || !errors.Is(callbackErrors[0], compaction.ErrCompaction) {
				t.Errorf("callback errors = %v, want one compaction error", callbackErrors)
			}
		})
	}
}

func TestRunner_OnRunErrorCallback_RunLive(t *testing.T) {
	runErr := errors.New("live run failed")
	for _, failure := range []string{"before_run", "early_reply_append", "start", "stream"} {
		t.Run(failure, func(t *testing.T) {
			var calls []error
			p, err := plugin.New(plugin.Config{
				Name: "observer",
				BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
					if failure == "before_run" {
						return nil, runErr
					}
					if failure == "early_reply_append" {
						return genai.NewContentFromText("reply", genai.RoleModel), nil
					}
					return nil, nil
				},
				OnRunErrorCallback: func(ctx agent.InvocationContext, err error) {
					if ctx.Session().ID() != "session" {
						t.Errorf("callback session = %q, want session", ctx.Session().ID())
					}
					calls = append(calls, err)
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			a := &mockLiveAgent{
				Agent: must(agent.New(agent.Config{Name: "assistant"})),
				runLiveFn: func(agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
					if failure == "before_run" || failure == "early_reply_append" {
						t.Error("RunLive started after an early exit")
					}
					if failure == "start" {
						return nil, nil, runErr
					}
					return &dummyLiveSession{}, func(yield func(*session.Event, error) bool) {
						yield(nil, runErr)
					}, nil
				},
			}
			svc := session.InMemoryService()
			if failure == "early_reply_append" {
				svc = &beforeRunSessionService{Service: svc, err: runErr}
			}
			r, err := New(Config{
				AppName: "test", Agent: a, SessionService: svc, AutoCreateSession: true,
				PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}},
			})
			if err != nil {
				t.Fatal(err)
			}
			sess, events, gotErr := r.RunLive(t.Context(), "user", "session", agent.LiveRunConfig{})
			if failure == "stream" {
				if gotErr != nil || sess == nil || events == nil {
					t.Fatalf("RunLive returned session %v, events %v, error %v", sess, events, gotErr)
				}
				for _, err := range events {
					gotErr = err
				}
			} else if sess != nil || events != nil {
				t.Error("RunLive returned a session or iterator after an error")
			}
			if !errors.Is(gotErr, runErr) || len(calls) != 1 || calls[0] != gotErr {
				t.Errorf("RunLive error = %v, callback errors = %v; want the same error once", gotErr, calls)
			}
		})
	}
}

func TestRunner_OnRunErrorCallback_SuccessfulRunLive(t *testing.T) {
	for _, earlyReply := range []bool{false, true} {
		t.Run(map[bool]string{false: "stream", true: "early_reply"}[earlyReply], func(t *testing.T) {
			calls := 0
			p, err := plugin.New(plugin.Config{
				Name: "observer",
				BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
					if earlyReply {
						return genai.NewContentFromText("reply", genai.RoleModel), nil
					}
					return nil, nil
				},
				OnRunErrorCallback: func(agent.InvocationContext, error) { calls++ },
			})
			if err != nil {
				t.Fatal(err)
			}
			a := &mockLiveAgent{
				Agent: must(agent.New(agent.Config{Name: "assistant"})),
				runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
					if earlyReply {
						t.Error("RunLive started after an early reply")
					}
					return &dummyLiveSession{}, func(yield func(*session.Event, error) bool) {
						ev := session.NewEvent(ctx, ctx.InvocationID())
						ev.Author = "assistant"
						ev.Content = genai.NewContentFromText("reply", genai.RoleModel)
						yield(ev, nil)
					}, nil
				},
			}
			r, err := New(Config{
				AppName: "test", Agent: a, SessionService: session.InMemoryService(), AutoCreateSession: true,
				PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}},
			})
			if err != nil {
				t.Fatal(err)
			}
			_, events, err := r.RunLive(t.Context(), "user", "session", agent.LiveRunConfig{})
			if err != nil {
				t.Fatal(err)
			}
			count := 0
			for ev, err := range events {
				if err != nil {
					t.Fatal(err)
				}
				if ev != nil {
					count++
				}
			}
			if count != 1 || calls != 0 {
				t.Errorf("events = %d, error callbacks = %d; want one event and no error callbacks", count, calls)
			}
		})
	}
}
