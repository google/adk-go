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
	"fmt"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

func TestRunner_OnRunError(t *testing.T) {
	sentinel := errors.New("test failure")
	tests := []struct {
		name              string
		agentErrors       []error
		returnAfterErrors bool
		beforeRunErr      error
		onEventErr        error
		appendErr         error
		userMessageErr    error
		afterModelErr     error
		wantErr           error
		wantAfterRun      int
	}{
		{name: "success", wantAfterRun: 1},
		{name: "agent continues after errors", agentErrors: []error{sentinel, context.Canceled}, wantErr: sentinel, wantAfterRun: 1},
		{name: "agent returns after error", agentErrors: []error{sentinel}, returnAfterErrors: true, wantErr: sentinel, wantAfterRun: 1},
		{name: "before run error", beforeRunErr: sentinel, wantErr: sentinel, wantAfterRun: 1},
		{name: "on event error", onEventErr: sentinel, wantErr: sentinel, wantAfterRun: 1},
		{name: "persistence error", appendErr: sentinel, wantErr: sentinel, wantAfterRun: 1},
		{name: "user message error", userMessageErr: sentinel, wantErr: sentinel, wantAfterRun: 0},
		{name: "after model error", afterModelErr: sentinel, wantErr: sentinel, wantAfterRun: 1},
	}
	callers := []struct {
		name string
		stop bool
	}{
		{name: "caller_drains"},
		{name: "caller_stops", stop: true},
	}
	for _, kind := range []string{"custom", "node", "live"} {
		for _, tc := range tests {
			if (tc.afterModelErr != nil && kind != "node") || (tc.userMessageErr != nil && kind == "live") {
				continue
			}
			for _, caller := range callers {
				t.Run(kind+"/"+tc.name+"/"+caller.name, func(t *testing.T) {
					var observed [2][]error
					afterCalls := 0
					var invocationID string
					replacement := genai.NewContentFromText("replacement", genai.RoleUser)
					observe := func(index int) plugin.OnRunErrorCallback {
						return func(ctx agent.InvocationContext, err error) {
							if err == nil {
								t.Error("error callback received nil")
							}
							if afterCalls != 0 {
								t.Error("error callback ran after cleanup")
							}
							if kind != "live" && tc.userMessageErr == nil && ctx.UserContent() != replacement {
								t.Error("error callback did not receive rewritten user content")
							}
							if ctx.InvocationID() == "" {
								t.Error("missing invocation ID")
							}
							if invocationID != "" && invocationID != ctx.InvocationID() {
								t.Error("invocation ID changed")
							}
							invocationID = ctx.InvocationID()
							observed[index] = append(observed[index], err)
						}
					}
					p, err := plugin.New(plugin.Config{
						Name:               "observer",
						OnRunErrorCallback: observe(0),
						AfterRunCallback:   func(agent.InvocationContext) { afterCalls++ },
						BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
							return nil, tc.beforeRunErr
						},
						OnEventCallback: func(agent.InvocationContext, *session.Event) (*session.Event, error) {
							return nil, tc.onEventErr
						},
						OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
							return replacement, tc.userMessageErr
						},
					})
					if err != nil {
						t.Fatal(err)
					}
					second, err := plugin.New(plugin.Config{Name: "second", OnRunErrorCallback: observe(1)})
					if err != nil {
						t.Fatal(err)
					}
					empty, err := plugin.New(plugin.Config{Name: "no error callback"})
					if err != nil {
						t.Fatal(err)
					}
					sequence := func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
						return func(yield func(*session.Event, error) bool) {
							for _, err := range tc.agentErrors {
								if !yield(nil, err) {
									return
								}
							}
							if tc.returnAfterErrors {
								return
							}
							ev := session.NewEvent(ctx, ctx.InvocationID())
							ev.Author = "root"
							ev.Content = genai.NewContentFromText("answer", genai.RoleModel)
							yield(ev, nil)
						}
					}
					a := must(agent.New(agent.Config{Name: "root", Run: sequence}))
					if kind == "node" {
						m := &runErrorModel{}
						if len(tc.agentErrors) > 0 {
							m.err = tc.agentErrors[0]
						}
						a = must(llmagent.New(llmagent.Config{
							Name: "root", Model: m,
							AfterModelCallbacks: []llmagent.AfterModelCallback{func(agent.Context, *model.LLMResponse, error) (*model.LLMResponse, error) {
								return nil, tc.afterModelErr
							}},
						}))
					}
					if kind == "live" {
						a = &mockLiveAgent{Agent: a, runLiveFn: func(ctx agent.InvocationContext) (agent.LiveSession, iter.Seq2[*session.Event, error], error) {
							return &dummyLiveSession{}, sequence(ctx), nil
						}}
					}
					svc := &beforeRunSessionService{Service: session.InMemoryService(), err: tc.appendErr}
					r, err := New(Config{
						AppName: "app", Agent: a, SessionService: svc, AutoCreateSession: true,
						PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p, empty, second}},
					})
					if err != nil {
						t.Fatal(err)
					}
					events := r.Run(t.Context(), "user", "session", genai.NewContentFromText("question", genai.RoleUser), agent.RunConfig{})
					var setupErr error
					if kind == "live" {
						_, events, setupErr = r.RunLive(t.Context(), "user", "session", agent.LiveRunConfig{})
					}
					errorsSeen, eventsSeen := 0, 0
					check := func(err error) {
						t.Helper()
						errorsSeen++
						for i := range observed {
							if len(observed[i]) != errorsSeen {
								t.Fatalf("plugin %d notified %d times, want %d", i, len(observed[i]), errorsSeen)
							}
							if observed[i][errorsSeen-1] != err {
								t.Error("plugin did not receive the caller's exact error")
							}
						}
						wantErr := tc.wantErr
						if errorsSeen <= len(tc.agentErrors) {
							wantErr = tc.agentErrors[errorsSeen-1]
						}
						if !errors.Is(err, wantErr) {
							t.Error("unexpected error identity")
						}
					}
					if setupErr != nil {
						check(setupErr)
					} else {
						if len(observed[0]) != 0 {
							t.Fatal("error reported before iteration")
						}
						for ev, err := range events {
							if err != nil {
								check(err)
							} else if ev != nil {
								eventsSeen++
							}
							if caller.stop {
								break
							}
						}
					}
					if tc.wantErr == nil {
						if errorsSeen != 0 || len(observed[0]) != 0 || len(observed[1]) != 0 {
							t.Error("successful run reported an error")
						}
						if eventsSeen == 0 {
							t.Error("successful run yielded no events")
						}
					} else if errorsSeen == 0 {
						t.Error("expected a run error")
					}
					if len(tc.agentErrors) > 0 && kind != "node" && !caller.stop {
						wantEvents := 1
						if tc.returnAfterErrors {
							wantEvents = 0
						}
						if errorsSeen != len(tc.agentErrors) || eventsSeen != wantEvents {
							t.Errorf("stream counts = %d errors, %d events; want %d, %d", errorsSeen, eventsSeen, len(tc.agentErrors), wantEvents)
						}
					}
					// User-message and synchronous live failures predate the existing AfterRun lifecycle.
					wantAfter := tc.wantAfterRun
					if setupErr != nil {
						wantAfter = 0
					}
					if afterCalls != wantAfter {
						t.Errorf("AfterRun calls = %d, want %d", afterCalls, wantAfter)
					}
				})
			}
		}
	}
}

type runErrorModel struct{ err error }

func (*runErrorModel) Name() string { return "test" }
func (m *runErrorModel) GenerateContent(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if m.err != nil {
			yield(nil, m.err)
			return
		}
		yield(&model.LLMResponse{Content: genai.NewContentFromText("answer", genai.RoleModel)}, nil)
	}
}

func TestRunner_OnRunError_ChildError(t *testing.T) {
	for _, forward := range []bool{false, true} {
		t.Run(fmt.Sprintf("forward=%t", forward), func(t *testing.T) {
			sentinel := errors.New("child failure")
			child := must(agent.New(agent.Config{Name: "child", Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) { yield(nil, sentinel) }
			}}))
			root := must(agent.New(agent.Config{Name: "root", SubAgents: []agent.Agent{child}, Run: func(ctx agent.InvocationContext) iter.Seq2[*session.Event, error] {
				return func(yield func(*session.Event, error) bool) {
					for ev, err := range child.Run(ctx) {
						if forward && !yield(ev, err) {
							return
						}
					}
					ev := session.NewEvent(ctx, ctx.InvocationID())
					ev.Author = "root"
					yield(ev, nil)
				}
			}}))
			errorCalls, afterAgentCalls, afterRunCalls := 0, 0, 0
			p, err := plugin.New(plugin.Config{
				Name: "observer", OnRunErrorCallback: func(_ agent.InvocationContext, err error) {
					errorCalls++
					if err != sentinel {
						t.Error("error identity changed")
					}
				}, AfterAgentCallback: func(agent.Context) (*genai.Content, error) { afterAgentCalls++; return nil, nil },
				AfterRunCallback: func(agent.InvocationContext) { afterRunCalls++ },
			})
			if err != nil {
				t.Fatal(err)
			}
			r, err := New(Config{
				AppName: "app", Agent: root, SessionService: session.InMemoryService(), AutoCreateSession: true,
				PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}},
			})
			if err != nil {
				t.Fatal(err)
			}
			errorsSeen, eventsSeen := 0, 0
			for _, err := range r.Run(t.Context(), "user", "session", nil, agent.RunConfig{}) {
				if err != nil {
					errorsSeen++
				} else {
					eventsSeen++
				}
			}
			want := 0
			if forward {
				want = 1
			}
			if errorCalls != want || errorsSeen != want {
				t.Errorf("callback/caller error counts = %d/%d, want %d", errorCalls, errorsSeen, want)
			}
			if afterAgentCalls != 2 || afterRunCalls != 1 || eventsSeen != 1 {
				t.Errorf("agent/run/event counts = %d/%d/%d, want 2/1/1", afterAgentCalls, afterRunCalls, eventsSeen)
			}
		})
	}
}
