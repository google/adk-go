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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

func TestRunner_BeforeRunShortCircuit(t *testing.T) {
	beforeErr := errors.New("before run failed")
	onEventErr := errors.New("on event failed")
	appendErr := errors.New("append failed")
	tests := []struct {
		name        string
		beforeErr   error
		onEventErr  error
		appendErr   error
		withContent bool
		replace     bool
		partial     bool
		rewriteUser bool
		withoutUser bool
	}{
		{name: "reply", withContent: true},
		{name: "on event replaces reply", withContent: true, replace: true},
		{name: "partial reply is not persisted", withContent: true, partial: true},
		{name: "before run error", beforeErr: beforeErr},
		{name: "before run error takes precedence over reply", beforeErr: beforeErr, withContent: true},
		{name: "on event error", withContent: true, onEventErr: onEventErr},
		{name: "append error", withContent: true, appendErr: appendErr},
		{name: "rewritten user input", withContent: true, rewriteUser: true},
		{name: "no user message", withContent: true, withoutUser: true},
	}

	for _, kind := range []string{"custom", "llm"} {
		for _, tc := range tests {
			t.Run(kind+"/"+tc.name, func(t *testing.T) {
				ctx := t.Context()
				var a agent.Agent
				var err error
				if kind == "llm" {
					a, err = llmagent.New(llmagent.Config{Name: "assistant", Model: &noopModel{}})
				} else {
					a, err = agent.New(agent.Config{
						Name: "assistant",
						Run: func(agent.InvocationContext) iter.Seq2[*session.Event, error] {
							return func(func(*session.Event, error) bool) {}
						},
					})
				}
				if err != nil {
					t.Fatal(err)
				}

				userContent := genai.NewContentFromText("original question", genai.RoleUser)
				wantUserContent := userContent
				if tc.rewriteUser {
					wantUserContent = genai.NewContentFromText("sanitized question", genai.RoleUser)
				}
				wantUserEvents := 1
				if tc.withoutUser {
					userContent = nil
					wantUserEvents = 0
				}
				cachedReply := genai.NewContentFromText("cached answer", genai.RoleModel)
				wantReply := cachedReply
				if tc.replace {
					wantReply = genai.NewContentFromText("modified answer", genai.RoleModel)
				}
				onEventCalls, afterRunCalls := 0, 0
				p, err := plugin.New(plugin.Config{
					Name: "cache",
					OnUserMessageCallback: func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
						return wantUserContent, nil
					},
					BeforeRunCallback: func(agent.InvocationContext) (*genai.Content, error) {
						if tc.withContent {
							return cachedReply, tc.beforeErr
						}
						return nil, tc.beforeErr
					},
					BeforeAgentCallback: func(agent.Context) (*genai.Content, error) {
						t.Error("agent execution was not short-circuited")
						return nil, errors.New("unexpected agent execution")
					},
					OnEventCallback: func(ic agent.InvocationContext, ev *session.Event) (*session.Event, error) {
						onEventCalls++
						if ev.Author != a.Name() {
							t.Errorf("OnEvent author = %q, want %q", ev.Author, a.Name())
						}
						if diff := cmp.Diff(cachedReply, ev.Content); diff != "" {
							t.Errorf("OnEvent content mismatch (-want +got):\n%s", diff)
						}
						if got := ic.Session().Events().Len(); got != wantUserEvents {
							t.Errorf("history length before OnEvent = %d, want %d", got, wantUserEvents)
						}
						if tc.onEventErr != nil {
							return nil, tc.onEventErr
						}
						if tc.replace || tc.partial {
							modified := *ev
							modified.Content = wantReply
							modified.Partial = tc.partial
							modified.Actions.StateDelta = map[string]any{"reply_processed": true}
							return &modified, nil
						}
						return nil, nil
					},
					AfterRunCallback: func(agent.InvocationContext) {
						afterRunCalls++
					},
				})
				if err != nil {
					t.Fatal(err)
				}
				svc := session.InMemoryService()
				if tc.appendErr != nil {
					svc = &beforeRunAppendErrorService{Service: svc, err: tc.appendErr}
				}
				r, err := New(Config{
					AppName: "test", Agent: a, SessionService: svc, AutoCreateSession: true,
					PluginConfig: PluginConfig{Plugins: []*plugin.Plugin{p}},
				})
				if err != nil {
					t.Fatal(err)
				}

				var gotEvent *session.Event
				var gotErr error
				yieldCount := 0
				for ev, err := range r.Run(ctx, "user", "session", userContent, agent.RunConfig{}) {
					yieldCount++
					gotEvent, gotErr = ev, err
				}
				if yieldCount != 1 {
					t.Errorf("yield count = %d, want 1", yieldCount)
				}
				wantErr := tc.beforeErr
				if wantErr == nil {
					wantErr = tc.onEventErr
				}
				if wantErr == nil {
					wantErr = tc.appendErr
				}
				if !errors.Is(gotErr, wantErr) {
					t.Errorf("error = %v, want %v", gotErr, wantErr)
				}
				if wantErr == nil {
					if gotErr != nil {
						t.Fatalf("Run: %v", gotErr)
					}
					if gotEvent == nil {
						t.Fatal("no reply event")
					}
					if gotEvent.Author != a.Name() || gotEvent.Partial != tc.partial {
						t.Errorf("reply author/partial = %q/%v, want %q/%v", gotEvent.Author, gotEvent.Partial, a.Name(), tc.partial)
					}
					if diff := cmp.Diff(wantReply, gotEvent.Content); diff != "" {
						t.Errorf("reply mismatch (-want +got):\n%s", diff)
					}
				} else if gotEvent != nil {
					t.Errorf("error path yielded a content event: %+v", gotEvent)
				}
				wantOnEventCalls := 1
				if tc.beforeErr != nil {
					wantOnEventCalls = 0
				}
				if onEventCalls != wantOnEventCalls || afterRunCalls != 1 {
					t.Errorf("OnEvent/AfterRun calls = %d/%d, want %d/1", onEventCalls, afterRunCalls, wantOnEventCalls)
				}

				fresh, err := svc.Get(ctx, &session.GetRequest{AppName: "test", UserID: "user", SessionID: "session"})
				if err != nil {
					t.Fatal(err)
				}
				wantStoredEvents := wantUserEvents
				if wantErr == nil && !tc.partial {
					wantStoredEvents++
				}
				if got := fresh.Session.Events().Len(); got != wantStoredEvents {
					t.Fatalf("stored events = %d, want %d", got, wantStoredEvents)
				}
				if wantUserEvents > 0 {
					userEvent := fresh.Session.Events().At(0)
					if userEvent.Author != "user" {
						t.Errorf("user event author = %q", userEvent.Author)
					}
					if diff := cmp.Diff(wantUserContent, userEvent.Content); diff != "" {
						t.Errorf("stored user input mismatch (-want +got):\n%s", diff)
					}
				}
				if wantStoredEvents > wantUserEvents {
					storedReply := fresh.Session.Events().At(wantUserEvents)
					if diff := cmp.Diff(gotEvent, storedReply); diff != "" {
						t.Errorf("stored reply mismatch (-want +got):\n%s", diff)
					}
				}
				if tc.replace {
					if value, err := fresh.Session.State().Get("reply_processed"); err != nil || value != true {
						t.Errorf("OnEvent state delta was not persisted: value=%v, err=%v", value, err)
					}
				}
			})
		}
	}
}

type beforeRunAppendErrorService struct {
	session.Service
	err error
}

func (s *beforeRunAppendErrorService) AppendEvent(ctx context.Context, sess session.Session, ev *session.Event) error {
	if ev.Author != "user" {
		return s.err
	}
	return s.Service.AppendEvent(ctx, sess, ev)
}
