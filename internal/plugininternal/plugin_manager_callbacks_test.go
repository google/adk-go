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

package plugininternal

import (
	"errors"
	"slices"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/plugin"
	"google.golang.org/adk/v2/session"
)

func newTestInvocation(t *testing.T) agent.InvocationContext {
	t.Helper()
	return icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
}

func TestPluginManager_CallbackErrors_NameThePluginAndCallback(t *testing.T) {
	errBoom := errors.New("boom")
	tests := []struct {
		name string
		// set makes cfg's callback of this row call record and return err.
		set     func(cfg *plugin.Config, record func(), err error)
		run     func(pm *PluginManager, ictx agent.InvocationContext) error
		wantMsg string
	}{
		{
			name: "RunOnUserMessageCallback",
			set: func(cfg *plugin.Config, record func(), err error) {
				cfg.OnUserMessageCallback = func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
					record()
					return nil, err
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) error {
				_, err := pm.RunOnUserMessageCallback(ictx, genai.NewContentFromText("hi", genai.RoleUser))
				return err
			},
			wantMsg: "Error in plugin 'bad' during 'on_user_message_callback' callback: boom",
		},
		{
			name: "RunBeforeRunCallback",
			set: func(cfg *plugin.Config, record func(), err error) {
				cfg.BeforeRunCallback = func(agent.InvocationContext) (*genai.Content, error) {
					record()
					return nil, err
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) error {
				_, err := pm.RunBeforeRunCallback(ictx)
				return err
			},
			wantMsg: "Error in plugin 'bad' during 'before_run_callback' callback: boom",
		},
		{
			name: "RunAfterRunCallback",
			set: func(cfg *plugin.Config, record func(), err error) {
				cfg.FailableAfterRunCallback = func(agent.InvocationContext) error {
					record()
					return err
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) error {
				return pm.RunAfterRunCallback(ictx)
			},
			wantMsg: "Error in plugin 'bad' during 'after_run_callback' callback: boom",
		},
		{
			name: "RunOnEventCallback",
			set: func(cfg *plugin.Config, record func(), err error) {
				cfg.OnEventCallback = func(agent.InvocationContext, *session.Event) (*session.Event, error) {
					record()
					return nil, err
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) error {
				_, err := pm.RunOnEventCallback(ictx, &session.Event{ID: "e1"})
				return err
			},
			wantMsg: "Error in plugin 'bad' during 'on_event_callback' callback: boom",
		},
		{
			name: "RunBeforeAgentCallback",
			set: func(cfg *plugin.Config, record func(), err error) {
				cfg.BeforeAgentCallback = func(agent.Context) (*genai.Content, error) {
					record()
					return nil, err
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) error {
				_, err := pm.RunBeforeAgentCallback(icontext.NewCallbackContext(ictx))
				return err
			},
			wantMsg: "Error in plugin 'bad' during 'before_agent_callback' callback: boom",
		},
		{
			name: "RunAfterAgentCallback",
			set: func(cfg *plugin.Config, record func(), err error) {
				cfg.AfterAgentCallback = func(agent.Context) (*genai.Content, error) {
					record()
					return nil, err
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) error {
				_, err := pm.RunAfterAgentCallback(icontext.NewCallbackContext(ictx))
				return err
			},
			wantMsg: "Error in plugin 'bad' during 'after_agent_callback' callback: boom",
		},
		{
			name: "RunBeforeModelCallback",
			set: func(cfg *plugin.Config, record func(), err error) {
				cfg.BeforeModelCallback = func(agent.Context, *model.LLMRequest) (*model.LLMResponse, error) {
					record()
					return nil, err
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) error {
				_, err := pm.RunBeforeModelCallback(icontext.NewCallbackContext(ictx), &model.LLMRequest{})
				return err
			},
			wantMsg: "Error in plugin 'bad' during 'before_model_callback' callback: boom",
		},
		{
			name: "RunAfterModelCallback",
			set: func(cfg *plugin.Config, record func(), err error) {
				cfg.AfterModelCallback = func(agent.Context, *model.LLMResponse, error) (*model.LLMResponse, error) {
					record()
					return nil, err
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) error {
				_, err := pm.RunAfterModelCallback(icontext.NewCallbackContext(ictx), &model.LLMResponse{}, nil)
				return err
			},
			wantMsg: "Error in plugin 'bad' during 'after_model_callback' callback: boom",
		},
		{
			name: "RunOnModelErrorCallback",
			set: func(cfg *plugin.Config, record func(), err error) {
				cfg.OnModelErrorCallback = func(agent.Context, *model.LLMRequest, error) (*model.LLMResponse, error) {
					record()
					return nil, err
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) error {
				_, err := pm.RunOnModelErrorCallback(icontext.NewCallbackContext(ictx), &model.LLMRequest{}, errors.New("model down"))
				return err
			},
			wantMsg: "Error in plugin 'bad' during 'on_model_error_callback' callback: boom",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			cfgFor := func(name string, err error) plugin.Config {
				cfg := plugin.Config{Name: name}
				tt.set(&cfg, func() { calls = append(calls, name) }, err)
				return cfg
			}
			pm := newTestManager(t, cfgFor("ok", nil), cfgFor("bad", errBoom), cfgFor("after", nil))

			err := tt.run(pm, newTestInvocation(t))

			if err == nil {
				t.Fatalf("%s() error = nil, want %q", tt.name, tt.wantMsg)
			}
			if got := err.Error(); got != tt.wantMsg {
				t.Errorf("%s() error = %q, want %q", tt.name, got, tt.wantMsg)
			}
			if !errors.Is(err, errBoom) {
				t.Errorf("errors.Is(%s(), errBoom) = false, want true", tt.name)
			}
			if errors.Unwrap(err) != errBoom {
				t.Errorf("errors.Unwrap(%s()) = %v, want errBoom", tt.name, errors.Unwrap(err))
			}
			if err == errBoom {
				t.Errorf("%s() returned errBoom itself, want it wrapped", tt.name)
			}
			if want := []string{"ok", "bad"}; !slices.Equal(calls, want) {
				t.Errorf("callbacks called = %v, want %v", calls, want)
			}
		})
	}
}

func TestPluginManager_RunCallbacks_FirstNonNilResultStopsTheChain(t *testing.T) {
	contentIf := func(answer bool) *genai.Content {
		if answer {
			return genai.NewContentFromText("cached", genai.RoleModel)
		}
		return nil
	}
	textOf := func(c *genai.Content) string {
		if c == nil || len(c.Parts) == 0 {
			return ""
		}
		return c.Parts[0].Text
	}
	tests := []struct {
		name string
		// set makes cfg's callback of this row call record and, when answer is
		// true, return a value.
		set  func(cfg *plugin.Config, record func(), answer bool)
		run  func(pm *PluginManager, ictx agent.InvocationContext) (string, error)
		want string
	}{
		{
			name: "RunOnUserMessageCallback",
			set: func(cfg *plugin.Config, record func(), answer bool) {
				cfg.OnUserMessageCallback = func(agent.InvocationContext, *genai.Content) (*genai.Content, error) {
					record()
					return contentIf(answer), nil
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) (string, error) {
				c, err := pm.RunOnUserMessageCallback(ictx, genai.NewContentFromText("hi", genai.RoleUser))
				return textOf(c), err
			},
			want: "cached",
		},
		{
			name: "RunBeforeRunCallback",
			set: func(cfg *plugin.Config, record func(), answer bool) {
				cfg.BeforeRunCallback = func(agent.InvocationContext) (*genai.Content, error) {
					record()
					return contentIf(answer), nil
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) (string, error) {
				c, err := pm.RunBeforeRunCallback(ictx)
				return textOf(c), err
			},
			want: "cached",
		},
		{
			name: "RunOnEventCallback",
			set: func(cfg *plugin.Config, record func(), answer bool) {
				cfg.OnEventCallback = func(agent.InvocationContext, *session.Event) (*session.Event, error) {
					record()
					if answer {
						return &session.Event{ID: "e2"}, nil
					}
					return nil, nil
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) (string, error) {
				ev, err := pm.RunOnEventCallback(ictx, &session.Event{ID: "e1"})
				if ev == nil {
					return "", err
				}
				return ev.ID, err
			},
			want: "e2",
		},
		{
			name: "RunBeforeAgentCallback",
			set: func(cfg *plugin.Config, record func(), answer bool) {
				cfg.BeforeAgentCallback = func(agent.Context) (*genai.Content, error) {
					record()
					return contentIf(answer), nil
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) (string, error) {
				c, err := pm.RunBeforeAgentCallback(icontext.NewCallbackContext(ictx))
				return textOf(c), err
			},
			want: "cached",
		},
		{
			name: "RunAfterAgentCallback",
			set: func(cfg *plugin.Config, record func(), answer bool) {
				cfg.AfterAgentCallback = func(agent.Context) (*genai.Content, error) {
					record()
					return contentIf(answer), nil
				}
			},
			run: func(pm *PluginManager, ictx agent.InvocationContext) (string, error) {
				c, err := pm.RunAfterAgentCallback(icontext.NewCallbackContext(ictx))
				return textOf(c), err
			},
			want: "cached",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls []string
			cfgFor := func(name string, answer bool) plugin.Config {
				cfg := plugin.Config{Name: name}
				tt.set(&cfg, func() { calls = append(calls, name) }, answer)
				return cfg
			}
			pm := newTestManager(t, cfgFor("p1", false), cfgFor("p2", true), cfgFor("p3", false))

			got, err := tt.run(pm, newTestInvocation(t))
			if err != nil {
				t.Fatalf("%s() error = %v, want nil", tt.name, err)
			}
			if got != tt.want {
				t.Errorf("%s() result = %q, want %q", tt.name, got, tt.want)
			}
			if want := []string{"p1", "p2"}; !slices.Equal(calls, want) {
				t.Errorf("callbacks called = %v, want %v", calls, want)
			}
		})
	}
}
