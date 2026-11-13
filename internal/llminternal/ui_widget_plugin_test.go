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

package llminternal

import (
	"context"
	"errors"
	"iter"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/internal/agent/runconfig"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/plugininternal/plugincontext"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// modelCallbackPlugins is a plugin manager that runs the given function for
// the model callback it names. The embedded interface is nil: a call to a
// method not overridden here would panic, and the flow under test makes none.
type modelCallbackPlugins struct {
	pluginManager
	before func(agent.Context) (*model.LLMResponse, error)
	after  func(agent.Context) (*model.LLMResponse, error)
	onErr  func(agent.Context) (*model.LLMResponse, error)
}

func (p *modelCallbackPlugins) RunBeforeModelCallback(c agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
	if p.before == nil {
		return nil, nil
	}
	return p.before(c)
}

func (p *modelCallbackPlugins) RunAfterModelCallback(c agent.Context, _ *model.LLMResponse, _ error) (*model.LLMResponse, error) {
	if p.after == nil {
		return nil, nil
	}
	return p.after(c)
}

func (p *modelCallbackPlugins) RunOnModelErrorCallback(c agent.Context, _ *model.LLMRequest, _ error) (*model.LLMResponse, error) {
	if p.onErr == nil {
		return nil, nil
	}
	return p.onErr(c)
}

// A widget a plugin's model callback renders reaches the model response event
// of that call, the same as one an agent's model callback renders.
func TestPluginModelCallbackWidgetReachesModelResponseEvent(t *testing.T) {
	widget := session.UIWidget{ID: "w1", Provider: "mcp", Payload: map[string]any{"uri": "ui://app"}}
	render := func(c agent.Context) error { return agent.RenderUIWidget(c, widget) }
	reply := func() *model.LLMResponse {
		return &model.LLMResponse{Content: genai.NewContentFromText("model reply", genai.RoleModel)}
	}
	okModel := func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) { yield(reply(), nil) }
	}
	failModel := func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) { yield(nil, errors.New("model failed")) }
	}
	tests := []struct {
		name    string
		model   func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error]
		plugins *modelCallbackPlugins
	}{
		{
			name:  "before model",
			model: okModel,
			plugins: &modelCallbackPlugins{before: func(c agent.Context) (*model.LLMResponse, error) {
				return nil, render(c)
			}},
		},
		{
			name:  "before model returns a response",
			model: okModel,
			plugins: &modelCallbackPlugins{before: func(c agent.Context) (*model.LLMResponse, error) {
				if err := render(c); err != nil {
					return nil, err
				}
				return &model.LLMResponse{Content: genai.NewContentFromText("plugin reply", genai.RoleModel)}, nil
			}},
		},
		{
			name:  "after model",
			model: okModel,
			plugins: &modelCallbackPlugins{after: func(c agent.Context) (*model.LLMResponse, error) {
				return nil, render(c)
			}},
		},
		{
			name:  "on model error",
			model: failModel,
			plugins: &modelCallbackPlugins{onErr: func(c agent.Context) (*model.LLMResponse, error) {
				if err := render(c); err != nil {
					return nil, err
				}
				return reply(), nil
			}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := Flow{Model: &mockModelForTest{name: "test-model", generateContent: tt.model}}
			base := runconfig.ToContext(t.Context(), &runconfig.RunConfig{})
			base = context.WithValue(base, plugincontext.PluginManagerCtxKey, tt.plugins)
			ctx := icontext.NewInvocationContext(base,
				icontext.InvocationContextParams{InvocationID: "inv_1", Agent: &mockAgent{name: "agent_1"}})

			var got [][]session.UIWidget
			for ev, err := range f.Run(ctx) {
				if err != nil {
					t.Fatalf("Run() error = %v", err)
				}
				got = append(got, ev.Actions.RenderUIWidgets)
			}

			want := [][]session.UIWidget{{{ID: "w1", Provider: "mcp", Payload: map[string]any{"uri": "ui://app"}}}}
			if diff := cmp.Diff(want, got); diff != "" {
				t.Errorf("RenderUIWidgets of the yielded events mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
