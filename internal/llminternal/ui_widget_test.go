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
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// Merged parallel function responses carry the widgets of every response in
// response order, without dropping a repeated ID, as adk-python's
// merge_parallel_function_response_events does.
func TestMergeEventActions_RenderUIWidgets(t *testing.T) {
	tests := []struct {
		name  string
		base  []session.UIWidget
		other []session.UIWidget
		want  []session.UIWidget
	}{
		{
			name:  "concatenates and keeps a repeated ID",
			base:  []session.UIWidget{{ID: "w1", Provider: "mcp", Payload: map[string]any{"a": 1}}},
			other: []session.UIWidget{{ID: "w1", Provider: "mcp", Payload: map[string]any{"a": 2}}, {ID: "w2", Provider: "mcp"}},
			want: []session.UIWidget{
				{ID: "w1", Provider: "mcp", Payload: map[string]any{"a": 1}},
				{ID: "w1", Provider: "mcp", Payload: map[string]any{"a": 2}},
				{ID: "w2", Provider: "mcp"},
			},
		},
		{
			name:  "only the second response has widgets",
			other: []session.UIWidget{{ID: "w1", Provider: "mcp"}},
			want:  []session.UIWidget{{ID: "w1", Provider: "mcp"}},
		},
		{
			name: "no widgets stay nil",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mergeEventActions(&session.EventActions{RenderUIWidgets: tt.base}, &session.EventActions{RenderUIWidgets: tt.other})
			if diff := cmp.Diff(tt.want, got.RenderUIWidgets); diff != "" {
				t.Errorf("merged RenderUIWidgets mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// A live streaming tool runs under cancelledToolContext, which embeds the tool
// context; a widget it renders lands in the tool context's actions.
func TestCancelledToolContext_ForwardsRenderUIWidget(t *testing.T) {
	ic := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{})
	actions := &session.EventActions{}
	toolCtx := agent.NewToolContext(ic, "fc", actions, nil)
	cancelCtx, cancel := toolCtx.WithAgentCancel()
	defer cancel()
	streaming := &cancelledToolContext{Context: toolCtx, cancelCtx: cancelCtx}

	if err := agent.RenderUIWidget(streaming, session.UIWidget{ID: "w1", Provider: "mcp"}); err != nil {
		t.Fatalf("RenderUIWidget() error = %v, want nil", err)
	}

	want := []session.UIWidget{{ID: "w1", Provider: "mcp"}}
	if diff := cmp.Diff(want, actions.RenderUIWidgets); diff != "" {
		t.Errorf("tool context RenderUIWidgets mismatch (-want +got):\n%s", diff)
	}
}

// A widget a model callback renders reaches the model response event of that
// call, as in adk-python, whose model callbacks share that event's actions
// (flows/llm_flows/core/_finalizer.py).
func TestModelCallbackWidgetReachesModelResponseEvent(t *testing.T) {
	widget := session.UIWidget{ID: "w1", Provider: "mcp", Payload: map[string]any{"uri": "ui://app"}}
	render := func(ctx agent.Context) error { return agent.RenderUIWidget(ctx, widget) }
	reply := func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(&model.LLMResponse{Content: genai.NewContentFromText("model reply", genai.RoleModel)}, nil)
		}
	}
	streamedReply := func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
		return func(yield func(*model.LLMResponse, error) bool) {
			if !yield(&model.LLMResponse{Content: genai.NewContentFromText("model ", genai.RoleModel), Partial: true}, nil) {
				return
			}
			if !yield(&model.LLMResponse{Content: genai.NewContentFromText("reply", genai.RoleModel), Partial: true}, nil) {
				return
			}
			yield(&model.LLMResponse{Content: genai.NewContentFromText("model reply", genai.RoleModel)}, nil)
		}
	}
	renderedOnFirstChunk := false
	wantW1 := session.UIWidget{ID: "w1", Provider: "mcp", Payload: map[string]any{"uri": "ui://app"}}
	tests := []struct {
		name      string
		model     func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error]
		runConfig *agent.RunConfig
		flow      Flow
		want      [][]session.UIWidget
	}{
		{
			name:  "before model",
			model: reply,
			flow: Flow{BeforeModelCallbacks: []BeforeModelCallback{
				func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
					return nil, render(ctx)
				},
			}},
			want: [][]session.UIWidget{{wantW1}},
		},
		{
			name:  "before model returns a response",
			model: reply,
			flow: Flow{BeforeModelCallbacks: []BeforeModelCallback{
				func(ctx agent.Context, _ *model.LLMRequest) (*model.LLMResponse, error) {
					if err := render(ctx); err != nil {
						return nil, err
					}
					return &model.LLMResponse{Content: genai.NewContentFromText("callback reply", genai.RoleModel)}, nil
				},
			}},
			want: [][]session.UIWidget{{wantW1}},
		},
		{
			name:  "after model",
			model: reply,
			flow: Flow{AfterModelCallbacks: []AfterModelCallback{
				func(ctx agent.Context, _ *model.LLMResponse, _ error) (*model.LLMResponse, error) {
					return nil, render(ctx)
				},
			}},
			want: [][]session.UIWidget{{wantW1}},
		},
		{
			name:  "after model replaces the response",
			model: reply,
			flow: Flow{AfterModelCallbacks: []AfterModelCallback{
				func(ctx agent.Context, _ *model.LLMResponse, _ error) (*model.LLMResponse, error) {
					if err := render(ctx); err != nil {
						return nil, err
					}
					return &model.LLMResponse{Content: genai.NewContentFromText("replacement", genai.RoleModel)}, nil
				},
			}},
			want: [][]session.UIWidget{{wantW1}},
		},
		{
			// The chunks of a streamed call share its actions, so a widget
			// rendered on the first chunk is on every event after it.
			name:      "after model on the first streamed chunk",
			model:     streamedReply,
			runConfig: &agent.RunConfig{StreamingMode: agent.StreamingModeSSE},
			flow: Flow{AfterModelCallbacks: []AfterModelCallback{
				func(ctx agent.Context, _ *model.LLMResponse, _ error) (*model.LLMResponse, error) {
					if renderedOnFirstChunk {
						return nil, nil
					}
					renderedOnFirstChunk = true
					return nil, render(ctx)
				},
			}},
			want: [][]session.UIWidget{{wantW1}, {wantW1}, {wantW1}},
		},
		{
			name: "on model error",
			model: func(context.Context, *model.LLMRequest, bool) iter.Seq2[*model.LLMResponse, error] {
				return func(yield func(*model.LLMResponse, error) bool) {
					yield(nil, errors.New("model failed"))
				}
			},
			flow: Flow{OnModelErrorCallbacks: []OnModelErrorCallback{
				func(ctx agent.Context, _ *model.LLMRequest, _ error) (*model.LLMResponse, error) {
					if err := render(ctx); err != nil {
						return nil, err
					}
					return &model.LLMResponse{Content: genai.NewContentFromText("model reply", genai.RoleModel)}, nil
				},
			}},
			want: [][]session.UIWidget{{wantW1}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := tt.flow
			f.Model = &mockModelForTest{name: "test-model", generateContent: tt.model}
			ctx := icontext.NewInvocationContext(
				runconfig.ToContext(t.Context(), &runconfig.RunConfig{}),
				icontext.InvocationContextParams{InvocationID: "inv_1", Agent: &mockAgent{name: "agent_1"}, RunConfig: tt.runConfig},
			)

			var got [][]session.UIWidget
			for ev, err := range f.Run(ctx) {
				if err != nil {
					t.Fatalf("Run() error = %v", err)
				}
				got = append(got, ev.Actions.RenderUIWidgets)
			}

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("RenderUIWidgets of the yielded events mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
