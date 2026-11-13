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

package agent

import (
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

func TestRenderUIWidget_OrderAndDuplicates(t *testing.T) {
	tests := []struct {
		name  string
		build func(ic InvocationContext, actions *session.EventActions) Context
	}{
		{
			name: "tool context",
			build: func(ic InvocationContext, actions *session.EventActions) Context {
				return NewToolContext(ic, "call-1", actions, nil)
			},
		},
		{
			name:  "callback context",
			build: NewCallbackContext,
		},
		{
			name:  "callback context with artifact tracking",
			build: NewCallbackContextWithArtifactTracking,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actions := &session.EventActions{}
			ctx := tt.build(&invocationContext{Context: t.Context()}, actions)
			w1 := session.UIWidget{ID: "w1", Provider: "mcp", Payload: map[string]any{"uri": "ui://1"}}
			w2 := session.UIWidget{ID: "w2", Provider: "mcp"}

			if err := RenderUIWidget(ctx, w1); err != nil {
				t.Fatalf("RenderUIWidget(w1) error = %v, want nil", err)
			}
			if err := RenderUIWidget(ctx, w2); err != nil {
				t.Fatalf("RenderUIWidget(w2) error = %v, want nil", err)
			}
			want := []session.UIWidget{
				{ID: "w1", Provider: "mcp", Payload: map[string]any{"uri": "ui://1"}},
				{ID: "w2", Provider: "mcp"},
			}
			if diff := cmp.Diff(want, actions.RenderUIWidgets); diff != "" {
				t.Fatalf("RenderUIWidgets after two renders mismatch (-want +got):\n%s", diff)
			}

			err := RenderUIWidget(ctx, session.UIWidget{ID: "w1", Provider: "other"})
			const wantErr = "UI widget with ID 'w1' already exists in the current event actions."
			if err == nil || err.Error() != wantErr {
				t.Errorf("RenderUIWidget(duplicate w1) error = %v, want %q", err, wantErr)
			}
			if diff := cmp.Diff(want, actions.RenderUIWidgets); diff != "" {
				t.Errorf("RenderUIWidgets after refused duplicate mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestRenderUIWidget_UnsupportedContexts(t *testing.T) {
	tests := []struct {
		name  string
		build func(t *testing.T) Context
	}{
		{
			name:  "NewContext",
			build: func(t *testing.T) Context { return NewContext(&invocationContext{Context: t.Context()}) },
		},
		{
			name:  "Promote",
			build: func(t *testing.T) Context { return Promote(&invocationContext{Context: t.Context()}) },
		},
		{
			name:  "StrictContextMock",
			build: func(t *testing.T) Context { return &StrictContextMock{Ctx: t.Context()} },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := RenderUIWidget(tt.build(t), session.UIWidget{ID: "w1", Provider: "mcp"})
			if !errors.Is(err, ErrUnsupportedContext) {
				t.Errorf("RenderUIWidget() error = %v, want ErrUnsupportedContext", err)
			}
		})
	}
}

// A before- or after-agent callback that renders a widget but returns no
// content and changes no state or artifact yields no event of its own, as in
// adk-python (agents/base_agent.py), so the widget is not delivered.
func TestAgentCallbacksDropWidgetOnlyRender(t *testing.T) {
	var renderErrs []error
	renderOnly := func(ctx Context) (*genai.Content, error) {
		renderErrs = append(renderErrs, RenderUIWidget(ctx, session.UIWidget{ID: "w1", Provider: "mcp"}))
		return nil, nil
	}
	tests := []struct {
		name   string
		config Config
	}{
		{name: "before_agent", config: Config{BeforeAgentCallbacks: []BeforeAgentCallback{renderOnly}}},
		{name: "after_agent", config: Config{AfterAgentCallbacks: []AfterAgentCallback{renderOnly}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			renderErrs = nil
			custom := &customAgent{}
			tt.config.Name = "test"
			tt.config.Run = custom.Run
			testAgent, err := New(tt.config)
			if err != nil {
				t.Fatalf("New() error = %v", err)
			}
			ctx := &invocationContext{
				Context: t.Context(),
				agent:   testAgent,
				session: &mockSession{sessionID: "test-session"},
			}
			var got []string
			for event, err := range testAgent.Run(ctx) {
				if err != nil {
					t.Fatalf("Run() error = %v", err)
				}
				got = append(got, event.Content.Parts[0].Text)
			}
			if diff := cmp.Diff([]error{nil}, renderErrs, cmp.Comparer(func(a, b error) bool { return a == b })); diff != "" {
				t.Errorf("RenderUIWidget() errors in the callback mismatch (-want +got):\n%s", diff)
			}
			// Only the agent's own event: the callback adds none.
			if diff := cmp.Diff([]string{"hello"}, got); diff != "" {
				t.Errorf("event texts mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
