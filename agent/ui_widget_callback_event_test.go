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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/session"
)

// A before- or after-agent callback that renders a widget and returns content
// yields an event with that content and that widget, as in adk-python, where
// the event takes the callback context's actions.
func TestAgentCallbacksDeliverWidgetWithContent(t *testing.T) {
	renderAndReply := func(ctx Context) (*genai.Content, error) {
		if err := RenderUIWidget(ctx, session.UIWidget{ID: "w1", Provider: "mcp", Payload: map[string]any{"uri": "ui://app"}}); err != nil {
			return nil, err
		}
		return genai.NewContentFromText("from callback", genai.RoleModel), nil
	}
	tests := []struct {
		name      string
		config    Config
		wantTexts []string
		wantEvent int // index of the callback's event among the yielded events
	}{
		{
			name:      "before_agent",
			config:    Config{BeforeAgentCallbacks: []BeforeAgentCallback{renderAndReply}},
			wantTexts: []string{"from callback"},
			wantEvent: 0,
		},
		{
			name:      "after_agent",
			config:    Config{AfterAgentCallbacks: []AfterAgentCallback{renderAndReply}},
			wantTexts: []string{"hello", "from callback"},
			wantEvent: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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
			var texts []string
			var widgets [][]session.UIWidget
			for event, err := range testAgent.Run(ctx) {
				if err != nil {
					t.Fatalf("Run() error = %v", err)
				}
				texts = append(texts, event.Content.Parts[0].Text)
				widgets = append(widgets, event.Actions.RenderUIWidgets)
			}

			if diff := cmp.Diff(tt.wantTexts, texts); diff != "" {
				t.Fatalf("event texts mismatch (-want +got):\n%s", diff)
			}
			want := []session.UIWidget{{ID: "w1", Provider: "mcp", Payload: map[string]any{"uri": "ui://app"}}}
			if diff := cmp.Diff(want, widgets[tt.wantEvent]); diff != "" {
				t.Errorf("callback event RenderUIWidgets mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
