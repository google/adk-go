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
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// An OnEventCallback replacement is merged onto the original event field by
// field: the original's widgets survive a replacement without any, and a
// replacement's own widgets win.
func TestRunner_OnEventReplacement_MergesRenderUIWidgets(t *testing.T) {
	tests := []struct {
		name        string
		replacement func() *session.Event
		want        []session.UIWidget
	}{
		{
			name: "replacement without widgets keeps the original's",
			replacement: func() *session.Event {
				return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("replaced", genai.RoleModel)}}
			},
			want: []session.UIWidget{{ID: "w1", Provider: "mcp", Payload: map[string]any{"uri": "ui://1"}}},
		},
		{
			name: "replacement widgets win",
			replacement: func() *session.Event {
				return &session.Event{Actions: session.EventActions{RenderUIWidgets: []session.UIWidget{{ID: "w2", Provider: "mcp"}}}}
			},
			want: []session.UIWidget{{ID: "w2", Provider: "mcp"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := cmOnEventPlugin(t, func(agent.InvocationContext, *session.Event) (*session.Event, error) {
				return tt.replacement(), nil
			})
			yielded, stored, _ := epClassicLoop.run(t, func(ev *session.Event) {
				ev.Content = genai.NewContentFromText("original", genai.RoleModel)
				ev.Actions.RenderUIWidgets = []session.UIWidget{{ID: "w1", Provider: "mcp", Payload: map[string]any{"uri": "ui://1"}}}
			}, p)

			if diff := cmp.Diff(tt.want, epOnlyEvent(t, "yielded", yielded).Actions.RenderUIWidgets); diff != "" {
				t.Errorf("yielded RenderUIWidgets mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.want, epOnlyEvent(t, "stored", stored).Actions.RenderUIWidgets); diff != "" {
				t.Errorf("stored RenderUIWidgets mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
