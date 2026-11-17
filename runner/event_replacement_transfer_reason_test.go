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

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// An OnEventCallback replacement is merged onto the original event field by
// field, and a transfer reason travels with its target: a replacement that
// names a target replaces the original's reason, an empty one included, as
// adk-python's _get_output_event replaces the whole actions object
// (runners.py).
func TestRunner_OnEventReplacement_TransferReason(t *testing.T) {
	tests := []struct {
		name        string
		original    session.EventActions
		replacement func() *session.Event
		wantTarget  string
		wantReason  string
	}{
		{
			name:     "replacement reason over original target",
			original: session.EventActions{TransferToAgent: "A"},
			replacement: func() *session.Event {
				return &session.Event{Actions: session.EventActions{TransferReason: "p"}}
			},
			wantTarget: "A",
			wantReason: "p",
		},
		{
			name:     "replacement without actions keeps the original's",
			original: session.EventActions{TransferToAgent: "A", TransferReason: "r"},
			replacement: func() *session.Event {
				return &session.Event{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("replaced", genai.RoleModel)}}
			},
			wantTarget: "A",
			wantReason: "r",
		},
		{
			name:     "replacement target drops the original's reason",
			original: session.EventActions{TransferToAgent: "A", TransferReason: "r"},
			replacement: func() *session.Event {
				return &session.Event{Actions: session.EventActions{TransferToAgent: "B"}}
			},
			wantTarget: "B",
			wantReason: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := cmOnEventPlugin(t, func(agent.InvocationContext, *session.Event) (*session.Event, error) {
				return tt.replacement(), nil
			})
			yielded, stored, _ := epClassicLoop.run(t, func(ev *session.Event) {
				ev.Content = genai.NewContentFromText("original", genai.RoleModel)
				ev.Actions.TransferToAgent = tt.original.TransferToAgent
				ev.Actions.TransferReason = tt.original.TransferReason
			}, p)

			got := epOnlyEvent(t, "yielded", yielded).Actions
			if got.TransferToAgent != tt.wantTarget || got.TransferReason != tt.wantReason {
				t.Errorf("yielded (TransferToAgent, TransferReason) = (%q, %q), want (%q, %q)", got.TransferToAgent, got.TransferReason, tt.wantTarget, tt.wantReason)
			}
			got = epOnlyEvent(t, "stored", stored).Actions
			if got.TransferToAgent != tt.wantTarget || got.TransferReason != tt.wantReason {
				t.Errorf("stored (TransferToAgent, TransferReason) = (%q, %q), want (%q, %q)", got.TransferToAgent, got.TransferReason, tt.wantTarget, tt.wantReason)
			}
		})
	}
}
