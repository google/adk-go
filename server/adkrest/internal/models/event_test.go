// Copyright 2025 Google LLC
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

package models

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

// TestFromSessionEvent_EventActions verifies that all EventActions fields —
// including TransferToAgent, Escalate, and SkipSummarization — are correctly
// mapped when converting a session.Event to a REST API Event.
//
// Regression test for https://github.com/google/adk-go/issues/509.
func TestFromSessionEvent_EventActions(t *testing.T) {
	tests := []struct {
		name    string
		actions session.EventActions
		want    EventActions
	}{
		{
			name: "all fields populated",
			actions: session.EventActions{
				StateDelta:        map[string]any{"key": "value"},
				ArtifactDelta:     map[string]int64{"file.txt": 2},
				TransferToAgent:   "agent-b",
				Escalate:          true,
				SkipSummarization: true,
			},
			want: EventActions{
				StateDelta:        map[string]any{"key": "value"},
				ArtifactDelta:     map[string]int64{"file.txt": 2},
				TransferToAgent:   "agent-b",
				Escalate:          true,
				SkipSummarization: true,
			},
		},
		{
			name: "TransferToAgent only",
			actions: session.EventActions{
				TransferToAgent: "orchestrator",
			},
			want: EventActions{
				TransferToAgent: "orchestrator",
			},
		},
		{
			name: "Escalate only",
			actions: session.EventActions{
				Escalate: true,
			},
			want: EventActions{
				Escalate: true,
			},
		},
		{
			name: "SkipSummarization only",
			actions: session.EventActions{
				SkipSummarization: true,
			},
			want: EventActions{
				SkipSummarization: true,
			},
		},
		{
			name:    "zero value",
			actions: session.EventActions{},
			want:    EventActions{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src := session.Event{Actions: tc.actions}
			got := FromSessionEvent(src)
			if diff := cmp.Diff(tc.want, got.Actions); diff != "" {
				t.Errorf("FromSessionEvent().Actions mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestToSessionEvent_EventActions verifies the reverse mapping: REST API Event
// back to session.Event, covering all three orchestration fields.
func TestToSessionEvent_EventActions(t *testing.T) {
	tests := []struct {
		name    string
		actions EventActions
		want    session.EventActions
	}{
		{
			name: "all fields populated",
			actions: EventActions{
				StateDelta:        map[string]any{"k": 1},
				ArtifactDelta:     map[string]int64{"img.png": 3},
				TransferToAgent:   "sub-agent",
				Escalate:          true,
				SkipSummarization: true,
			},
			want: session.EventActions{
				StateDelta:        map[string]any{"k": 1},
				ArtifactDelta:     map[string]int64{"img.png": 3},
				TransferToAgent:   "sub-agent",
				Escalate:          true,
				SkipSummarization: true,
			},
		},
		{
			name: "TransferToAgent only",
			actions: EventActions{
				TransferToAgent: "root-agent",
			},
			want: session.EventActions{
				TransferToAgent: "root-agent",
			},
		},
		{
			name: "Escalate only",
			actions: EventActions{
				Escalate: true,
			},
			want: session.EventActions{
				Escalate: true,
			},
		},
		{
			name: "SkipSummarization only",
			actions: EventActions{
				SkipSummarization: true,
			},
			want: session.EventActions{
				SkipSummarization: true,
			},
		},
		{
			name:    "zero value",
			actions: EventActions{},
			want:    session.EventActions{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			src := Event{Actions: tc.actions}
			got := ToSessionEvent(src)
			if diff := cmp.Diff(tc.want, got.Actions); diff != "" {
				t.Errorf("ToSessionEvent().Actions mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestFromSessionEvent_RoundTrip verifies that a session.Event round-trips
// through FromSessionEvent → ToSessionEvent without losing any EventActions fields.
func TestFromSessionEvent_RoundTrip(t *testing.T) {
	original := session.Event{
		ID:           "evt-1",
		InvocationID: "inv-1",
		Author:       "agent",
		LLMResponse: model.LLMResponse{
			Content: genai.NewContentFromText("hello", genai.RoleModel),
		},
		Actions: session.EventActions{
			StateDelta:        map[string]any{"step": "done"},
			ArtifactDelta:     map[string]int64{"report.pdf": 1},
			TransferToAgent:   "handoff-agent",
			Escalate:          true,
			SkipSummarization: true,
		},
	}

	got := ToSessionEvent(FromSessionEvent(original))

	if diff := cmp.Diff(original.Actions, got.Actions); diff != "" {
		t.Errorf("round-trip EventActions mismatch (-want +got):\n%s", diff)
	}
}
