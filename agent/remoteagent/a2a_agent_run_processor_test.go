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

package remoteagent

import (
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"github.com/sjzsdu/adk-go/agent"
	icontext "github.com/sjzsdu/adk-go/internal/context"
	"github.com/sjzsdu/adk-go/internal/utils"
	"github.com/sjzsdu/adk-go/model"
	"github.com/sjzsdu/adk-go/server/adka2a"
	"github.com/sjzsdu/adk-go/session"
)

func TestA2AAgentRunProcessor_aggregatePartial(t *testing.T) {
	newPartialEvent := func(text string) *session.Event {
		return &session.Event{LLMResponse: model.LLMResponse{
			Partial: true,
			Content: genai.NewContentFromText(text, genai.RoleModel),
		}}
	}
	newCompletedEvent := func(parts ...*genai.Part) *session.Event {
		e := &session.Event{LLMResponse: model.LLMResponse{TurnComplete: true}}
		if len(parts) > 0 {
			e.Content = genai.NewContentFromParts(parts, genai.RoleModel)
		}
		return e
	}
	withADKPartial := func(event *a2a.TaskArtifactUpdateEvent, partial bool) *a2a.TaskArtifactUpdateEvent {
		event.Metadata = map[string]any{adka2a.ToA2AMetaKey("partial"): partial}
		return event
	}

	task := &a2a.Task{ID: "t1"}
	tests := []struct {
		name       string
		events     []a2a.Event
		wantEvents []*session.Event
	}{
		{
			name: "emit aggregated on final status",
			events: []a2a.Event{
				a2a.NewArtifactUpdateEvent(task, "a1", a2a.TextPart{Text: "Hel"}),
				a2a.NewArtifactUpdateEvent(task, "a1", a2a.TextPart{Text: "lo"}),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantEvents: []*session.Event{
				newPartialEvent("Hel"),
				newPartialEvent("lo"),
				newCompletedEvent(genai.NewPartFromText("Hello")),
			},
		},
		{
			name: "do not aggregate when ADK events",
			events: []a2a.Event{
				withADKPartial(a2a.NewArtifactUpdateEvent(task, "a1", a2a.TextPart{Text: "Hel"}), true),
				withADKPartial(a2a.NewArtifactUpdateEvent(task, "a1", a2a.TextPart{Text: "lo"}), true),
				withADKPartial(a2a.NewArtifactUpdateEvent(task, "a1", a2a.TextPart{Text: "Hello"}), false),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantEvents: []*session.Event{
				newPartialEvent("Hel"),
				newPartialEvent("lo"),
				{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Hello", genai.RoleModel)}},
				newCompletedEvent(),
			},
		},
		{
			name: "aggregation reset by final snapshot",
			events: []a2a.Event{
				a2a.NewArtifactUpdateEvent(task, "a1", a2a.TextPart{Text: "ignore me"}),
				&a2a.Task{
					ID:        task.ID,
					Artifacts: []*a2a.Artifact{{Parts: a2a.ContentParts{a2a.TextPart{Text: "done"}}}},
					Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
				},
			},
			wantEvents: []*session.Event{
				newPartialEvent("ignore me"),
				newCompletedEvent(genai.NewPartFromText("done")),
			},
		},
		{
			name: "aggregation reset by non-final snapshot",
			events: []a2a.Event{
				a2a.NewArtifactUpdateEvent(task, "a1", a2a.TextPart{Text: "foo"}),
				&a2a.Task{ID: task.ID},
				a2a.NewArtifactUpdateEvent(task, "a1", a2a.TextPart{Text: "bar"}),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantEvents: []*session.Event{
				newPartialEvent("foo"),
				newPartialEvent("bar"),
				newCompletedEvent(genai.NewPartFromText("bar")),
			},
		},
		{
			name: "thoughts aggregation",
			events: []a2a.Event{
				a2a.NewArtifactUpdateEvent(task, "a1", a2a.TextPart{
					Text:     "thinking...",
					Metadata: map[string]any{adka2a.ToA2AMetaKey("thought"): true},
				}),
				a2a.NewArtifactUpdateEvent(task, "a1", a2a.TextPart{Text: "done"}),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantEvents: []*session.Event{
				{LLMResponse: model.LLMResponse{
					Partial: true,
					Content: &genai.Content{Parts: []*genai.Part{{Thought: true, Text: "thinking..."}}, Role: genai.RoleModel},
				}},
				newPartialEvent("done"),
				newCompletedEvent(
					&genai.Part{Thought: true, Text: "thinking..."},
					&genai.Part{Text: "done"},
				),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			agnt := utils.Must(agent.New(agent.Config{}))
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent: agnt,
			})

			p := newRunProcessor(A2AConfig{}, nil)
			var gotEvents []*session.Event

			for _, event := range tc.events {

				adkEvent, err := adka2a.ToSessionEvent(ctx, event)
				if err != nil {
					t.Fatalf("ToSessionEvent failed: %v", err)
				}

				if adkEvent == nil {
					// Handle Task snapshot resetting aggregation even if it doesn't produce an event
					if _, ok := event.(*a2a.Task); ok {
						p.aggregatePartial(ctx, event, nil)
					}
					continue
				}

				if agg := p.aggregatePartial(ctx, event, adkEvent); agg != nil {
					gotEvents = append(gotEvents, agg)
				}
				gotEvents = append(gotEvents, adkEvent)
			}

			if diff := cmp.Diff(tc.wantEvents, gotEvents, cmp.AllowUnexported(session.Event{}), cmpopts.IgnoreFields(session.Event{}, "ID", "Timestamp", "InvocationID", "Author", "Branch", "CustomMetadata")); diff != "" {
				t.Errorf("mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
