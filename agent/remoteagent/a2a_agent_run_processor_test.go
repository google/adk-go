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

	"google.golang.org/adk/agent"
	icontext "google.golang.org/adk/internal/context"
	"google.golang.org/adk/internal/utils"
	"google.golang.org/adk/model"
	"google.golang.org/adk/server/adka2a"
	"google.golang.org/adk/session"
)

func TestA2AAgentRunProcessor_aggregatePartial(t *testing.T) {
	type updateFlags struct {
		append    bool
		lastChunk bool
	}
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
	newArtifactUpdate := func(aid a2a.ArtifactID, flags updateFlags, text string) *a2a.TaskArtifactUpdateEvent {
		return &a2a.TaskArtifactUpdateEvent{
			TaskID:    task.ID,
			ContextID: task.ContextID,
			Artifact:  &a2a.Artifact{ID: aid, Parts: a2a.ContentParts{a2a.TextPart{Text: text}}},
			LastChunk: flags.lastChunk,
			Append:    flags.append,
		}
	}

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
	newEvent := func(parts ...*genai.Part) *session.Event {
		e := &session.Event{LLMResponse: model.LLMResponse{Partial: false}}
		if len(parts) > 0 {
			e.Content = genai.NewContentFromParts(parts, genai.RoleModel)
		}
		return e
	}
	withADKPartial := func(event *a2a.TaskArtifactUpdateEvent, partial bool) *a2a.TaskArtifactUpdateEvent {
		event.Metadata = map[string]any{adka2a.ToA2AMetaKey("partial"): partial}
		return event
	}

	aid1, aid2 := a2a.NewArtifactID(), a2a.NewArtifactID()
	tests := []struct {
		name       string
		events     []a2a.Event
		wantEvents []*session.Event
	}{
		{
			name: "do not aggregate when ADK events",
			events: []a2a.Event{
				withADKPartial(a2a.NewArtifactUpdateEvent(task, aid1, a2a.TextPart{Text: "Hel"}), true),
				withADKPartial(a2a.NewArtifactUpdateEvent(task, aid1, a2a.TextPart{Text: "lo"}), true),
				withADKPartial(a2a.NewArtifactUpdateEvent(task, aid1, a2a.TextPart{Text: "Hello"}), false),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantEvents: []*session.Event{
				newPartialEvent("Hel"),
				newPartialEvent("lo"),
				newEvent(genai.NewPartFromText("Hello")),
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
				newEvent(genai.NewPartFromText("bar")),
				newCompletedEvent(),
			},
		},
		{
			name: "[append=true, lastChunk=false] emit aggregated on final status",
			events: []a2a.Event{
				newArtifactUpdate(aid1, updateFlags{append: true}, "Hel"),
				newArtifactUpdate(aid1, updateFlags{append: true}, "lo"),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantEvents: []*session.Event{
				newPartialEvent("Hel"),
				newPartialEvent("lo"),
				newEvent(genai.NewPartFromText("Hello")),
				newCompletedEvent(),
			},
		},
		{
			name: "[append=true, lastChunk=false] emit multiple aggregated on final status",
			events: []a2a.Event{
				newArtifactUpdate(aid1, updateFlags{append: true}, "Foo"),
				newArtifactUpdate(aid2, updateFlags{append: true}, "Bar"),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantEvents: []*session.Event{
				newPartialEvent("Foo"),
				newPartialEvent("Bar"),
				newEvent(genai.NewPartFromText("Foo")),
				newEvent(genai.NewPartFromText("Bar")),
				newCompletedEvent(),
			},
		},
		{
			name: "last updated aggregation is emitted last",
			events: []a2a.Event{
				newArtifactUpdate(aid1, updateFlags{append: true}, "Foo"),
				newArtifactUpdate(aid2, updateFlags{append: true}, "Bar"),
				newArtifactUpdate(aid1, updateFlags{append: true}, "Baz"),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantEvents: []*session.Event{
				newPartialEvent("Foo"),
				newPartialEvent("Bar"),
				newPartialEvent("Baz"),
				newEvent(genai.NewPartFromText("Bar")),
				newEvent(genai.NewPartFromText("FooBaz")),
				newCompletedEvent(),
			},
		},
		{
			name: "[append=true, lastChunk=true] results in partial followed by non-partial",
			events: []a2a.Event{
				newArtifactUpdate(aid1, updateFlags{append: true}, "Hel"),
				newArtifactUpdate(aid1, updateFlags{append: true, lastChunk: true}, "lo"),
				newArtifactUpdate(aid2, updateFlags{append: true}, "bar"),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantEvents: []*session.Event{
				newPartialEvent("Hel"),
				newPartialEvent("lo"),
				newEvent(genai.NewPartFromText("Hello")),
				newPartialEvent("bar"),
				newEvent(genai.NewPartFromText("bar")),
				newCompletedEvent(),
			},
		},
		{
			name: "[append=false, lastChunk=true] results in non-partial",
			events: []a2a.Event{
				newArtifactUpdate(aid1, updateFlags{append: true}, "Hel"),
				newArtifactUpdate(aid1, updateFlags{append: false, lastChunk: true}, "Hello"),
				newArtifactUpdate(aid2, updateFlags{append: true}, "bar"),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantEvents: []*session.Event{
				newPartialEvent("Hel"),
				newEvent(genai.NewPartFromText("Hello")),
				newPartialEvent("bar"),
				newEvent(genai.NewPartFromText("bar")),
				newCompletedEvent(),
			},
		},
		{
			name: "[append=false, lastChunk=true] as first event non-partial",
			events: []a2a.Event{
				newArtifactUpdate(aid1, updateFlags{append: false, lastChunk: true}, "Hello"),
				newArtifactUpdate(aid2, updateFlags{append: true}, "bar"),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantEvents: []*session.Event{
				newEvent(genai.NewPartFromText("Hello")),
				newPartialEvent("bar"),
				newEvent(genai.NewPartFromText("bar")),
				newCompletedEvent(),
			},
		},
		{
			name: "[append=false, lastChunk=false] resets aggregation",
			events: []a2a.Event{
				newArtifactUpdate(aid1, updateFlags{append: true}, "Foo"),
				newArtifactUpdate(aid1, updateFlags{append: false}, "Bar"),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted),
			},
			wantEvents: []*session.Event{
				newPartialEvent("Foo"),
				newPartialEvent("Bar"),
				newEvent(genai.NewPartFromText("Bar")),
				newCompletedEvent(),
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
				newEvent(
					&genai.Part{Thought: true, Text: "thinking..."},
					&genai.Part{Text: "done"},
				),
				newCompletedEvent(),
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

				gotEvents = append(gotEvents, p.aggregatePartial(ctx, event, adkEvent)...)
			}
			opts := []cmp.Option{
				cmpopts.IgnoreFields(session.Event{}, "ID", "Timestamp", "InvocationID", "Author", "Branch", "CustomMetadata", "Actions"),
			}
			if diff := cmp.Diff(tc.wantEvents, gotEvents, opts...); diff != "" {
				t.Errorf("mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
