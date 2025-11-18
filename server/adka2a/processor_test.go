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

package adka2a

import (
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func modelResponseFromParts(parts ...*genai.Part) model.LLMResponse {
	return model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}}
}

func newArtifactLastChunkEvent(task *a2a.Task) *a2a.TaskArtifactUpdateEvent {
	ev := a2a.NewArtifactUpdateEvent(task, a2a.NewArtifactID())
	ev.LastChunk = true
	return ev
}

func newFinalStatusUpdate(task *a2a.Task, state a2a.TaskState, msg *a2a.Message) *a2a.TaskStatusUpdateEvent {
	ev := a2a.NewStatusUpdateEvent(task, state, msg)
	ev.Final = true
	return ev
}

func TestEventProcessor_Process(t *testing.T) {
	artifactIDPlaceholder := a2a.NewArtifactID()
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}

	testCases := []struct {
		name      string
		events    []*session.Event
		processed []*a2a.TaskArtifactUpdateEvent
		terminal  []a2a.Event
	}{
		{
			name: "skip if no response",
			events: []*session.Event{
				{ID: "125", InvocationID: "345", Actions: session.EventActions{Escalate: true}},
				{ID: "127", InvocationID: "345", Branch: "b", Author: "a"},
			},
			terminal: []a2a.Event{newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil)},
		},
		{
			name: "skip if no content parts",
			events: []*session.Event{
				{LLMResponse: model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel}}},
				{LLMResponse: model.LLMResponse{Interrupted: true}},
			},
			terminal: []a2a.Event{newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil)},
		},
		{
			name: "multi-part artifact update",
			events: []*session.Event{{
				LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello"), genai.NewPartFromText(", world!")),
			}},
			processed: []*a2a.TaskArtifactUpdateEvent{
				a2a.NewArtifactEvent(task, a2a.TextPart{Text: "Hello"}, a2a.TextPart{Text: ", world!"}),
			},
			terminal: []a2a.Event{
				newArtifactLastChunkEvent(task),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
		},
		{
			name: "multiple artifact updates",
			events: []*session.Event{
				{
					LLMResponse: modelResponseFromParts(genai.NewPartFromExecutableCode("get_the_answer()", genai.LanguagePython)),
				},
				{
					LLMResponse: modelResponseFromParts(genai.NewPartFromCodeExecutionResult(genai.OutcomeOK, "42")),
				},
				{
					LLMResponse: modelResponseFromParts(genai.NewPartFromText("The answer is 42")),
				},
			},
			processed: []*a2a.TaskArtifactUpdateEvent{
				a2a.NewArtifactEvent(task, a2a.DataPart{
					Data:     map[string]any{"code": "get_the_answer()", "language": string(genai.LanguagePython)},
					Metadata: map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeCodeExecutableCode},
				}),
				a2a.NewArtifactUpdateEvent(task, artifactIDPlaceholder, a2a.DataPart{
					Data:     map[string]any{"outcome": string(genai.OutcomeOK), "output": "42"},
					Metadata: map[string]any{a2aDataPartMetaTypeKey: a2aDataPartTypeCodeExecResult},
				}),
				a2a.NewArtifactUpdateEvent(task, artifactIDPlaceholder, a2a.TextPart{Text: "The answer is 42"}),
			},
			terminal: []a2a.Event{
				newArtifactLastChunkEvent(task),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
		},
		{
			name: "failed without artifacts",
			events: []*session.Event{
				{LLMResponse: model.LLMResponse{ErrorCode: "1", ErrorMessage: "failed"}},
			},
			terminal: []a2a.Event{
				toTaskFailedUpdateEvent(
					task, errorFromResponse(&model.LLMResponse{ErrorCode: "1", ErrorMessage: "failed"}),
					map[string]any{ToA2AMetaKey("error_code"): "1"},
				),
			},
		},
		{
			name: "the first failure is returned",
			events: []*session.Event{
				{LLMResponse: model.LLMResponse{ErrorCode: "1", ErrorMessage: "failed 1"}},
				{LLMResponse: model.LLMResponse{ErrorCode: "2", ErrorMessage: "failed 2"}},
			},
			terminal: []a2a.Event{
				toTaskFailedUpdateEvent(
					task, errorFromResponse(&model.LLMResponse{ErrorCode: "1", ErrorMessage: "failed 1"}),
					map[string]any{ToA2AMetaKey("error_code"): "1"},
				),
			},
		},
		{
			name: "failed with artifacts",
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("The answer is"))},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("42"))},
				{LLMResponse: model.LLMResponse{ErrorCode: "1", ErrorMessage: "failed"}},
			},
			processed: []*a2a.TaskArtifactUpdateEvent{
				a2a.NewArtifactEvent(task, a2a.TextPart{Text: "The answer is"}),
				a2a.NewArtifactUpdateEvent(task, artifactIDPlaceholder, a2a.TextPart{Text: "42"}),
			},
			terminal: []a2a.Event{
				newArtifactLastChunkEvent(task),
				toTaskFailedUpdateEvent(
					task, errorFromResponse(&model.LLMResponse{ErrorCode: "1", ErrorMessage: "failed"}),
					map[string]any{ToA2AMetaKey("error_code"): "1"},
				),
			},
		},
		{
			name: "failed before receiving all parts",
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("The answer is"))},
				{LLMResponse: model.LLMResponse{ErrorCode: "1", ErrorMessage: "failed"}},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("42"))},
			},
			processed: []*a2a.TaskArtifactUpdateEvent{
				a2a.NewArtifactEvent(task, a2a.TextPart{Text: "The answer is"}),
				a2a.NewArtifactUpdateEvent(task, artifactIDPlaceholder, a2a.TextPart{Text: "42"}),
			},
			terminal: []a2a.Event{
				newArtifactLastChunkEvent(task),
				toTaskFailedUpdateEvent(
					task, errorFromResponse(&model.LLMResponse{ErrorCode: "1", ErrorMessage: "failed"}),
					map[string]any{ToA2AMetaKey("error_code"): "1"},
				),
			},
		},
		{
			name: "input_required not produced for a normal function call",
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromFunctionCall("get_weather", map[string]any{"city": "Warsaw"}))},
			},
			processed: []*a2a.TaskArtifactUpdateEvent{
				a2a.NewArtifactEvent(task, a2a.DataPart{
					Data: map[string]any{"name": "get_weather", "args": map[string]any{"city": "Warsaw"}},
					Metadata: map[string]any{
						a2aDataPartMetaTypeKey:        a2aDataPartTypeFunctionCall,
						a2aDataPartMetaLongRunningKey: false,
					},
				}),
			},
			terminal: []a2a.Event{
				newArtifactLastChunkEvent(task),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
		},
		{
			name: "input_required produced for long running function",
			events: []*session.Event{
				{
					LongRunningToolIDs: []string{"get_weather"},
					LLMResponse: modelResponseFromParts(&genai.Part{
						FunctionCall: &genai.FunctionCall{ID: "get_weather", Name: "weather", Args: map[string]any{"city": "Warsaw"}},
					}),
				},
			},
			processed: []*a2a.TaskArtifactUpdateEvent{
				a2a.NewArtifactEvent(task, a2a.DataPart{
					Data: map[string]any{"id": "get_weather", "name": "weather", "args": map[string]any{"city": "Warsaw"}},
					Metadata: map[string]any{
						a2aDataPartMetaTypeKey:        a2aDataPartTypeFunctionCall,
						a2aDataPartMetaLongRunningKey: true,
					},
				}),
			},
			terminal: []a2a.Event{
				newArtifactLastChunkEvent(task),
				newFinalStatusUpdate(task, a2a.TaskStateInputRequired, nil),
			},
		},
		{
			name: "long running tool call before receiving all parts",
			events: []*session.Event{
				{
					LongRunningToolIDs: []string{"get_weather"},
					LLMResponse: modelResponseFromParts(&genai.Part{
						FunctionCall: &genai.FunctionCall{ID: "get_weather", Name: "weather", Args: map[string]any{"city": "Warsaw"}},
					}),
				},
				{
					LLMResponse: modelResponseFromParts(genai.NewPartFromText("This will take a while")),
				},
			},
			processed: []*a2a.TaskArtifactUpdateEvent{
				a2a.NewArtifactEvent(task, a2a.DataPart{
					Data: map[string]any{"id": "get_weather", "name": "weather", "args": map[string]any{"city": "Warsaw"}},
					Metadata: map[string]any{
						a2aDataPartMetaTypeKey:        a2aDataPartTypeFunctionCall,
						a2aDataPartMetaLongRunningKey: true,
					},
				}),
				a2a.NewArtifactUpdateEvent(task, artifactIDPlaceholder, a2a.TextPart{Text: "This will take a while"}),
			},
			terminal: []a2a.Event{
				newArtifactLastChunkEvent(task),
				newFinalStatusUpdate(task, a2a.TaskStateInputRequired, nil),
			},
		},
	}

	for _, tc := range testCases {
		ignoreFields := []cmp.Option{
			cmpopts.IgnoreFields(a2a.Message{}, "ID"),
			cmpopts.IgnoreFields(a2a.Artifact{}, "ID"),
			cmpopts.IgnoreFields(a2a.TaskStatus{}, "Timestamp"),
		}
		t.Run(tc.name, func(t *testing.T) {
			reqCtx := &a2asrv.RequestContext{TaskID: task.ID, ContextID: task.ContextID}
			processor := newEventProcessor(reqCtx, invocationMeta{})

			var gotEvents []*a2a.TaskArtifactUpdateEvent
			for _, event := range tc.events {
				got, err := processor.process(t.Context(), event)
				if err != nil {
					t.Fatalf("processor.process() error = %v, want nil", err)
				}
				if got != nil {
					gotEvents = append(gotEvents, got)
				}
			}

			if diff := cmp.Diff(tc.processed, gotEvents, ignoreFields...); diff != "" {
				t.Fatalf("processor.process() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", gotEvents, tc.events, diff)
			}

			gotTerminal := processor.makeTerminalEvents()
			if diff := cmp.Diff(tc.terminal, gotTerminal, ignoreFields...); diff != "" {
				t.Fatalf("processor.makeTerminalEvents() wrong result (+got,-want)\ngot = %v\nwant = %v\ndiff = %s", gotTerminal, tc.terminal, diff)
			}
		})
	}
}

func TestEventProcessor_ArtifactUpdates(t *testing.T) {
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}
	events := []*session.Event{
		{
			LLMResponse: modelResponseFromParts(genai.NewPartFromExecutableCode("find_cat()", genai.LanguagePython)),
		},
		{
			LLMResponse: modelResponseFromParts(
				genai.NewPartFromCodeExecutionResult(genai.OutcomeOK, "https://cats.com/image.png"),
				genai.NewPartFromText("A cat image was downloaded to /home/me/cat.png"),
				genai.NewPartFromFunctionCall("download", map[string]any{"from": "https://cats.com/image.png", "to": "/home/me/cat.png"}),
			),
		},
		{
			LLMResponse: modelResponseFromParts(genai.NewPartFromFunctionResponse("download", map[string]any{"status": "ok"})),
		},
		{
			LLMResponse: modelResponseFromParts(genai.NewPartFromText("A cat image was downloaded to /home/me/cat.png")),
		},
	}

	reqCtx := &a2asrv.RequestContext{TaskID: task.ID, ContextID: task.ContextID}
	processor := newEventProcessor(reqCtx, invocationMeta{})
	got := make([]*a2a.TaskArtifactUpdateEvent, len(events))
	for i, event := range events {
		processed, err := processor.process(t.Context(), event)
		if err != nil {
			t.Fatalf("processor.process() error for %d-th = %v, want nil", i, err)
		}
		if processed != nil {
			got[i] = processed
		}
	}

	if len(events) != len(got) {
		t.Fatalf("processor.process() returned %d events, want %d\nevents = %v", len(got), len(events), got)
	}
	if got[0].Append || got[0].LastChunk {
		t.Fatalf("processor.process()[0] = %+v, want {Append=false, LastChunk=false}", got[0])
	}
	wantID := got[0].Artifact.ID
	for i := range len(got) - 1 {
		event := got[i+1]
		if event.LastChunk {
			t.Fatalf("processor.process()[%d] = %+v, want LastChunk=false", i, event)
		}
		if event.Artifact.ID != wantID {
			t.Fatalf("processor.process()[%d] ID = %v, got %v", i, event.Artifact.ID, wantID)
		}
	}

	terminal := processor.makeTerminalEvents()
	finalUpdate, ok := terminal[0].(*a2a.TaskArtifactUpdateEvent)
	if len(terminal) != 2 || !ok {
		t.Fatalf("processor.makeTerminalEvents() = %v, want [finalArtifactChunk, finalStatusUpdate]", terminal)
	}
	if !(finalUpdate.Append && finalUpdate.LastChunk) {
		t.Fatalf("finalArtifactUpdate = %+v, want {Append=true, LastChunk=true}", finalUpdate)
	}
}
