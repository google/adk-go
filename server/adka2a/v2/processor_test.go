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

	"github.com/a2aproject/a2a-go/v2/a2a"
	"github.com/a2aproject/a2a-go/v2/a2asrv"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/model"
	"google.golang.org/adk/session"
)

func modelResponseFromParts(parts ...*genai.Part) model.LLMResponse {
	return model.LLMResponse{Content: &genai.Content{Role: genai.RoleModel, Parts: parts}}
}

func modelPartialResponseFromParts(parts ...*genai.Part) model.LLMResponse {
	resp := modelResponseFromParts(parts...)
	resp.Partial = true
	return resp
}

func newNonPartialArtifactEvent(task *a2a.Task, parts ...*a2a.Part) *a2a.TaskArtifactUpdateEvent {
	ev := a2a.NewArtifactEvent(task, parts...)
	// It is important for events to be explicitely marked as ADK partial or non-partial.
	// This signals to consumers that the remote agent is running its own aggregation logic.
	ev.Metadata = map[string]any{metadataPartialKey: false}
	return ev
}

func newNonPartialArtifactUpdateEvent(task *a2a.Task, parts ...*a2a.Part) *a2a.TaskArtifactUpdateEvent {
	ev := newNonPartialArtifactEvent(task, parts...)
	ev.Append = true
	return ev
}

func newDiscardPartialArtifactUpdate(task *a2a.Task) *a2a.TaskArtifactUpdateEvent {
	ev := newLegacyPartialArtifactUpdate(task, "", []*a2a.Part{a2a.NewDataPart(map[string]any{})})
	ev.LastChunk = true
	return ev
}

func newFinalStatusUpdate(task *a2a.Task, state a2a.TaskState, msg *a2a.Message) *a2a.TaskStatusUpdateEvent {
	return a2a.NewStatusUpdateEvent(task, state, msg)
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
				{ID: "125", InvocationID: "345"},
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
				newNonPartialArtifactEvent(task, a2a.NewTextPart("Hello"), a2a.NewTextPart(", world!")),
			},
			terminal: []a2a.Event{newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil)},
		},
		{
			name: "multiple artifact updates",
			events: []*session.Event{
				{LLMResponse: modelResponseFromParts(genai.NewPartFromExecutableCode("get_the_answer()", genai.LanguagePython))},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromCodeExecutionResult(genai.OutcomeOK, "42"))},
				{LLMResponse: modelResponseFromParts(genai.NewPartFromText("The answer is 42"))},
			},
			processed: []*a2a.TaskArtifactUpdateEvent{
				newNonPartialArtifactEvent(task, func() *a2a.Part {
					p := a2a.NewDataPart(map[string]any{"code": "get_the_answer()", "language": string(genai.LanguagePython)})
					p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeCodeExecutableCode)
					return p
				}()),
				newNonPartialArtifactUpdateEvent(task, func() *a2a.Part {
					p := a2a.NewDataPart(map[string]any{"outcome": string(genai.OutcomeOK), "output": "42"})
					p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeCodeExecResult)
					return p
				}()),
				newNonPartialArtifactUpdateEvent(task, a2a.NewTextPart("The answer is 42")),
			},
			terminal: []a2a.Event{newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil)},
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
				newNonPartialArtifactEvent(task, a2a.NewTextPart("The answer is")),
				newNonPartialArtifactUpdateEvent(task, a2a.NewTextPart("42")),
			},
			terminal: []a2a.Event{
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
				newNonPartialArtifactEvent(task, a2a.NewTextPart("The answer is")),
				newNonPartialArtifactUpdateEvent(task, a2a.NewTextPart("42")),
			},
			terminal: []a2a.Event{
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
				newNonPartialArtifactEvent(task, func() *a2a.Part {
					p := a2a.NewDataPart(map[string]any{"name": "get_weather", "args": map[string]any{"city": "Warsaw"}})
					p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeFunctionCall)
					p.SetMeta(a2aDataPartMetaLongRunningKey, false)
					return p
				}()),
			},
			terminal: []a2a.Event{
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
			processed: nil,
			terminal: []a2a.Event{
				newFinalStatusUpdate(task, a2a.TaskStateInputRequired, &a2a.Message{
					Role: a2a.MessageRoleAgent,
					Parts: []*a2a.Part{
						func() *a2a.Part {
							p := a2a.NewDataPart(map[string]any{"id": "get_weather", "name": "weather", "args": map[string]any{"city": "Warsaw"}})
							p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeFunctionCall)
							p.SetMeta(a2aDataPartMetaLongRunningKey, true)
							return p
						}(),
					},
				}),
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
				newNonPartialArtifactEvent(task, a2a.NewTextPart("This will take a while")),
			},

			terminal: []a2a.Event{
				newFinalStatusUpdate(task, a2a.TaskStateInputRequired, &a2a.Message{
					Role: a2a.MessageRoleAgent,
					Parts: []*a2a.Part{
						func() *a2a.Part {
							p := a2a.NewDataPart(map[string]any{"id": "get_weather", "name": "weather", "args": map[string]any{"city": "Warsaw"}})
							p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeFunctionCall)
							p.SetMeta(a2aDataPartMetaLongRunningKey, true)
							return p
						}(),
					},
				}),
			},
		},
		{
			name: "long running tool response without ID",
			events: []*session.Event{
				{
					LongRunningToolIDs: []string{"get_weather"},
					LLMResponse: modelResponseFromParts(&genai.Part{
						FunctionCall: &genai.FunctionCall{ID: "get_weather", Name: "weather", Args: map[string]any{"city": "Warsaw"}},
					}),
				},
				{
					// No LongRunningToolIDs here
					LLMResponse: modelResponseFromParts(&genai.Part{
						FunctionResponse: &genai.FunctionResponse{ID: "get_weather", Name: "weather", Response: map[string]any{"status": "pending"}},
					}),
				},
			},
			processed: nil,
			terminal: []a2a.Event{
				newFinalStatusUpdate(task, a2a.TaskStateInputRequired, &a2a.Message{
					Role: a2a.MessageRoleAgent,
					Parts: []*a2a.Part{
						func() *a2a.Part {
							p := a2a.NewDataPart(map[string]any{"id": "get_weather", "name": "weather", "args": map[string]any{"city": "Warsaw"}})
							p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeFunctionCall)
							p.SetMeta(a2aDataPartMetaLongRunningKey, true)
							return p
						}(),
						func() *a2a.Part {
							p := a2a.NewDataPart(map[string]any{"id": "get_weather", "name": "weather", "response": map[string]any{"status": "pending"}})
							p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeFunctionResponse)
							return p
						}(),
					},
				}),
			},
		},
		{
			name: "actions in completed event meta",
			events: []*session.Event{
				{ID: "125", InvocationID: "345", Actions: session.EventActions{Escalate: true, TransferToAgent: "a-2"}},
			},
			terminal: []a2a.Event{
				&a2a.TaskStatusUpdateEvent{
					TaskID:    task.ID,
					ContextID: task.ContextID,
					Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
					Metadata:  map[string]any{metadataEscalateKey: true, metadataTransferToAgentKey: "a-2"},
				},
			},
		},
		{
			name: "last agent transfer is returned",
			events: []*session.Event{
				{ID: "125", InvocationID: "345", Actions: session.EventActions{TransferToAgent: "a-2"}},
				{ID: "126", InvocationID: "346", Actions: session.EventActions{TransferToAgent: "a-3"}},
			},
			terminal: []a2a.Event{
				&a2a.TaskStatusUpdateEvent{
					TaskID:    task.ID,
					ContextID: task.ContextID,
					Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted},
					Metadata:  map[string]any{metadataTransferToAgentKey: "a-3"},
				},
			},
		},
		{
			name: "actions not overwritten by subsequent events",
			events: []*session.Event{
				{
					LLMResponse: modelResponseFromParts(genai.NewPartFromText("The answer is")),
					Actions:     session.EventActions{Escalate: true, TransferToAgent: "a-2"},
				},
				{LLMResponse: model.LLMResponse{ErrorCode: "1", ErrorMessage: "failed"}},
			},
			processed: []*a2a.TaskArtifactUpdateEvent{
				newNonPartialArtifactEvent(task, a2a.NewTextPart("The answer is")),
			},
			terminal: []a2a.Event{
				toTaskFailedUpdateEvent(
					task, errorFromResponse(&model.LLMResponse{ErrorCode: "1", ErrorMessage: "failed"}),
					map[string]any{ToA2AMetaKey("error_code"): "1", metadataEscalateKey: true, metadataTransferToAgentKey: "a-2"},
				),
			},
		},
		{
			name: "partial events parts marked",
			events: []*session.Event{
				{
					LLMResponse: modelPartialResponseFromParts(
						genai.NewPartFromText("The answer is"),
						genai.NewPartFromExecutableCode("get_the_answer()", genai.LanguagePython)),
				},
				{LLMResponse: modelPartialResponseFromParts(
					genai.NewPartFromCodeExecutionResult(genai.OutcomeOK, "42"),
					genai.NewPartFromText("42"),
				)},
			},
			processed: []*a2a.TaskArtifactUpdateEvent{
				newLegacyPartialArtifactUpdate(task, artifactIDPlaceholder, []*a2a.Part{
					func() *a2a.Part {
						p := a2a.NewTextPart("The answer is")
						p.SetMeta(ToA2AMetaKey("partial"), true)
						return p
					}(),
					func() *a2a.Part {
						p := a2a.NewDataPart(map[string]any{"code": "get_the_answer()", "language": string(genai.LanguagePython)})
						p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeCodeExecutableCode)
						p.SetMeta(ToA2AMetaKey("partial"), true)
						return p
					}(),
				}),
				newLegacyPartialArtifactUpdate(task, artifactIDPlaceholder, []*a2a.Part{
					func() *a2a.Part {
						p := a2a.NewDataPart(map[string]any{"outcome": string(genai.OutcomeOK), "output": "42"})
						p.SetMeta(a2aDataPartMetaTypeKey, a2aDataPartTypeCodeExecResult)
						p.SetMeta(ToA2AMetaKey("partial"), true)
						return p
					}(),
					func() *a2a.Part {
						p := a2a.NewTextPart("42")
						p.SetMeta(ToA2AMetaKey("partial"), true)
						return p
					}(),
				}),
			},
			terminal: []a2a.Event{
				newDiscardPartialArtifactUpdate(task),
				newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil),
			},
		},
		{
			name: "metadata merging",
			events: []*session.Event{{
				InvocationID: "test-invocation-id",
				LLMResponse:  modelResponseFromParts(genai.NewPartFromText("Hello")),
			}},
			processed: []*a2a.TaskArtifactUpdateEvent{
				func() *a2a.TaskArtifactUpdateEvent {
					ev := newNonPartialArtifactEvent(task, a2a.NewTextPart("Hello"))
					ev.Metadata[ToA2AMetaKey("invocation_id")] = "test-invocation-id"
					return ev
				}(),
			},
			terminal: []a2a.Event{newFinalStatusUpdate(task, a2a.TaskStateCompleted, nil)},
		},
	}

	for _, tc := range testCases {
		ignoreFields := []cmp.Option{
			cmpopts.IgnoreFields(a2a.Message{}, "ID"),
			cmpopts.IgnoreFields(a2a.Artifact{}, "ID"),
			cmpopts.IgnoreFields(a2a.TaskStatus{}, "Timestamp"),
		}
		t.Run(tc.name, func(t *testing.T) {
			reqCtx := &a2asrv.ExecutorContext{TaskID: task.ID, ContextID: task.ContextID}
			processor := newEventProcessor(reqCtx, invocationMeta{}, nil, newLegacyArtifactMaker(reqCtx))

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

			gotTerminal := makeTerminalEvents(processor)
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

	reqCtx := &a2asrv.ExecutorContext{TaskID: task.ID, ContextID: task.ContextID}
	processor := newEventProcessor(reqCtx, invocationMeta{}, nil, newLegacyArtifactMaker(reqCtx))
	got := make([]*a2a.TaskArtifactUpdateEvent, len(events))
	for i, event := range events {
		processed, err := processor.process(t.Context(), event)
		if err != nil {
			t.Fatalf("processor.process() error for %d-th = %v, want nil", i, err)
		}
		got[i] = processed
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

	terminal := makeTerminalEvents(processor)
	if len(terminal) != 1 {
		t.Fatalf("processor.makeTerminalEvents() = %v, want [finalStatusUpdate]", terminal)
	}
}

func TestEventProcessor_PartialEventsAreDiscardedAsAnArtifact(t *testing.T) {
	task := &a2a.Task{ID: a2a.NewTaskID(), ContextID: a2a.NewContextID()}

	events := []*session.Event{
		{LLMResponse: modelPartialResponseFromParts(genai.NewPartFromText("Hello, "))},
		{LLMResponse: modelPartialResponseFromParts(genai.NewPartFromText("world!"))},
		{LLMResponse: modelResponseFromParts(genai.NewPartFromText("Hello, world!"))},
	}

	reqCtx := &a2asrv.ExecutorContext{TaskID: task.ID, ContextID: task.ContextID}
	processor := newEventProcessor(reqCtx, invocationMeta{}, nil, newLegacyArtifactMaker(reqCtx))
	got := make([]*a2a.TaskArtifactUpdateEvent, len(events))
	for i, event := range events {
		processed, err := processor.process(t.Context(), event)
		if err != nil {
			t.Fatalf("processor.process() error for %d-th = %v, want nil", i, err)
		}
		got[i] = processed
	}

	if len(events) != len(got) {
		t.Fatalf("processor.process() returned %d events, want %d\nevents = %v", len(got), len(events), got)
	}

	lastI := len(events) - 1
	if got[0].Append || got[0].LastChunk {
		t.Fatalf("processor.process()[0] = %+v, want {Append=false, LastChunk=false}", got[0])
	}
	if got[lastI].Append || got[lastI].LastChunk {
		t.Fatalf("processor.process()[lastI] = %+v, want {Append=false, LastChunk=false}", got[lastI])
	}

	if got[0].Artifact.ID != got[1].Artifact.ID {
		t.Fatal("processor.process() partial events applied to different artifact")
	}
	if got[0].Artifact.ID == got[lastI].Artifact.ID {
		t.Fatal("processor.process() final event applied to partial artifact")
	}

	for i, event := range got {
		if event.LastChunk {
			t.Fatalf("processor.process()[%d] = %+v, want LastChunk=false", i, event)
		}
	}

	terminal := makeTerminalEvents(processor)
	finalUpdate, ok := terminal[0].(*a2a.TaskArtifactUpdateEvent)
	if len(terminal) != 2 || !ok {
		t.Fatalf("processor.makeTerminalEvents() = %v, want [finalArtifactChunk, finalStatusUpdate]", terminal)
	}
	wantEvent := &a2a.TaskArtifactUpdateEvent{
		TaskID:    task.ID,
		ContextID: task.ContextID,
		Artifact: &a2a.Artifact{
			ID: got[0].Artifact.ID,
			Parts: a2a.ContentParts{func() *a2a.Part {
				p := a2a.NewDataPart(map[string]any{})
				p.SetMeta(metadataPartialKey, true)
				return p
			}()},
			Metadata: map[string]any{metadataPartialKey: true},
		},
		Metadata:  map[string]any{metadataPartialKey: true},
		Append:    false,
		LastChunk: true,
	}
	if diff := cmp.Diff(wantEvent, finalUpdate); diff != "" {
		t.Fatalf("makeTerminalEvents() wrong result (+got,-want) diff = %s", diff)
	}
}

func makeTerminalEvents(processor *eventProcessor) []a2a.Event {
	result := make([]a2a.Event, 0, 2)
	if finalUpdate := processor.eventToArtifact.makeFinalUpdate(); finalUpdate != nil {
		result = append(result, finalUpdate)
	}
	result = append(result, processor.makeFinalStatusUpdate())
	return result
}
