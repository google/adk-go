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

package llminternal_test

import (
	"context"
	"encoding/json"
	"iter"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool/toolconfirmation"
)

type testModel struct {
	model.LLM
}

// Test behavior around Agent's IncludeContents.
func TestContentsRequestProcessor_IncludeContents(t *testing.T) {
	const agentName = "testAgent"
	testModel := &testModel{}

	emptyEvent := []*session.Event{}
	helloAndGoodBye := []*session.Event{
		{
			Author: "user", // Not in the current turn in multi-agent scenario. See buildContentsCurrentTurnContextOnly.
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("hello", "user"),
			},
		},
		{
			Author: "user",
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("good bye", "user"),
			},
		},
	}
	agentTransfer := []*session.Event{
		{
			Author: "anotherAgent", // History.
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromFunctionCall("func1", nil, "model"),
			},
		},
		{
			Author: "anotherAgent",
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromFunctionResponse("func1", nil, "user"),
			},
		},
		{
			Author: "anotherAgent", // Beginning of the current turn started by another agent.
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("transfer to testAgent", "model"),
			},
		},
		{
			Author: agentName, // See python flows/llm_flows/base_llm_flow.py BaseLlmFlow._run_one_step_async.
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromFunctionCall("func1", nil, "model"),
			},
		},
	}
	robot := []*session.Event{
		{
			Author: agentName,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("do func1", "user"),
			},
		},
		{
			Author: agentName,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromFunctionCall("func1", nil, "model"),
			},
		},
		{
			Author: agentName,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromFunctionResponse("func1", nil, "user"),
			},
		},
	}

	t.Parallel()
	testCases := []struct {
		name            string
		includeContents llmagent.IncludeContents
		events          []*session.Event
		want            []*genai.Content
	}{
		{
			name:            "empty",
			includeContents: "default",
			events:          emptyEvent,
		},
		{
			name:            "empty",
			includeContents: "none",
			events:          emptyEvent,
		},
		{
			name:            "helloAndGoodBye",
			includeContents: "",
			events:          helloAndGoodBye,
			want: []*genai.Content{
				genai.NewContentFromText("hello", "user"),
				genai.NewContentFromText("good bye", "user"),
			},
		},
		{
			name:            "helloAndGoodBye",
			includeContents: "default", // default == ""
			events:          helloAndGoodBye,
			want: []*genai.Content{
				genai.NewContentFromText("hello", "user"),
				genai.NewContentFromText("good bye", "user"),
			},
		},
		{
			name:            "helloAndGoodBye",
			includeContents: "none",
			events:          helloAndGoodBye,
			want: []*genai.Content{
				genai.NewContentFromText("good bye", "user"),
			},
		},
		{
			name:            "agentTransfer",
			includeContents: "",
			events:          agentTransfer,
			want: []*genai.Content{
				// events from other agents are converted by convertForeignEvent.
				{
					Parts: []*genai.Part{
						{Text: "For context:"},
						{Text: "[anotherAgent] called tool `func1` with parameters: null"},
					},
					Role: "user",
				},
				{
					Parts: []*genai.Part{
						{Text: "For context:"},
						{Text: "[anotherAgent] `func1` tool returned result: null"},
					},
					Role: "user",
				},
				{
					Parts: []*genai.Part{
						{Text: "For context:"},
						{Text: "[anotherAgent] said: transfer to testAgent"},
					},
					Role: "user",
				},
				genai.NewContentFromFunctionCall("func1", nil, "model"),
			},
		},
		{
			name:            "agentTransfer",
			includeContents: "none",
			events:          agentTransfer,
			want: []*genai.Content{
				{
					Parts: []*genai.Part{
						{Text: "For context:"},
						{Text: "[anotherAgent] said: transfer to testAgent"},
					},
					Role: "user",
				},
				genai.NewContentFromFunctionCall("func1", nil, "model"),
			},
		},
		{
			name:            "robot",
			includeContents: "default",
			events:          robot,
			want: []*genai.Content{
				genai.NewContentFromText("do func1", "user"),
				genai.NewContentFromFunctionCall("func1", nil, "model"),
				genai.NewContentFromFunctionResponse("func1", nil, "user"),
			},
		},
		{
			name:            "robot",
			includeContents: "none",
			events:          robot,
			want: []*genai.Content{
				genai.NewContentFromText("do func1", "user"),
				genai.NewContentFromFunctionCall("func1", nil, "model"),
				genai.NewContentFromFunctionResponse("func1", nil, "user"),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name+"/include_contents="+string(tc.includeContents), func(t *testing.T) {
			testAgent := utils.Must(llmagent.New(llmagent.Config{
				Name:            agentName,
				Model:           testModel,
				IncludeContents: tc.includeContents,
			}))

			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent: testAgent,
				Session: &fakeSession{
					events: tc.events,
				},
			})

			req := &model.LLMRequest{}
			for ev, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
				if ev != nil {
					t.Fatal("ContentsRequestProcessor generated an unexpected event")
				}
				if err != nil {
					t.Fatalf("contentRequestProcessor failed: %v", err)
				}
			}
			got := req.Contents
			if diff := cmp.Diff(wantWithContinuation(tc.want), got); diff != "" {
				t.Errorf("LLMRequest after contentsRequestProcessor mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestContentsRequestProcessor_IncludeContentsNone_IsolationScopePivot
// guards the include_contents="none" pivot against an out-of-scope
// event: a scoped event must not be chosen as the turn start, otherwise
// the unscoped agent would see an empty/truncated turn. Mirrors
// adk-python's _should_include_event_in_context gate in
// _get_current_turn_contents.
func TestContentsRequestProcessor_IncludeContentsNone_IsolationScopePivot(t *testing.T) {
	t.Parallel()
	events := []*session.Event{
		{
			Author: "user",
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("unscoped turn", "user"),
			},
		},
		{
			Author:         "user",
			IsolationScope: "task-1",
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("scoped task", "user"),
			},
		},
	}
	testAgent := utils.Must(llmagent.New(llmagent.Config{
		Name:            "testAgent",
		Model:           &testModel{},
		IncludeContents: "none",
	}))
	// Unscoped agent: the trailing scoped event must be skipped as the
	// pivot, so the unscoped user turn is what the agent sees.
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:   testAgent,
		Session: &fakeSession{events: events},
	})

	req := &model.LLMRequest{}
	for _, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
		if err != nil {
			t.Fatalf("contentsRequestProcessor failed: %v", err)
		}
	}
	want := []*genai.Content{genai.NewContentFromText("unscoped turn", "user")}
	if diff := cmp.Diff(wantWithContinuation(want), req.Contents); diff != "" {
		t.Errorf("contents mismatch (-want +got):\n%s", diff)
	}
}

// TestContentsRequestProcessor_StrictIsolationFilterExcludesForeignScope
// verifies that the contents from the different scope are not included
// in the content for LLM.
func TestContentsRequestProcessor_StrictIsolationFilterExcludesForeignScope(t *testing.T) {
	t.Parallel()
	const myScope = "task-mine"
	events := []*session.Event{
		// Foreign-scoped user event: must be filtered out.
		{
			Author:         "user",
			IsolationScope: "garbage-scope",
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("foreign payload", "user"),
			},
		},
		// In-scope user event: must reach the LLM.
		{
			Author:         "user",
			IsolationScope: myScope,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("my task brief", "user"),
			},
		},
	}
	testAgent := utils.Must(llmagent.New(llmagent.Config{
		Name:  "scopedAgent",
		Model: &testModel{},
	}))
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:          testAgent,
		IsolationScope: myScope,
		Session:        &fakeSession{events: events},
	})

	req := &model.LLMRequest{}
	for _, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
		if err != nil {
			t.Fatalf("contentsRequestProcessor failed: %v", err)
		}
	}
	want := []*genai.Content{genai.NewContentFromText("my task brief", "user")}
	if diff := cmp.Diff(wantWithContinuation(want), req.Contents); diff != "" {
		t.Errorf("contents mismatch (-want +got):\n%s", diff)
	}
}

func TestContentsRequestProcessor_TaskInputFromOriginatingFC(t *testing.T) {
	t.Parallel()
	const taskAgentName = "taskAgent"
	const coordName = "coord"
	const fcID = "fc-1"
	args := map[string]any{"goal": "summarize"}
	events := []*session.Event{
		{
			Author: coordName,
			LLMResponse: model.LLMResponse{
				Content: &genai.Content{
					Role: "model",
					Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{
						ID: fcID, Name: taskAgentName, Args: args,
					}}},
				},
			},
		},
		{
			Author:         taskAgentName,
			IsolationScope: fcID,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("working on it", "model"),
			},
		},
	}

	argsJSON, err := json.Marshal(args)
	if err != nil {
		t.Fatalf("marshal args: %v", err)
	}

	t.Run("task mode prepends FC args without nudge", func(t *testing.T) {
		taskAgent := utils.Must(llmagent.New(llmagent.Config{
			Name:  taskAgentName,
			Model: &testModel{},
			Mode:  llmagent.ModeTask,
		}))
		ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
			Agent:          taskAgent,
			IsolationScope: fcID,
			Session:        &fakeSession{events: events},
		})
		req := &model.LLMRequest{}
		for _, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
			if err != nil {
				t.Fatalf("contentsRequestProcessor failed: %v", err)
			}
		}
		want := []*genai.Content{
			{Role: genai.RoleUser, Parts: []*genai.Part{{Text: string(argsJSON)}}},
			genai.NewContentFromText("working on it", "model"),
		}
		if diff := cmp.Diff(wantWithContinuation(want), req.Contents); diff != "" {
			t.Errorf("contents mismatch (-want +got):\n%s", diff)
		}
	})

	t.Run("single_turn mode appends nudge after FC args", func(t *testing.T) {
		stAgent := utils.Must(llmagent.New(llmagent.Config{
			Name:            taskAgentName,
			Model:           &testModel{},
			Mode:            llmagent.ModeSingleTurn,
			IncludeContents: "none",
		}))
		ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
			Agent:          stAgent,
			IsolationScope: fcID,
			Session:        &fakeSession{events: events},
		})
		req := &model.LLMRequest{}
		for _, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
			if err != nil {
				t.Fatalf("contentsRequestProcessor failed: %v", err)
			}
		}
		wantLeading := &genai.Content{
			Role: genai.RoleUser,
			Parts: []*genai.Part{
				{Text: string(argsJSON)},
				{Text: llminternal.SingleTurnNudge},
			},
		}
		if len(req.Contents) == 0 {
			t.Fatal("expected at least one content; got none")
		}
		if diff := cmp.Diff(wantLeading, req.Contents[0]); diff != "" {
			t.Errorf("leading content mismatch (-want +got):\n%s", diff)
		}
	})
}

func TestContentsRequestProcessor_TaskInputFromUserContentFallback(t *testing.T) {
	t.Parallel()
	const taskAgentName = "taskAgent"
	const scope = "wf:node-1"
	userInput := genai.NewContentFromText("do the thing", genai.RoleUser)

	// Session has no event carrying a FunctionCall with ID == scope.
	events := []*session.Event{
		{
			Author:         taskAgentName,
			IsolationScope: scope,
			LLMResponse: model.LLMResponse{
				Content: genai.NewContentFromText("ack", "model"),
			},
		},
	}

	taskAgent := utils.Must(llmagent.New(llmagent.Config{
		Name:  taskAgentName,
		Model: &testModel{},
		Mode:  llmagent.ModeTask,
	}))
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:          taskAgent,
		IsolationScope: scope,
		UserContent:    userInput,
		Session:        &fakeSession{events: events},
	})
	req := &model.LLMRequest{}
	for _, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
		if err != nil {
			t.Fatalf("contentsRequestProcessor failed: %v", err)
		}
	}
	want := []*genai.Content{
		{Role: genai.RoleUser, Parts: userInput.Parts},
		genai.NewContentFromText("ack", "model"),
	}
	if diff := cmp.Diff(wantWithContinuation(want), req.Contents); diff != "" {
		t.Errorf("contents mismatch (-want +got):\n%s", diff)
	}
}

func TestContentsRequestProcessor(t *testing.T) {
	const agentName = "testAgent"
	testModel := &testModel{}

	t.Parallel()
	testCases := []struct {
		name           string
		branch         string
		isolationScope string
		events         []*session.Event
		want           []*genai.Content
	}{
		{
			name:   "NilEvent",
			events: nil,
			want:   nil,
		},
		{
			name:   "EmptyEvents",
			events: []*session.Event{},
			want:   nil,
		},
		{
			name: "UserAndAgentEvents",
			events: []*session.Event{
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("Hello", "user"),
					},
				},
				{
					Author: "testAgent",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("Hi there", "model"),
					},
				},
			},
			want: []*genai.Content{
				genai.NewContentFromText("Hello", "user"),
				genai.NewContentFromText("Hi there", "model"),
			},
		},
		{
			name: "anotherAgentEvent",
			events: []*session.Event{
				{
					Author: "anotherAgent",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("Foreign message", "model"),
					},
				},
			},
			want: []*genai.Content{
				{
					Role: "user",
					Parts: []*genai.Part{
						{Text: "For context:"},
						{Text: "[anotherAgent] said: Foreign message"},
					},
				},
			},
		},
		{
			name: "ExcludeToolConfirmation",
			events: []*session.Event{
				{
					Author: "AgentA",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{
									FunctionCall: &genai.FunctionCall{
										ID:   "call_confirm_123",
										Name: "adk_request_confirmation",
										Args: map[string]any{"message": "Confirm delete?"},
									},
								},
							},
						},
					},
				},
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role: "user",
							Parts: []*genai.Part{
								{
									FunctionResponse: &genai.FunctionResponse{
										ID:       "call_confirm_123",
										Name:     "adk_request_confirmation",
										Response: map[string]any{"confirmed": true},
									},
								},
							},
						},
					},
				},
			},
			want: nil,
		},
		{
			name:   "FilterByBranch",
			branch: "branch1.task1",
			events: []*session.Event{
				{
					Author: "user",
					Branch: "branch1",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("In branch 1", "user"),
					},
				},
				{
					Author: "user",
					Branch: "branch1.task1",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("In branch 1 and task 1", "user"),
					},
				},
				{
					Author: "user",
					Branch: "branch12",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("In branch 12", "user"),
					},
				},
				{
					Author: "user",
					Branch: "branch2",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("In branch 2", "user"),
					},
				},
				{
					Author: "user",
					Branch: "",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("empty branch", "user"),
					},
				},
			},
			want: []*genai.Content{
				genai.NewContentFromText("In branch 1", "user"),
				genai.NewContentFromText("In branch 1 and task 1", "user"),
				genai.NewContentFromText("empty branch", "user"),
			},
		},
		{
			// Isolation scope is exact-match (unlike branch, where empty
			// is universally visible): a scoped agent sees ONLY events
			// with the same scope, not unscoped ones.
			name:           "FilterByIsolationScope_Scoped",
			isolationScope: "task-1",
			events: []*session.Event{
				{
					Author:         "user",
					IsolationScope: "task-1",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("in task 1", "user"),
					},
				},
				{
					Author:         "user",
					IsolationScope: "task-2",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("in task 2", "user"),
					},
				},
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("unscoped", "user"),
					},
				},
			},
			want: []*genai.Content{
				genai.NewContentFromText("in task 1", "user"),
			},
		},
		{
			// An unscoped agent (empty scope) sees ONLY unscoped events.
			name: "FilterByIsolationScope_Unscoped",
			events: []*session.Event{
				{
					Author:         "user",
					IsolationScope: "task-1",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("in task 1", "user"),
					},
				},
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						Content: genai.NewContentFromText("unscoped", "user"),
					},
				},
			},
			want: []*genai.Content{
				genai.NewContentFromText("unscoped", "user"),
			},
		},
		{
			name: "AuthEvent",
			events: []*session.Event{
				{
					Author: agentName,
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{FunctionCall: &genai.FunctionCall{Name: "adk_request_credential"}},
							},
						},
					},
				},
			},
			want: nil,
		},
		{
			name: "EventWithoutContent",
			events: []*session.Event{
				{Author: "user"},
			},
			want: nil,
		},
		{
			name: "TranscriptionAggregation",
			events: []*session.Event{
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						InputTranscription: &genai.Transcription{Text: "hello ", Finished: false},
					},
				},
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						InputTranscription: &genai.Transcription{Text: "world", Finished: true},
					},
				},
				{
					Author: "testAgent",
					LLMResponse: model.LLMResponse{
						OutputTranscription: &genai.Transcription{Text: "hi ", Finished: false},
					},
				},
				{
					Author: "testAgent",
					LLMResponse: model.LLMResponse{
						OutputTranscription: &genai.Transcription{Text: "there", Finished: true},
					},
				},
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						InputTranscription: &genai.Transcription{Text: "ok", Finished: true},
					},
				},
			},
			want: []*genai.Content{
				genai.NewContentFromText("hello world", "user"),
				genai.NewContentFromText("hi there", "model"),
				genai.NewContentFromText("ok", "user"),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			testAgent := utils.Must(llmagent.New(llmagent.Config{
				Name:  "testAgent",
				Model: testModel,
			}))

			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent:          testAgent,
				Branch:         tc.branch,
				IsolationScope: tc.isolationScope,
				Session: &fakeSession{
					events: tc.events,
				},
			})

			req := &model.LLMRequest{}
			for ev, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
				if ev != nil {
					t.Fatal("ContentsRequestProcessor generated an unexpected event")
				}
				if err != nil {
					t.Fatalf("contentRequestProcessor failed: %v", err)
				}
			}
			got := req.Contents
			if diff := cmp.Diff(wantWithContinuation(tc.want), got); diff != "" {
				t.Errorf("LLMRequest after contentRequestProcessor mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestConvertForeignEvent(t *testing.T) {
	t.Parallel()
	now := time.Now()
	testCases := []struct {
		name  string
		event *session.Event
		want  *session.Event
	}{
		{
			name: "Text",
			event: &session.Event{
				Timestamp: now,
				Author:    "foreign",
				LLMResponse: model.LLMResponse{
					Content: genai.NewContentFromText("hello", "model"),
				},
				Branch: "b",
			},
			want: &session.Event{
				Timestamp: now,
				Author:    "user",
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Role: "user",
						Parts: []*genai.Part{
							{Text: "For context:"},
							{Text: "[foreign] said: hello"},
						},
					},
				},
				Branch: "b",
			},
		},
		{
			name: "FunctionCall",
			event: &session.Event{
				Timestamp: now,
				Author:    "foreign",
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Role: "model",
						Parts: []*genai.Part{
							{FunctionCall: &genai.FunctionCall{Name: "test", Args: map[string]any{"a": "b"}}},
						},
					},
				},
				Branch: "b",
			},
			want: &session.Event{
				Timestamp: now,
				Author:    "user",
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Role: "user",
						Parts: []*genai.Part{
							{Text: "For context:"},
							{Text: "[foreign] called tool `test` with parameters: {\"a\":\"b\"}"},
						},
					},
				},
				Branch: "b",
			},
		},
		{
			name: "FunctionResponse",
			event: &session.Event{
				Timestamp: now,
				Author:    "foreign",
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Role: "model",
						Parts: []*genai.Part{
							{FunctionResponse: &genai.FunctionResponse{Name: "test", Response: map[string]any{"c": "d"}}},
						},
					},
				},
				Branch: "b",
			},
			want: &session.Event{
				Timestamp: now,
				Author:    "user",
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Role: "user",
						Parts: []*genai.Part{
							{Text: "For context:"},
							{Text: "[foreign] `test` tool returned result: {\"c\":\"d\"}"},
						},
					},
				},
				Branch: "b",
			},
		},
		{
			name: "ThoughtOnlyEvent",
			event: &session.Event{
				Timestamp: now,
				Author:    "foreign",
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Role: "model",
						Parts: []*genai.Part{
							{Text: "internal reasoning", Thought: true},
							{Text: "more thoughts", Thought: true},
						},
					},
				},
				Branch: "b",
			},
			want: nil, // thought-only event produces no useful context
		},
		{
			name: "MixedThoughtAndTextEvent",
			event: &session.Event{
				Timestamp: now,
				Author:    "foreign",
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Role: "model",
						Parts: []*genai.Part{
							{Text: "internal reasoning", Thought: true},
							{Text: "actual response"},
						},
					},
				},
				Branch: "b",
			},
			want: &session.Event{
				Timestamp: now,
				Author:    "user",
				LLMResponse: model.LLMResponse{
					Content: &genai.Content{
						Role: "user",
						Parts: []*genai.Part{
							{Text: "For context:"},
							{Text: "[foreign] said: actual response"},
						},
					},
				},
				Branch: "b",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := llminternal.ConvertForeignEvent(tc.event)
			if diff := cmp.Diff(tc.want, got, cmp.AllowUnexported(genai.FunctionCall{}, genai.FunctionResponse{})); diff != "" {
				t.Errorf("convertForeignEvent() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestContentsRequestProcessor_NonLLMAgent(t *testing.T) {
	testAgent := utils.Must(agent.New(agent.Config{
		Name: "test_agent",
	}))

	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent: testAgent,
	})

	req := &model.LLMRequest{}

	for ev, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
		if ev != nil {
			t.Fatal("ContentsRequestProcessor generated an unexpected event")
		}
		if err != nil {
			t.Fatalf("contentRequestProcessor failed: %v", err)
		}
	}
	got := req
	want := &model.LLMRequest{}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("LLMRequest after contentRequestProcessor mismatch (-want +got):\n%s", diff)
	}
}

func TestContentsRequestProcessor_Rearrange(t *testing.T) {
	const agentName = "test_agent"
	testModel := &testModel{}

	// --- Reusable Test Data ---
	// Basic Call/Response
	fcBasic := &genai.FunctionCall{
		ID:   "call_123",
		Name: "search_tool",
		Args: map[string]any{"query": "test"},
	}
	frBasic := &genai.FunctionResponse{
		ID:       "call_123",
		Name:     "search_tool",
		Response: map[string]any{"results": []string{"item1", "item2"}},
	}

	// LRO Call/Responses
	fcLRO := &genai.FunctionCall{
		ID:   "long_call_123",
		Name: "long_running_tool",
		Args: map[string]any{"task": "process"},
	}
	frLROInter := &genai.FunctionResponse{
		ID:       "long_call_123",
		Name:     "long_running_tool",
		Response: map[string]any{"status": "processing", "progress": 50},
	}
	frLROFinal := &genai.FunctionResponse{
		ID:       "long_call_123",
		Name:     "long_running_tool",
		Response: map[string]any{"status": "completed", "result": "done"},
	}

	// Mixed LRO/Normal Calls/Responses
	fcLROMixed := &genai.FunctionCall{
		ID:   "lro_call_456",
		Name: "long_running_tool",
		Args: map[string]any{"task": "analyze"},
	}
	fcNormalMixed := &genai.FunctionCall{
		ID:   "normal_call_789",
		Name: "search_tool",
		Args: map[string]any{"query": "test"},
	}
	frLROInterMixed := &genai.FunctionResponse{
		ID:       "lro_call_456",
		Name:     "long_running_tool",
		Response: map[string]any{"status": "processing", "progress": 25},
	}
	frNormalMixed := &genai.FunctionResponse{
		ID:       "normal_call_789",
		Name:     "search_tool",
		Response: map[string]any{"results": []string{"item1", "item2"}},
	}
	frLROFinalMixed := &genai.FunctionResponse{
		ID:       "lro_call_456",
		Name:     "long_running_tool",
		Response: map[string]any{"status": "completed", "analysis": "done"},
	}

	// History LRO Call/Responses
	fcHistLRO := &genai.FunctionCall{
		ID:   "history_call_123",
		Name: "long_running_tool",
		Args: map[string]any{"task": "process"},
	}
	frHistLROInter := &genai.FunctionResponse{
		ID:       "history_call_123",
		Name:     "long_running_tool",
		Response: map[string]any{"status": "processing", "progress": 50},
	}
	frHistLROFinal := &genai.FunctionResponse{
		ID:       "history_call_123",
		Name:     "long_running_tool",
		Response: map[string]any{"status": "completed", "result": "done"},
	}

	// History Mixed Call/Responses
	fcHistLROMixed := &genai.FunctionCall{
		ID:   "history_lro_123",
		Name: "long_running_tool",
		Args: map[string]any{"task": "analyze"},
	}
	fcHistNormalMixed := &genai.FunctionCall{
		ID:   "history_normal_456",
		Name: "search_tool",
		Args: map[string]any{"query": "data"},
	}
	frHistLROInterMixed := &genai.FunctionResponse{
		ID:       "history_lro_123",
		Name:     "long_running_tool",
		Response: map[string]any{"status": "processing", "progress": 30},
	}
	frHistNormalMixed := &genai.FunctionResponse{
		ID:       "history_normal_456",
		Name:     "search_tool",
		Response: map[string]any{"results": []string{"result1", "result2"}},
	}
	frHistLROFinalMixed := &genai.FunctionResponse{
		ID:       "history_lro_123",
		Name:     "long_running_tool",
		Response: map[string]any{"status": "completed", "analysis": "finished"},
	}

	// Preserve Content Call/Responses
	fcPreserve := &genai.FunctionCall{
		ID:   "preserve_test",
		Name: "long_running_tool",
		Args: map[string]any{"test": "value"},
	}
	frPreserveInter := &genai.FunctionResponse{
		ID:       "preserve_test",
		Name:     "long_running_tool",
		Response: map[string]any{"status": "processing"},
	}
	frPreserveFinal := &genai.FunctionResponse{
		ID:       "preserve_test",
		Name:     "long_running_tool",
		Response: map[string]any{"output": "preserved"},
	}

	// Human-in-the-loop confirmation call/responses
	fcHITLApproved := &genai.FunctionCall{
		ID:   "hitl_approved_call",
		Name: "request_vacation_days",
		Args: map[string]any{"days": 5, "user_id": "user-123"},
	}
	frHITLApprovedPending := &genai.FunctionResponse{
		ID:       "hitl_approved_call",
		Name:     "request_vacation_days",
		Response: map[string]any{"status": "Manager approval is required.", "request_id": "req-1"},
	}
	frHITLApprovedFinal := &genai.FunctionResponse{
		ID:       "hitl_approved_call",
		Name:     "request_vacation_days",
		Response: map[string]any{"status": "The time off request is accepted.", "days_approved": 5, "request_id": "req-1"},
	}
	fcHITLApprovalRequest := &genai.FunctionCall{
		ID:   "hitl_approved_confirmation",
		Name: toolconfirmation.FunctionCallName,
		Args: map[string]any{
			"originalFunctionCall": fcHITLApproved,
			"toolConfirmation": toolconfirmation.ToolConfirmation{
				Confirmed: false,
				Hint:      "Please approve or reject the tool call request_vacation_days().",
			},
		},
	}
	frHITLApprovalConfirmed := &genai.FunctionResponse{
		ID:       "hitl_approved_confirmation",
		Name:     toolconfirmation.FunctionCallName,
		Response: map[string]any{"confirmed": true},
	}
	fcHITLDenied := &genai.FunctionCall{
		ID:   "hitl_denied_call",
		Name: "request_vacation_days",
		Args: map[string]any{"days": 10, "user_id": "user-123"},
	}
	frHITLDeniedPending := &genai.FunctionResponse{
		ID:       "hitl_denied_call",
		Name:     "request_vacation_days",
		Response: map[string]any{"status": "Manager approval is required.", "request_id": "req-2"},
	}
	frHITLDeniedFinal := &genai.FunctionResponse{
		ID:       "hitl_denied_call",
		Name:     "request_vacation_days",
		Response: map[string]any{"status": "The time off request is rejected.", "days_approved": 0, "request_id": "req-2"},
	}
	fcHITLDenialRequest := &genai.FunctionCall{
		ID:   "hitl_denied_confirmation",
		Name: toolconfirmation.FunctionCallName,
		Args: map[string]any{
			"originalFunctionCall": fcHITLDenied,
			"toolConfirmation": toolconfirmation.ToolConfirmation{
				Confirmed: false,
				Hint:      "Please approve or reject the tool call request_vacation_days().",
			},
		},
	}
	frHITLDenialRejected := &genai.FunctionResponse{
		ID:       "hitl_denied_confirmation",
		Name:     toolconfirmation.FunctionCallName,
		Response: map[string]any{"confirmed": false},
	}

	// Error Call/Response
	frOrphaned := &genai.FunctionResponse{
		ID:       "no_matching_call",
		Name:     "orphaned_tool",
		Response: map[string]any{"error": "no matching call"},
	}

	// --- Test Cases ---
	testCases := []struct {
		name    string
		events  []*session.Event
		want    []*genai.Content
		wantErr string // Use string to check for specific error messages
	}{
		{
			name:   "NilEvent",
			events: nil,
			want:   nil,
		},
		{
			name:   "EmptyEvents",
			events: []*session.Event{},
			want:   nil,
		},
		{
			name: "EventWithoutContent",
			events: []*session.Event{
				{Author: "user"},
			},
			want: nil,
		},
		{
			name: "Basic function call no rearrangement",
			events: []*session.Event{
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Search for test", "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: NewContentFromFunctionCall(fcBasic, "model")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frBasic, "user")}},
			},
			want: []*genai.Content{
				genai.NewContentFromText("Search for test", "user"),
				NewContentFromFunctionCall(fcBasic, "model"),
				NewContentFromFunctionResponse(frBasic, "user"),
			},
		},
		{
			name: "Rearrangement with intermediate response",
			events: []*session.Event{
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Run long process", "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: NewContentFromFunctionCall(fcLRO, "model")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frLROInter, "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Still processing...", "model")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frLROFinal, "user")}},
			},
			want: []*genai.Content{
				genai.NewContentFromText("Run long process", "user"),
				NewContentFromFunctionCall(fcLRO, "model"),
				NewContentFromFunctionResponse(frLROFinal, "user"),
			},
		},
		{
			name: "Rearrangement preserves unrelated function events",
			events: []*session.Event{
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Run long process and search", "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: NewContentFromFunctionCall(fcLRO, "model")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frLROInter, "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: NewContentFromFunctionCall(fcBasic, "model")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frBasic, "user")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frLROFinal, "user")}},
			},
			want: []*genai.Content{
				genai.NewContentFromText("Run long process and search", "user"),
				NewContentFromFunctionCall(fcLRO, "model"),
				NewContentFromFunctionResponse(frLROFinal, "user"),
				NewContentFromFunctionCall(fcBasic, "model"),
				NewContentFromFunctionResponse(frBasic, "user"),
			},
		},
		{
			name: "HITL confirmation approved preserves resumed tool response",
			events: []*session.Event{
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Request five vacation days", "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: NewContentFromFunctionCall(fcHITLApproved, "model")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frHITLApprovedPending, "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: NewContentFromFunctionCall(fcHITLApprovalRequest, "model")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frHITLApprovalConfirmed, "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frHITLApprovedFinal, "user")}},
			},
			want: []*genai.Content{
				genai.NewContentFromText("Request five vacation days", "user"),
				NewContentFromFunctionCall(fcHITLApproved, "model"),
				NewContentFromFunctionResponse(frHITLApprovedFinal, "user"),
			},
		},
		{
			name: "HITL confirmation denied preserves rejected tool response",
			events: []*session.Event{
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Request ten vacation days", "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: NewContentFromFunctionCall(fcHITLDenied, "model")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frHITLDeniedPending, "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: NewContentFromFunctionCall(fcHITLDenialRequest, "model")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frHITLDenialRejected, "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frHITLDeniedFinal, "user")}},
			},
			want: []*genai.Content{
				genai.NewContentFromText("Request ten vacation days", "user"),
				NewContentFromFunctionCall(fcHITLDenied, "model"),
				NewContentFromFunctionResponse(frHITLDeniedFinal, "user"),
			},
		},
		{
			name: "Rearrangement with mixed LRO and normal calls",
			events: []*session.Event{
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Analyze data and search for info", "user")}},
				{
					Author: agentName,
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role:  "model",
							Parts: []*genai.Part{{FunctionCall: fcLROMixed}, {FunctionCall: fcNormalMixed}},
						},
					},
				},
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role:  "user",
							Parts: []*genai.Part{{FunctionResponse: frLROInterMixed}, {FunctionResponse: frNormalMixed}},
						},
					},
				},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Analysis in progress, search completed", "model")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frLROFinalMixed, "user")}},
			},
			want: []*genai.Content{
				genai.NewContentFromText("Analyze data and search for info", "user"),
				{Role: "model", Parts: []*genai.Part{{FunctionCall: fcLROMixed}, {FunctionCall: fcNormalMixed}}},
				{Role: "user", Parts: []*genai.Part{{FunctionResponse: frLROFinalMixed}, {FunctionResponse: frNormalMixed}}},
			},
		},
		{
			name: "Rearrangement in history (non-final event)",
			events: []*session.Event{
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Start long process", "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: NewContentFromFunctionCall(fcHistLRO, "model")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frHistLROInter, "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Still processing...", "model")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frHistLROFinal, "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Process completed successfully!", "model")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Great! What's next?", "user")}},
			},
			want: []*genai.Content{
				genai.NewContentFromText("Start long process", "user"),
				NewContentFromFunctionCall(fcHistLRO, "model"),
				NewContentFromFunctionResponse(frHistLROFinal, "user"),
				genai.NewContentFromText("Still processing...", "model"),
				genai.NewContentFromText("Process completed successfully!", "model"),
				genai.NewContentFromText("Great! What's next?", "user"),
			},
		},
		{
			name: "Mixed rearrangement in history (non-final event)",
			events: []*session.Event{
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Analyze and search simultaneously", "user")}},
				{
					Author: agentName,
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role:  "model",
							Parts: []*genai.Part{{FunctionCall: fcHistLROMixed}, {FunctionCall: fcHistNormalMixed}},
						},
					},
				},
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role:  "user",
							Parts: []*genai.Part{{FunctionResponse: frHistLROInterMixed}, {FunctionResponse: frHistNormalMixed}},
						},
					},
				},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Analysis continuing, search done", "model")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: NewContentFromFunctionResponse(frHistLROFinalMixed, "user")}},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Both tasks completed successfully!", "model")}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Perfect! What should we do next?", "user")}},
			},
			want: []*genai.Content{
				genai.NewContentFromText("Analyze and search simultaneously", "user"),
				{Role: "model", Parts: []*genai.Part{{FunctionCall: fcHistLROMixed}, {FunctionCall: fcHistNormalMixed}}},
				{Role: "user", Parts: []*genai.Part{{FunctionResponse: frHistLROFinalMixed}, {FunctionResponse: frHistNormalMixed}}},
				genai.NewContentFromText("Analysis continuing, search done", "model"),
				genai.NewContentFromText("Both tasks completed successfully!", "model"),
				genai.NewContentFromText("Perfect! What should we do next?", "user"),
			},
		},
		{
			name: "Rearrangement preserves mixed text parts",
			events: []*session.Event{
				{Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Before function call", "user")}},
				{
					Author: agentName,
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role:  "model",
							Parts: []*genai.Part{{Text: "I'll process this for you"}, {FunctionCall: fcPreserve}},
						},
					},
				},
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role:  "user",
							Parts: []*genai.Part{{Text: "Intermediate prefix"}, {FunctionResponse: frPreserveInter}, {Text: "Processing..."}},
						},
					},
				},
				{Author: agentName, LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("Still working on it...", "model")}},
				{
					Author: "user",
					LLMResponse: model.LLMResponse{
						Content: &genai.Content{
							Role:  "user",
							Parts: []*genai.Part{{Text: "Final prefix"}, {FunctionResponse: frPreserveFinal}, {Text: "Final suffix"}},
						},
					},
				},
			},
			want: []*genai.Content{
				genai.NewContentFromText("Before function call", "user"),
				{Role: "model", Parts: []*genai.Part{{Text: "I'll process this for you"}, {FunctionCall: fcPreserve}}},
				{Role: "user", Parts: []*genai.Part{
					{Text: "Intermediate prefix"},
					{FunctionResponse: frPreserveFinal},
					{Text: "Processing..."},
					{Text: "Final prefix"},
					{Text: "Final suffix"},
				}},
			},
		},
		{
			name: "Error on function response without matching call",
			events: []*session.Event{
				{Author: "user", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{Text: "Regular message"}}}}},
				{Author: "user", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: frOrphaned}}}}},
			},
			want:    nil,
			wantErr: "no function call event found",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			testAgent := utils.Must(llmagent.New(llmagent.Config{
				Name:  agentName,
				Model: testModel,
			}))

			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent: testAgent,
				Session: &fakeSession{
					events: tc.events,
				},
			})

			req := &model.LLMRequest{}
			for ev, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
				if ev != nil {
					t.Fatal("ContentsRequestProcessor generated an unexpected event")
				}
				if tc.wantErr != "" {
					if err == nil {
						t.Fatal("ContentsRequestProcessor succeeded; expected an error")
					}
					if !strings.Contains(err.Error(), tc.wantErr) {
						t.Errorf("Expected error to contain %q, got: %v", tc.wantErr, err)
					}
					return // Test is done
				}

				if err != nil {
					t.Fatalf("ContentsRequestProcessor failed: %v", err)
				}
			}

			got := req.Contents
			if diff := cmp.Diff(wantWithContinuation(tc.want), got); diff != "" {
				t.Errorf("LLMRequest.Contents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// NewContentFromFunctionCall creates a new Content struct with a single FunctionCall part.
// It assigns the provided role to the Content.
func NewContentFromFunctionCall(fc *genai.FunctionCall, role string) *genai.Content {
	return &genai.Content{
		Role:  role,
		Parts: []*genai.Part{{FunctionCall: fc}},
	}
}

// NewContentFromFunctionResponse creates a new Content struct with a single FunctionResponse part.
// It assigns the provided role to the Content.
func NewContentFromFunctionResponse(fr *genai.FunctionResponse, role string) *genai.Content {
	return &genai.Content{
		Role:  role,
		Parts: []*genai.Part{{FunctionResponse: fr}},
	}
}

type fakeSession struct {
	events []*session.Event
}

func (s *fakeSession) State() session.State {
	return nil
}

func (s *fakeSession) ID() string {
	return "test_session"
}

func (s *fakeSession) UserID() string {
	return "test_user"
}

func (s *fakeSession) AppName() string {
	return "test_app"
}

func (s *fakeSession) LastUpdateTime() time.Time {
	return time.Time{}
}

func (s *fakeSession) Events() session.Events {
	return s
}

func (s *fakeSession) Len() int {
	return len(s.events)
}

func (s *fakeSession) All() iter.Seq[*session.Event] {
	return slices.Values(s.events)
}

func (s *fakeSession) AllBackward() iter.Seq[*session.Event] {
	return nil
}

func (s *fakeSession) Append(ctx context.Context, e ...*session.Event) error {
	return nil
}

func (s *fakeSession) At(i int) *session.Event {
	return s.events[i]
}

var (
	_ session.Session = (*fakeSession)(nil)
	_ session.Events  = (*fakeSession)(nil)
)

func wantWithContinuation(want []*genai.Content) []*genai.Content {
	if len(want) > 0 {
		if last := want[len(want)-1]; last != nil && last.Role != "user" {
			res := make([]*genai.Content, len(want), len(want)+1)
			copy(res, want)
			res = append(res, genai.NewContentFromText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", "user"))
			return res
		}
	}
	return want
}
