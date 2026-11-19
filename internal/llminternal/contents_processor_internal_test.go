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

package llminternal

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

func TestDropOrphanedFunctionResponses_LogAndClone(t *testing.T) {
	event := &session.Event{LLMResponse: model.LLMResponse{
		ModelVersion: "model-version", FinishReason: genai.FinishReasonStop, TurnComplete: true,
		Content: &genai.Content{Role: "user", Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{ID: "stale\nforged", Response: map[string]any{"secret": "payload"}}},
			{Text: "Keep this", FunctionResponse: &genai.FunctionResponse{ID: "other"}},
			{Text: "Keep this too"},
		}},
	}}
	want := *event
	want.LLMResponse.Content = genai.NewContentFromText("Keep this", "user")
	want.LLMResponse.Content.Parts = append(want.LLMResponse.Content.Parts, &genai.Part{Text: "Keep this too"})
	output := captureLog(t, func() {
		got, _ := dropOrphanedFunctionResponses([]*session.Event{event}, nil)
		if diff := cmp.Diff([]*session.Event{&want}, got); diff != "" {
			t.Errorf("pruned events mismatch (-want +got):\n%s", diff)
		}
	})
	if want := "adk: dropping function responses with no matching function call: [\"stale\\nforged\" \"other\"]\n"; output != want {
		t.Errorf("log = %q, want %q", output, want)
	}
	if len(event.Content.Parts) != 3 || event.Content.Parts[0].FunctionResponse == nil || event.Content.Parts[1].FunctionResponse == nil || event.Content.Parts[2] == nil {
		t.Fatal("pruning mutated the original event parts")
	}
}

func TestDropOrphanedFunctionCalls(t *testing.T) {
	call := func(id string) *genai.Part {
		return &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: "tool"}, ThoughtSignature: []byte("sig-" + id)}
	}
	response := func(id string) *genai.Part {
		return &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: id, Name: "tool"}}
	}
	event := func(role string, parts ...*genai.Part) *session.Event {
		return &session.Event{LLMResponse: model.LLMResponse{Content: &genai.Content{Role: role, Parts: parts}}}
	}

	parallel := event("model", &genai.Part{Text: "Calling tools."}, call("answered"), call("orphan_1"), call(""))
	onlyOrphan := event("model", call("orphan_2"))
	longRunning := event("model", call("long_running"), call("confirmation"))
	longRunning.LongRunningToolIDs = []string{"long_running", "confirmation"}
	answer := event("user", response("answered"))
	user := event("user", &genai.Part{Text: "Are you still there?"})
	events := []*session.Event{parallel, answer, onlyOrphan, longRunning, user}

	wantParallel := cloneEvent(parallel)
	wantParallel.LLMResponse.Content.Parts = []*genai.Part{{Text: "Calling tools."}, call("answered"), call("")}
	want := []*session.Event{wantParallel, answer, longRunning, user}

	output := captureLog(t, func() {
		got := dropOrphanedFunctionCalls(events)
		if diff := cmp.Diff(want, got); diff != "" {
			t.Fatalf("pruned events mismatch (-want +got):\n%s", diff)
		}
		for i, ev := range got {
			if ev == parallel {
				t.Errorf("got[%d] is the stored event; want a clone because a call was dropped from it", i)
			}
		}
		if got[2] != longRunning || got[3] != user {
			t.Error("events without a dropped call were copied; want the stored events returned as is")
		}
	})
	if want := "adk: dropping function calls with no matching function response: [\"orphan_1\" \"orphan_2\"]\n"; output != want {
		t.Errorf("log = %q, want %q", output, want)
	}
	if len(parallel.Content.Parts) != 4 || parallel.Content.Parts[2].FunctionCall == nil || len(onlyOrphan.Content.Parts) != 1 {
		t.Fatal("pruning mutated the original event parts")
	}
}

func TestDropOrphanedFunctionCalls_NothingToDrop(t *testing.T) {
	events := []*session.Event{
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromFunctionCall("tool", nil, "model")}},
		{LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "answered", Name: "tool"}}}}}},
		{LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "answered", Name: "tool"}}}}}},
		{LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("hello", "user")}},
	}
	output := captureLog(t, func() {
		got := dropOrphanedFunctionCalls(events)
		if !slices.Equal(got, events) {
			t.Error("dropOrphanedFunctionCalls changed a history with no orphaned call; want the input events returned unchanged")
		}
	})
	if output != "" {
		t.Errorf("log = %q, want nothing", output)
	}
}

func TestPromptTokenEstimator_SuffixDoesNotLogPairedResponseAsOrphan(t *testing.T) {
	const agentName = "worker"
	svc := session.InMemoryService()
	created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
	if err != nil {
		t.Fatal(err)
	}
	sess := created.Session
	events := []*session.Event{
		{Author: agentName, LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
			{FunctionCall: &genai.FunctionCall{ID: "paired", Name: "ping"}},
		}}}},
		{Author: agentName, LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "user", Parts: []*genai.Part{
			{FunctionResponse: &genai.FunctionResponse{ID: "paired", Name: "ping", Response: map[string]any{"result": strings.Repeat("x", 4000)}}},
		}}}},
	}
	for _, event := range events {
		if err := svc.AppendEvent(t.Context(), sess, event); err != nil {
			t.Fatal(err)
		}
	}
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:   &mockLLMAgent{Agent: utils.Must(agent.New(agent.Config{Name: agentName})), s: &State{}},
		Session: sess,
	})
	output := captureLog(t, func() {
		promptTokenEstimator(ctx)(events[1:])
	})
	if output != "" {
		t.Errorf("paired response in suffix produced a stale-response notice: %q", output)
	}
}

// TestContentsRequestProcessor_DropOrphanedResponses pins which function
// responses the contents build drops because no earlier call matches them,
// and what it logs about them, as adk-python's
// _rearranger.drop_orphaned_function_responses does.
func TestContentsRequestProcessor_DropOrphanedResponses(t *testing.T) {
	const agentName = "test_agent"
	text := func(s string) *genai.Part { return &genai.Part{Text: s} }
	call := func(id string) *genai.Part {
		return &genai.Part{FunctionCall: &genai.FunctionCall{ID: id, Name: "t"}}
	}
	resp := func(id, name, r string) *genai.Part {
		return &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: id, Name: name, Response: map[string]any{"r": r}}}
	}
	event := func(author, role string, parts ...*genai.Part) *session.Event {
		return &session.Event{Author: author, LLMResponse: model.LLMResponse{Content: &genai.Content{Role: role, Parts: parts}}}
	}
	user := func(s string) *session.Event { return event("user", "user", text(s)) }
	content := func(role string, parts ...*genai.Part) *genai.Content {
		return &genai.Content{Role: role, Parts: parts}
	}
	scopedCall := event(agentName, "model", call("c1"))
	scopedCall.IsolationScope = "task"

	tests := []struct {
		name            string
		includeContents IncludeContents
		events          []*session.Event
		want            []*genai.Content
		wantErr         string
		wantLog         string
	}{
		{
			// Fails if a call that comes after a response counts for it.
			name: "response before its call is dropped",
			events: []*session.Event{
				user("go"),
				event("user", "user", resp("x", "t", "early")),
				event(agentName, "model", call("x")),
				event("user", "user", resp("x", "t", "late")),
			},
			want: []*genai.Content{
				content("user", text("go")),
				content("model", call("x")),
				content("user", resp("x", "t", "late")),
			},
			wantLog: "adk: dropping function responses with no matching function call: [\"x\"]\n",
		},
		{
			// Fails if the log line is not written for a dropped response.
			name: "response with no call anywhere is dropped and logged",
			events: []*session.Event{
				user("go"),
				event("user", "user", resp("zz", "t", "x")),
				user("next"),
			},
			want: []*genai.Content{
				content("user", text("go")),
				content("user", text("next")),
			},
			wantLog: "adk: dropping function responses with no matching function call: [\"zz\"]\n",
		},
		{
			// Fails if an id-less orphan is kept or logged as "".
			name: "id-less orphan is logged as <missing-id>",
			events: []*session.Event{
				user("hello"),
				event("user", "user", resp("", "tool", "x")),
			},
			want: []*genai.Content{
				content("user", text("hello")),
			},
			wantLog: "adk: dropping function responses with no matching function call: [\"<missing-id>\"]\n",
		},
		{
			// Fails if a response whose call is outside the history is kept
			// before the last event, or is logged.
			name: "mid-history response whose call is outside the history",
			events: []*session.Event{
				scopedCall,
				user("go"),
				event("user", "user", text("note"), resp("c1", "t", "x")),
				user("next"),
			},
			want: []*genai.Content{
				content("user", text("go")),
				content("user", text("note")),
				content("user", text("next")),
			},
		},
		{
			// Fails if the trailing response whose call precedes the
			// include_contents "none" slice is dropped (adk-python drops it).
			name:            "none-mode pivot",
			includeContents: IncludeContentsNone,
			events: []*session.Event{
				event(agentName, "model", call("t1")),
				user("Q2"),
				event(agentName, "user", text("note"), resp("t1", "t", "x")),
			},
			wantErr: "no function call event found",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			svc := session.InMemoryService()
			created, err := svc.Create(t.Context(), &session.CreateRequest{AppName: "app", UserID: "user"})
			if err != nil {
				t.Fatal(err)
			}
			sess := created.Session
			for _, ev := range tc.events {
				if err := svc.AppendEvent(t.Context(), sess, ev); err != nil {
					t.Fatal(err)
				}
			}
			// AppendEvent assigns event ids, so the snapshot is taken after it.
			before, err := json.Marshal(slices.Collect(sess.Events().All()))
			if err != nil {
				t.Fatal(err)
			}
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent:   &mockLLMAgent{Agent: utils.Must(agent.New(agent.Config{Name: agentName})), s: &State{IncludeContents: tc.includeContents}},
				Session: sess,
			})
			req := &model.LLMRequest{}
			var gotErr error
			gotLog := captureLog(t, func() {
				for _, err := range ContentsRequestProcessor(ctx, req, &Flow{}) {
					gotErr = err
				}
			})

			if tc.wantErr != "" {
				if gotErr == nil || !strings.Contains(gotErr.Error(), tc.wantErr) {
					t.Fatalf("ContentsRequestProcessor error = %v, want one containing %q", gotErr, tc.wantErr)
				}
			} else if gotErr != nil {
				t.Fatalf("ContentsRequestProcessor failed: %v", gotErr)
			}
			if diff := cmp.Diff(tc.want, req.Contents); diff != "" {
				t.Errorf("LLMRequest.Contents mismatch (-want +got):\n%s", diff)
			}
			if gotLog != tc.wantLog {
				t.Errorf("log = %q, want %q", gotLog, tc.wantLog)
			}
			after, err := json.Marshal(slices.Collect(sess.Events().All()))
			if err != nil {
				t.Fatal(err)
			}
			if string(before) != string(after) {
				t.Errorf("ContentsRequestProcessor mutated session events (-before +after):\n%s", cmp.Diff(string(before), string(after)))
			}
		})
	}
}

// TestRearrangeEventsForFunctionResponsesInHistory_CompactionSummaryIsNotStale
// pins that a compaction summary between two updates of one call is kept: it
// is not a model reply the newer update supersedes. The public path drops
// compaction records when compaction is not configured, so the step is called
// directly.
func TestRearrangeEventsForFunctionResponsesInHistory_CompactionSummaryIsNotStale(t *testing.T) {
	event := func(role string, parts ...*genai.Part) *session.Event {
		return &session.Event{Author: "test_agent", InvocationID: "inv1", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: role, Parts: parts}}}
	}
	resp := func(r string) *genai.Part {
		return &genai.Part{FunctionResponse: &genai.FunctionResponse{ID: "c1", Name: "watch", Response: map[string]any{"r": r}}}
	}
	callEvent := event("model", &genai.Part{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "watch"}})
	progress := event("user", resp("progress"))
	summary := event("model", &genai.Part{Text: "Summary of earlier turns."})
	summary.Actions.Compaction = &session.EventCompaction{CompactedContent: genai.NewContentFromText("Summary of earlier turns.", "model")}
	done := event("user", resp("done"))
	finished := event("model", &genai.Part{Text: "Finished."})

	got, err := rearrangeEventsForFunctionResponsesInHistory([]*session.Event{callEvent, progress, summary, done, finished})
	if err != nil {
		t.Fatalf("rearrangeEventsForFunctionResponsesInHistory() failed: %v", err)
	}
	want := []*session.Event{callEvent, done, summary, finished}
	if !slices.Equal(got, want) {
		t.Errorf("rearrangeEventsForFunctionResponsesInHistory() = %v, want [call, done response, compaction summary, Finished] (the input events themselves)", got)
	}
}

// TestBuildContents_ClonedCallEventIdentity pins that the contents build
// recognizes a copy of a session event as part of the history. The trailing
// event answers its own call, so its response is dropped, as adk-python
// drops it; when the copy counts as a different event, the call in the stored
// event counts as outside the history, the response is kept and the build
// fails.
func TestBuildContents_ClonedCallEventIdentity(t *testing.T) {
	u := &session.Event{ID: "e1", Author: "user", LLMResponse: model.LLMResponse{Content: genai.NewContentFromText("go", "user")}}
	trailing := &session.Event{ID: "e2", Author: "test_agent", LLMResponse: model.LLMResponse{Content: &genai.Content{Role: "model", Parts: []*genai.Part{
		{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "tool"}},
		{FunctionResponse: &genai.FunctionResponse{ID: "c1", Name: "tool", Response: map[string]any{"r": "same"}}},
	}}}}
	allEvents := []*session.Event{u, trailing}
	events := []*session.Event{u, cloneEvent(trailing)}

	got, err := buildContentsDefaultWithCallSource("test_agent", "", "", events, allEvents, false, nil)
	if err != nil {
		t.Fatalf("buildContentsDefaultWithCallSource() failed: %v", err)
	}
	want := []*genai.Content{genai.NewContentFromText("go", "user")}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("buildContentsDefaultWithCallSource() mismatch (-want +got):\n%s", diff)
	}
}
