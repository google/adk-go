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
