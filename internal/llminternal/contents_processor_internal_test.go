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
		got := dropOrphanedFunctionResponses([]*session.Event{event}, nil)
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
