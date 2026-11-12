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

package llminternal_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// A rewound turn stays in the session but must not reach the model. Ports
// adk-python's test_rewind_events_are_filtered_out
// (tests/unittests/flows/llm_flows/context/test_contents.py).
func TestContentsRequestProcessor_DropsRewoundEvents(t *testing.T) {
	t.Parallel()
	turn := func(invocationID, author, text, role string) *session.Event {
		return &session.Event{
			InvocationID: invocationID,
			Author:       author,
			LLMResponse:  model.LLMResponse{Content: genai.NewContentFromText(text, genai.Role(role))},
		}
	}
	events := []*session.Event{
		turn("inv1", "user", "First message", "user"),
		turn("inv1", "testAgent", "First response", "model"),
		turn("inv2", "user", "Second message", "user"),
		turn("inv2", "testAgent", "Second response", "model"),
		{InvocationID: "rewind_inv", Author: "user", Actions: session.EventActions{RewindBeforeInvocationID: "inv2"}},
		turn("inv3", "user", "Third message", "user"),
	}
	testAgent := utils.Must(llmagent.New(llmagent.Config{Name: "testAgent", Model: &testModel{}}))
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:   testAgent,
		Session: &fakeSession{events: events},
	})

	req := &model.LLMRequest{}
	for _, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
		if err != nil {
			t.Fatalf("ContentsRequestProcessor failed: %v", err)
		}
	}

	want := []*genai.Content{
		genai.NewContentFromText("First message", "user"),
		genai.NewContentFromText("First response", "model"),
		genai.NewContentFromText("Third message", "user"),
	}
	if diff := cmp.Diff(want, req.Contents); diff != "" {
		t.Errorf("contents mismatch (-want +got):\n%s", diff)
	}
}
