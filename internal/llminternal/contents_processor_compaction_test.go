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
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent/llmagent"
	icontext "google.golang.org/adk/v2/internal/context"
	"google.golang.org/adk/v2/internal/llminternal"
	"google.golang.org/adk/v2/internal/utils"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
)

// compactionEpoch anchors the synthetic timestamps in this file. Only the
// relative ordering of the events matters.
var compactionEpoch = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func compactionAt(n int) time.Time {
	return compactionEpoch.Add(time.Duration(n) * time.Second)
}

func compactionTextEvent(author string, ts int, text string) *session.Event {
	role := "model"
	if author == "user" {
		role = "user"
	}
	return &session.Event{
		Author:      author,
		Timestamp:   compactionAt(ts),
		LLMResponse: model.LLMResponse{Content: genai.NewContentFromText(text, genai.Role(role))},
	}
}

func compactionSummaryEvent(ts, start, end int, summary string) *session.Event {
	return &session.Event{
		Author:    "user",
		Timestamp: compactionAt(ts),
		Actions: session.EventActions{
			Compaction: &session.EventCompaction{
				StartTimestamp:   compactionAt(start),
				EndTimestamp:     compactionAt(end),
				CompactedContent: genai.NewContentFromText(summary, "model"),
			},
		},
	}
}

// TestContentsRequestProcessor_Compaction checks the prompt the model actually
// receives once a session holds compaction events: covered turns are replaced
// by the summary, and everything else is untouched.
func TestContentsRequestProcessor_Compaction(t *testing.T) {
	t.Parallel()

	const agentName = "testAgent"
	testModel := &testModel{}

	testCases := []struct {
		name   string
		events []*session.Event
		want   []*genai.Content
	}{
		{
			name: "summary replaces the turns it covers",
			events: []*session.Event{
				compactionTextEvent("user", 1, "q1"),
				compactionTextEvent(agentName, 2, "a1"),
				compactionTextEvent("user", 3, "q2"),
				compactionTextEvent(agentName, 4, "a2"),
				compactionSummaryEvent(5, 1, 4, "Earlier: the user asked two questions."),
				compactionTextEvent("user", 6, "q3"),
			},
			want: []*genai.Content{
				genai.NewContentFromText("Earlier: the user asked two questions.", "model"),
				genai.NewContentFromText("q3", "user"),
			},
		},
		{
			name: "turns outside the range survive alongside the summary",
			events: []*session.Event{
				compactionTextEvent("user", 1, "q1"),
				compactionTextEvent(agentName, 2, "a1"),
				compactionSummaryEvent(3, 1, 2, "Earlier: one exchange."),
				compactionTextEvent("user", 4, "q2"),
				compactionTextEvent(agentName, 5, "a2"),
			},
			want: []*genai.Content{
				genai.NewContentFromText("Earlier: one exchange.", "model"),
				genai.NewContentFromText("q2", "user"),
				genai.NewContentFromText("a2", "model"),
			},
		},
		{
			name: "a subsumed summary is dropped, only the wider one is sent",
			events: []*session.Event{
				compactionTextEvent("user", 1, "q1"),
				compactionTextEvent(agentName, 2, "a1"),
				compactionSummaryEvent(3, 1, 2, "narrow summary"),
				compactionTextEvent("user", 4, "q2"),
				compactionTextEvent(agentName, 5, "a2"),
				compactionSummaryEvent(6, 1, 5, "wide summary"),
			},
			want: []*genai.Content{
				genai.NewContentFromText("wide summary", "model"),
			},
		},
		{
			name: "no compaction events leaves history untouched",
			events: []*session.Event{
				compactionTextEvent("user", 1, "q1"),
				compactionTextEvent(agentName, 2, "a1"),
			},
			want: []*genai.Content{
				genai.NewContentFromText("q1", "user"),
				genai.NewContentFromText("a1", "model"),
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			testAgent := utils.Must(llmagent.New(llmagent.Config{
				Name:  agentName,
				Model: testModel,
			}))
			ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
				Agent:   testAgent,
				Session: &fakeSession{events: tc.events},
			})

			req := &model.LLMRequest{}
			for ev, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
				if ev != nil {
					t.Fatal("ContentsRequestProcessor generated an unexpected event")
				}
				if err != nil {
					t.Fatalf("ContentsRequestProcessor failed: %v", err)
				}
			}

			if diff := cmp.Diff(wantWithContinuation(tc.want), req.Contents); diff != "" {
				t.Errorf("LLMRequest contents mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestContentsRequestProcessor_CompactionKeepsToolPairing covers the paused
// long-running tool case: the call is summarized away but its result arrives
// later, so the call has to be restored or prompt assembly fails.
func TestContentsRequestProcessor_CompactionKeepsToolPairing(t *testing.T) {
	t.Parallel()

	const agentName = "testAgent"

	call := &session.Event{
		Author:    agentName,
		Timestamp: compactionAt(2),
		LLMResponse: model.LLMResponse{Content: &genai.Content{
			Role:  "model",
			Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{ID: "c1", Name: "long_job"}}},
		}},
		LongRunningToolIDs: []string{"c1"},
	}
	placeholder := &session.Event{
		Author:    "user",
		Timestamp: compactionAt(3),
		LLMResponse: model.LLMResponse{Content: &genai.Content{
			Role:  "user",
			Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "c1", Name: "long_job", Response: map[string]any{"status": "pending"}}}},
		}},
	}
	result := &session.Event{
		Author:    "user",
		Timestamp: compactionAt(6),
		LLMResponse: model.LLMResponse{Content: &genai.Content{
			Role:  "user",
			Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{ID: "c1", Name: "long_job", Response: map[string]any{"status": "done"}}}},
		}},
	}

	events := []*session.Event{
		compactionTextEvent("user", 1, "start the job"),
		call,
		placeholder,
		compactionSummaryEvent(5, 1, 3, "Earlier: the user started a long job."),
		result,
	}

	testAgent := utils.Must(llmagent.New(llmagent.Config{Name: agentName, Model: &testModel{}}))
	ctx := icontext.NewInvocationContext(t.Context(), icontext.InvocationContextParams{
		Agent:   testAgent,
		Session: &fakeSession{events: events},
	})

	req := &model.LLMRequest{}
	for ev, err := range llminternal.ContentsRequestProcessor(ctx, req, &llminternal.Flow{}) {
		if ev != nil {
			t.Fatal("ContentsRequestProcessor generated an unexpected event")
		}
		if err != nil {
			t.Fatalf("ContentsRequestProcessor failed: %v", err)
		}
	}

	// The recovered call must precede the surviving response, or the model sees
	// a response to a call it was never shown.
	var sawCall, sawResponse bool
	for _, c := range req.Contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil && p.FunctionCall.ID == "c1" {
				sawCall = true
			}
			if p.FunctionResponse != nil && p.FunctionResponse.ID == "c1" {
				if !sawCall {
					t.Error("function response for c1 appears before its call was recovered")
				}
				sawResponse = true
			}
		}
	}
	if !sawCall {
		t.Errorf("compacted long-running call was not recovered; contents: %v", req.Contents)
	}
	if !sawResponse {
		t.Errorf("surviving function response is missing; contents: %v", req.Contents)
	}
}
