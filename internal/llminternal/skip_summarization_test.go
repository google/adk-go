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
	"context"
	"iter"
	"testing"

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

type skipSummaryArgs struct{}
type skipSummaryResult struct {
	Result string `json:"result"`
}

func skipSummaryFunc(ctx agent.Context, _ skipSummaryArgs) (skipSummaryResult, error) {
	ctx.Actions().SkipSummarization = true
	return skipSummaryResult{Result: "tool output"}, nil
}

// skipSummaryModel returns a single function call and, if invoked again,
// fails the test: SkipSummarization is expected to end the run after one
// function response, so the model must never be asked to summarize it.
type skipSummaryModel struct {
	model.LLM
	t     *testing.T
	calls int
}

func (m *skipSummaryModel) Name() string { return "skip-summary-model" }

func (m *skipSummaryModel) GenerateContent(ctx context.Context, req *model.LLMRequest, useStream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		m.calls++
		if m.calls > 1 {
			m.t.Fatalf("model called %d times; SkipSummarization should end the run after the first function response", m.calls)
			return
		}
		yield(&model.LLMResponse{
			Content: &genai.Content{
				Role: "model",
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							ID:   "call_1",
							Name: "skip_summary",
							Args: map[string]any{},
						},
					},
				},
			},
		}, nil)
	}
}

// TestHandleFunctionCalls_SkipSummarizationDisplaysResult verifies that when
// a tool sets SkipSummarization, the parent agent loop still terminates on
// the function response event (the flag's documented effect), but the
// terminal event carries the tool's result as a visible text part rather
// than a bare, unrendered FunctionResponse.
func TestHandleFunctionCalls_SkipSummarizationDisplaysResult(t *testing.T) {
	skipSummaryTool, err := functiontool.New(functiontool.Config{
		Name:        "skip_summary",
		Description: "returns a result and skips summarization",
	}, skipSummaryFunc)
	if err != nil {
		t.Fatal(err)
	}

	m := &skipSummaryModel{t: t}

	a, err := llmagent.New(llmagent.Config{
		Name:        "tester",
		Description: "Tester agent",
		Instruction: "You are a tester agent.",
		Model:       m,
		Tools:       []tool.Tool{skipSummaryTool},
	})
	if err != nil {
		t.Fatal(err)
	}

	sessionService := session.InMemoryService()
	if _, err := sessionService.Create(t.Context(), &session.CreateRequest{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	}); err != nil {
		t.Fatal(err)
	}

	r, err := runner.New(runner.Config{
		Agent:          a,
		SessionService: sessionService,
		AppName:        "testApp",
	})
	if err != nil {
		t.Fatal(err)
	}

	it := r.Run(t.Context(), "testUser", "testSession", &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{genai.NewPartFromText("go")},
	}, agent.RunConfig{StreamingMode: agent.StreamingModeSSE})

	var events []*session.Event
	for ev, err := range it {
		if err != nil {
			t.Fatal(err)
		}
		events = append(events, ev)
	}

	if len(events) != 2 {
		t.Fatalf("expected 2 events (function call, function response), got %d", len(events))
	}

	respEvent := events[1]
	if !respEvent.IsFinalResponse() {
		t.Errorf("expected function response event to be the final response, IsFinalResponse() = false")
	}

	var gotFunctionResponse, gotText bool
	var text string
	for _, part := range respEvent.Content.Parts {
		if part.FunctionResponse != nil {
			gotFunctionResponse = true
		}
		if part.Text != "" {
			gotText = true
			text = part.Text
		}
	}
	if !gotFunctionResponse {
		t.Errorf("expected final event to retain the FunctionResponse part")
	}
	if !gotText {
		t.Errorf("expected final event to carry a visible text part with the tool result")
	}
	if text != "tool output" {
		t.Errorf("text part = %q, want %q", text, "tool output")
	}
}
