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

	"google.golang.org/genai"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/internal/testutil"
	"google.golang.org/adk/v2/runner"
	"google.golang.org/adk/v2/session"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/agenttool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// runSkipSummarizationScenario runs rootAgent (backed by a singleCallMockModel
// that calls toolName once) to completion and returns the collected events.
func runSkipSummarizationScenario(t *testing.T, rootAgent agent.Agent) []*session.Event {
	t.Helper()

	sessionService := session.InMemoryService()
	if _, err := sessionService.Create(t.Context(), &session.CreateRequest{
		AppName:   "testApp",
		UserID:    "testUser",
		SessionID: "testSession",
	}); err != nil {
		t.Fatal(err)
	}

	r, err := runner.New(runner.Config{
		Agent:          rootAgent,
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
	return events
}

// TestHandleFunctionCalls_SkipSummarization_AgentToolDisplaysResult verifies
// that when agenttool sets SkipSummarization, the parent agent loop still
// terminates on the function response event (the flag's documented effect),
// but the terminal event carries the sub-agent's answer as a visible text
// part rather than a bare, unrendered FunctionResponse.
func TestHandleFunctionCalls_SkipSummarization_AgentToolDisplaysResult(t *testing.T) {
	subAgentModel := &testutil.MockModel{
		Responses: []*genai.Content{
			genai.NewContentFromText("sub-agent answer", genai.RoleModel),
		},
	}
	subAgent, err := llmagent.New(llmagent.Config{
		Name:        "sub_agent",
		Description: "answers questions",
		Instruction: "Answer the question.",
		Model:       subAgentModel,
	})
	if err != nil {
		t.Fatal(err)
	}

	agentTool := agenttool.New(subAgent, &agenttool.Config{SkipSummarization: true})

	rootModel := &singleCallMockModel{
		toolName: agentTool.Name(),
		fcID:     "call_1",
		args:     map[string]any{"request": "what is the answer?"},
	}
	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "root",
		Description: "root agent",
		Instruction: "Delegate to the sub-agent.",
		Model:       rootModel,
		Tools:       []tool.Tool{agentTool},
	})
	if err != nil {
		t.Fatal(err)
	}

	events := runSkipSummarizationScenario(t, rootAgent)

	if len(events) != 2 {
		t.Fatalf("expected 2 events (function call, function response), got %d", len(events))
	}
	if rootModel.calls != 1 {
		t.Errorf("root model called %d times, want 1: SkipSummarization should end the run after the first function response", rootModel.calls)
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
		t.Errorf("expected final event to carry a visible text part with the sub-agent's answer")
	}
	if text != "sub-agent answer" {
		t.Errorf("text part = %q, want %q", text, "sub-agent answer")
	}
}

// TestHandleFunctionCalls_SkipSummarization_PlainToolResultNotDisplayed
// verifies that SkipSummarization set by a tool other than agenttool still
// ends the parent agent loop, but does NOT get its result surfaced as text.
// SkipSummarization is also used by UI/widget and pending-confirmation tools
// to suppress an internal acknowledgement that was never meant to be shown;
// unconditionally displaying it would leak that payload into the transcript
// and bypass client-side filters that strip FunctionResponse parts.
func TestHandleFunctionCalls_SkipSummarization_PlainToolResultNotDisplayed(t *testing.T) {
	const toolName = "widget_ack"

	widgetTool, err := functiontool.New(functiontool.Config{
		Name:        toolName,
		Description: "acknowledges a widget interaction",
	}, func(ctx agent.Context, _ struct{}) (map[string]string, error) {
		ctx.Actions().SkipSummarization = true
		return map[string]string{"status": "ok"}, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	rootModel := &singleCallMockModel{toolName: toolName, fcID: "call_1"}
	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "root",
		Description: "root agent",
		Instruction: "Use the widget tool.",
		Model:       rootModel,
		Tools:       []tool.Tool{widgetTool},
	})
	if err != nil {
		t.Fatal(err)
	}

	events := runSkipSummarizationScenario(t, rootAgent)

	if len(events) != 2 {
		t.Fatalf("expected 2 events (function call, function response), got %d", len(events))
	}
	if rootModel.calls != 1 {
		t.Errorf("root model called %d times, want 1: SkipSummarization should end the run after the first function response", rootModel.calls)
	}

	respEvent := events[1]
	if !respEvent.IsFinalResponse() {
		t.Errorf("expected function response event to be the final response, IsFinalResponse() = false")
	}
	if len(respEvent.Content.Parts) != 1 || respEvent.Content.Parts[0].FunctionResponse == nil {
		t.Errorf("expected final event to carry only the FunctionResponse part, got %d parts: %+v", len(respEvent.Content.Parts), respEvent.Content.Parts)
	}
}
