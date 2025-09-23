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

package tool_test

import (
	"context"
	"strings"
	"testing"

	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

func TestNewLongRunningFunctionTool(t *testing.T) {
	type SumArgs struct {
		A int `json:"a"` // an integer to sum
		B int `json:"b"` // another integer to sum
	}
	type SumResult struct {
		Result string `json:"result"` // the operation result
	}

	handler := func(ctx context.Context, input SumArgs) SumResult {
		return SumResult{Result: "Processing sum"}
	}
	sumTool, err := tool.NewLongRunningFunctionTool(tool.FunctionToolConfig{
		Name:        "sum",
		Description: "sums two integers",
	}, handler)
	if err != nil {
		t.Fatalf("TestNewLongRunningFunctionTool failed: %v", err)
	}
	if sumTool.Name() != "sum" {
		t.Fatalf("TestNewLongRunningFunctionTool failed: wrong name")
	}
	if sumTool.Description() != "sums two integers" {
		t.Fatalf("TestNewLongRunningFunctionTool failed: wrong description")
	}
	if sumTool.IsLongRunning() == false {
		t.Fatalf("TestNewLongRunningFunctionTool failed: wrong value for IsLongRunning")
	}
	functionTool, ok := sumTool.(toolinternal.FunctionTool)
	if !ok {
		t.Fatalf("TestNewLongRunningFunctionTool failed: could not convert to FunctionTool")
	}
	if !strings.Contains(functionTool.Declaration().Description, "NOTE: This is a long-running operation") {
		t.Fatalf("TestNewLongRunningFunctionTool failed: wrong description note")
	}

	_ = sumTool // use the tool
}

func NewContentFromFunctionResponseWithID(name string, response map[string]any, id string, role string) *genai.Content {
	content := genai.NewContentFromFunctionResponse(name, response, genai.Role(role))
	content.Parts[0].FunctionResponse.ID = id
	return content
}

type IncArgs struct {
}

func TestLongRunningFunctionFlow(t *testing.T) {
	functionCalled := 0
	increaseByOne := func(ctx context.Context, x IncArgs) map[string]string {
		functionCalled++
		return map[string]string{"status": "pending"}
	}
	testLongRunningFunctionFlow(t, increaseByOne, "status", &functionCalled)
}

func TestLongRunningStringFunctionFlow(t *testing.T) {
	functionCalled := 0
	increaseByOne := func(ctx context.Context, x IncArgs) string {
		functionCalled++
		return "pending"
	}
	testLongRunningFunctionFlow(t, increaseByOne, "result", &functionCalled)
}

// --- Test Suite ---
func testLongRunningFunctionFlow[Out any](t *testing.T, increaseByOne func(ctx context.Context, x IncArgs) Out, resultKey string, callCount *int) {
	// 1. Setup
	responses := []*genai.Content{
		genai.NewContentFromFunctionCall("increaseByOne", map[string]any{}, "model"),
		genai.NewContentFromText("response1", "model"),
		genai.NewContentFromText("response2", "model"),
		genai.NewContentFromText("response3", "model"),
		genai.NewContentFromText("response4", "model"),
	}
	mockModel := &testutil.MockModel{Responses: responses}

	longRunningTool, err := tool.NewLongRunningFunctionTool(tool.FunctionToolConfig{
		Name:        "increaseByOne",
		Description: "increaseByOne",
	}, increaseByOne)
	if err != nil {
		t.Fatalf("failed to create longRunningTool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:  "long_running_agent",
		Model: mockModel,
		Tools: []tool.Tool{longRunningTool},
	})
	if err != nil {
		t.Fatalf("failed to create llm agent: %v", err)
	}
	runner := testutil.NewTestAgentRunner(t, a)

	// 2. Initial Run
	eventStream := runner.Run(t, "test_session", "test1")
	eventParts, err := testutil.CollectParts(eventStream)
	if err != nil {
		t.Fatalf("failed to collect events: %v", err)
	}

	// 3. Assertions for Initial Run
	if len(mockModel.Requests) != 2 {
		t.Errorf("got %d requests, want 2", len(mockModel.Requests))
	}
	if *callCount != 1 {
		t.Errorf("function called %d times, want 1", *callCount)
	}

	// Assert first request
	expectedReqText1 := "test1"
	if len(mockModel.Requests[0].Contents) != 1 {
		t.Errorf("got %d requests content, want 1", len(mockModel.Requests[0].Contents))
	}
	if len(mockModel.Requests[0].Contents[0].Parts) != 1 {
		t.Errorf("got %d requests content parts, want 1", len(mockModel.Requests[0].Contents[0].Parts))
	}
	if mockModel.Requests[0].Contents[0].Parts[0].Text != expectedReqText1 {
		t.Errorf("request 1 mismatch:\ngot: %#v\nwant: %#v", mockModel.Requests[0].Contents[0].Parts[0].Text, expectedReqText1)
	}

	// Assert second request
	if len(mockModel.Requests[1].Contents) != 3 {
		t.Errorf("got %d requests content, want 3", len(mockModel.Requests[1].Contents))
	}
	if len(mockModel.Requests[1].Contents[2].Parts) != 1 {
		t.Errorf("got %d requests content parts, want 1", len(mockModel.Requests[1].Contents[2].Parts))
	}
	functionResponse := mockModel.Requests[1].Contents[2].Parts[0].FunctionResponse
	if functionResponse == nil {
		t.Fatalf("request 2 mismatch:\ngot: nil function response ")
	}
	if functionResponse.Name != "increaseByOne" {
		t.Errorf("got %q function response name, want increaseByOne", functionResponse.Name)
	}
	if functionResponse.Response == nil {
		t.Errorf("got %q function response, want map", functionResponse.Response)
	}
	if val, ok := functionResponse.Response[resultKey]; !ok || val != "pending" {
		t.Errorf("function response, incorrect %q value", resultKey)
	}

	// Assert parts
	if len(eventParts) != 3 {
		t.Fatalf("got %d events parts, want 3", len(eventParts))
	}
	functionCallEventPart := eventParts[0]
	functionResponseEventPart := eventParts[1]
	llmResponseEventPart := eventParts[2]
	if functionCallEventPart.FunctionCall.Name != "increaseByOne" || len(functionCallEventPart.FunctionCall.Args) != 0 {
		t.Errorf("Invalid functionCallEventPart")
	}
	if functionResponseEventPart.FunctionResponse.Name != "increaseByOne" {
		t.Errorf("Invalid functionResponseEventPart")
	}
	if val, ok := functionResponseEventPart.FunctionResponse.Response[resultKey]; !ok || val != "pending" {
		t.Errorf("Invalid functionResponseEventPart")
	}
	if llmResponseEventPart.Text != "response1" {
		t.Errorf("Invalid llmResponseEventPart")
	}
	idFromTheFunctionCallEvent := functionCallEventPart.FunctionCall.ID

	testCases := []struct {
		name              string         // Name for the Run subtest
		inputContent      *genai.Content // The content to send
		wantReqCount      int            // Expected len(mockModel.Requests)
		wantEventCount    int            // Expected len(eventParts)
		wantEventText     string         // Expected eventParts[0].Text
		wantResponseKey   string         // Expected key in fuction response
		wantResponseValue any            // Expected value in fuction response
	}{
		{
			name: "function response still waiting",
			inputContent: NewContentFromFunctionResponseWithID(
				"increaseByOne", map[string]any{"status": "still waiting"}, idFromTheFunctionCallEvent, "user",
			),
			wantReqCount:      3,
			wantEventCount:    1,
			wantEventText:     "response2",
			wantResponseKey:   "status",
			wantResponseValue: "still waiting",
		},
		{
			name: "function response result 2",
			inputContent: NewContentFromFunctionResponseWithID(
				"increaseByOne", map[string]any{"result": 2}, idFromTheFunctionCallEvent, "user",
			),
			wantReqCount:      4,
			wantEventCount:    1,
			wantEventText:     "response3",
			wantResponseKey:   "result",
			wantResponseValue: 2,
		},
		{
			name: "function response result 3",
			inputContent: NewContentFromFunctionResponseWithID(
				"increaseByOne", map[string]any{"result": 3}, idFromTheFunctionCallEvent, "user",
			),
			wantReqCount:      5,
			wantEventCount:    1,
			wantEventText:     "response4",
			wantResponseKey:   "result",
			wantResponseValue: 3,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			eventStream := runner.RunContent(t, "test_session", tc.inputContent)
			eventParts, err := testutil.CollectParts(eventStream)
			if err != nil {
				t.Fatalf("failed to collect events: %v", err)
			}

			// Assert against the values from the test case struct
			if len(mockModel.Requests) != tc.wantReqCount {
				t.Fatalf("got %d requests, want %d", len(mockModel.Requests), tc.wantReqCount)
			}
			latestRequestContents := mockModel.Requests[len(mockModel.Requests)-1].Contents
			// content should still be 3 since the function responses are merged into one in contents_processor
			if len(latestRequestContents) != 3 {
				t.Fatalf("got %d latest request contents size, want %d", len(latestRequestContents), 3)
			}
			latestRequestFunctionResponse := latestRequestContents[len(latestRequestContents)-1].Parts[0].FunctionResponse
			if latestRequestFunctionResponse.Name != "increaseByOne" {
				t.Errorf("got %q latestRequest Function response name want %q",
					latestRequestFunctionResponse.Name, "increaseByOne")
			}
			val, ok := latestRequestFunctionResponse.Response[tc.wantResponseKey]
			if !ok {
				t.Fatalf("Function response map missing expected key: %q", tc.wantResponseKey)
			}
			if val != tc.wantResponseValue {
				t.Errorf("Function response value mismatch for key %q:\n  got: %#v\n want: %#v",
					tc.wantResponseKey, val, tc.wantResponseValue)
			}
			if len(eventParts) != tc.wantEventCount {
				t.Fatalf("got %d events parts, want %d", len(eventParts), tc.wantEventCount)
			}
			// This check is now safe because the Fatalf above would have stopped the test
			if len(eventParts) > 0 && eventParts[0].Text != tc.wantEventText {
				t.Fatalf("got event part text %q, want %q", eventParts[0].Text, tc.wantEventText)
			}
		})
	}

	// Should still be one
	if *callCount != 1 {
		t.Errorf("function called %d times, want 1", *callCount)
	}
}

func TestLongRunningToolIDsAreSet(t *testing.T) {
	// 1. Setup
	responses := []*genai.Content{
		genai.NewContentFromFunctionCall("increaseByOne", map[string]any{}, "model"),
		genai.NewContentFromText("response1", "model"),
	}
	mockModel := &testutil.MockModel{Responses: responses}
	functionCalled := 0

	type IncArgs struct {
	}

	increaseByOne := func(ctx context.Context, x IncArgs) map[string]string {
		functionCalled++
		return map[string]string{"status": "pending"}
	}

	longRunningTool, err := tool.NewLongRunningFunctionTool(tool.FunctionToolConfig{
		Name:        "increaseByOne",
		Description: "increaseByOne",
	}, increaseByOne)
	if err != nil {
		t.Fatalf("failed to create longRunningTool: %v", err)
	}

	a, err := llmagent.New(llmagent.Config{
		Name:  "hello_world_agent",
		Model: mockModel,
		Tools: []tool.Tool{longRunningTool},
	})
	if err != nil {
		t.Fatalf("failed to create llm agent: %v", err)
	}
	runner := testutil.NewTestAgentRunner(t, a)

	// 2. Initial Run
	eventStream := runner.Run(t, "test_session", "test1")
	events, err := testutil.CollectEvents(eventStream)
	if err != nil {
		t.Fatalf("failed to collect events: %v", err)
	}

	if len(events) != 3 { // first event is function call, seconds is function response, third is llm message back
		t.Errorf("got %d events, want 3", len(events))
	}

	// Assert responses
	functionCallEvent := events[0]
	functionResponseEvent := events[1]
	llmResponseEvent := events[2]
	// First event should have LongRunningToolIDs field
	if functionCallEvent.LongRunningToolIDs == nil || len(functionCallEvent.LongRunningToolIDs) != 1 {
		t.Fatalf("Invalid LongRunningToolIDs for functionCallEvent")
	}
	if functionResponseEvent.LongRunningToolIDs != nil {
		t.Errorf("Invalid LongRunningToolIDs for functionResponseEvent")
	}
	if len(llmResponseEvent.LongRunningToolIDs) != 0 {
		t.Errorf("Invalid LongRunningToolIDs for llmResponseEvent")
	}
	if functionCallEvent.LongRunningToolIDs[0] != functionCallEvent.LLMResponse.Content.Parts[0].FunctionCall.ID {
		t.Fatalf("Invalid LongRunningToolIDs for functionCallEvent got %q expected %q",
			functionCallEvent.LongRunningToolIDs[0],
			functionCallEvent.LLMResponse.Content.Parts[0].FunctionCall.ID)
	}
}
