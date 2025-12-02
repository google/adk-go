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

package functiontool_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/internal/toolinternal"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

func TestNewLongRunningFunctionTool(t *testing.T) {
	type SumArgs struct {
		A int `json:"a"` // an integer to sum
		B int `json:"b"` // another integer to sum
	}
	type SumResult struct {
		Result string `json:"result"` // the operation result
	}

	handler := func(ctx tool.Context, input SumArgs) (SumResult, error) {
		return SumResult{Result: "Processing sum"}, nil
	}
	sumTool, err := functiontool.New(functiontool.Config{
		Name:          "sum",
		Description:   "sums two integers",
		IsLongRunning: true,
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

func NewContentFromFunctionResponseWithID(name string, response map[string]any, id, role string) *genai.Content {
	content := genai.NewContentFromFunctionResponse(name, response, genai.Role(role))
	content.Parts[0].FunctionResponse.ID = id
	return content
}

type IncArgs struct{}

func TestLongRunningFunctionFlow(t *testing.T) {
	functionCalled := 0
	increaseByOne := func(ctx tool.Context, x IncArgs) (map[string]string, error) {
		functionCalled++
		return map[string]string{"status": "pending"}, nil
	}
	testLongRunningFunctionFlow(t, increaseByOne, "status", &functionCalled)
}

func TestLongRunningStringFunctionFlow(t *testing.T) {
	functionCalled := 0
	increaseByOne := func(ctx tool.Context, x IncArgs) (string, error) {
		functionCalled++
		return "pending", nil
	}
	testLongRunningFunctionFlow(t, increaseByOne, "result", &functionCalled)
}

// --- Test Suite ---
func testLongRunningFunctionFlow[Out any](t *testing.T, increaseByOne func(ctx tool.Context, x IncArgs) (Out, error), resultKey string, callCount *int) {
	// 1. Setup
	responses := []*genai.Content{
		genai.NewContentFromFunctionCall("increaseByOne", map[string]any{}, "model"),
		genai.NewContentFromText("response1", "model"),
		genai.NewContentFromText("response2", "model"),
		genai.NewContentFromText("response3", "model"),
		genai.NewContentFromText("response4", "model"),
	}
	mockModel := &testutil.MockModel{Responses: responses}

	longRunningTool, err := functiontool.New(functiontool.Config{
		Name:          "increaseByOne",
		Description:   "increaseByOne",
		IsLongRunning: true,
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
		// Marshal the slice into a readable JSON string
		requestsJSON, _ := json.MarshalIndent(mockModel.Requests, "", "  ")
		t.Fatalf("got %d requests, want 2;\n- requests:\n%s", len(mockModel.Requests), requestsJSON)
	}
	if *callCount != 1 {
		t.Errorf("function called %d times, want 1", *callCount)
	}

	// Assert first request
	wantFirsteq := []*genai.Content{
		genai.NewContentFromText("test1", "user"),
	}
	if diff := cmp.Diff(wantFirsteq, mockModel.Requests[0].Contents); diff != "" {
		t.Errorf("LLMRequest.Contents mismatch (-want +got):\n%s", diff)
	}

	// Assert second request
	wantSecondReq := []*genai.Content{
		genai.NewContentFromText("test1", "user"),
		genai.NewContentFromFunctionCall("increaseByOne", map[string]any{}, "model"),
		genai.NewContentFromFunctionResponse("increaseByOne", map[string]any{resultKey: "pending"}, "user"),
	}
	if diff := cmp.Diff(wantSecondReq, mockModel.Requests[1].Contents); diff != "" {
		t.Errorf("LLMRequest.Contents mismatch (-want +got):\n%s", diff)
	}

	wantEventParts := []*genai.Part{
		genai.NewPartFromFunctionCall("increaseByOne", map[string]any{}),
		genai.NewPartFromFunctionResponse("increaseByOne", map[string]any{resultKey: "pending"}),
		genai.NewPartFromText("response1"),
	}
	if diff := cmp.Diff(wantEventParts, eventParts, cmpopts.IgnoreFields(genai.FunctionCall{}, "ID"),
		cmpopts.IgnoreFields(genai.FunctionResponse{}, "ID")); diff != "" {
		t.Errorf("Event parts mismatch (-want +got):\n%s", diff)
	}

	functionCallEventPart := eventParts[0]
	idFromTheFunctionCallEvent := functionCallEventPart.FunctionCall.ID

	testCases := []struct {
		name           string         // Name for the Run subtest
		inputContent   *genai.Content // The content to send
		wantReqCount   int            // Expected len(mockModel.Requests)
		wantEventCount int            // Expected len(eventParts)
		wantEventText  string         // Expected eventParts[0].Text
		wantContent    *genai.Content // Expected output content
	}{
		{
			name: "function response still waiting",
			inputContent: NewContentFromFunctionResponseWithID(
				"increaseByOne", map[string]any{"status": "still waiting"}, idFromTheFunctionCallEvent, "user",
			),
			wantReqCount:   3,
			wantEventCount: 1,
			wantEventText:  "response2",
			wantContent:    genai.NewContentFromFunctionResponse("increaseByOne", map[string]any{"status": "still waiting"}, "user"),
		},
		{
			name: "function response result 2",
			inputContent: NewContentFromFunctionResponseWithID(
				"increaseByOne", map[string]any{"result": 2}, idFromTheFunctionCallEvent, "user",
			),
			wantReqCount:   4,
			wantEventCount: 1,
			wantEventText:  "response3",
			wantContent:    genai.NewContentFromFunctionResponse("increaseByOne", map[string]any{"result": 2}, "user"),
		},
		{
			name: "function response result 3",
			inputContent: NewContentFromFunctionResponseWithID(
				"increaseByOne", map[string]any{"result": 3}, idFromTheFunctionCallEvent, "user",
			),
			wantReqCount:   5,
			wantEventCount: 1,
			wantEventText:  "response4",
			wantContent:    genai.NewContentFromFunctionResponse("increaseByOne", map[string]any{"result": 3}, "user"),
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

			if diff := cmp.Diff(tc.wantContent, latestRequestContents[len(latestRequestContents)-1]); diff != "" {
				t.Errorf("LLMRequest.Content mismatch (-want +got):\n%s", diff)
			}

			if len(eventParts) != tc.wantEventCount {
				// Marshal the slice into a readable JSON string
				partsJSON, _ := json.MarshalIndent(eventParts, "", "  ")
				t.Fatalf("got %d events parts, want %d;\n- parts:\n%s", len(eventParts), tc.wantEventCount, partsJSON)
			}
			// This check is now safe because the Fatalf above would have stopped the test
			if len(eventParts) > 0 && eventParts[0].Text != tc.wantEventText {
				t.Errorf("got event part text %q, want %q", eventParts[0].Text, tc.wantEventText)
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

	type IncArgs struct{}

	increaseByOne := func(ctx tool.Context, x IncArgs) (map[string]string, error) {
		functionCalled++
		return map[string]string{"status": "pending"}, nil
	}

	longRunningTool, err := functiontool.New(functiontool.Config{
		Name:          "increaseByOne",
		Description:   "increaseByOne",
		IsLongRunning: true,
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
		// Marshal the slice into a readable JSON string
		eventsJSON, _ := json.MarshalIndent(events, "", "  ")
		t.Fatalf("got %d for events length, want 3;\n- events:\n%s", len(events), eventsJSON)
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
