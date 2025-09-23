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

	handler := func(ctx tool.Context, input SumArgs) SumResult {
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

// --- Test Suite ---
func TestAsyncFunction(t *testing.T) {
	// 1. Setup
	responses := []*genai.Content{
		&genai.Content{Parts: []*genai.Part{
			genai.NewPartFromFunctionCall("increaseByOne", map[string]any{}),
		}},
		&genai.Content{Parts: []*genai.Part{genai.NewPartFromText("response1")}},
		&genai.Content{Parts: []*genai.Part{genai.NewPartFromText("response2")}},
		&genai.Content{Parts: []*genai.Part{genai.NewPartFromText("response3")}},
		&genai.Content{Parts: []*genai.Part{genai.NewPartFromText("response4")}},
	}
	mockModel := &testutil.MockModel{Responses: responses}
	functionCalled := 0

	type IncArgs struct {
	}

	increaseByOne := func(ctx tool.Context, x IncArgs) map[string]string {
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
	events := runner.Run(t, "test_session", "test1")
	textParts, err := testutil.CollectTextParts(events)
	if err != nil {
		t.Fatalf("failed to collect text parts: %v", err)
	}

	// 3. Assertions for Initial Run
	if len(mockModel.Requests) != 2 {
		t.Errorf("got %d requests, want 2", len(mockModel.Requests))
	}
	if functionCalled != 1 {
		t.Errorf("function called %d times, want 1", functionCalled)
	}
	if len(textParts) != 1 {
		t.Errorf("got %d text parts, want 1", len(textParts))
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

}
