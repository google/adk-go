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

package exitlooptool_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"google.golang.org/genai"

	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/agent/workflowagents/loopagent"
	"google.golang.org/adk/internal/testutil"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/exitlooptool"
)

// --- Test Suite ---
func TestExitLoopToolExitsLoopAgent(t *testing.T) {
	// Define the structure for our test cases
	testCases := []struct {
		name          string
		mockResponses []*genai.Content
		maxIterations uint
		want          []*genai.Content
	}{
		{
			name: "ExitLoopToolStopsMidLoop",
			mockResponses: []*genai.Content{
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromFunctionCall("exit_loop", map[string]any{}, "model"),
				genai.NewContentFromText("this should not be processed", "model"),
				genai.NewContentFromText("this should not be processed", "model"),
			},
			maxIterations: 5,
			want: []*genai.Content{
				// Results from first GenerateStream call
				genai.NewContentFromText("response1", "model"),
				genai.NewContentFromText("response2", "model"),
				genai.NewContentFromFunctionCall("exit_loop", map[string]any{}, "model"),
				// Result from the tool execution
				genai.NewContentFromFunctionResponse("exit_loop", map[string]any{}, "user"),
			},
		},
		{
			name: "MaxIterationsStopsLoop",
			mockResponses: []*genai.Content{
				// First iteration
				genai.NewContentFromText("iteration 1 response", "model"),
				// Second iteration
				genai.NewContentFromText("iteration 2 response", "model"),
				// This won't be reached
				genai.NewContentFromText("iteration 3 response", "model"),
			},
			maxIterations: 2,
			want: []*genai.Content{
				genai.NewContentFromText("iteration 1 response", "model"),
				genai.NewContentFromText("iteration 2 response", "model"),
			},
		},
		{
			name: "ExitLoopToolStopsImmediately",
			mockResponses: []*genai.Content{
				genai.NewContentFromFunctionCall("exit_loop", map[string]any{}, "model"),
				genai.NewContentFromText("this should not be processed", "model"),
				genai.NewContentFromText("this should not be processed", "model"),
			},
			maxIterations: 3,
			want: []*genai.Content{
				genai.NewContentFromFunctionCall("exit_loop", map[string]any{}, "model"),
				genai.NewContentFromFunctionResponse("exit_loop", map[string]any{}, "user"),
			},
		},
	}

	// Iterate over the test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 1. Setup
			mockModel := &testutil.MockModel{Responses: tc.mockResponses}
			exitLoopTool, err := exitlooptool.New()
			if err != nil {
				t.Fatalf("failed to create exit tool: %v", err)
			}

			a, err := llmagent.New(llmagent.Config{
				Name:  "simple agent",
				Model: mockModel,
				Tools: []tool.Tool{exitLoopTool},
			})
			if err != nil {
				t.Fatalf("failed to create llm agent: %v", err)
			}

			looper, err := loopagent.New(loopagent.Config{
				AgentConfig: agent.Config{
					Name:      "looper",
					SubAgents: []agent.Agent{a},
				},
				MaxIterations: tc.maxIterations,
			})
			if err != nil {
				t.Fatalf("failed to create loop agent: %v", err)
			}
			runner := testutil.NewTestAgentRunner(t, looper)

			// 2. Execution and Assertion
			eventCount := 0
			ev := runner.Run(t, "id", "message")

			for got, err := range ev {
				if err != nil {
					// Check if an error was expected
					t.Fatalf("runner returned unexpected error: %v", err)
					// If error was expected, we can stop here or check for a specific error type.
					return
				}

				if eventCount >= len(tc.want) {
					t.Fatalf("stream generated more values than the expected %d. Got: %+v", len(tc.want), got.Content)
				}

				if diff := cmp.Diff(tc.want[eventCount], got.Content, cmpopts.IgnoreFields(genai.FunctionCall{}, "ID"),
					cmpopts.IgnoreFields(genai.FunctionResponse{}, "ID")); diff != "" {
					t.Errorf("LoopAgent Run() mismatch (-want +got):\n%s", diff)
				}
				eventCount++
			}

			// Final check on the number of events
			if eventCount != len(tc.want) {
				t.Errorf("unexpected stream length, want %d got %d", len(tc.want), eventCount)
			}
		})
	}
}
