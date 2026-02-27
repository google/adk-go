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

package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// TestToolCallIDPreservation tests that tool_call_id is correctly preserved
// between requests and responses (Fix #1)
func TestToolCallIDPreservation(t *testing.T) {
	// Mock server that returns tool calls with specific IDs
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++

		if callCount == 1 {
			// First call: model returns tool calls
			resp := ChatCompletionResponse{
				ID:      "test-1",
				Object:  "chat.completion",
				Created: time.Now().Unix(),
				Model:   "test-model",
				Choices: []Choice{{
					Message: OpenAIMessage{
						Role: "assistant",
						ToolCalls: []ToolCall{
							{
								ID:   "call_abc123",
								Type: "function",
								Function: FunctionCall{
									Name:      "get_weather",
									Arguments: `{"location":"SF"}`,
								},
							},
							{
								ID:   "call_xyz789",
								Type: "function",
								Function: FunctionCall{
									Name:      "get_time",
									Arguments: `{"timezone":"PST"}`,
								},
							},
						},
					},
					FinishReason: "tool_calls",
				}},
				Usage: Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
			}
			json.NewEncoder(w).Encode(resp)
		} else {
			// Second call: verify that tool response has matching IDs
			var req ChatCompletionRequest
			json.NewDecoder(r.Body).Decode(&req)

			// Check that tool messages have correct tool_call_id
			toolMsgCount := 0
			for _, msg := range req.Messages {
				if msg.Role == "tool" {
					toolMsgCount++
					if msg.ToolCallID != "call_abc123" && msg.ToolCallID != "call_xyz789" {
						t.Errorf("Invalid tool_call_id: %s", msg.ToolCallID)
					}
				}
			}

			if toolMsgCount != 2 {
				t.Errorf("Expected 2 tool messages, got %d", toolMsgCount)
			}

			// Return final response
			resp := ChatCompletionResponse{
				ID:      "test-2",
				Object:  "chat.completion",
				Created: time.Now().Unix(),
				Model:   "test-model",
				Choices: []Choice{{
					Message: OpenAIMessage{
						Role:    "assistant",
						Content: "Weather is 72°F, time is 3PM",
					},
					FinishReason: "stop",
				}},
				Usage: Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 30},
			}
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer server.Close()

	cfg := &Config{BaseURL: server.URL}
	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatal(err)
	}

	// First request - get tool calls
	req1 := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What's the weather and time?", "user"),
		},
		Tools: map[string]any{
			"get_weather": map[string]any{"description": "Get weather"},
			"get_time":    map[string]any{"description": "Get time"},
		},
	}

	ctx := context.Background()
	var toolCallIDs []string

	for resp, err := range m.GenerateContent(ctx, req1, false) {
		if err != nil {
			t.Fatal(err)
		}

		// Verify IDs are preserved in response
		for _, part := range resp.Content.Parts {
			if part.FunctionCall != nil {
				toolCallIDs = append(toolCallIDs, part.FunctionCall.ID)
				t.Logf("✓ Received function call with ID: %s", part.FunctionCall.ID)
			}
		}
	}

	if len(toolCallIDs) != 2 {
		t.Fatalf("Expected 2 tool calls, got %d", len(toolCallIDs))
	}

	// Verify IDs match what server sent
	if toolCallIDs[0] != "call_abc123" || toolCallIDs[1] != "call_xyz789" {
		t.Errorf("Tool call IDs not preserved: got %v", toolCallIDs)
	} else {
		t.Log("✓ Tool call IDs preserved correctly in response")
	}

	// Second request - send tool responses with correct IDs
	req2 := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What's the weather and time?", "user"),
			{
				Role: "model",
				Parts: []*genai.Part{
					{FunctionCall: &genai.FunctionCall{
						ID:   "call_abc123",
						Name: "get_weather",
						Args: map[string]any{"location": "SF"},
					}},
					{FunctionCall: &genai.FunctionCall{
						ID:   "call_xyz789",
						Name: "get_time",
						Args: map[string]any{"timezone": "PST"},
					}},
				},
			},
			{
				Role: "function",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "call_abc123",
						Name:     "get_weather",
						Response: map[string]any{"temp": "72F"},
					}},
				},
			},
			{
				Role: "function",
				Parts: []*genai.Part{
					{FunctionResponse: &genai.FunctionResponse{
						ID:       "call_xyz789",
						Name:     "get_time",
						Response: map[string]any{"time": "3PM"},
					}},
				},
			},
		},
	}

	for resp, err := range m.GenerateContent(ctx, req2, false) {
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("✓ Second request succeeded with tool responses")
		_ = resp
	}

	t.Log("✓ Tool call ID preservation test PASSED")
}

// TestParallelToolCallsInStreaming tests that parallel tool calls are correctly
// merged using index field in streaming mode (Fix #2)
func TestParallelToolCallsInStreaming(t *testing.T) {
	om := &openaiModel{}

	// Simulate streaming chunks with parallel tool calls
	var aggregated []ToolCall

	// First chunk: start of tool call 0
	aggregated = om.mergeToolCall(aggregated, ToolCall{
		Index: 0,
		ID:    "call_1",
		Type:  "function",
		Function: FunctionCall{
			Name:      "weather",
			Arguments: `{"loc`,
		},
	})

	// Second chunk: start of tool call 1
	aggregated = om.mergeToolCall(aggregated, ToolCall{
		Index: 1,
		ID:    "call_2",
		Type:  "function",
		Function: FunctionCall{
			Name:      "time",
			Arguments: `{"tz`,
		},
	})

	// Third chunk: continue tool call 0
	aggregated = om.mergeToolCall(aggregated, ToolCall{
		Index: 0,
		Function: FunctionCall{
			Arguments: `ation":"SF"}`,
		},
	})

	// Fourth chunk: continue tool call 1
	aggregated = om.mergeToolCall(aggregated, ToolCall{
		Index: 1,
		Function: FunctionCall{
			Arguments: `":"PST"}`,
		},
	})

	// Verify results
	if len(aggregated) != 2 {
		t.Fatalf("Expected 2 tool calls, got %d", len(aggregated))
	}

	if aggregated[0].Function.Arguments != `{"location":"SF"}` {
		t.Errorf("Tool call 0 args incorrect: %s", aggregated[0].Function.Arguments)
	} else {
		t.Log("✓ Tool call 0 merged correctly")
	}

	if aggregated[1].Function.Arguments != `{"tz":"PST"}` {
		t.Errorf("Tool call 1 args incorrect: %s", aggregated[1].Function.Arguments)
	} else {
		t.Log("✓ Tool call 1 merged correctly")
	}

	t.Log("✓ Parallel tool calls streaming test PASSED")
}

// TestMessageSequenceValidation tests that invalid message sequences are rejected (Fix #4)
func TestMessageSequenceValidation(t *testing.T) {
	tests := []struct {
		name      string
		messages  []OpenAIMessage
		shouldErr bool
	}{
		{
			name: "valid sequence with tool calls and responses",
			messages: []OpenAIMessage{
				{Role: "user", Content: "hello"},
				{Role: "assistant", ToolCalls: []ToolCall{
					{ID: "call_1", Type: "function", Function: FunctionCall{Name: "test", Arguments: "{}"}},
				}},
				{Role: "tool", ToolCallID: "call_1", Content: "result"},
				{Role: "assistant", Content: "done"},
			},
			shouldErr: false,
		},
		{
			name: "invalid: user message between tool call and response",
			messages: []OpenAIMessage{
				{Role: "assistant", ToolCalls: []ToolCall{
					{ID: "call_1", Type: "function", Function: FunctionCall{Name: "test", Arguments: "{}"}},
				}},
				{Role: "user", Content: "wait!"}, // ❌ Invalid!
				{Role: "tool", ToolCallID: "call_1", Content: "result"},
			},
			shouldErr: true,
		},
		{
			name: "invalid: unresolved tool calls at end",
			messages: []OpenAIMessage{
				{Role: "user", Content: "hello"},
				{Role: "assistant", ToolCalls: []ToolCall{
					{ID: "call_1", Type: "function", Function: FunctionCall{Name: "test", Arguments: "{}"}},
				}},
				// Missing tool response!
			},
			shouldErr: true,
		},
		{
			name: "valid: multiple parallel tool calls with all responses",
			messages: []OpenAIMessage{
				{Role: "assistant", ToolCalls: []ToolCall{
					{ID: "call_1", Type: "function", Function: FunctionCall{Name: "test1", Arguments: "{}"}},
					{ID: "call_2", Type: "function", Function: FunctionCall{Name: "test2", Arguments: "{}"}},
				}},
				{Role: "tool", ToolCallID: "call_1", Content: "result1"},
				{Role: "tool", ToolCallID: "call_2", Content: "result2"},
			},
			shouldErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateMessageSequence(tt.messages)
			if tt.shouldErr && err == nil {
				t.Errorf("Expected error but got none")
			} else if !tt.shouldErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			} else if err == nil {
				t.Logf("✓ Valid sequence accepted")
			} else {
				t.Logf("✓ Invalid sequence rejected: %v", err)
			}
		})
	}

	t.Log("✓ Message sequence validation test PASSED")
}
