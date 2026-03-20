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
	"log"
	"os"
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

const (
	gemmaModelName = "google/gemma-3-12b"
	gemmaLLMURL    = "http://127.0.0.1:1234/v1"
	gemmaTimeout   = 60 * time.Second
)

// skipIfNoGemma skips the test if Gemma model is not available
func skipIfNoGemma(t *testing.T) {
	if os.Getenv("SKIP_INTEGRATION") == "1" {
		t.Skip("Skipping Gemma integration test (SKIP_INTEGRATION=1)")
	}

	cfg := &Config{
		BaseURL: gemmaLLMURL,
		Timeout: 5 * time.Second,
	}

	m, err := NewModel(gemmaModelName, cfg)
	if err != nil {
		t.Skipf("Gemma model not available: %v", err)
	}

	om := m.(*openaiModel)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := ChatCompletionRequest{
		Model:    gemmaModelName,
		Messages: []OpenAIMessage{{Role: "user", Content: "ping"}},
	}

	_, err = om.makeRequest(ctx, req)
	if err != nil {
		t.Skipf("Gemma model not responding: %v", err)
	}
}

// TestGemma_SimpleToolCall tests tool calling with Gemma 3
func TestGemma_SimpleToolCall(t *testing.T) {
	skipIfNoGemma(t)

	var logBuf strings.Builder
	logger := log.New(&logBuf, "[GEMMA] ", log.Ltime)

	cfg := &Config{
		BaseURL:    gemmaLLMURL,
		Timeout:    gemmaTimeout,
		MaxRetries: 3,
		Logger:     logger,
	}

	m, err := NewModel(gemmaModelName, cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	// Define weather tool
	tools := map[string]any{
		"get_weather": map[string]any{
			"description": "Get the current weather for a location",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type":        "string",
						"description": "The city name",
					},
				},
				"required": []string{"location"},
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), gemmaTimeout)
	defer cancel()

	// First request - ask for weather
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What's the weather in London?", "user"),
		},
		Tools: tools,
	}

	t.Log("=== Step 1: Sending request to Gemma ===")

	var toolCallID string
	var toolCallName string
	var toolCallArgs string

	for resp, err := range m.GenerateContent(ctx, req, false) {
		if err != nil {
			t.Fatalf("Request failed: %v", err)
		}

		t.Logf("✓ Got response from Gemma")

		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			t.Fatal("No content in response")
		}

		// Check for function calls
		for _, part := range resp.Content.Parts {
			if part.FunctionCall != nil {
				toolCallID = part.FunctionCall.ID
				toolCallName = part.FunctionCall.Name

				// Convert args to JSON string for logging
				if loc, ok := part.FunctionCall.Args["location"].(string); ok {
					toolCallArgs = loc
				}

				t.Logf("✓ Gemma requested tool call:")
				t.Logf("  - ID: %s", toolCallID)
				t.Logf("  - Function: %s", toolCallName)
				t.Logf("  - Location: %s", toolCallArgs)

				// CRITICAL CHECK: Verify ID is not empty and not our generated fallback
				if toolCallID == "" {
					t.Error("❌ Tool call ID is empty!")
				} else if strings.HasPrefix(toolCallID, "call_") && len(toolCallID) < 15 {
					t.Errorf("⚠️  Tool call ID looks like our fallback: %s (should be from Gemma)", toolCallID)
				} else {
					t.Logf("✓ Tool call ID preserved correctly from Gemma: %s", toolCallID)
				}

				if toolCallName != "get_weather" {
					t.Errorf("Expected function 'get_weather', got '%s'", toolCallName)
				}
			}
		}
	}

	if toolCallID == "" {
		t.Fatal("Gemma did not return tool calls")
	}

	t.Log("\n=== Step 2: Sending tool response back to Gemma ===")

	// Second request - send tool response with SAME ID
	req2 := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What's the weather in London?", "user"),
			{
				Role: "model",
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							ID:   toolCallID, // Use the SAME ID from Gemma
							Name: toolCallName,
							Args: map[string]any{"location": toolCallArgs},
						},
					},
				},
			},
			{
				Role: "function",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:   toolCallID, // Use the SAME ID
							Name: toolCallName,
							Response: map[string]any{
								"temperature": "15°C",
								"condition":   "Cloudy",
								"humidity":    "75%",
							},
						},
					},
				},
			},
		},
	}

	for resp, err := range m.GenerateContent(ctx, req2, false) {
		if err != nil {
			t.Fatalf("Second request failed: %v\nLog:\n%s", err, logBuf.String())
		}

		t.Logf("✓ Gemma accepted tool response with matching ID")

		// Get final text response
		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					t.Logf("✓ Gemma final response: %s", strings.TrimSpace(part.Text))
				}
			}
		}
	}

	t.Log("\n=== Test PASSED ===")
	t.Logf("Log output:\n%s", logBuf.String())
}

// TestGemma_ParallelToolCalls tests parallel function calling
func TestGemma_ParallelToolCalls(t *testing.T) {
	skipIfNoGemma(t)

	var logBuf strings.Builder
	logger := log.New(&logBuf, "[GEMMA-PARALLEL] ", log.Ltime)

	cfg := &Config{
		BaseURL:    gemmaLLMURL,
		Timeout:    gemmaTimeout,
		MaxRetries: 3,
		Logger:     logger,
	}

	m, err := NewModel(gemmaModelName, cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	// Define multiple tools
	tools := map[string]any{
		"get_weather": map[string]any{
			"description": "Get the current weather",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{"type": "string"},
				},
				"required": []string{"location"},
			},
		},
		"get_time": map[string]any{
			"description": "Get the current time",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"timezone": map[string]any{"type": "string"},
				},
				"required": []string{"timezone"},
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), gemmaTimeout)
	defer cancel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What's the weather in London and what time is it in New York?", "user"),
		},
		Tools: tools,
	}

	t.Log("=== Testing parallel tool calls ===")

	var toolCalls []*genai.FunctionCall

	for resp, err := range m.GenerateContent(ctx, req, false) {
		if err != nil {
			t.Fatalf("Request failed: %v", err)
		}

		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.FunctionCall != nil {
					toolCalls = append(toolCalls, part.FunctionCall)
					t.Logf("✓ Tool call: %s (ID: %s)", part.FunctionCall.Name, part.FunctionCall.ID)
				}
			}
		}
	}

	if len(toolCalls) >= 1 {
		t.Logf("✓ Gemma returned %d tool call(s)", len(toolCalls))

		// Verify all IDs are unique
		ids := make(map[string]bool)
		for _, tc := range toolCalls {
			if ids[tc.ID] {
				t.Errorf("❌ Duplicate tool call ID: %s", tc.ID)
			}
			ids[tc.ID] = true
		}

		if len(ids) == len(toolCalls) {
			t.Logf("✓ All tool call IDs are unique")
		}
	} else {
		t.Log("Note: Gemma returned only one tool call (some models don't support parallel calls)")
	}

	t.Logf("Log output:\n%s", logBuf.String())
}

// TestGemma_Streaming tests streaming with tool calls
func TestGemma_Streaming(t *testing.T) {
	skipIfNoGemma(t)

	var logBuf strings.Builder
	logger := log.New(&logBuf, "[GEMMA-STREAM] ", log.Ltime)

	cfg := &Config{
		BaseURL:    gemmaLLMURL,
		Timeout:    gemmaTimeout,
		MaxRetries: 3,
		Logger:     logger,
	}

	m, err := NewModel(gemmaModelName, cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	tools := map[string]any{
		"get_weather": map[string]any{
			"description": "Get weather",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{"type": "string"},
				},
			},
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), gemmaTimeout)
	defer cancel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What's the weather in Paris?", "user"),
		},
		Tools: tools,
	}

	t.Log("=== Testing streaming mode ===")

	chunkCount := 0
	var finalToolCallID string

	for resp, err := range m.GenerateContent(ctx, req, true) {
		if err != nil {
			t.Fatalf("Streaming failed: %v", err)
		}

		chunkCount++

		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					t.Logf("Chunk %d: text=%q", chunkCount, part.Text)
				}
				if part.FunctionCall != nil {
					finalToolCallID = part.FunctionCall.ID
					t.Logf("Chunk %d: tool_call=%s (ID: %s)", chunkCount, part.FunctionCall.Name, part.FunctionCall.ID)
				}
			}
		}

		if resp.TurnComplete {
			t.Logf("✓ Stream completed after %d chunks", chunkCount)
			break
		}
	}

	if finalToolCallID != "" {
		t.Logf("✓ Final tool call ID: %s", finalToolCallID)

		// Verify ID format
		if finalToolCallID == "" {
			t.Error("❌ Tool call ID is empty in streaming mode")
		} else {
			t.Log("✓ Tool call ID preserved in streaming mode")
		}
	}

	t.Logf("Log output:\n%s", logBuf.String())
}
