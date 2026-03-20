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
	qwenModelName = "Qwen/Qwen3.5-4B"
	qwenLLMURL    = "http://127.0.0.1:8000/v1" // vLLM default port
	qwenTimeout   = 120 * time.Second           // Qwen thinking mode can be slow
)

// skipIfNoQwen skips the test if Qwen model is not available.
func skipIfNoQwen(t *testing.T) {
	if os.Getenv("SKIP_INTEGRATION") == "1" {
		t.Skip("Skipping Qwen integration test (SKIP_INTEGRATION=1)")
	}

	url := os.Getenv("QWEN_URL")
	if url == "" {
		url = qwenLLMURL
	}

	cfg := &Config{
		BaseURL: url,
		Timeout: 5 * time.Second,
	}

	modelName := os.Getenv("QWEN_MODEL")
	if modelName == "" {
		modelName = qwenModelName
	}

	m, err := NewModel(modelName, cfg)
	if err != nil {
		t.Skipf("Qwen model not available: %v", err)
	}

	om := m.(*openaiModel)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req := ChatCompletionRequest{
		Model:    modelName,
		Messages: []OpenAIMessage{{Role: "user", Content: "ping"}},
	}

	_, err = om.makeRequest(ctx, req)
	if err != nil {
		t.Skipf("Qwen model not responding: %v", err)
	}
}

func qwenURL() string {
	if url := os.Getenv("QWEN_URL"); url != "" {
		return url
	}
	return qwenLLMURL
}

func qwenModel() string {
	if m := os.Getenv("QWEN_MODEL"); m != "" {
		return m
	}
	return qwenModelName
}

// TestQwen_SimpleToolCall tests tool calling with Qwen 3.5.
func TestQwen_SimpleToolCall(t *testing.T) {
	skipIfNoQwen(t)

	var logBuf strings.Builder
	logger := log.New(&logBuf, "[QWEN] ", log.Ltime)

	cfg := &Config{
		BaseURL:      qwenURL(),
		Timeout:      qwenTimeout,
		MaxRetries:   3,
		Logger:       logger,
		DebugLogging: true,
	}

	m, err := NewModel(qwenModel(), cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	// Define weather tool.
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

	ctx, cancel := context.WithTimeout(context.Background(), qwenTimeout)
	defer cancel()

	// First request — ask for weather.
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What's the weather in London?", "user"),
		},
		Tools: tools,
	}

	t.Log("=== Step 1: Sending request to Qwen ===")

	var toolCallID string
	var toolCallName string
	var toolCallArgs string

	for resp, err := range m.GenerateContent(ctx, req, false) {
		if err != nil {
			t.Fatalf("Request failed: %v\nLog:\n%s", err, logBuf.String())
		}

		t.Log("Got response from Qwen")

		if resp.Content == nil || len(resp.Content.Parts) == 0 {
			t.Fatal("No content in response")
		}

		// Verify no <think> blocks leaked through.
		for _, part := range resp.Content.Parts {
			if part.Text != "" && strings.Contains(part.Text, "<think>") {
				t.Error("Response contains unstripped <think> block")
			}
		}

		// Check for function calls.
		for _, part := range resp.Content.Parts {
			if part.FunctionCall != nil {
				toolCallID = part.FunctionCall.ID
				toolCallName = part.FunctionCall.Name

				if loc, ok := part.FunctionCall.Args["location"].(string); ok {
					toolCallArgs = loc
				}

				t.Logf("Qwen requested tool call:")
				t.Logf("  - ID: %s", toolCallID)
				t.Logf("  - Function: %s", toolCallName)
				t.Logf("  - Location: %s", toolCallArgs)

				if toolCallID == "" {
					t.Error("Tool call ID is empty")
				}

				if toolCallName != "get_weather" {
					t.Errorf("Expected function 'get_weather', got '%s'", toolCallName)
				}
			}
		}
	}

	if toolCallID == "" {
		t.Fatalf("Qwen did not return tool calls\nLog:\n%s", logBuf.String())
	}

	t.Log("\n=== Step 2: Sending tool response back to Qwen ===")

	// Second request — send tool response with SAME ID.
	req2 := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What's the weather in London?", "user"),
			{
				Role: "model",
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							ID:   toolCallID,
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
							ID:   toolCallID,
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

		t.Log("Qwen accepted tool response with matching ID")

		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					text := strings.TrimSpace(part.Text)
					t.Logf("Qwen final response: %s", text)
					if strings.Contains(text, "<think>") {
						t.Error("Final response contains unstripped <think> block")
					}
				}
			}
		}
	}

	t.Log("\n=== Test PASSED ===")
}

// TestQwen_ThinkingBlockStripping tests that <think> blocks are stripped.
func TestQwen_ThinkingBlockStripping(t *testing.T) {
	skipIfNoQwen(t)

	cfg := &Config{
		BaseURL: qwenURL(),
		Timeout: qwenTimeout,
	}

	m, err := NewModel(qwenModel(), cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), qwenTimeout)
	defer cancel()

	// Simple question that should trigger thinking.
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What is 2+2?", "user"),
		},
	}

	for resp, err := range m.GenerateContent(ctx, req, false) {
		if err != nil {
			t.Fatalf("Request failed: %v", err)
		}

		if resp.Content != nil {
			for _, part := range resp.Content.Parts {
				if part.Text != "" {
					if strings.Contains(part.Text, "<think>") {
						t.Errorf("Response contains <think> block: %q", part.Text[:min(len(part.Text), 200)])
					}
					t.Logf("Response (clean): %s", strings.TrimSpace(part.Text))
				}
			}
		}
	}
}

// TestQwen_Streaming tests streaming with Qwen.
func TestQwen_Streaming(t *testing.T) {
	skipIfNoQwen(t)

	var logBuf strings.Builder
	logger := log.New(&logBuf, "[QWEN-STREAM] ", log.Ltime)

	cfg := &Config{
		BaseURL:    qwenURL(),
		Timeout:    qwenTimeout,
		MaxRetries: 3,
		Logger:     logger,
	}

	m, err := NewModel(qwenModel(), cfg)
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

	ctx, cancel := context.WithTimeout(context.Background(), qwenTimeout)
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
			t.Logf("Stream completed after %d chunks", chunkCount)

			// Verify final response has no <think> blocks.
			if resp.Content != nil {
				for _, part := range resp.Content.Parts {
					if part.Text != "" && strings.Contains(part.Text, "<think>") {
						t.Error("Final streaming response contains <think> block")
					}
				}
			}
			break
		}
	}

	if finalToolCallID != "" {
		t.Logf("Final tool call ID: %s", finalToolCallID)
	}
}

// TestQwen_ParallelToolCalls tests parallel function calling with Qwen.
func TestQwen_ParallelToolCalls(t *testing.T) {
	skipIfNoQwen(t)

	var logBuf strings.Builder
	logger := log.New(&logBuf, "[QWEN-PARALLEL] ", log.Ltime)

	cfg := &Config{
		BaseURL:    qwenURL(),
		Timeout:    qwenTimeout,
		MaxRetries: 3,
		Logger:     logger,
	}

	m, err := NewModel(qwenModel(), cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

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

	ctx, cancel := context.WithTimeout(context.Background(), qwenTimeout)
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
					t.Logf("Tool call: %s (ID: %s)", part.FunctionCall.Name, part.FunctionCall.ID)
				}
			}
		}
	}

	if len(toolCalls) >= 1 {
		t.Logf("Qwen returned %d tool call(s)", len(toolCalls))

		ids := make(map[string]bool)
		for _, tc := range toolCalls {
			if ids[tc.ID] {
				t.Errorf("Duplicate tool call ID: %s", tc.ID)
			}
			ids[tc.ID] = true
		}

		if len(ids) == len(toolCalls) {
			t.Log("All tool call IDs are unique")
		}
	} else {
		t.Log("Note: Qwen returned no tool calls")
	}
}
