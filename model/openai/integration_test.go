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
	"fmt"
	"log"
	"os"
	"strings"
	"testing"
	"time"

	"google.golang.org/adk/model"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

// Integration tests with real LLM (LM Studio/Ollama)
// These tests require a local LLM server running at http://127.0.0.1:1234

const (
	localLLMURL   = "http://127.0.0.1:1234"
	testModelName = "mistralai/mistral-7b-instruct-v0.3" // Available model
	testTimeout   = 60 * time.Second
)

// skipIfNoLocalLLM skips the test if local LLM is not available
func skipIfNoLocalLLM(t *testing.T) {
	if os.Getenv("SKIP_INTEGRATION") == "1" {
		t.Skip("Skipping integration test (SKIP_INTEGRATION=1)")
	}

	// Quick health check
	cfg := &Config{
		BaseURL: localLLMURL,
		Timeout: 5 * time.Second,
	}

	m, err := NewModel(testModelName, cfg)
	if err != nil {
		t.Skipf("Local LLM not available: %v", err)
	}

	// Try a simple request via openaiModel
	om := m.(*openaiModel)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := ChatCompletionRequest{
		Model:    testModelName,
		Messages: []OpenAIMessage{{Role: "user", Content: "ping"}},
	}

	_, err = om.makeRequest(ctx, req)
	if err != nil {
		t.Skipf("Local LLM not responding: %v", err)
	}
}

// TestIntegrationSimpleToolCall tests a simple tool call with real LLM
func TestIntegrationSimpleToolCall(t *testing.T) {
	skipIfNoLocalLLM(t)

	var logBuf strings.Builder
	logger := log.New(&logBuf, "[INTEGRATION] ", log.Ltime)

	// Simple weather tool
	weatherTool := &simpleTool{
		name:        "get_weather",
		description: "Get the current weather for a location",
		execFunc: func(args map[string]any) (map[string]any, error) {
			location, _ := args["location"].(string)
			return map[string]any{
				"location":    location,
				"temperature": 72,
				"condition":   "sunny",
			}, nil
		},
	}

	cfg := &Config{
		BaseURL:    localLLMURL,
		Timeout:    testTimeout,
		MaxRetries: 3,
		Logger:     logger,
	}

	m, err := NewModel(testModelName, cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	// Register tool
	tools := map[string]any{
		"get_weather": weatherTool,
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Parts: []*genai.Part{
					genai.NewPartFromText("What's the weather in San Francisco?"),
				},
				Role: "user",
			},
		},
	}

	// Convert to OpenAI format with tools
	messages, err := om.convertToOpenAIMessages(ctx, req)
	if err != nil {
		t.Fatalf("Failed to convert messages: %v", err)
	}

	// Create request with tools
	chatReq := ChatCompletionRequest{
		Model:    testModelName,
		Messages: messages,
		Tools:    convertToolsToOpenAI(tools),
	}

	// Make request
	respData, err := om.makeRequest(ctx, chatReq)
	if err != nil {
		t.Logf("Request failed (expected if model doesn't support tools): %v", err)
		t.Logf("Log output:\n%s", logBuf.String())
		// Don't fail - some models don't support function calling
		return
	}

	var response ChatCompletionResponse
	if err := json.Unmarshal(respData, &response); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	t.Logf("✓ Got response from local LLM")
	t.Logf("Model: %s", response.Model)

	if len(response.Choices) == 0 {
		t.Fatal("No choices in response")
	}

	t.Logf("Response: %+v", response.Choices[0].Message)

	if len(response.Choices[0].Message.ToolCalls) > 0 {
		t.Logf("✓ Model returned tool calls: %d", len(response.Choices[0].Message.ToolCalls))
		for i, tc := range response.Choices[0].Message.ToolCalls {
			t.Logf("  Tool call %d: %s(%s)", i+1, tc.Function.Name, tc.Function.Arguments)
		}
	} else {
		t.Logf("Note: Model did not return tool calls (may not support function calling)")
	}

	t.Logf("Log output:\n%s", logBuf.String())
}

// TestIntegrationToolCallChain tests chaining multiple tool calls
func TestIntegrationToolCallChain(t *testing.T) {
	skipIfNoLocalLLM(t)

	var logBuf strings.Builder
	logger := log.New(&logBuf, "[CHAIN] ", log.Ltime)

	// Multiple tools that can be chained
	searchTool := &simpleTool{
		name:        "search",
		description: "Search for information",
		execFunc: func(args map[string]any) (map[string]any, error) {
			query, _ := args["query"].(string)
			return map[string]any{
				"results": []string{
					fmt.Sprintf("Result 1 for %s", query),
					fmt.Sprintf("Result 2 for %s", query),
				},
			}, nil
		},
	}

	summarizeTool := &simpleTool{
		name:        "summarize",
		description: "Summarize text",
		execFunc: func(args map[string]any) (map[string]any, error) {
			text, _ := args["text"].(string)
			return map[string]any{
				"summary": fmt.Sprintf("Summary of: %s", text),
			}, nil
		},
	}

	cfg := &Config{
		BaseURL:    localLLMURL,
		Timeout:    testTimeout,
		MaxRetries: 3,
		Logger:     logger,
	}

	m, err := NewModel(testModelName, cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	tools := map[string]any{
		"search":    searchTool,
		"summarize": summarizeTool,
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Parts: []*genai.Part{
					genai.NewPartFromText("Search for 'Go programming' and summarize the results"),
				},
				Role: "user",
			},
		},
	}

	messages, err := om.convertToOpenAIMessages(ctx, req)
	if err != nil {
		t.Fatalf("Failed to convert messages: %v", err)
	}

	chatReq := ChatCompletionRequest{
		Model:    testModelName,
		Messages: messages,
		Tools:    convertToolsToOpenAI(tools),
	}

	respData, err := om.makeRequest(ctx, chatReq)
	if err != nil {
		t.Logf("Request failed: %v", err)
		t.Logf("Log output:\n%s", logBuf.String())
		return
	}

	var response ChatCompletionResponse
	if err := json.Unmarshal(respData, &response); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	t.Logf("✓ Got response for chain request")

	if len(response.Choices) == 0 {
		t.Log("No choices in response - skipping tool execution")
		return
	}

	if len(response.Choices[0].Message.ToolCalls) > 0 {
		t.Logf("✓ Model suggested %d tool calls", len(response.Choices[0].Message.ToolCalls))

		// Execute tool calls
		executor := NewToolExecutor(tools, &ToolExecutorConfig{
			ParallelExecution: false, // Sequential for chain
			Timeout:           30 * time.Second,
			MaxRetries:        2,
			Logger:            logger,
		})

		results, err := executor.ExecuteToolCalls(ctx, response.Choices[0].Message.ToolCalls, nilToolContext())
		if err != nil {
			t.Fatalf("Failed to execute tools: %v", err)
		}

		t.Logf("✓ Executed %d tools successfully", len(results))
		for i, r := range results {
			t.Logf("  Result %d (%s): %+v", i+1, r.Name, r.Response)
		}
	}

	t.Logf("Log output:\n%s", logBuf.String())
}

// TestIntegrationToolError tests handling of tool execution errors
func TestIntegrationToolError(t *testing.T) {
	skipIfNoLocalLLM(t)

	var logBuf strings.Builder
	logger := log.New(&logBuf, "[ERROR_TEST] ", log.Ltime)

	// Tool that always fails
	failingTool := &simpleTool{
		name:        "failing_tool",
		description: "A tool that fails",
		execFunc: func(args map[string]any) (map[string]any, error) {
			return nil, fmt.Errorf("intentional tool failure for testing")
		},
	}

	cfg := &Config{
		BaseURL:    localLLMURL,
		Timeout:    testTimeout,
		MaxRetries: 3,
		Logger:     logger,
	}

	m, err := NewModel(testModelName, cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	tools := map[string]any{
		"failing_tool": failingTool,
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Parts: []*genai.Part{
					genai.NewPartFromText("Use the failing_tool"),
				},
				Role: "user",
			},
		},
	}

	messages, err := om.convertToOpenAIMessages(ctx, req)
	if err != nil {
		t.Fatalf("Failed to convert messages: %v", err)
	}

	chatReq := ChatCompletionRequest{
		Model:    testModelName,
		Messages: messages,
		Tools:    convertToolsToOpenAI(tools),
	}

	respData, err := om.makeRequest(ctx, chatReq)
	if err != nil {
		t.Logf("Request failed: %v", err)
		return
	}

	var response ChatCompletionResponse
	if err := json.Unmarshal(respData, &response); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if len(response.Choices) == 0 {
		t.Log("No choices in response - skipping tool execution")
		return
	}

	if len(response.Choices[0].Message.ToolCalls) > 0 {
		executor := NewToolExecutor(tools, &ToolExecutorConfig{
			ParallelExecution: false,
			Timeout:           30 * time.Second,
			MaxRetries:        2,
			Logger:            logger,
		})

		results, err := executor.ExecuteToolCalls(ctx, response.Choices[0].Message.ToolCalls, nilToolContext())
		if err != nil {
			t.Fatalf("ExecuteToolCalls returned error: %v", err)
		}

		// Verify errors are captured in results
		for _, r := range results {
			if r.Error != nil {
				t.Logf("✓ Tool error properly captured: %v", r.Error)

				// Verify error is in response
				if errMsg, ok := r.Response["error"].(string); ok {
					t.Logf("✓ Error in response: %s", errMsg)
				}
			}
		}
	}

	t.Logf("Log output:\n%s", logBuf.String())
}

// convertToolsToOpenAI converts tools to OpenAI format
func convertToolsToOpenAI(tools map[string]any) []Tool {
	result := []Tool{}
	for _, t := range tools {
		if toolImpl, ok := t.(tool.Tool); ok {
			result = append(result, Tool{
				Type: "function",
				Function: Function{
					Name:        toolImpl.Name(),
					Description: toolImpl.Description(),
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"location": map[string]any{
								"type":        "string",
								"description": "The location",
							},
							"query": map[string]any{
								"type":        "string",
								"description": "The search query",
							},
							"text": map[string]any{
								"type":        "string",
								"description": "The text to process",
							},
						},
					},
				},
			})
		}
	}
	return result
}
