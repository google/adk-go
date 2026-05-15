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

//go:build lmstudio

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
	"google.golang.org/genai"
)

const (
	lmStudioURL   = "http://127.0.0.1:1234/v1"
	qwenModelName = "qwen/qwen3.5-9b"
)

func lmStudioConfig() *Config {
	return &Config{
		BaseURL:      lmStudioURL,
		DebugLogging: true,
		Logger:       log.New(os.Stderr, "[QWEN] ", log.LstdFlags),
		Timeout:      5 * time.Minute, // Qwen 3.5 with thinking needs time on local hardware
	}
}

// TestLive_Qwen35_SimpleChat tests basic chat without tools.
func TestLive_Qwen35_SimpleChat(t *testing.T) {
	m, err := NewModel(qwenModelName, lmStudioConfig())
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	maxTokens := int32(4096)
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What is 2+2? Answer in one word.", "user"),
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: maxTokens,
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("You are a helpful assistant. Be concise."),
				},
			},
		},
	}

	t.Log("=== SYNC (non-streaming) ===")
	for resp, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent error: %v", err)
		}
		logResponse(t, "SYNC", resp)

		// Verify no think tags leaked
		for _, part := range resp.Content.Parts {
			if strings.Contains(part.Text, "<think>") {
				t.Error("LEAKED: <think> tags found in sync response!")
			}
		}
	}

	t.Log("\n=== STREAMING ===")
	var partialCount int
	var finalText string
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if resp.Partial {
			partialCount++
			for _, part := range resp.Content.Parts {
				if strings.Contains(part.Text, "<think>") {
					t.Error("LEAKED: <think> tags found in streaming partial!")
				}
			}
		} else if resp.TurnComplete {
			for _, part := range resp.Content.Parts {
				finalText += part.Text
			}
			logResponse(t, "STREAM-FINAL", resp)
		}
	}
	t.Logf("Streaming: %d partials, final text: %q", partialCount, finalText)

	if strings.Contains(finalText, "<think>") {
		t.Error("LEAKED: <think> tags found in streaming final response!")
	}
}

// TestLive_Qwen35_ToolCalling tests function calling with Qwen 3.5.
func TestLive_Qwen35_ToolCalling(t *testing.T) {
	m, err := NewModel(qwenModelName, lmStudioConfig())
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	// Define tools via map (as ADK passes them)
	tools := map[string]any{
		"get_weather": map[string]any{
			"description": "Get the current weather for a specific location",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type":        "string",
						"description": "The city name, e.g. Paris, London",
					},
				},
				"required": []string{"location"},
			},
		},
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What's the weather in Tokyo?", "user"),
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("You are a helpful assistant. Use the get_weather tool when asked about weather."),
				},
			},
		},
		Tools: tools,
	}

	t.Log("=== TOOL CALLING (sync) ===")
	for resp, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent error: %v", err)
		}
		logResponse(t, "TOOL-SYNC", resp)

		// Check for function calls
		hasFuncCall := false
		for _, part := range resp.Content.Parts {
			if part.FunctionCall != nil {
				hasFuncCall = true
				t.Logf("TOOL CALL: %s(%v) [id=%s]",
					part.FunctionCall.Name,
					part.FunctionCall.Args,
					part.FunctionCall.ID)
			}
		}
		if !hasFuncCall {
			t.Log("WARNING: Model did not return tool calls (may have answered directly)")
		}
	}

	t.Log("\n=== TOOL CALLING (streaming) ===")
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if resp.TurnComplete {
			logResponse(t, "TOOL-STREAM-FINAL", resp)
			for _, part := range resp.Content.Parts {
				if part.FunctionCall != nil {
					t.Logf("STREAM TOOL CALL: %s(%v) [id=%s]",
						part.FunctionCall.Name,
						part.FunctionCall.Args,
						part.FunctionCall.ID)
				}
			}
		}
	}
}

// TestLive_Qwen35_MultiTurn tests multi-turn conversation with tool results.
func TestLive_Qwen35_MultiTurn(t *testing.T) {
	m, err := NewModel(qwenModelName, lmStudioConfig())
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	tools := map[string]any{
		"get_weather": map[string]any{
			"description": "Get the current weather for a specific location",
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

	// Turn 1: User asks about weather
	t.Log("=== TURN 1: User asks about weather ===")
	req1 := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What's the weather in Paris?", "user"),
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("You are a weather assistant. Always use get_weather tool."),
				},
			},
		},
		Tools: tools,
	}

	var turn1Response *model.LLMResponse
	for resp, err := range m.GenerateContent(context.Background(), req1, false) {
		if err != nil {
			t.Fatalf("Turn 1 error: %v", err)
		}
		turn1Response = resp
		logResponse(t, "TURN1", resp)
	}

	// Check if model returned a tool call
	var toolCallID, toolCallName string
	for _, part := range turn1Response.Content.Parts {
		if part.FunctionCall != nil {
			toolCallID = part.FunctionCall.ID
			toolCallName = part.FunctionCall.Name
			t.Logf("Turn 1 tool call: %s (id=%s)", toolCallName, toolCallID)
		}
	}

	if toolCallID == "" {
		t.Log("Model did not return tool call, skipping multi-turn")
		return
	}

	// Turn 2: Provide tool result and get final answer
	t.Log("\n=== TURN 2: Provide tool result ===")
	req2 := &model.LLMRequest{
		Contents: []*genai.Content{
			// Original user message
			genai.NewContentFromText("What's the weather in Paris?", "user"),
			// Model's tool call (from turn 1)
			turn1Response.Content,
			// Tool result
			{
				Role: "function",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:   toolCallID,
							Name: toolCallName,
							Response: map[string]any{
								"location":    "Paris",
								"temperature": "18°C",
								"condition":   "Partly cloudy",
								"humidity":    "72%",
							},
						},
					},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("You are a weather assistant. Always use get_weather tool."),
				},
			},
		},
		Tools: tools,
	}

	for resp, err := range m.GenerateContent(context.Background(), req2, false) {
		if err != nil {
			t.Fatalf("Turn 2 error: %v", err)
		}
		logResponse(t, "TURN2", resp)

		// Final answer should contain weather info and no think tags
		for _, part := range resp.Content.Parts {
			if part.Text != "" {
				if strings.Contains(part.Text, "<think>") {
					t.Error("LEAKED: <think> tags in final answer!")
				}
				if strings.Contains(strings.ToLower(part.Text), "paris") {
					t.Log("Final answer mentions Paris")
				}
			}
		}
	}
}

// TestLive_Qwen35_ThinkTagDetection directly checks if the raw API returns think tags.
func TestLive_Qwen35_ThinkTagDetection(t *testing.T) {
	m, err := NewModel(qwenModelName, lmStudioConfig())
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}
	om := m.(*openaiModel)

	// Ask a question that should trigger thinking
	messages := []OpenAIMessage{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is 15 * 17? Show your reasoning."},
	}

	maxTokens := int32(4096)
	chatReq := ChatCompletionRequest{
		Model:     qwenModelName,
		Messages:  messages,
		MaxTokens: &maxTokens,
	}

	t.Log("=== RAW API CALL (checking for <think> tags) ===")
	respData, err := om.makeRequest(context.Background(), chatReq)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}

	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(respData, &chatResp); err != nil {
		t.Fatalf("Failed to parse: %v", err)
	}

	if len(chatResp.Choices) == 0 {
		t.Fatal("No choices")
	}

	rawContent := fmt.Sprintf("%v", chatResp.Choices[0].Message.Content)
	t.Logf("RAW content length: %d", len(rawContent))

	hasThinkTags := strings.Contains(rawContent, "<think>")
	t.Logf("Contains <think> tags: %v", hasThinkTags)

	if hasThinkTags {
		// Show first 500 chars of raw content
		preview := rawContent
		if len(preview) > 500 {
			preview = preview[:500] + "..."
		}
		t.Logf("RAW preview:\n%s", preview)

		// Now test that our adapter strips them
		cleaned := stripThinkTags(rawContent)
		t.Logf("CLEANED length: %d (removed %d chars)", len(cleaned), len(rawContent)-len(cleaned))
		t.Logf("CLEANED:\n%s", cleaned)

		if strings.Contains(cleaned, "<think>") {
			t.Error("stripThinkTags FAILED to remove think tags!")
		} else {
			t.Log("stripThinkTags correctly removed think tags")
		}
	} else {
		t.Log("No <think> tags in raw response (thinking may be disabled in LM Studio)")
		t.Logf("Content:\n%s", rawContent)
	}
}

// TestLive_Qwen35_ParallelToolCalls tests that the model can call multiple tools at once.
func TestLive_Qwen35_ParallelToolCalls(t *testing.T) {
	m, err := NewModel(qwenModelName, lmStudioConfig())
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	tools := map[string]any{
		"get_weather": map[string]any{
			"description": "Get the current weather for a city",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type":        "string",
						"description": "City name",
					},
				},
				"required": []string{"location"},
			},
		},
		"get_time": map[string]any{
			"description": "Get the current time in a timezone",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"timezone": map[string]any{
						"type":        "string",
						"description": "Timezone like Europe/Paris",
					},
				},
				"required": []string{"timezone"},
			},
		},
	}

	maxTokens := int32(4096)
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What's the weather AND current time in London?", "user"),
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: maxTokens,
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("You are a helpful assistant. Use ALL available tools when relevant. Call multiple tools in parallel when possible."),
				},
			},
		},
		Tools: tools,
	}

	t.Log("=== PARALLEL TOOL CALLS (sync) ===")
	for resp, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("Error: %v", err)
		}
		logResponse(t, "PARALLEL", resp)

		funcCallCount := 0
		for _, part := range resp.Content.Parts {
			if part.FunctionCall != nil {
				funcCallCount++
				t.Logf("  Tool call #%d: %s(id=%s, args=%v)",
					funcCallCount, part.FunctionCall.Name, part.FunctionCall.ID, part.FunctionCall.Args)
			}
		}
		t.Logf("Total tool calls: %d", funcCallCount)
		if funcCallCount >= 2 {
			t.Log("Model returned parallel tool calls!")
		} else if funcCallCount == 1 {
			t.Log("Model returned single tool call (may call second tool in next turn)")
		} else {
			t.Log("WARNING: No tool calls returned")
		}
	}

	t.Log("\n=== PARALLEL TOOL CALLS (streaming) ===")
	for resp, err := range m.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("Stream error: %v", err)
		}
		if resp.TurnComplete {
			logResponse(t, "PARALLEL-STREAM", resp)
			funcCallCount := 0
			for _, part := range resp.Content.Parts {
				if part.FunctionCall != nil {
					funcCallCount++
					t.Logf("  Stream tool call #%d: %s(id=%s, args=%v)",
						funcCallCount, part.FunctionCall.Name, part.FunctionCall.ID, part.FunctionCall.Args)
				}
			}
			t.Logf("Stream total tool calls: %d", funcCallCount)
		}
	}
}

// TestLive_Qwen35_NoParamsTool tests a tool with no parameters.
func TestLive_Qwen35_NoParamsTool(t *testing.T) {
	m, err := NewModel(qwenModelName, lmStudioConfig())
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	tools := map[string]any{
		"get_current_date": map[string]any{
			"description": "Get today's date. Takes no parameters.",
			"parameters": map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
	}

	maxTokens := int32(4096)
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What is today's date?", "user"),
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens: maxTokens,
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("You must use the get_current_date tool to answer date questions."),
				},
			},
		},
		Tools: tools,
	}

	t.Log("=== NO-PARAMS TOOL ===")
	for resp, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("Error: %v", err)
		}
		logResponse(t, "NO-PARAMS", resp)

		for _, part := range resp.Content.Parts {
			if part.FunctionCall != nil {
				t.Logf("Tool call: %s(args=%v, id=%s)", part.FunctionCall.Name, part.FunctionCall.Args, part.FunctionCall.ID)
				if part.FunctionCall.Name != "get_current_date" {
					t.Errorf("Expected get_current_date, got %s", part.FunctionCall.Name)
				}
			}
		}
	}
}

// TestLive_Qwen35_JSONMode tests JSON structured output.
func TestLive_Qwen35_JSONMode(t *testing.T) {
	m, err := NewModel(qwenModelName, lmStudioConfig())
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	maxTokens := int32(4096)
	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("List 3 European capitals with their countries.", "user"),
		},
		Config: &genai.GenerateContentConfig{
			MaxOutputTokens:  maxTokens,
			ResponseMIMEType: "application/json",
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("You are a helpful assistant. Return data as JSON array of objects with 'city' and 'country' fields."),
				},
			},
		},
	}

	t.Log("=== JSON MODE ===")
	for resp, err := range m.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("Error: %v", err)
		}
		logResponse(t, "JSON", resp)

		for _, part := range resp.Content.Parts {
			if part.Text != "" {
				// Check think tags stripped
				if strings.Contains(part.Text, "<think>") {
					t.Error("LEAKED: <think> tags in JSON response!")
				}
				// Try to parse as JSON
				var parsed any
				if err := json.Unmarshal([]byte(part.Text), &parsed); err != nil {
					t.Errorf("Response is NOT valid JSON: %v\nText: %s", err, part.Text)
				} else {
					pretty, _ := json.MarshalIndent(parsed, "", "  ")
					t.Logf("Valid JSON response:\n%s", string(pretty))
				}
			}
		}
	}
}

func logResponse(t *testing.T, prefix string, resp *model.LLMResponse) {
	t.Helper()
	if resp == nil {
		t.Logf("[%s] nil response", prefix)
		return
	}
	t.Logf("[%s] TurnComplete=%v Partial=%v FinishReason=%v",
		prefix, resp.TurnComplete, resp.Partial, resp.FinishReason)

	if resp.Content != nil {
		t.Logf("[%s] Role=%s Parts=%d", prefix, resp.Content.Role, len(resp.Content.Parts))
		for i, part := range resp.Content.Parts {
			if part.Text != "" {
				text := part.Text
				if len(text) > 200 {
					text = text[:200] + "..."
				}
				t.Logf("[%s]   Part[%d] Text: %q", prefix, i, text)
			}
			if part.FunctionCall != nil {
				t.Logf("[%s]   Part[%d] FunctionCall: %s(id=%s, args=%v)",
					prefix, i, part.FunctionCall.Name, part.FunctionCall.ID, part.FunctionCall.Args)
			}
		}
	}

	if resp.UsageMetadata != nil {
		t.Logf("[%s] Tokens: prompt=%d completion=%d total=%d",
			prefix,
			resp.UsageMetadata.PromptTokenCount,
			resp.UsageMetadata.CandidatesTokenCount,
			resp.UsageMetadata.TotalTokenCount)
	}
}
