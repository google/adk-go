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
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// Test message conversion roundtrip
func TestMessageConversionRoundtrip(t *testing.T) {
	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	tests := []struct {
		name    string
		content *genai.Content
	}{
		{
			name:    "simple text",
			content: genai.NewContentFromText("Hello world", "user"),
		},
		{
			name: "function call",
			content: genai.NewContentFromFunctionCall("get_weather", map[string]any{
				"location": "London",
			}, "model"),
		},
		{
			name: "function response",
			content: &genai.Content{
				Role: "function",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:   "call_test123", // Required ID
							Name: "get_weather",
							Response: map[string]any{
								"temperature": "20C",
								"condition":   "sunny",
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Convert to OpenAI format
			msgs, err := om.convertContent(tt.content)
			if err != nil {
				t.Fatalf("Failed to convert content: %v", err)
			}

			if len(msgs) == 0 {
				t.Fatal("No messages produced")
			}

			// For simple verification, just ensure we got messages
			for _, msg := range msgs {
				if msg.Role == "" {
					t.Error("Message role should not be empty")
				}
			}
		})
	}
}

// TestStripThinkTags tests removal of <think> blocks from reasoning model output (Qwen 3.5, QwQ).
func TestStripThinkTags(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "no think tags",
			input:    "Hello world",
			expected: "Hello world",
		},
		{
			name:     "simple think block",
			input:    "<think>\nLet me reason about this...\n</think>\n\nThe answer is 42.",
			expected: "The answer is 42.",
		},
		{
			name:     "think block with multiline reasoning",
			input:    "<think>\nStep 1: analyze\nStep 2: compute\nStep 3: verify\n</think>\n\nResult: correct.",
			expected: "Result: correct.",
		},
		{
			name:     "unclosed think block (truncated output)",
			input:    "<think>\nI'm still thinking about this and the output was cut off by max_tok",
			expected: "",
		},
		{
			name:     "text before think block",
			input:    "Sure! <think>\nreasoning here\n</think>\n\nHere is my answer.",
			expected: "Sure! Here is my answer.",
		},
		{
			name:     "only think block, no answer",
			input:    "<think>\njust reasoning\n</think>",
			expected: "",
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "think keyword but not a tag",
			input:    "I think this is correct",
			expected: "I think this is correct",
		},
		{
			name:     "orphaned close tag after tool call",
			input:    "</think>\n\nHere is the answer based on tool results.",
			expected: "Here is the answer based on tool results.",
		},
		{
			name:     "orphaned close tag with leading whitespace",
			input:    "  </think>  \n\nAnswer",
			expected: "Answer",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := stripThinkTags(tt.input)
			if result != tt.expected {
				t.Errorf("stripThinkTags(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

// TestConvertToLLMResponse_StripsThinkTags verifies that <think> tags are stripped from
// sync (non-streaming) model responses.
func TestConvertToLLMResponse_StripsThinkTags(t *testing.T) {
	m := &openaiModel{name: "qwen3.5-9b"}

	msg := &OpenAIMessage{
		Role:    "assistant",
		Content: "<think>\nLet me analyze this.\n</think>\n\nThe capital of France is Paris.",
	}

	resp, err := m.convertToLLMResponse(msg, nil, nil)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(resp.Content.Parts) != 1 {
		t.Fatalf("Expected 1 part, got %d", len(resp.Content.Parts))
	}

	text := resp.Content.Parts[0].Text
	if text != "The capital of France is Paris." {
		t.Errorf("Expected clean text, got %q", text)
	}
}

// TestConvertContent_StripsThinkFromHistory verifies that <think> tags are stripped
// from historical assistant messages (Qwen 3.5 requirement).
func TestConvertContent_StripsThinkFromHistory(t *testing.T) {
	m := &openaiModel{name: "qwen3.5-9b"}

	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			genai.NewPartFromText("<think>\nSome reasoning\n</think>\n\nClean answer here."),
		},
	}

	msgs, err := m.convertContent(content)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(msgs) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(msgs))
	}

	text, ok := msgs[0].Content.(string)
	if !ok {
		t.Fatalf("Expected string content, got %T", msgs[0].Content)
	}

	if text != "Clean answer here." {
		t.Errorf("Expected think tags stripped from history, got %q", text)
	}
}

// TestConvertContent_UserMessageUnchanged verifies that user messages are NOT stripped.
func TestConvertContent_UserMessageUnchanged(t *testing.T) {
	m := &openaiModel{name: "qwen3.5-9b"}

	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			genai.NewPartFromText("Can you <think> about this?"),
		},
	}

	msgs, err := m.convertContent(content)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	text, ok := msgs[0].Content.(string)
	if !ok {
		t.Fatalf("Expected string content, got %T", msgs[0].Content)
	}

	if text != "Can you <think> about this?" {
		t.Errorf("User message should NOT be modified, got %q", text)
	}
}

// TestConvertToOpenAIMessages_Stateless tests the stateless conversion with system instruction,
// JSON mode, and multi-turn contents.
func TestConvertToOpenAIMessages_Stateless(t *testing.T) {
	cfg := &Config{
		BaseURL: "http://localhost:1234/v1",
	}

	m, err := NewModel("test-model", cfg)
	if err != nil {
		t.Fatalf("Failed to create model: %v", err)
	}

	om := m.(*openaiModel)

	t.Run("system instruction", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("Hello", "user"),
			},
			Config: &genai.GenerateContentConfig{
				SystemInstruction: &genai.Content{
					Parts: []*genai.Part{
						genai.NewPartFromText("You are a helpful assistant."),
					},
				},
			},
		}

		msgs, err := om.convertToOpenAIMessages(req)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(msgs) != 2 {
			t.Fatalf("Expected 2 messages (system + user), got %d", len(msgs))
		}

		if msgs[0].Role != "system" {
			t.Errorf("Expected first message role 'system', got %q", msgs[0].Role)
		}
		if msgs[0].Content != "You are a helpful assistant." {
			t.Errorf("Unexpected system content: %v", msgs[0].Content)
		}
		if msgs[1].Role != "user" {
			t.Errorf("Expected second message role 'user', got %q", msgs[1].Role)
		}
	})

	t.Run("JSON mode adds instruction", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("Give me data", "user"),
			},
			Config: &genai.GenerateContentConfig{
				ResponseMIMEType: "application/json",
			},
		}

		msgs, err := om.convertToOpenAIMessages(req)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(msgs) != 2 {
			t.Fatalf("Expected 2 messages (system JSON + user), got %d", len(msgs))
		}

		if msgs[0].Role != "system" {
			t.Errorf("Expected system message for JSON mode, got %q", msgs[0].Role)
		}
		sysContent, ok := msgs[0].Content.(string)
		if !ok || sysContent != "You must respond with valid JSON." {
			t.Errorf("Unexpected JSON mode system content: %v", msgs[0].Content)
		}
	})

	t.Run("multi-turn contents", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("Hello", "user"),
				genai.NewContentFromText("Hi there!", "model"),
				genai.NewContentFromText("How are you?", "user"),
			},
			Config: &genai.GenerateContentConfig{},
		}

		msgs, err := om.convertToOpenAIMessages(req)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(msgs) != 3 {
			t.Fatalf("Expected 3 messages, got %d", len(msgs))
		}

		expectedRoles := []string{"user", "assistant", "user"}
		for i, role := range expectedRoles {
			if msgs[i].Role != role {
				t.Errorf("Message %d: expected role %q, got %q", i, role, msgs[i].Role)
			}
		}
	})

	t.Run("no contents", func(t *testing.T) {
		req := &model.LLMRequest{
			Contents: nil,
			Config:   &genai.GenerateContentConfig{},
		}

		msgs, err := om.convertToOpenAIMessages(req)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if len(msgs) != 0 {
			t.Errorf("Expected 0 messages for nil contents, got %d", len(msgs))
		}
	})
}
